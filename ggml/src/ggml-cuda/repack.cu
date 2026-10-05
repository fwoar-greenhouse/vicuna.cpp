#include "repack.cuh"

#include <algorithm>
#include <cstring>
#include <vector>

bool ggml_cuda_repack_eligible(const ggml_tensor * t) {
    return t->view_src == nullptr && ggml_cuda_repack_type_supported(t->type) && ggml_is_contiguous(t) &&
        t->ne[0] % ggml_blck_size(t->type) == 0;
}

bool ggml_cuda_tensor_is_repacked(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    return base->buffer && ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_get_type(base->buffer)) && ggml_cuda_repack_eligible(base);
}

// One thread per block. Moves blocks [b0, b0 + nblocks) between a GGUF layout copy (orig, starts at block b0)
// and the repacked tensor (rp, whole tensor).
template <bool to_repacked>
static __global__ void k_repack(char * __restrict__ orig, char * __restrict__ rp, const ggml_cuda_repack_layout L,
        const int64_t nkb, const int64_t ne1, const int64_t b0, const int64_t nblocks) {
    const int64_t ib = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t b  = b0 + ib;
    const int64_t g  = b / nkb;       // row in the whole tensor
    const int64_t kb = b - g*nkb;
    const int64_t rs = g % ne1;       // row in the 2D slice
    const int64_t s0 = g - rs + (rs / GGML_CUDA_REPACK_ROWS)*GGML_CUDA_REPACK_ROWS; // first row of the stripe
    const int     i  = rs % GGML_CUDA_REPACK_ROWS;
    const int     r  = min((int64_t) GGML_CUDA_REPACK_ROWS, ne1 - (rs - i));

    char * sp = rp + s0*nkb*L.bs;
    uint16_t * ob = (uint16_t *) (orig + ib*L.bs);

    // halfword j of the block body is halfword j of the block, skipping the rest
    const int ro = L.rest_off/2;
    const int rn = L.rest/2;
    for (int c = 0; c < L.nchunk; ++c) {
        uint16_t * cp = (uint16_t *) (sp + ggml_cuda_repack_chunk_offset(c, kb, i, r, nkb));
        uint16_t v[8];
        if (to_repacked) {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int j = 8*c + k;
                v[k] = ob[j < ro ? j : j + rn];
            }
            *(uint4 *) cp = *(const uint4 *) v;
        } else {
            *(uint4 *) v = *(const uint4 *) cp;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int j = 8*c + k;
                ob[j < ro ? j : j + rn] = v[k];
            }
        }
    }
    if (L.rest > 0) {
        uint16_t * rest = (uint16_t *) (sp + ggml_cuda_repack_rest_offset(L, kb, i, r, nkb));
        for (int k = 0; k < rn; ++k) {
            if (to_repacked) {
                rest[k] = ob[ro + k];
            } else {
                ob[ro + k] = rest[k];
            }
        }
    }
}

// Moves the blocks [b0, b1) between host memory in the GGUF layout and the repacked tensor.
static void ggml_cuda_repack_blocks(const ggml_tensor * t, void * host, const int64_t b0, const int64_t b1, const bool to_repacked) {
    const ggml_cuda_repack_layout L = ggml_cuda_repack_get_layout(t->type);
    const int64_t nkb = t->ne[0] / ggml_blck_size(t->type);
    GGML_ASSERT((size_t) L.bs == ggml_type_size(t->type));

    const int64_t max_blocks = std::max<int64_t>(1, (32 << 20) / L.bs);
    const int64_t nscratch   = std::min(b1 - b0, max_blocks);
    char * scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, nscratch*L.bs));

    cudaStream_t stream = cudaStreamPerThread;
    for (int64_t c0 = b0; c0 < b1; c0 += nscratch) {
        const int64_t n = std::min(nscratch, b1 - c0);
        char * h = (char *) host + (c0 - b0)*L.bs;
        const int nth = 256;
        const int64_t nbl = (n + nth - 1) / nth;
        if (to_repacked) {
            CUDA_CHECK(cudaMemcpyAsync(scratch, h, n*L.bs, cudaMemcpyHostToDevice, stream));
            k_repack<true><<<nbl, nth, 0, stream>>>(scratch, (char *) t->data, L, nkb, t->ne[1], c0, n);
            CUDA_CHECK(cudaGetLastError());
        } else {
            k_repack<false><<<nbl, nth, 0, stream>>>(scratch, (char *) t->data, L, nkb, t->ne[1], c0, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h, scratch, n*L.bs, cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaFree(scratch));
}

// The base tensor and the offset relative to its data.
static const ggml_tensor * ggml_cuda_repack_base(const ggml_tensor * t, size_t & offset) {
    if (t->view_src) {
        offset += (const char *) t->data - (const char *) t->view_src->data;
        return t->view_src;
    }
    return t;
}

void ggml_cuda_repack_set_tensor(ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const ggml_tensor * t = ggml_cuda_repack_base(tensor, offset);
    GGML_ASSERT(ggml_cuda_repack_eligible(t));
    GGML_ASSERT(offset + size <= ggml_nbytes(t));
    if (size == 0) {
        return;
    }
    const size_t  bs = ggml_type_size(t->type);
    const int64_t b0 = offset / bs;
    const int64_t b1 = (offset + size + bs - 1) / bs;
    if (b0*bs == offset && b1*bs == offset + size) {
        ggml_cuda_repack_blocks(t, (void *) data, b0, b1, true);
        return;
    }
    // partial blocks: read, patch, write
    std::vector<char> tmp((b1 - b0)*bs);
    ggml_cuda_repack_blocks(t, tmp.data(), b0, b1, false);
    memcpy(tmp.data() + (offset - b0*bs), data, size);
    ggml_cuda_repack_blocks(t, tmp.data(), b0, b1, true);
}

void ggml_cuda_repack_get_tensor(const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const ggml_tensor * t = ggml_cuda_repack_base(tensor, offset);
    GGML_ASSERT(ggml_cuda_repack_eligible(t));
    GGML_ASSERT(offset + size <= ggml_nbytes(t));
    if (size == 0) {
        return;
    }
    const size_t  bs = ggml_type_size(t->type);
    const int64_t b0 = offset / bs;
    const int64_t b1 = (offset + size + bs - 1) / bs;
    if (b0*bs == offset && b1*bs == offset + size) {
        ggml_cuda_repack_blocks(t, data, b0, b1, false);
        return;
    }
    std::vector<char> tmp((b1 - b0)*bs);
    ggml_cuda_repack_blocks(t, tmp.data(), b0, b1, false);
    memcpy(data, tmp.data() + (offset - b0*bs), size);
}

// A view of a repacked tensor can be read if it is a range of whole stripes of each matrix (the last one may end early).
static bool ggml_cuda_repack_view_ok(const ggml_tensor * t) {
    const ggml_tensor * base = t->view_src;
    if (!base) {
        return true;
    }
    if (t->type != base->type || t->ne[0] != base->ne[0] || t->nb[1] != base->nb[1] || t->nb[2] != base->nb[2] || t->nb[3] != base->nb[3]) {
        return false;
    }
    const size_t offs = (const char *) t->data - (const char *) base->data;
    if (offs % base->nb[1] != 0 || offs >= base->nb[2]) {
        return false;
    }
    const int64_t r0 = offs / base->nb[1];
    const int64_t r1 = r0 + t->ne[1];
    return r0 % GGML_CUDA_REPACK_ROWS == 0 && r1 <= base->ne[1] && (r1 % GGML_CUDA_REPACK_ROWS == 0 || r1 == base->ne[1]) &&
        t->ne[2] <= base->ne[2] && t->ne[3] <= base->ne[3];
}

bool ggml_cuda_repack_supports_op(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    if (!src0 || !ggml_cuda_tensor_is_repacked(src0) || !ggml_cuda_repack_view_ok(src0)) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && ggml_cuda_tensor_is_repacked(op->src[i])) {
            return false;
        }
    }
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->ne[1] <= GGML_CUDA_REPACK_MMVQ_MAX_COLS;
        case GGML_OP_MUL_MAT_ID:
            return op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->ne[2] <= GGML_CUDA_REPACK_MMVQ_MAX_COLS;
        default:
            return false;
    }
}

