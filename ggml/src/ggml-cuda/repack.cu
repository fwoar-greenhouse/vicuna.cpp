#include "repack.cuh"
#include "dequantize.cuh"

#include <algorithm>
#include <cstring>
#include <type_traits>
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
        case GGML_OP_MUL_MAT_ID:
            return op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            return src0->ne[2] == 1 && src0->ne[3] == 1 && op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[1]) &&
                op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------------------------
// Dequantization and GET_ROWS on repacked tensors.

// Loads the GGUF bytes of block kb of row i of a stripe into blk.
template <ggml_type type>
static __device__ __forceinline__ void repack_load_block(const char * sx, const int64_t kb, const int i, const int r, const int64_t nkb, int * blk) {
    constexpr ggml_cuda_repack_layout L = ggml_cuda_repack_get_layout(type);
    static_assert(L.rest_off == 16*L.nchunk || L.rest == 0, "rest must follow the chunks");
#pragma unroll
    for (int c = 0; c < L.nchunk; ++c) {
        const int4 v = *(const int4 *) (sx + ggml_cuda_repack_chunk_offset(c, kb, i, r, nkb));
        blk[4*c + 0] = v.x;
        blk[4*c + 1] = v.y;
        blk[4*c + 2] = v.z;
        blk[4*c + 3] = v.w;
    }
    if constexpr (L.rest == 2) {
        blk[4*L.nchunk] = *(const uint16_t *) (sx + ggml_cuda_repack_rest_offset(L, kb, i, r, nkb));
    }
}

// Calls out(j, v) for the 8 values 8j..8j+7 of a block, with the same arithmetic as dequantize.cuh.
template <ggml_type type, typename F>
static __device__ __forceinline__ void repack_dequant_block(const int * blk, const F & out) {
    if constexpr (type == GGML_TYPE_Q5_K) {
        const block_q5_K * b = (const block_q5_K *) blk;
        const float dall = __low2half(b->dm);
        const float dmin = __high2half(b->dm);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint8_t sc, m;
            get_scale_min_k4(s, b->scales, sc, m);
            const float d1 = dall * sc;
            const float m1 = dmin * m;
            const uint8_t * ql = b->qs + 32*(s/2);
            const uint8_t   hm = 1 << s;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v[8];
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int k = 8*j + l;
                    const int q = (s % 2 == 0 ? ql[k] & 0xF : ql[k] >> 4) + (b->qh[k] & hm ? 16 : 0);
                    v[l] = d1 * q - m1;
                }
                out(4*s + j, v);
            }
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const block_q6_K * b = (const block_q6_K *) blk;
        const float d = b->d;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint8_t * ql = b->ql + 64*n + 32*(k % 2);
                const uint8_t * qh = b->qh + 32*n;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float v[8];
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const int e  = 8*j + l;
                        const int8_t * sc = b->scales + 8*n + e/16 + 2*k;
                        const int q = (k < 2 ? ql[e] & 0xF : ql[e] >> 4) | (((qh[e] >> (2*k)) & 3) << 4);
                        v[l] = d * sc[0] * ((int8_t) q - 32);
                    }
                    out(16*n + 4*k + j, v);
                }
            }
        }
    } else {
        static_assert(type == GGML_TYPE_COUNT, "no repacked dequantization for this type");
    }
}

template <typename dst_t>
static __device__ __forceinline__ void repack_store8(dst_t * dst, const float * v) {
    if constexpr (std::is_same_v<dst_t, half>) {
        half2 h[4];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            h[l] = make_half2(v[2*l], v[2*l + 1]);
        }
        *(int4 *) dst = *(const int4 *) h;
    } else {
#pragma unroll
        for (int l = 0; l < 8; l += 4) {
            *(float4 *) (dst + l) = make_float4(v[l], v[l + 1], v[l + 2], v[l + 3]);
        }
    }
}

// One wave per stripe and block column, lane = row. dst is contiguous [ne3][ne2][ne1][ne0].
template <ggml_type type, typename dst_t>
static __global__ void k_dequant_repack(const char * __restrict__ x, dst_t * __restrict__ dst, const int64_t nkb, const int64_t ne1,
        const int64_t ne2, const int64_t nb2, const int64_t nb3) {
    constexpr ggml_cuda_repack_layout L = ggml_cuda_repack_get_layout(type);
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const int64_t s0 = (int64_t) blockIdx.x*GGML_CUDA_REPACK_ROWS;
    const int     r  = min((int64_t) GGML_CUDA_REPACK_ROWS, ne1 - s0);
    const int     i  = threadIdx.x;
    const int64_t kb = (int64_t) blockIdx.y*blockDim.y + threadIdx.y;
    if (i >= r || kb >= nkb) {
        return;
    }
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const char * sx = x + i2*nb2 + i3*nb3 + s0*nkb*L.bs;
    dst_t * drow = dst + ((int64_t) blockIdx.z*ne1 + s0 + i)*nkb*qk;
    {
        int blk[(L.bs + 3)/4];
        repack_load_block<type>(sx, kb, i, r, nkb, blk);
        repack_dequant_block<type>(blk, [&](const int j, const float * v) {
            repack_store8(drow + kb*qk + 8*j, v);
        });
    }
}

template <typename dst_t>
static void ggml_cuda_repack_dequantize_t(const ggml_tensor * src0, dst_t * dst, cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_tensor_is_repacked(src0));
    const int64_t nkb  = src0->ne[0] / ggml_blck_size(src0->type);
    const dim3 block(GGML_CUDA_REPACK_ROWS, 4, 1);
    const dim3 grid((src0->ne[1] + GGML_CUDA_REPACK_ROWS - 1) / GGML_CUDA_REPACK_ROWS, (nkb + block.y - 1) / block.y, src0->ne[2]*src0->ne[3]);
    switch (src0->type) {
        case GGML_TYPE_Q5_K:
            k_dequant_repack<GGML_TYPE_Q5_K><<<grid, block, 0, stream>>>((const char *) src0->data, dst, nkb, src0->ne[1], src0->ne[2], src0->nb[2], src0->nb[3]);
            break;
        case GGML_TYPE_Q6_K:
            k_dequant_repack<GGML_TYPE_Q6_K><<<grid, block, 0, stream>>>((const char *) src0->data, dst, nkb, src0->ne[1], src0->ne[2], src0->nb[2], src0->nb[3]);
            break;
        default:
            GGML_ABORT("unsupported repack type %s", ggml_type_name(src0->type));
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_repack_dequantize_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    ggml_cuda_repack_dequantize_t(src0, dst, stream);
}

// One block of 64 threads per row to get, thread t takes the block columns t, t+64, ...
template <ggml_type type>
static __global__ void k_get_rows_repack(const char * __restrict__ x, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int64_t nkb, const int64_t ne1, const int64_t stride_ids, const int64_t stride_dst) {
    constexpr ggml_cuda_repack_layout L = ggml_cuda_repack_get_layout(type);
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const int64_t row = ids[blockIdx.x*stride_ids];
    const int64_t s0  = row - row % GGML_CUDA_REPACK_ROWS;
    const int     r   = min((int64_t) GGML_CUDA_REPACK_ROWS, ne1 - s0);
    const char * sx = x + s0*nkb*L.bs;
    float * drow = dst + blockIdx.x*stride_dst;
    for (int64_t kb = threadIdx.x; kb < nkb; kb += blockDim.x) {
        int blk[(L.bs + 3)/4];
        repack_load_block<type>(sx, kb, row - s0, r, nkb, blk);
        repack_dequant_block<type>(blk, [&](const int j, const float * v) {
            repack_store8(drow + kb*qk + 8*j, v);
        });
    }
}

void ggml_cuda_repack_get_rows(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_tensor_is_repacked(src0) && src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(dst));
    const int64_t nkb  = src0->ne[0] / ggml_blck_size(src0->type);
    const int64_t nids = ggml_nelements(src1);
    GGML_ASSERT(ggml_is_contiguous(src1) && ggml_nrows(dst) == nids && dst->nb[2] == dst->nb[1]*dst->ne[1] && dst->nb[3] == dst->nb[2]*dst->ne[2]);
    const int64_t stride_dst = dst->nb[1] / sizeof(float);
    switch (src0->type) {
        case GGML_TYPE_Q5_K:
            k_get_rows_repack<GGML_TYPE_Q5_K><<<nids, 64, 0, stream>>>((const char *) src0->data, (const int32_t *) src1->data,
                (float *) dst->data, nkb, src0->ne[1], 1, stride_dst);
            break;
        case GGML_TYPE_Q6_K:
            k_get_rows_repack<GGML_TYPE_Q6_K><<<nids, 64, 0, stream>>>((const char *) src0->data, (const int32_t *) src1->data,
                (float *) dst->data, nkb, src0->ne[1], 1, stride_dst);
            break;
        default:
            GGML_ABORT("unsupported repack type %s", ggml_type_name(src0->type));
    }
    CUDA_CHECK(cudaGetLastError());
}
