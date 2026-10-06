#include "argsort.cuh"
#include "top-k.cuh"

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}


// Small k: each thread keeps its KT best (key, index) pairs in registers, sorted (equal keys: lower index first).
// A warp then takes the best of its threads k times (warp max, the winner drops its head).
// Pass 1 writes the k best of each warp, pass 2 merges them in one block per row (warps, then warp 0).

template <int KT>
static __device__ __forceinline__ void top_k_small_insert(uint32_t (&key)[KT], int (&idx)[KT], uint32_t k, int i) {
#pragma unroll
    for (int j = 0; j < KT; ++j) {
        const bool     b  = k > key[j] || (k == key[j] && i < idx[j]);
        const uint32_t tk = key[j];
        const int      ti = idx[j];
        key[j] = b ? k : tk;
        idx[j] = b ? i : ti;
        k = b ? tk : k;
        i = b ? ti : i;
    }
}

template <int mask>
static __device__ __forceinline__ uint32_t top_k_small_warp_max(uint32_t v) {
    const uint32_t o = __float_as_uint(ggml_cuda_shfl_xor64<mask>(__uint_as_float(v)));
    v = o > v ? o : v;
    if constexpr (mask > 1) {
        return top_k_small_warp_max<mask/2>(v);
    }
    return v;
}

// k rounds of warp max over the list heads, the winner drops its head; lane r < k gets the r-th best.
// Equal keys in several lanes go to the lowest lane.
template <int KT>
static __device__ __forceinline__ void top_k_small_warp_merge(uint32_t (&key)[KT], int (&idx)[KT], const int k,
        uint32_t & out_key, int & out_idx) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int lane = threadIdx.x % warp_size;
    for (int r = 0; r < k; ++r) {
        const uint32_t best = top_k_small_warp_max<warp_size/2>(key[0]);
        const int      win  = __ffsll((unsigned long long) __ballot(key[0] == best)) - 1;
        const int      bidx = __shfl(idx[0], win, warp_size);
        if (lane == r) {
            out_key = best;
            out_idx = bidx;
        }
        if (lane == win) {
#pragma unroll
            for (int j = 0; j < KT - 1; ++j) {
                key[j] = key[j + 1];
                idx[j] = idx[j + 1];
            }
            key[KT - 1] = 0;
            idx[KT - 1] = INT_MAX;
        }
    }
}

// the k best of the block: each warp takes its k best, then warp 0 merges them; thread r < k gets the r-th best
template <int KT, int BLOCK_SIZE>
static __device__ __forceinline__ void top_k_small_block_merge(uint32_t (&key)[KT], int (&idx)[KT], const int k,
        uint32_t & out_key, int & out_idx) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = BLOCK_SIZE / warp_size;
    __shared__ uint32_t s_key[nwarps*KT];
    __shared__ int      s_idx[nwarps*KT];

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;

    uint32_t wk = 0;
    int      wi = INT_MAX;
    top_k_small_warp_merge<KT>(key, idx, k, wk, wi);
    if (lane < k) {
        s_key[warp*k + lane] = wk;
        s_idx[warp*k + lane] = wi;
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < KT; ++j) {
        key[j] = 0;
        idx[j] = INT_MAX;
    }
    for (int c = lane; c < nwarps*k; c += warp_size) {
        top_k_small_insert<KT>(key, idx, s_key[c], s_idx[c]);
    }
    top_k_small_warp_merge<KT>(key, idx, k, out_key, out_idx);
}

template <int KT, int BLOCK_SIZE>
static __global__ void top_k_small_pass1(const float * __restrict__ src, uint32_t * __restrict__ cand_key,
        int * __restrict__ cand_idx, const int ncols, const int k) {
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;

    uint32_t key[KT];
    int      idx[KT];
#pragma unroll
    for (int j = 0; j < KT; ++j) {
        key[j] = 0;
        idx[j] = INT_MAX;
    }

    for (int col = blockIdx.x * BLOCK_SIZE + tid; col < ncols; col += gridDim.x * BLOCK_SIZE) {
        const uint32_t kk = top_k_float_to_ordered(row_src[col]);
        if (kk >= key[KT - 1]) {
            top_k_small_insert<KT>(key, idx, kk, col);
        }
    }

    // the k best of each warp, merged in pass 2
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int lane = tid % warp_size;
    uint32_t out_key = 0;
    int      out_idx = INT_MAX;
    top_k_small_warp_merge<KT>(key, idx, k, out_key, out_idx);
    if (lane < k) {
        const size_t o = (((size_t) row * gridDim.x + blockIdx.x) * (BLOCK_SIZE/warp_size) + tid/warp_size) * k + lane;
        cand_key[o] = out_key;
        cand_idx[o] = out_idx;
    }
}

template <int KT, int BLOCK_SIZE>
static __global__ void top_k_small_pass2(const uint32_t * __restrict__ cand_key, const int * __restrict__ cand_idx,
        int * __restrict__ dst, const int ncand, const int k) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    uint32_t key[KT];
    int      idx[KT];
#pragma unroll
    for (int j = 0; j < KT; ++j) {
        key[j] = 0;
        idx[j] = INT_MAX;
    }
    for (int c = tid; c < ncand; c += BLOCK_SIZE) {
        top_k_small_insert<KT>(key, idx, cand_key[(size_t) row * ncand + c], cand_idx[(size_t) row * ncand + c]);
    }

    uint32_t out_key = 0;
    int      out_idx = INT_MAX;
    top_k_small_block_merge<KT, BLOCK_SIZE>(key, idx, k, out_key, out_idx);
    if (tid < k) {
        dst[(size_t) row * k + tid] = out_idx;
    }
}

template <int KT>
static void top_k_small_cuda(ggml_cuda_pool & pool, const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    // about 8 values per thread
    const int nblocks = std::max(1, std::min((ncols + 8*BLOCK_SIZE - 1) / (8*BLOCK_SIZE), 128));

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int ncand     = nblocks * (BLOCK_SIZE/warp_size) * k;

    ggml_cuda_pool_alloc<uint32_t> cand_key(pool, (size_t) nrows * ncand);
    ggml_cuda_pool_alloc<int>      cand_idx(pool, (size_t) nrows * ncand);

    top_k_small_pass1<KT, BLOCK_SIZE><<<dim3(nblocks, nrows), BLOCK_SIZE, 0, stream>>>(src, cand_key.get(), cand_idx.get(), ncols, k);
    top_k_small_pass2<KT, BLOCK_SIZE><<<nrows, BLOCK_SIZE, 0, stream>>>(cand_key.get(), cand_idx.get(), dst, ncand, k);
}


void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
    if (ncols > 1024 && k <= 16 && nrows <= 64) {
        // the list length sets the cost of each insert
        if (k <= 4) {
            top_k_small_cuda<4>(pool, src0_d, dst_d, ncols, nrows, k, stream);
        } else if (k <= 8) {
            top_k_small_cuda<8>(pool, src0_d, dst_d, ncols, nrows, k, stream);
        } else if (k <= 12) {
            top_k_small_cuda<12>(pool, src0_d, dst_d, ncols, nrows, k, stream);
        } else {
            top_k_small_cuda<16>(pool, src0_d, dst_d, ncols, nrows, k, stream);
        }
    } else if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
    }
}
