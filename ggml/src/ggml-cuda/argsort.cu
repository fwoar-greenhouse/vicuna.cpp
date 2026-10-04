#include "argsort.cuh"

// Bitonic sort implementation
template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

// One compare-exchange of the bitonic network at (k, j) for column col.
template<ggml_sort_order order>
static inline __device__ void bitonic_step(const float * x_row, int * dst_row, const int ncols, const int col, const int k, const int j) {
    const int ixj = col ^ j;
    if (ixj <= col) {
        return;
    }
    if ((col & k) == 0) {
        if (dst_row[col] >= ncols ||
            (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                x_row[dst_row[col]] < x_row[dst_row[ixj]]))
        ) {
            ggml_cuda_swap(dst_row[col], dst_row[ixj]);
        }
    } else {
        if (dst_row[ixj] >= ncols ||
            (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                x_row[dst_row[col]] > x_row[dst_row[ixj]]))
        ) {
            ggml_cuda_swap(dst_row[col], dst_row[ixj]);
        }
    }
}

// Bitonic sort of one row per block. Each thread owns the columns
// threadIdx.x + i * blockDim.x, so rows wider than the block (up to the
// shared memory limit) sort with several columns per thread. Every
// (k, j) stage runs all owned columns before the barrier; a pair
// (col, col ^ j) is exchanged by the owner of its lower index only.
template<ggml_sort_order order>
static __global__ void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad) {
    const int row = blockIdx.x;

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
        dst_row[col] = col;
    }

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
                bitonic_step<order>(x_row, dst_row, ncols, col, k, j);
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        dst[row * ncols + col] = dst_row[col];
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    // one thread per column up to the block limit, several columns per
    // thread beyond it; shared memory is the remaining bound
    const dim3 block_dims(ncols_pad < CUDA_ARGSORT_BLOCK_SIZE ? ncols_pad : CUDA_ARGSORT_BLOCK_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_ASC>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else if (order == GGML_SORT_ORDER_DESC) {
        k_argsort_f32_i32<GGML_SORT_ORDER_DESC>
            <<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_cuda_bitonic(src0_d, (int *) dst_d, ncols, nrows, order, stream);
}
