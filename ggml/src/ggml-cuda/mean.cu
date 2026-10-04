#include "mean.cuh"
#include "reduce_rows.cuh"


template <typename T> __global__ void divide_by_count(T * result, size_t count) {
    *result /= static_cast<T>(count);
}

void ggml_cuda_op_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

// Special case for reducing vectors

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    // Heuristic for block size selection to optimize occupancy.
    // See discussion in: https://github.com/ggml-org/llama.cpp/pull/15132
    dim3 block_dims;
    if ((nrows / nsm) < 2) {
        block_dims = dim3(512, 1, 1);
    } else {
        block_dims = dim3(ncols < 1024 ? 32 : 128, 1, 1);
    }
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    if (ggml_is_contiguous(src0)) {
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/true>, launch_params, src0_d, dst_d, ncols);
        return;
    }

    const char * src0_d_bytes = (const char *) src0->data;
    ggml_cuda_kernel_launch(reduce_rows_f32_strided</*norm=*/true>, launch_params, src0_d_bytes, dst_d, ncols,
            src0->ne[1], src0->ne[2], src0->nb[1], src0->nb[2], src0->nb[3]);
}
