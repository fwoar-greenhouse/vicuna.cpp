#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CONCAT_CPY_MAX 8

struct concat_cpy_args {
    float * dst[CONCAT_CPY_MAX];
    int     off[CONCAT_CPY_MAX]; // first concat column of the copied window
    int     nb1[CONCAT_CPY_MAX]; // dst stride between sequences, in floats
    int     ne0;                 // window width
    int     n;
};

// CONCAT along dim 0 with few columns, followed by CPYs of column windows of the result (conv state update)
bool ggml_cuda_concat_cpy_supported(const ggml_tensor * cat, const ggml_tensor * const * cpys, int n_cpy);

void ggml_cuda_op_concat_cpy(ggml_backend_cuda_context & ctx, ggml_tensor * cat, ggml_tensor * const * cpys, int n_cpy);
