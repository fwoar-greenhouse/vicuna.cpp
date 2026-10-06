#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne01, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Two mat-vecs with few rows and the same src1 in one launch, with epilogues:
// dst_a = op_a(src0_a*src1 + bias_a) * scale_a (bias_a, scale_a: one value per row), dst_b = op_b(src0_b*src1).
bool ggml_cuda_mul_mat_vec_q_pair_supported(const ggml_tensor * mm_a, const ggml_tensor * mm_b);

void ggml_cuda_mul_mat_vec_q_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * mm_b,
        float * dst_a, int64_t stride_col_dst_a, const float * bias_a, const float * scale_a, ggml_unary_op op_a,
        float * dst_b, int64_t stride_col_dst_b, ggml_unary_op op_b);
