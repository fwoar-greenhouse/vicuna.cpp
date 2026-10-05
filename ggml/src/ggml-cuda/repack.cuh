#pragma once

#include "common.cuh"

// Weight repack for MI100 ("S64" layout, see docs/backend/MI100-parity.md, item 1 design).
// Each 2D matrix (each ne2/ne3 slice) is cut into stripes of 64 rows, the last stripe has ne1 % 64 rows.
// A stripe starts at the same byte offset as in the GGUF layout and has the same size.
// A block is split into 16 byte chunks and a rest. In a stripe of r rows and nkb blocks per row,
// chunk c of block kb of row i is at ((c*nkb + kb)*r + i)*16. The rests follow the chunks,
// in groups of G = 16/rest blocks: group g is [row][n_g blocks] with n_g = min(G, nkb - g*G).

#define GGML_CUDA_REPACK_ROWS 64

// Matrices with fewer rows stay in the GGUF layout.
#define GGML_CUDA_REPACK_MIN_ROWS 256

struct ggml_cuda_repack_layout {
    int bs;       // bytes per block
    int nchunk;   // 16 byte chunks per block
    int rest;     // bytes per block that are not in a chunk
    int rest_off; // offset of the rest in the block
};

static constexpr __host__ __device__ ggml_cuda_repack_layout ggml_cuda_repack_get_layout(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return { 18,  1, 2,   0};
        case GGML_TYPE_Q8_0: return { 34,  2, 2,   0};
        case GGML_TYPE_Q4_K: return {144,  9, 0,   0};
        case GGML_TYPE_Q5_K: return {176, 11, 0,   0};
        case GGML_TYPE_Q6_K: return {210, 13, 2, 208};
        case GGML_TYPE_IQ4_XS: return {136, 8, 8,  0};
        default:             return {  0,  0, 0,   0};
    }
}

// True if the type has a repack layout and readers.
static constexpr __host__ __device__ bool ggml_cuda_repack_type_supported(const ggml_type type) {
    return ggml_cuda_repack_get_layout(type).bs != 0;
}

// Byte offset of chunk c of block kb of row i in a stripe of r rows.
static __host__ __device__ __forceinline__ int64_t ggml_cuda_repack_chunk_offset(
        const int c, const int64_t kb, const int i, const int r, const int64_t nkb) {
    return ((c*nkb + kb)*r + i)*16;
}

// Byte offset of the rest of block kb of row i in a stripe of r rows.
static __host__ __device__ __forceinline__ int64_t ggml_cuda_repack_rest_offset(
        const ggml_cuda_repack_layout L, const int64_t kb, const int i, const int r, const int64_t nkb) {
    const int     G  = 16 / L.rest;
    const int64_t g  = kb / G;
    const int64_t ng = min((int64_t) G, nkb - g*G);
    return (int64_t) L.nchunk*nkb*r*16 + (g*G*r + i*ng + (kb - g*G))*L.rest;
}

// get_int_from_table_16 for kvalues_iq4nl with the table in the instructions (no loads)
static __device__ __forceinline__ int2 ggml_cuda_repack_iq4nl_lut(const int q4) {
    constexpr uint32_t t0 = 0xBFAD9881; // -127, -104, -83, -65
    constexpr uint32_t t1 = 0xF6EADDCF; //  -49,  -35, -22, -10
    constexpr uint32_t t2 = 0x26190D01; //    1,   13,  25,  38
    constexpr uint32_t t3 = 0x71594535; //   53,   69,  89, 113
    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;
    const uint32_t v_even_low  = __builtin_amdgcn_perm(t1, t0, q_even & 0x07070707);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(t1, t0, q_odd  & 0x07070707);
    const uint32_t v_even_high = __builtin_amdgcn_perm(t3, t2, q_even & 0x07070707);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(t3, t2, q_odd  & 0x07070707);
    const uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    const uint32_t mask_odd  = 0x03020100 | ((q_odd  & 0x08080808) >> 1);
    return make_int2(__builtin_amdgcn_perm(v_even_high, v_even_low, mask_even), __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd));
}

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft);

// The tensor (or the tensor a view points to) is stored in the repacked layout.
bool ggml_cuda_tensor_is_repacked(const ggml_tensor * t);

// A tensor with this type and shape is repacked when it is put into a repack buffer.
bool ggml_cuda_repack_eligible(const ggml_tensor * t);

// Buffer functions for the repack buffer type, data is in the GGUF layout.
void ggml_cuda_repack_set_tensor(ggml_tensor * t, const void * data, size_t offset, size_t size);
void ggml_cuda_repack_get_tensor(const ggml_tensor * t, void * data, size_t offset, size_t size);

// Ops that can read a repacked src.
bool ggml_cuda_repack_supports_op(const ggml_tensor * op);

// Largest number of columns for the repacked GEMV.
#define GGML_CUDA_REPACK_MMVQ_MAX_COLS 8

// MUL_MAT uses the repacked GEMV up to this many columns, MMQ above (MUL_MAT_ID: up to 8 tokens).
static constexpr int ggml_cuda_repack_mmvq_max_cols(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return 6;
        case GGML_TYPE_Q8_0: return 8;
        case GGML_TYPE_Q4_K: return 8;
        case GGML_TYPE_Q5_K: return 8;
        case GGML_TYPE_Q6_K: return 8;
        case GGML_TYPE_IQ4_XS: return 7;
        default:             return GGML_CUDA_REPACK_MMVQ_MAX_COLS;
    }
}

// The same for a weight: all columns for matrices with few rows (too few MMQ tiles), at most 6 below 16M weights.
static int ggml_cuda_repack_mmvq_max_cols(const ggml_tensor * src0) {
    if (src0->ne[1] < 1024) {
        return GGML_CUDA_REPACK_MMVQ_MAX_COLS;
    }
    const int n = ggml_cuda_repack_mmvq_max_cols(src0->type);
    return src0->ne[0]*src0->ne[1] < 16*1024*1024 ? std::min(n, 6) : n;
}

// GEMV on a repacked src0 (MUL_MAT up to 8 columns, MUL_MAT_ID up to 8 tokens), y is q8_1, arguments as in mmvq.cu.
void ggml_cuda_mul_mat_vec_q_repack(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        int ncols_dst, int stride_col_y, int stride_col_dst,
        int nchannels_y, int nchannels_dst, int stride_channel_y, int stride_channel_dst,
        int nsamples_dst, int64_t stride_sample_y, int64_t stride_sample_dst,
        int ids_stride, cudaStream_t stream);

// Dequantizes a repacked src0 into a contiguous tensor of type dst_type (f16, bf16 or f32) with the same shape.
void ggml_cuda_repack_dequantize(const ggml_tensor * src0, void * dst, ggml_type dst_type, cudaStream_t stream);

// GET_ROWS from a 2D repacked src0 into f32 rows.
void ggml_cuda_repack_get_rows(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, cudaStream_t stream);
