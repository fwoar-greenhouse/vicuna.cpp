#pragma once

#include "common.cuh"

// Weight repack for MI100 ("S64" layout, see docs/backend/MI100-parity.md, item 1 design).
// Each 2D matrix (each ne2/ne3 slice) is cut into stripes of 64 rows, the last stripe has ne1 % 64 rows.
// A stripe starts at the same byte offset as in the GGUF layout and has the same size.
// A block is split into 16 byte chunks and a rest. In a stripe of r rows and nkb blocks per row,
// chunk c of block kb of row i is at ((c*nkb + kb)*r + i)*16. The rests follow the chunks,
// in groups of G = 16/rest blocks: group g is [row][n_g blocks] with n_g = min(G, nkb - g*G).

#define GGML_CUDA_REPACK_ROWS 64

struct ggml_cuda_repack_layout {
    int bs;       // bytes per block
    int nchunk;   // 16 byte chunks per block
    int rest;     // bytes per block that are not in a chunk
    int rest_off; // offset of the rest in the block
};

static constexpr __host__ __device__ ggml_cuda_repack_layout ggml_cuda_repack_get_layout(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q5_K: return {176, 11, 0,   0};
        case GGML_TYPE_Q6_K: return {210, 13, 2, 208};
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

