#pragma once

// MMQ x tile loaders for repacked weights (S64 layout, see repack.cuh).
// A tile of I rows is I/64 stripes. Each warp takes one stripe and a quarter of the block,
// lane = row, so every global load reads 64 x 16 contiguous bytes.
// The shared memory tile is the same as for the GGUF layout, so vec_dot does not change.

#include "repack.cuh"

template <bool store>
static __device__ __forceinline__ int4 mmq_ld4(int * __restrict__ xr, int & ir, const char * p) {
    int4 v;
    if constexpr (store) {
        v = make_int4(xr[ir + 0], xr[ir + 1], xr[ir + 2], xr[ir + 3]);
    } else {
        v = *(const int4 *) p;
        xr[ir + 0] = v.x;
        xr[ir + 1] = v.y;
        xr[ir + 2] = v.z;
        xr[ir + 3] = v.w;
    }
    ir += 4;
    return v;
}

template <ggml_type type, int J, bool fallback, bool store> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_repack(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride, int * __restrict__ xr) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr ggml_cuda_repack_layout L = ggml_cuda_repack_get_layout(type);
    // other configs (and the host pass) are never launched
    if constexpr (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT ||
                  warp_size != GGML_CUDA_REPACK_ROWS || nwarps != 4*(I/GGML_CUDA_REPACK_ROWS)) {
        GGML_UNUSED_VARS(x, x_tile, kbx0, i_max, stride, xr, sram_stride, L);
        return;
    } else {
        int ir = 0;

        // kbx0 = first block of the tile's first row + kb, the stride is the number of blocks per row
        const int nkb  = stride;
        const int kb   = kbx0 % nkb;
        const int st   = threadIdx.y / 4;
        const int part = threadIdx.y % 4;

        // rows past the matrix load the last stripe again, their results are not written
        int st_eff = st;
        int r      = GGML_CUDA_REPACK_ROWS;
        if (fallback) {
            st_eff = min(st, i_max / GGML_CUDA_REPACK_ROWS);
            r      = min(GGML_CUDA_REPACK_ROWS, i_max + 1 - st_eff*GGML_CUDA_REPACK_ROWS);
        }
        const int row = fallback ? min((int) threadIdx.x, r - 1) : threadIdx.x;
        const int i   = st*GGML_CUDA_REPACK_ROWS + threadIdx.x;

        const char * sx = x + ((int64_t) (kbx0 - kb) + (int64_t) st_eff*GGML_CUDA_REPACK_ROWS*nkb)*L.bs;
        const auto chunk = [&](const int c) {
            return sx + ggml_cuda_repack_chunk_offset(c, kb, row, r, nkb);
        };

        int * x_qs = (int *) x_tile;

        if constexpr (type == GGML_TYPE_Q4_K) {
            // part p: sub-blocks 2p and 2p+1, their d*sc and -dmin*m
            half2 * x_dm = (half2 *) (x_qs + MMQ_TILE_NE_K*2);
            const int p = part;

            const int4 m  = mmq_ld4<store>(xr, ir, chunk(0));
            const int4 q0 = mmq_ld4<store>(xr, ir, chunk(1 + 2*p));
            const int4 q1 = mmq_ld4<store>(xr, ir, chunk(2 + 2*p));

            if constexpr (store) {
                const int qs[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
                int v[16];
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    v[k + 0] = (qs[k] >> 0) & 0x0F0F0F0F;
                    v[k + 8] = (qs[k] >> 4) & 0x0F0F0F0F;
                }
                int4 * dst = (int4 *) (x_qs + i*sram_stride + 16*p);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    dst[k] = make_int4(v[4*k + 0], v[4*k + 1], v[4*k + 2], v[4*k + 3]);
                }

                const int scales[3] = {m.y, m.z, m.w};
                const half2 dm = (*(const half2 *) &m.x) * make_half2(1.0f, -1.0f);
                const int sc32 = unpack_scales_q45_K(scales, (2*p)/4 + 0);
                const int  m32 = unpack_scales_q45_K(scales, (2*p)/4 + 2);
                const uint8_t * sc8 = (const uint8_t *) &sc32;
                const uint8_t *  m8 = (const uint8_t *)  &m32;
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    const int s = 2*p + l;
                    x_dm[i*sram_stride + s] = dm*make_half2(sc8[s % 4], m8[s % 4]);
                }
            }
        } else if constexpr (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0) {
            // the tile is 8 blocks = one rest group; part p: blocks 2p and 2p+1.
            // Blocks past the end of the row (K % 256 != 0) load the last block again, their y values are 0.
            float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
            const int p  = part;
            int4 q[2][L.nchunk];
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const int kbb = min(kb + 2*p + b, nkb - 1);
#pragma unroll
                for (int c = 0; c < L.nchunk; ++c) {
                    q[b][c] = mmq_ld4<store>(xr, ir, sx + ggml_cuda_repack_chunk_offset(c, kbb, row, r, nkb));
                }
            }
            // the rest group of kb holds min(8, nkb - kb) d values
            const int ng = min(8, nkb - kb);
            const char * rp = sx + ggml_cuda_repack_rest_offset(L, kb, row, r, nkb);
            half d[2];
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const int j = min(2*p + b, ng - 1);
                d[b] = MMQ_LD(*(const half *) (rp + 2*j));
            }

            if constexpr (store) {
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const int * qv = (const int *) q[b];
                    int v[8];
                    if constexpr (type == GGML_TYPE_Q8_0) {
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            v[k] = qv[k];
                        }
                    } else {
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            v[k + 0] = __vsub4((qv[k] >> 0) & 0x0F0F0F0F, 0x08080808);
                            v[k + 4] = __vsub4((qv[k] >> 4) & 0x0F0F0F0F, 0x08080808);
                        }
                    }
                    int4 * dst = (int4 *) (x_qs + i*sram_stride + 8*(2*p + b));
                    dst[0] = make_int4(v[0], v[1], v[2], v[3]);
                    dst[1] = make_int4(v[4], v[5], v[6], v[7]);
                    x_df[i*sram_stride + 2*p + b] = d[b];
                }
            }
        } else if constexpr (type == GGML_TYPE_Q5_K) {
            // part p: sub-blocks 2p and 2p+1, their d*sc and -dmin*m
            half2 * x_dm = (half2 *) (x_qs + MMQ_TILE_NE_K*2);
            const int p = part;

            const int4 m  = mmq_ld4<store>(xr, ir, chunk(0));
            const int4 h0 = mmq_ld4<store>(xr, ir, chunk(1));
            const int4 h1 = mmq_ld4<store>(xr, ir, chunk(2));
            const int4 q0 = mmq_ld4<store>(xr, ir, chunk(3 + 2*p));
            const int4 q1 = mmq_ld4<store>(xr, ir, chunk(4 + 2*p));

            if constexpr (store) {
                const int qs[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
                const int qh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
                int v[16];
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    v[k + 0] = ((qs[k] >> 0) & 0x0F0F0F0F) | (((qh[k] >> (2*p + 0)) << 4) & 0x10101010);
                    v[k + 8] = ((qs[k] >> 4) & 0x0F0F0F0F) | (((qh[k] >> (2*p + 1)) << 4) & 0x10101010);
                }
                int4 * dst = (int4 *) (x_qs + i*sram_stride + 16*p);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    dst[k] = make_int4(v[4*k + 0], v[4*k + 1], v[4*k + 2], v[4*k + 3]);
                }

                const int scales[3] = {m.y, m.z, m.w};
                const half2 dm = (*(const half2 *) &m.x) * make_half2(1.0f, -1.0f);
                const int sc32 = unpack_scales_q45_K(scales, (2*p)/4 + 0);
                const int  m32 = unpack_scales_q45_K(scales, (2*p)/4 + 2);
                const uint8_t * sc8 = (const uint8_t *) &sc32;
                const uint8_t *  m8 = (const uint8_t *)  &m32;
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    const int s = 2*p + l;
                    x_dm[i*sram_stride + s] = dm*make_half2(sc8[s % 4], m8[s % 4]);
                }
            }
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            // part p: values 64p..64p+63 = half n = p/2, low (p even) or high (p odd) nibbles of 64 ql bytes
            float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
            int   * x_sc = (int   *) (x_df + MMQ_TILE_NE_K/QI6_K);
            const int p  = part;
            const int n  = p / 2;
            const int hn = p % 2;

            int4 ql[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                ql[k] = mmq_ld4<store>(xr, ir, chunk(4*n + k));
            }
            const int4 h0 = mmq_ld4<store>(xr, ir, chunk(8 + 2*n));
            const int4 h1 = mmq_ld4<store>(xr, ir, chunk(9 + 2*n));
            const int4 sc = mmq_ld4<store>(xr, ir, chunk(12));
            half d = 0.0f;
            if (p == 0) {
                d = MMQ_LD(*(const half *) (sx + ggml_cuda_repack_rest_offset(L, kb, row, r, nkb)));
            }

            if constexpr (store) {
                const int * qlv = (const int *) ql;
                const int qh[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
                int v[16];
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const int k  = 2*hn + j/8; // 32-value group in the half
                    const int lo = (qlv[j] >> (4*hn)) & 0x0F0F0F0F;
                    const int hi = ((qh[j % 8] >> (2*k)) << 4) & 0x30303030;
                    v[j] = __vsub4(lo | hi, 0x20202020);
                }
                int4 * dst = (int4 *) (x_qs + i*sram_stride + 16*p);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    dst[k] = make_int4(v[4*k + 0], v[4*k + 1], v[4*k + 2], v[4*k + 3]);
                }
                const int scv[4] = {sc.x, sc.y, sc.z, sc.w};
                x_sc[i*sram_stride + p] = scv[p];
                if (p == 0) {
                    x_df[i*sram_stride] = d;
                }
            }
            GGML_UNUSED(d);
        } else {
            static_assert(type == GGML_TYPE_COUNT, "no repacked MMQ loader for this type");
        }
    }
}
