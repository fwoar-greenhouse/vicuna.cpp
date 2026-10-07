#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// FlashAttention for large batches on CDNA (MFMA, wave64), head size 256, K/V as f16, q8_0 or q4_0.
// A CUDA block of 8 warps works on 128 Q columns (128/ncols2 Q rows x ncols2 Q heads that use the same K/V head) and steps over K/V in tiles of 64 rows.
// Warp w works on the Q columns 16*w..+15 and on all KV rows of a tile, as two halves of 32 rows.
// Shared memory holds one K tile (row-major) and one V tile (transposed, KV rows contiguous) as f16, with an XOR swizzle of the 16 byte granules.
// Each tile has two phases that end with a barrier: KQ reads the K tile while the V tile is written,
//     VKQ reads the V tile while the next K tile is written. The global loads for a tile are issued one phase before the data is used.
// The softmax max. is only updated if it grows by more than 2^10 (log2 domain), so the VKQ accumulators are rarely rescaled.

typedef _Float16 fattn_cdna_h4 __attribute__((ext_vector_type(4)));
typedef _Float16 fattn_cdna_h8 __attribute__((ext_vector_type(8)));
typedef float    fattn_cdna_f4 __attribute__((ext_vector_type(4)));

static constexpr int fattn_cdna_nbatch = 64; // KV rows per tile.

typedef _Float16 fattn_cdna_h2 __attribute__((ext_vector_type(2)));

// Raw data of one 32 value block of a K row.
template <ggml_type type>
struct fattn_cdna_raw_K {
    static constexpr int nx = type == GGML_TYPE_F16 ? 16 : (type == GGML_TYPE_Q8_0 ? 8 : 4);
    uint32_t x[nx];
    uint32_t d;
};

// Raw data of 8 values in 4 V rows.
template <ggml_type type>
struct fattn_cdna_raw_V {
    static constexpr int nx = type == GGML_TYPE_F16 ? 4 : (type == GGML_TYPE_Q8_0 ? 2 : 1);
    uint32_t x[4][nx];
    uint32_t d[4];
};

template <ggml_type type>
static constexpr __device__ float fattn_cdna_bias() {
    return type == GGML_TYPE_Q8_0 ? 128.0f : 8.0f;
}

template <ggml_type type>
static constexpr __device__ int fattn_cdna_block_size() {
    return type == GGML_TYPE_Q8_0 ? sizeof(block_q8_0) : sizeof(block_q4_0);
}

// u has two halves 1024 + q, returns the halves (q - bias)*d. The subtraction is exact, so the result is rounded once.
template <ggml_type type>
static __device__ __forceinline__ uint32_t fattn_cdna_dequant2(const uint32_t u, const fattn_cdna_h2 d) {
    constexpr _Float16 offset = -(1024.0f + fattn_cdna_bias<type>());
    fattn_cdna_h2 h = __builtin_bit_cast(fattn_cdna_h2, u);
    h = (h + fattn_cdna_h2{offset, offset})*d;
    return __builtin_bit_cast(uint32_t, h);
}

// Buffer resource for raw loads from p, p must be the same for all lanes.
static __device__ __forceinline__ __amdgpu_buffer_rsrc_t fattn_cdna_rsrc(const char * p) {
    const uint64_t pa = (uint64_t) p;
    const uint64_t pu = (uint64_t) (uint32_t) __builtin_amdgcn_readfirstlane((uint32_t) pa) | ((uint64_t) (uint32_t) __builtin_amdgcn_readfirstlane((uint32_t) (pa >> 32)) << 32);
    return __builtin_amdgcn_make_buffer_rsrc((void *) pu, (short) 0, 0x7fffffff, 0x00020000);
}

static __device__ __forceinline__ void fattn_cdna_load16(uint32_t * dst, const __amdgpu_buffer_rsrc_t r, const int voff, const int soff) {
    const auto t = __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0);
    dst[0] = t[0]; dst[1] = t[1]; dst[2] = t[2]; dst[3] = t[3];
}
static __device__ __forceinline__ void fattn_cdna_load8(uint32_t * dst, const __amdgpu_buffer_rsrc_t r, const int voff, const int soff) {
    const auto t = __builtin_amdgcn_raw_buffer_load_b64(r, voff, soff, 0);
    dst[0] = t[0]; dst[1] = t[1];
}
static __device__ __forceinline__ uint32_t fattn_cdna_load4(const __amdgpu_buffer_rsrc_t r, const int voff, const int soff) {
    return __builtin_amdgcn_raw_buffer_load_b32(r, voff, soff, 0);
}
static __device__ __forceinline__ uint32_t fattn_cdna_load2(const __amdgpu_buffer_rsrc_t r, const int voff, const int soff) {
    return __builtin_amdgcn_raw_buffer_load_b16(r, voff, soff, 0);
}

// K: the thread loads the block at byte offset voff (+ soff) of the K data.
template <ggml_type type>
static __device__ __forceinline__ void fattn_cdna_load_K(const __amdgpu_buffer_rsrc_t r, const int voff, const int soff, fattn_cdna_raw_K<type> & raw) {
    if constexpr (type == GGML_TYPE_F16) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            fattn_cdna_load16(raw.x + 4*k, r, voff + 16*k, soff);
        }
        raw.d = 0;
    } else {
        fattn_cdna_load16(raw.x, r, voff + 2, soff);
        if constexpr (type == GGML_TYPE_Q8_0) {
            fattn_cdna_load16(raw.x + 4, r, voff + 18, soff);
        }
        raw.d = fattn_cdna_load2(r, voff, soff);
    }
}

// Granule q (values 8*q..8*q+7) of a K block as f16.
template <ggml_type type>
static __device__ __forceinline__ uint4 fattn_cdna_convert_K(const fattn_cdna_raw_K<type> & raw, const int q) {
    uint4 out;
    if constexpr (type == GGML_TYPE_F16) {
        out = make_uint4(raw.x[4*q + 0], raw.x[4*q + 1], raw.x[4*q + 2], raw.x[4*q + 3]);
    } else {
        const fattn_cdna_h2 d = __builtin_bit_cast(fattn_cdna_h2, raw.d | (raw.d << 16));
        uint32_t u[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            if constexpr (type == GGML_TYPE_Q8_0) {
                u[k] = raw.x[2*q + k] ^ 0x80808080;
            } else {
                u[k] = (raw.x[2*(q % 2) + k] >> (4*(q/2))) & 0x0F0F0F0F;
            }
        }
        out.x = fattn_cdna_dequant2<type>(__builtin_amdgcn_perm(0x64646464, u[0], 0x04010400), d);
        out.y = fattn_cdna_dequant2<type>(__builtin_amdgcn_perm(0x64646464, u[0], 0x04030402), d);
        out.z = fattn_cdna_dequant2<type>(__builtin_amdgcn_perm(0x64646464, u[1], 0x04010400), d);
        out.w = fattn_cdna_dequant2<type>(__builtin_amdgcn_perm(0x64646464, u[1], 0x04030402), d);
    }
    return out;
}

// V: the thread loads 8 values of 4 rows, voff is the byte offset of the values in row 0 (block start for quantized types).
template <ggml_type type>
static __device__ __forceinline__ void fattn_cdna_load_V(const __amdgpu_buffer_rsrc_t r, const int voff, const int sub, const int soff, const int stride, fattn_cdna_raw_V<type> & raw) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if constexpr (type == GGML_TYPE_F16) {
            fattn_cdna_load16(raw.x[k], r, voff, soff + k*stride);
            raw.d[k] = 0;
        } else if constexpr (type == GGML_TYPE_Q8_0) {
            fattn_cdna_load8(raw.x[k], r, voff + 2 + 8*sub, soff + k*stride);
            raw.d[k] = fattn_cdna_load2(r, voff, soff + k*stride);
        } else {
            raw.x[k][0] = fattn_cdna_load4(r, voff + 2 + 4*sub, soff + k*stride);
            raw.d[k] = fattn_cdna_load2(r, voff, soff + k*stride);
        }
    }
}

// Value index within the D values of a V row for value e of the thread.
template <ggml_type type>
static __device__ __forceinline__ int fattn_cdna_V_index(const int dc, const int e) {
    if constexpr (type == GGML_TYPE_Q4_0) {
        return 32*(dc/4) + 4*(dc % 4) + (e % 4) + 16*(e/4);
    } else {
        return 8*dc + e;
    }
}

// Values e and e + 1 of the 4 rows as f16: out[0] = value e rows 0, 1, out[1] = value e rows 2, 3, out[2], out[3] the same for value e + 1.
template <ggml_type type>
static __device__ __forceinline__ void fattn_cdna_convert_V(const fattn_cdna_raw_V<type> & raw, const int e, uint32_t * out) {
    if constexpr (type == GGML_TYPE_F16) {
        out[0] = __builtin_amdgcn_perm(raw.x[1][e/2], raw.x[0][e/2], 0x05040100);
        out[1] = __builtin_amdgcn_perm(raw.x[3][e/2], raw.x[2][e/2], 0x05040100);
        out[2] = __builtin_amdgcn_perm(raw.x[1][e/2], raw.x[0][e/2], 0x07060302);
        out[3] = __builtin_amdgcn_perm(raw.x[3][e/2], raw.x[2][e/2], 0x07060302);
    } else {
        uint32_t u[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if constexpr (type == GGML_TYPE_Q8_0) {
                u[k] = raw.x[k][e/4] ^ 0x80808080;
            } else {
                u[k] = (raw.x[k][0] >> (4*(e/4))) & 0x0F0F0F0F;
            }
        }
        // Bytes e, e + 1 of rows 0, 1 (2, 3) interleaved, then spread to 0x64XX halves:
        const uint32_t sel = (e % 4) == 0 ? 0x05010400 : 0x07030602;
        const uint32_t t01 = __builtin_amdgcn_perm(u[1], u[0], sel);
        const uint32_t t23 = __builtin_amdgcn_perm(u[3], u[2], sel);
        const fattn_cdna_h2 d01 = __builtin_bit_cast(fattn_cdna_h2, raw.d[0] | (raw.d[1] << 16));
        const fattn_cdna_h2 d23 = __builtin_bit_cast(fattn_cdna_h2, raw.d[2] | (raw.d[3] << 16));
        out[0] = fattn_cdna_dequant2<type>(__builtin_amdgcn_perm(0x64646464, t01, 0x04010400), d01);
        out[1] = fattn_cdna_dequant2<type>(__builtin_amdgcn_perm(0x64646464, t23, 0x04010400), d23);
        out[2] = fattn_cdna_dequant2<type>(__builtin_amdgcn_perm(0x64646464, t01, 0x04030402), d01);
        out[3] = fattn_cdna_dequant2<type>(__builtin_amdgcn_perm(0x64646464, t23, 0x04030402), d23);
    }
}

// Barrier that only waits for shared memory accesses, the global loads for the next tile stay in flight.
//     The compiler would wait for all memory accesses before a s_barrier on gfx908, so it is inline asm.
static __device__ __forceinline__ void fattn_cdna_sync() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

template<int D, int ncols2, int nwarps, ggml_type type_K, ggml_type type_V>
__launch_bounds__(nwarps*64, 1)
static __global__ void flash_attn_ext_cdna(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t type_K_data, const int32_t type_V_data,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#if defined(FLASH_ATTN_AVAILABLE) && defined(AMD_MFMA_AVAILABLE)
    static_assert(D == 256, "bad D");
    static_assert(nwarps == 4 || nwarps == 8, "bad nwarps");
    constexpr int nrep    = 8/nwarps; // K blocks and V chunks per thread.
    constexpr int ncols   = 16*nwarps;
    constexpr int ncols1  = ncols/ncols2;
    constexpr int nbatch  = fattn_cdna_nbatch;
    constexpr int nh      = 2;    // Halves of 32 KV rows of a tile.
    constexpr float log2e = 1.4426950408889634f;
    // The max. in the log2 domain is shifted up by 3 (FATTN_KQ_MAX_OFFSET), it is only updated if it grows by more than max_slack.
    constexpr float max_offset = 3.0f;
    constexpr float max_slack  = 10.0f;

    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    __shared__ __align__(16) char smem[nbatch*D*2 + D*nbatch*2];
    char * K_s = smem;
    char * V_s = smem + nbatch*D*2;

    const int w   = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int l   = threadIdx.x;
    const int i   = l % 16;
    const int g   = l / 16;

    const int gqa_ratio    = ne02 / ne12;
    const int ntiles_z_gqa = gqa_ratio / ncols2;
    const int sequence     = blockIdx.z / (ne12*ntiles_z_gqa);
    const int z_KV         = (blockIdx.z - sequence*ne12*ntiles_z_gqa) / ntiles_z_gqa;
    const int zt_gqa       =  blockIdx.z - sequence*ne12*ntiles_z_gqa - z_KV*ntiles_z_gqa;
    const int head0        = z_KV*gqa_ratio + zt_gqa*ncols2;
    const int ic0          = blockIdx.x*ncols1;

    // Q column of this thread in the MFMA layouts:
    const int  jc     = 16*w + i;
    const int  j_Q    = ic0 + jc/ncols2;
    const bool col_ok = j_Q < int(ne01.z);

    // Q as B matrix for KQ: for the pair of MFMAs p the thread has the values 32*p + 8*g + 0..7 of column jc.
    fattn_cdna_h4 Q_B[D/16];
    {
        const float * Q_f = (const float *) (Q + nb03*sequence + nb02*(head0 + jc % ncols2) + nb01*(col_ok ? j_Q : 0));
        const float qscale = scale*log2e;
#pragma unroll
        for (int s = 0; s < D/16; ++s) {
            float4 tmp = *(const float4 *) (Q_f + 32*(s/2) + 8*g + 4*(s%2));
            tmp.x = col_ok ? tmp.x : 0.0f;
            tmp.y = col_ok ? tmp.y : 0.0f;
            tmp.z = col_ok ? tmp.z : 0.0f;
            tmp.w = col_ok ? tmp.w : 0.0f;
            Q_B[s] = fattn_cdna_h4{(_Float16) (qscale*tmp.x), (_Float16) (qscale*tmp.y), (_Float16) (qscale*tmp.z), (_Float16) (qscale*tmp.w)};
        }
    }

    K += nb13*sequence + nb12*z_KV;
    V += nb23*sequence + nb22*z_KV;

    // K tile loads/stores for the virtual warp vw = w + nwarps*r (8 in total): rows 16*(vw/2)..+15 and blocks 4*(vw%2)..+3.
    //     Within 8 consecutive lanes the rows are 0..3, 8..11 (+4) so that the 16 byte stores do not have bank conflicts.
    auto row_K_ld = [&](const int r) {
        return 16*((w + nwarps*r)/2) + (l % 4) + 8*((l/4) % 2) + 4*((l/8) % 2);
    };
    const int blk_K_ld = 4*(w % 2) + l/16;
    // V tile loads/stores: the thread has the KV rows 4*kg..4*kg+3 and the values dc*8..dc*8+7 (for q4_0 see fattn_cdna_V_index).
    const int kg_V = l % 16;
    auto dc_V = [&](const int r) {
        return 4*(w + nwarps*r) + l/16;
    };

    auto swz_K = [](const int row) {
        return (row & 3) | ((row >> 1) & 4);
    };

    auto store_K = [&](const fattn_cdna_raw_K<type_K> * r) {
#pragma unroll
        for (int rr = 0; rr < nrep; ++rr) {
            const int row = row_K_ld(rr);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 tmp = fattn_cdna_convert_K<type_K>(r[rr], q);
                *(uint4 *) (K_s + row*(2*D) + 16*((4*blk_K_ld + q) ^ swz_K(row))) = tmp;
            }
        }
    };
    auto store_V = [&](const fattn_cdna_raw_V<type_V> * r) {
#pragma unroll
        for (int rr = 0; rr < nrep; ++rr) {
#pragma unroll
            for (int e = 0; e < 8; e += 2) {
                uint32_t tmp[4];
                fattn_cdna_convert_V<type_V>(r[rr], e, tmp);
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int d = fattn_cdna_V_index<type_V>(dc_V(rr), e + k);
                    *(uint2 *) (V_s + d*(2*nbatch) + 16*((kg_V/2) ^ (d % 8)) + 8*(kg_V % 2)) = make_uint2(tmp[2*k + 0], tmp[2*k + 1]);
                }
            }
        }
    };

    const __amdgpu_buffer_rsrc_t rsrc_K = fattn_cdna_rsrc(K);
    const __amdgpu_buffer_rsrc_t rsrc_V = fattn_cdna_rsrc(V);
    const __amdgpu_buffer_rsrc_t rsrc_M = fattn_cdna_rsrc(mask + nb33*(sequence % ne33));
    const int voff_K = row_K_ld(0)*nb11 + blk_K_ld*(type_K == GGML_TYPE_F16 ? 64 : fattn_cdna_block_size<type_K>());
    const int voff_V = 4*kg_V*nb21 + (type_V == GGML_TYPE_F16 ? 16*dc_V(0) : (dc_V(0)/4)*fattn_cdna_block_size<type_V>());
    const int voff_M = (col_ok ? j_Q : int(ne01.z) - 1)*nb31 + 2*8*g;

    // Virtual warp w + nwarps*r has the K rows 16*(nwarps/2)*r further down, the V values 32*nwarps*r further right.
    auto load_K = [&](const int it, fattn_cdna_raw_K<type_K> * r) {
#pragma unroll
        for (int rr = 0; rr < nrep; ++rr) {
            fattn_cdna_load_K<type_K>(rsrc_K, voff_K, (it*nbatch + 8*nwarps*rr)*nb11, r[rr]);
        }
    };
    auto load_V = [&](const int it, fattn_cdna_raw_V<type_V> * r) {
#pragma unroll
        for (int rr = 0; rr < nrep; ++rr) {
            constexpr int dV = type_V == GGML_TYPE_F16 ? 2*32*nwarps : (nwarps)*fattn_cdna_block_size<type_V>();
            fattn_cdna_load_V<type_V>(rsrc_V, voff_V + dV*rr, dc_V(0) % 4, it*nbatch*nb21, nb21, r[rr]);
        }
    };
    auto load_mask = [&](const int it, uint4 * r) {
#pragma unroll
        for (int h = 0; h < nh; ++h) {
            fattn_cdna_load16((uint32_t *) &r[h], rsrc_M, voff_M + 64*h, 2*it*nbatch);
        }
    };

    // Shared memory addresses for the MFMA A matrices.
    // K for KQ tile m of half h (KV rows 32*h + 8*g + 4*m + 0..3 in the C matrix):
    //     lane i reads row 32*h + 8*(i/4) + 4*m + i%4, granule (4*p + g) ^ (i % 8).
    const int xs    = i % 8;
    const int row_K = 8*(i/4) + (i % 4);
    const int gx    = (g ^ (xs & 3));
    const int xb    = xs >> 2;
    const char * K_A_even = K_s + row_K*(2*D) + 16*gx + 64*xb;
    const char * K_A_odd  = K_s + row_K*(2*D) + 16*gx - 64*xb;
    // V^T for VKQ tile t: lane i reads row 16*t + i, granule (4*h + g) ^ (i % 8): the KV rows 32*h + 8*g + 0..7.
    const char * V_A[nh];
#pragma unroll
    for (int h = 0; h < nh; ++h) {
        V_A[h] = V_s + i*(2*nbatch) + 16*((4*h + g) ^ xs);
    }

    fattn_cdna_f4 VKQ[D/16];
#pragma unroll
    for (int t = 0; t < D/16; ++t) {
        VKQ[t] = fattn_cdna_f4{0.0f, 0.0f, 0.0f, 0.0f};
    }
    float KQ_max = -FLT_MAX/2.0f;
    float KQ_sum = 0.0f;

    ggml_cuda_pdl_sync();

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    const int ntiles    = k_VKQ_max / nbatch;
    int it = blockIdx.y;

    if (it < ntiles) {
        fattn_cdna_raw_K<type_K> rK[nrep];
        fattn_cdna_raw_V<type_V> rV[nrep];
        uint4 rM[nh];
        load_K(it, rK);
        load_V(it, rV);
        load_mask(it, rM);
        store_K(rK);
        fattn_cdna_sync();

        while (true) {
            // The last tile is loaded twice, the loads are not conditional so that the compiler does not need to wait for all loads.
            const int it_next = it + int(gridDim.y) < ntiles ? it + int(gridDim.y) : it;
            load_K(it_next, rK);

            fattn_cdna_f4 KQ[nh][2];
#pragma unroll
            for (int h = 0; h < nh; ++h) {
                KQ[h][0] = fattn_cdna_f4{0.0f, 0.0f, 0.0f, 0.0f};
                KQ[h][1] = fattn_cdna_f4{0.0f, 0.0f, 0.0f, 0.0f};
            }
            {
                // The K reads for p + 1 are issued before the MFMAs of p.
                auto load_K_A = [&](const int p, fattn_cdna_h8 * K_A) {
                    const char * base = p % 2 == 0 ? K_A_even : K_A_odd;
#pragma unroll
                    for (int hm = 0; hm < 2*nh; ++hm) {
                        K_A[hm] = *(const fattn_cdna_h8 *) (base + (32*(hm/2) + 4*(hm%2))*(2*D) + 64*p);
                    }
                };
                fattn_cdna_h8 K_A[2][2*nh];
                load_K_A(0, K_A[0]);
#pragma unroll
                for (int p = 0; p < D/32; ++p) {
                    if (p + 1 < D/32) {
                        load_K_A(p + 1, K_A[(p + 1) % 2]);
                    }
#pragma unroll
                    for (int hm = 0; hm < 2*nh; ++hm) {
                        const fattn_cdna_h8 a = K_A[p % 2][hm];
                        KQ[hm/2][hm%2] = __builtin_amdgcn_mfma_f32_16x16x16f16(__builtin_shufflevector(a, a, 0, 1, 2, 3), Q_B[2*p + 0], KQ[hm/2][hm%2], 0, 0, 0);
                        KQ[hm/2][hm%2] = __builtin_amdgcn_mfma_f32_16x16x16f16(__builtin_shufflevector(a, a, 4, 5, 6, 7), Q_B[2*p + 1], KQ[hm/2][hm%2], 0, 0, 0);
                    }
                }
            }

            store_V(rV);

            // Softmax, KQ[h][m][r] is the value of column jc and KV row 32*h + 8*g + 4*m + r:
            float s[8*nh];
#pragma unroll
            for (int h = 0; h < nh; ++h) {
                const fattn_cdna_h8 mh = __builtin_bit_cast(fattn_cdna_h8, rM[h]);
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    s[8*h + k] = KQ[h][k/4][k%4] + log2e*(float) mh[k];
                }
            }
            float KQ_max_new = s[0];
#pragma unroll
            for (int k = 1; k < 8*nh; ++k) {
                KQ_max_new = fmaxf(KQ_max_new, s[k]);
            }
            KQ_max_new = fmaxf(KQ_max_new, __shfl_xor(KQ_max_new, 16, 64));
            KQ_max_new = fmaxf(KQ_max_new, __shfl_xor(KQ_max_new, 32, 64));
            KQ_max_new += max_offset;
            const bool rescale = KQ_max_new > KQ_max + max_slack;
            if (__any(rescale)) {
                const float KQ_max_scale = rescale ? __builtin_amdgcn_exp2f(KQ_max - KQ_max_new) : 1.0f;
                KQ_max = rescale ? KQ_max_new : KQ_max;
                KQ_sum *= KQ_max_scale;
#pragma unroll
                for (int t = 0; t < D/16; ++t) {
                    VKQ[t] *= KQ_max_scale;
                }
            }
            fattn_cdna_h4 P_B[nh][2];
#pragma unroll
            for (int k = 0; k < 8*nh; ++k) {
                const float p = __builtin_amdgcn_exp2f(s[k] - KQ_max);
                KQ_sum += p;
                P_B[k/8][(k/4) % 2][k%4] = (_Float16) p;
            }

            fattn_cdna_sync();

            load_V(it_next, rV);
            load_mask(it_next, rM);

            {
                // The V reads for tile t + 1 are issued before the MFMAs of tile t.
                fattn_cdna_h8 V_A8[2][nh];
#pragma unroll
                for (int h = 0; h < nh; ++h) {
                    V_A8[0][h] = *(const fattn_cdna_h8 *) (V_A[h]);
                }
#pragma unroll
                for (int t = 0; t < D/16; ++t) {
                    if (t + 1 < D/16) {
#pragma unroll
                        for (int h = 0; h < nh; ++h) {
                            V_A8[(t + 1) % 2][h] = *(const fattn_cdna_h8 *) (V_A[h] + (t + 1)*16*(2*nbatch));
                        }
                    }
#pragma unroll
                    for (int h = 0; h < nh; ++h) {
                        const fattn_cdna_h8 a = V_A8[t % 2][h];
                        VKQ[t] = __builtin_amdgcn_mfma_f32_16x16x16f16(__builtin_shufflevector(a, a, 0, 1, 2, 3), P_B[h][0], VKQ[t], 0, 0, 0);
                        VKQ[t] = __builtin_amdgcn_mfma_f32_16x16x16f16(__builtin_shufflevector(a, a, 4, 5, 6, 7), P_B[h][1], VKQ[t], 0, 0, 0);
                    }
                }
            }

            store_K(rK);

            fattn_cdna_sync();

            if (it_next == it) {
                break;
            }
            it = it_next;
        }
    }

    KQ_sum += __shfl_xor(KQ_sum, 16, 64);
    KQ_sum += __shfl_xor(KQ_sum, 32, 64);

    const int head  = head0 + jc % ncols2;
    const int j_dst = (sequence*int(ne01.z) + j_Q)*ne02 + head;
    float * dst_j = dst + (int64_t(j_dst)*gridDim.y + blockIdx.y)*D;

    if (col_ok) {
        const float KQ_sum_inv = 1.0f/KQ_sum;
#pragma unroll
        for (int t = 0; t < D/16; ++t) {
            fattn_cdna_f4 tmp = VKQ[t];
            if (gridDim.y == 1) {
                tmp *= KQ_sum_inv;
            }
            *(float4 *) &dst_j[16*t + 4*g] = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
        }
        if (gridDim.y != 1 && g == 0) {
            dst_meta[j_dst*gridDim.y + blockIdx.y] = make_float2(KQ_max/log2e, KQ_sum);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap, type_K_data, type_V_data,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(AMD_MFMA_AVAILABLE)
}

template <int D, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_cdna_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = 8;
    constexpr int ncols1 = 16*nwarps/ncols2;
    fattn_kernel_t fattn_kernel = flash_attn_ext_cdna<D, ncols2, nwarps, type_K, type_V>;
    launch_fattn<D, ncols1, ncols2>(ctx, dst, fattn_kernel, nwarps, 0, fattn_cdna_nbatch, false, false, false, 64);
}

// Whether ggml_cuda_flash_attn_ext_cdna can be used for this FLASH_ATTN_EXT.
bool ggml_cuda_flash_attn_ext_cdna_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_cdna(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
