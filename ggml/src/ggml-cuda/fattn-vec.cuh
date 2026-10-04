#include "common.cuh"
#include "fattn-common.cuh"

// FlashAttention kernel for few Q columns (decode, small batches) with MFMA.
// A CUDA block works on one K/V head and 16 Q columns: ncols1 Q rows x ncols2 Q heads that use this K/V head (GQA).
// Each warp works on its own tiles of 16 KV rows. KQ = K*Q^T and VKQ += V^T*softmax(KQ) are 16x16x16 MFMAs with the Q columns as N.
// K and V are dequantized from VRAM straight into the MFMA operands, so they are read only once for all 16 Q columns.

typedef _Float16 fattn_vec_halfx4  __attribute__((ext_vector_type(4)));
typedef float    fattn_vec_floatx4 __attribute__((ext_vector_type(4)));

static constexpr __host__ __device__ int ggml_cuda_fattn_vec_get_nwarps() {
    return 4;
}

// The quantized blocks are only 2 or 4 byte aligned, gfx9 can do unaligned VRAM loads.
template <int nbytes>
static __device__ __forceinline__ void ggml_cuda_fattn_vec_load_unaligned(void * __restrict__ dst, const void * __restrict__ src) {
    __builtin_memcpy(dst, src, nbytes);
}

// Spread 4 bits of vh to bit 4 of each byte.
static __device__ __forceinline__ uint32_t ggml_cuda_fattn_vec_qh_to_bytes(const uint32_t vh) {
    return ((vh << 4) & 0x00000010) | ((vh << 11) & 0x00001000) | ((vh << 18) & 0x00100000) | ((vh << 25) & 0x10000000);
}

// Quantized values are loaded as unsigned bytes q, value = (q - bias)*d + m.
template <ggml_type type>
static constexpr __device__ int ggml_cuda_fattn_vec_q_bias() {
    return type == GGML_TYPE_Q8_0 ? 128 : (type == GGML_TYPE_Q4_0 ? 8 : (type == GGML_TYPE_Q5_0 ? 16 : 0));
}

template <ggml_type type>
static constexpr __device__ bool ggml_cuda_fattn_vec_q_has_min() {
    return type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1;
}

// nv consecutive values of a row (all in one block) as loaded from VRAM.
// The loads and the unpacking are separate so that the loads for the next KV tile can be issued early.
template <ggml_type type, int nv>
struct ggml_cuda_fattn_vec_q_raw {
    static_assert(nv == 4 || nv == 8 || nv == 16 || nv == 32, "bad nv");
    static constexpr int nw = type == GGML_TYPE_Q8_0 || nv < 32 ? nv/4 : 4;

    uint32_t qs[nw];
    uint32_t qh; // 5 bit types only
    half2    dm; // Scale and min (only for types with a min).
};

template <ggml_type type, int nv>
static __device__ __forceinline__ void ggml_cuda_fattn_vec_load_q(
        const char * __restrict__ row, const int v0, ggml_cuda_fattn_vec_q_raw<type, nv> & raw) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        const block_q8_0 * x = (const block_q8_0 *) row + v0/QK8_0;
        ggml_cuda_fattn_vec_load_unaligned<nv>(raw.qs, x->qs + v0 % QK8_0);
        raw.dm = __halves2half2(x->d, x->d);
    } else {
        // 4/5 bit types: values 0-15 of a block are in the low nibbles, values 16-31 in the high nibbles.
        static_assert(type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1, "bad type");
        constexpr int qs_offset = type == GGML_TYPE_Q4_0 ? 2 : (type == GGML_TYPE_Q4_1 ? 4 : (type == GGML_TYPE_Q5_0 ? 6 : 8));
        constexpr int nbytes    = type == GGML_TYPE_Q4_0 ? sizeof(block_q4_0) : (type == GGML_TYPE_Q4_1 ? sizeof(block_q4_1) :
                                 (type == GGML_TYPE_Q5_0 ? sizeof(block_q5_0) : sizeof(block_q5_1)));
        const char * x = row + (v0/32)*nbytes;

        ggml_cuda_fattn_vec_load_unaligned<4*ggml_cuda_fattn_vec_q_raw<type, nv>::nw>(raw.qs, x + qs_offset + (nv < 32 ? v0 % 16 : 0));
        if constexpr (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            ggml_cuda_fattn_vec_load_unaligned<4>(&raw.qh, x + qs_offset - 4);
        }
        if constexpr (ggml_cuda_fattn_vec_q_has_min<type>()) {
            raw.dm = *(const half2 *) x;
        } else {
            raw.dm = __halves2half2(*(const half *) x, *(const half *) x);
        }
    }
}

// Unpack the values loaded by ggml_cuda_fattn_vec_load_q as unsigned bytes, 4 per uint32_t.
template <ggml_type type, int nv>
static __device__ __forceinline__ void ggml_cuda_fattn_vec_unpack_q(
        const ggml_cuda_fattn_vec_q_raw<type, nv> & raw, const int v0, uint32_t * __restrict__ q) {
    if constexpr (type == GGML_TYPE_Q8_0) {
#pragma unroll
        for (int l = 0; l < nv/4; ++l) {
            q[l] = raw.qs[l] ^ 0x80808080;
        }
    } else {
        if constexpr (nv == 32) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                q[l + 0] = (raw.qs[l] >> 0) & 0x0F0F0F0F;
                q[l + 4] = (raw.qs[l] >> 4) & 0x0F0F0F0F;
            }
        } else {
            const int shift = 4*((v0 % 32) / 16);
#pragma unroll
            for (int l = 0; l < nv/4; ++l) {
                q[l] = (raw.qs[l] >> shift) & 0x0F0F0F0F;
            }
        }
        if constexpr (type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            const uint32_t qh = raw.qh >> (v0 % 32);
#pragma unroll
            for (int l = 0; l < nv/4; ++l) {
                q[l] |= ggml_cuda_fattn_vec_qh_to_bytes(qh >> (4*l));
            }
        }
    }
}

// u has 2 values as 0x64XX halves (1024 + q), convert them with scale d and min m:
template <ggml_type type>
static __device__ __forceinline__ half2 ggml_cuda_fattn_vec_dequant_h2(const uint32_t u, const half2 d, const half2 m) {
    constexpr float offset = -(1024.0f + ggml_cuda_fattn_vec_q_bias<type>());
    half2 h;
    memcpy(&h, &u, sizeof(h));
    h = (h + make_half2(offset, offset))*d;
    if constexpr (ggml_cuda_fattn_vec_q_has_min<type>()) {
        h += m;
    }
    return h;
}

static __device__ __forceinline__ fattn_vec_halfx4 ggml_cuda_fattn_vec_pack(const half2 lo, const half2 hi) {
    fattn_vec_halfx4 r;
    r[0] = __low2half(lo);
    r[1] = __high2half(lo);
    r[2] = __low2half(hi);
    r[3] = __high2half(hi);
    return r;
}

template<int D, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_vec_get_nwarps()*64, 1)
static __global__ void flash_attn_ext_vec(
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
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#if defined(FLASH_ATTN_AVAILABLE) && defined(AMD_MFMA_AVAILABLE)
    const char * GGML_CUDA_RESTRICT Q        = Q_ptr;
    const char * GGML_CUDA_RESTRICT K        = K_ptr;
    const char * GGML_CUDA_RESTRICT V        = V_ptr;
    const char * GGML_CUDA_RESTRICT mask     = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks    = sinks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max   = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst      = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta = dst_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int  ncols    = ncols1*ncols2;
    constexpr int  nwarps   = ggml_cuda_fattn_vec_get_nwarps();
    constexpr int  nbatch   = nwarps*16; // KV rows per iteration of the CUDA block.
    constexpr int  ncalls   = D/16;      // MFMAs per 16 KV rows, both for KQ and VKQ.
    constexpr int  nv_K     = D/4;       // K values per thread and row.
    constexpr int  nv_V     = D/16;      // V values per thread and row.
    constexpr bool K_f16    = type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_BF16;
    constexpr bool V_f16    = type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_BF16;
    constexpr int  nv_K_blk = nv_K < 32 ? nv_K : 32; // K values per thread and quantized block.

    // Registers for the raw data of one KV tile, without the prefetch of the next tile the loads are less overlapped.
    constexpr int  nregs_tile = (K_f16 ? nv_K/2 : nv_K/4 + 2*(nv_K/nv_K_blk)) + 4*(V_f16 ? nv_V/2 : nv_V/4 + 2) + 2;
    constexpr bool prefetch   = nregs_tile <= 72;
    constexpr bool Q_in_LDS   = D > 256; // Saves registers for large D.

    static_assert(ncols <= 16, "bad ncols");
    static_assert(D % 64 == 0 && D <= 512, "bad D");

    // z_KV == K/V head index, zt_gqa == Q head tile index per K/V head:
    const int gqa_ratio    = ne02 / ne12;
    const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
    const int sequence     = blockIdx.z / (ne12*ntiles_z_gqa);
    const int z_KV         = (blockIdx.z - sequence*ne12*ntiles_z_gqa) / ntiles_z_gqa;
    const int zt_gqa       =  blockIdx.z - sequence*ne12*ntiles_z_gqa - z_KV*ntiles_z_gqa;
    const int head0        = z_KV*gqa_ratio + zt_gqa*ncols2; // First Q head of this block.
    const int ic0          = blockIdx.x*ncols1;              // First Q column of this block.

    Q += nb03*sequence + nb02*head0 + nb01*ic0;
    K += nb13*sequence + nb12*z_KV;
    V += nb23*sequence + nb22*z_KV;

    const half * maskh = mask ? (const half *) (mask + nb33*(sequence % ne33) + nb31*ic0) : nullptr;
    const int stride_mask = nb31 / sizeof(half);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    // Column jc of the block is Q column ic0 + jc/ncols2 of Q head head0 + jc%ncols2.
    auto col_valid = [&](const int jc) {
        return jc < ncols && (ncols1 == 1 || ic0 + jc/ncols2 < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + jc%ncols2 < gqa_ratio);
    };

    // In the MFMA layouts a thread works on column jc = threadIdx.x % 16 and on rows/values 4*g..4*g+3 with g = threadIdx.x / 16.
    const int  jc     = threadIdx.x % 16;
    const int  g      = threadIdx.x / 16;
    const bool col_ok = col_valid(jc);

    // Q as B matrix for KQ: the thread has the values [g*nv_K, (g+1)*nv_K) of column jc, 4 per MFMA.
    // The order of the KQ dot product sum does not matter, K uses the same order.
    // The shared memory for Q is later used for combining VKQ.
    constexpr int nbytes_VKQ_s = 16*(D + 4)*sizeof(float);
    constexpr int nbytes_Q_s   = Q_in_LDS ? 16*D*sizeof(half) : 0;
    __shared__ __align__(16) char data_s[nbytes_VKQ_s > nbytes_Q_s ? nbytes_VKQ_s : nbytes_Q_s];
    fattn_vec_halfx4 * Q_s = (fattn_vec_halfx4 *) data_s;

    fattn_vec_halfx4 Q_B[Q_in_LDS ? 1 : ncalls];
    {
        const float * Q_f = (const float *) (Q + (jc/ncols2)*nb01 + (jc%ncols2)*nb02) + g*nv_K;
#pragma unroll
        for (int s0 = 0; s0 < ncalls; s0 += (Q_in_LDS ? nwarps : 1)) {
            const int s = s0 + (Q_in_LDS ? threadIdx.y : 0);
            float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (col_ok) {
                tmp = *(const float4 *) (Q_f + 4*s);
            }
            const fattn_vec_halfx4 tmp_h = {(_Float16) (scale*tmp.x), (_Float16) (scale*tmp.y), (_Float16) (scale*tmp.z), (_Float16) (scale*tmp.w)};
            if constexpr (Q_in_LDS) {
                Q_s[s*64 + threadIdx.x] = tmp_h;
            } else {
                Q_B[s] = tmp_h;
            }
        }
    }
    if constexpr (Q_in_LDS) {
        __syncthreads();
    }

    // VKQ[u][v] is the output of column jc for value (4*g + v)*nv_V + u.
    fattn_vec_floatx4 VKQ[ncalls];
#pragma unroll
    for (int u = 0; u < ncalls; ++u) {
        VKQ[u] = fattn_vec_floatx4{0.0f, 0.0f, 0.0f, 0.0f};
    }
    float KQ_max = -FLT_MAX/2.0f;
    float KQ_sum = 0.0f;

    ggml_cuda_pdl_sync();

    // Raw data of one tile of 16 KV rows:
    //     K as A matrix: values [g*nv_K, (g+1)*nv_K) of KV row k_VKQ_0 + jc.
    //     V as A matrix (V^T): values [jc*nv_V, (jc+1)*nv_V) of the KV rows k_VKQ_0 + 4*g + 0..3.
    struct tile_raw {
        uint32_t K_h[K_f16 ? nv_K/2 : 1];
        ggml_cuda_fattn_vec_q_raw<K_f16 ? GGML_TYPE_Q8_0 : type_K, nv_K_blk> K_q[K_f16 ? 1 : nv_K/nv_K_blk];
        uint32_t V_h[4][V_f16 ? nv_V/2 : 1];
        ggml_cuda_fattn_vec_q_raw<V_f16 ? GGML_TYPE_Q8_0 : type_V, nv_V> V_q[V_f16 ? 1 : 4];
        half2 mask[2];
    };

    auto load_tile = [&](const int k_VKQ_0, tile_raw & t) {
        const char * K_row = K + int64_t(k_VKQ_0 + jc)*nb11;
        if constexpr (K_f16) {
#pragma unroll
            for (int l = 0; l < nv_K/2; l += 4) {
                ggml_cuda_memcpy_1<16>(t.K_h + l, K_row + 2*g*nv_K + 4*l);
            }
        } else {
#pragma unroll
            for (int b = 0; b < nv_K/nv_K_blk; ++b) {
                ggml_cuda_fattn_vec_load_q(K_row, g*nv_K + b*nv_K_blk, t.K_q[b]);
            }
        }

        constexpr int V_cpy = nv_V/2 < 4 ? nv_V/2 : 4; // uint32_t per f16 copy
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const char * V_row = V + int64_t(k_VKQ_0 + 4*g + r)*nb21;
            if constexpr (V_f16) {
#pragma unroll
                for (int l = 0; l < nv_V/2; l += V_cpy) {
                    ggml_cuda_memcpy_1<V_cpy*4>(t.V_h[r] + l, V_row + 2*jc*nv_V + 4*l);
                }
            } else {
                ggml_cuda_fattn_vec_load_q(V_row, jc*nv_V, t.V_q[r]);
            }
        }

        // Branches with loads would need a wait for all loads, so always load something:
        //     columns that are out of bounds read the mask of column 0 and without mask some K data is read, the values are not used.
        const char * mask_src = maskh ? (const char *) (maskh + (col_ok ? jc/ncols2 : 0)*stride_mask + k_VKQ_0 + 4*g) : K_row;
        ggml_cuda_memcpy_1<8>(t.mask, mask_src);
    };

    auto process_tile = [&](const tile_raw & t) {
        // KQ[v] is the KQ value of column jc and KV row k_VKQ_0 + 4*g + v.
        fattn_vec_floatx4 KQ = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int b = 0; b < (K_f16 ? 1 : nv_K/nv_K_blk); ++b) {
            uint32_t K_q[nv_K_blk/4];
            half2 d;
            half2 m;
            if constexpr (!K_f16) {
                ggml_cuda_fattn_vec_unpack_q(t.K_q[b], g*nv_K + b*nv_K_blk, K_q);
                d = __low2half2 (t.K_q[b].dm);
                m = __high2half2(t.K_q[b].dm);
            }
#pragma unroll
            for (int s0 = 0; s0 < (K_f16 ? ncalls : nv_K_blk/4); ++s0) {
                const int s = K_f16 ? s0 : b*(nv_K_blk/4) + s0;
                fattn_vec_halfx4 K_A;
                if constexpr (type_K == GGML_TYPE_F16) {
                    memcpy(&K_A, t.K_h + 2*s, sizeof(K_A));
                } else if constexpr (type_K == GGML_TYPE_BF16) {
                    nv_bfloat162 tmp[2];
                    memcpy(tmp, t.K_h + 2*s, sizeof(tmp));
                    K_A = ggml_cuda_fattn_vec_pack(__float22half2_rn(ggml_cuda_cast<float2>(tmp[0])), __float22half2_rn(ggml_cuda_cast<float2>(tmp[1])));
                } else {
                    const uint32_t lo = __builtin_amdgcn_perm(0x64646464, K_q[s0], 0x04010400);
                    const uint32_t hi = __builtin_amdgcn_perm(0x64646464, K_q[s0], 0x04030402);
                    K_A = ggml_cuda_fattn_vec_pack(ggml_cuda_fattn_vec_dequant_h2<type_K>(lo, d, m), ggml_cuda_fattn_vec_dequant_h2<type_K>(hi, d, m));
                }
                KQ = __builtin_amdgcn_mfma_f32_16x16x16f16(K_A, Q_in_LDS ? Q_s[s*64 + threadIdx.x] : Q_B[s], KQ, 0, 0, 0);
            }
        }

        float KQ_max_new = KQ_max;
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            if (use_logit_softcap) {
                KQ[v] = logit_softcap*tanhf(KQ[v]);
            }
            const float mask_val = __half2float(v % 2 == 0 ? __low2half(t.mask[v/2]) : __high2half(t.mask[v/2]));
            KQ[v] += maskh ? slope*mask_val : 0.0f;
            KQ_max_new = fmaxf(KQ_max_new, KQ[v] + FATTN_KQ_MAX_OFFSET);
        }
        KQ_max_new = fmaxf(KQ_max_new, __shfl_xor(KQ_max_new, 16, 64));
        KQ_max_new = fmaxf(KQ_max_new, __shfl_xor(KQ_max_new, 32, 64));

        const float KQ_max_scale = expf(KQ_max - KQ_max_new);
        KQ_max = KQ_max_new;

        float P[4];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            P[v] = expf(KQ[v] - KQ_max);
        }
        KQ_sum = KQ_sum*KQ_max_scale + ((P[0] + P[1]) + (P[2] + P[3]));

        // The max. only changes rarely after the first tiles, scaling all accumulators is expensive.
        if (__any(KQ_max_scale != 1.0f)) {
#pragma unroll
            for (int u = 0; u < ncalls; ++u) {
                VKQ[u] *= KQ_max_scale;
            }
        }

        // The KQ C matrix has the layout of the B matrix for VKQ.
        const fattn_vec_halfx4 P_B = {(_Float16) P[0], (_Float16) P[1], (_Float16) P[2], (_Float16) P[3]};

        uint32_t V_q[4][V_f16 ? 1 : nv_V/4];
        half2 d01;
        half2 d23;
        half2 m01;
        half2 m23;
        if constexpr (!V_f16) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                ggml_cuda_fattn_vec_unpack_q(t.V_q[r], jc*nv_V, V_q[r]);
            }
            d01 = __lows2half2 (t.V_q[0].dm, t.V_q[1].dm);
            d23 = __lows2half2 (t.V_q[2].dm, t.V_q[3].dm);
            m01 = __highs2half2(t.V_q[0].dm, t.V_q[1].dm);
            m23 = __highs2half2(t.V_q[2].dm, t.V_q[3].dm);
        }

#pragma unroll
        for (int u = 0; u < ncalls; ++u) {
            // Value u of the 4 KV rows:
            fattn_vec_halfx4 V_A;
            if constexpr (V_f16) {
                const int w = u/2;
                const uint32_t sel = u % 2 ? 0x07060302 : 0x05040100;
                const uint32_t r01 = __builtin_amdgcn_perm(t.V_h[1][w], t.V_h[0][w], sel);
                const uint32_t r23 = __builtin_amdgcn_perm(t.V_h[3][w], t.V_h[2][w], sel);
                if constexpr (type_V == GGML_TYPE_F16) {
                    half2 lo;
                    half2 hi;
                    memcpy(&lo, &r01, sizeof(lo));
                    memcpy(&hi, &r23, sizeof(hi));
                    V_A = ggml_cuda_fattn_vec_pack(lo, hi);
                } else {
                    nv_bfloat162 lo;
                    nv_bfloat162 hi;
                    memcpy(&lo, &r01, sizeof(lo));
                    memcpy(&hi, &r23, sizeof(hi));
                    V_A = ggml_cuda_fattn_vec_pack(__float22half2_rn(ggml_cuda_cast<float2>(lo)), __float22half2_rn(ggml_cuda_cast<float2>(hi)));
                }
            } else {
                const int w = u/4;
                const int k = u%4;
                const uint32_t sel = k | (12 << 8) | ((4 + k) << 16) | (12 << 24);
                const uint32_t r01 = __builtin_amdgcn_perm(V_q[1][w], V_q[0][w], sel) | 0x64006400;
                const uint32_t r23 = __builtin_amdgcn_perm(V_q[3][w], V_q[2][w], sel) | 0x64006400;
                V_A = ggml_cuda_fattn_vec_pack(ggml_cuda_fattn_vec_dequant_h2<type_V>(r01, d01, m01), ggml_cuda_fattn_vec_dequant_h2<type_V>(r23, d23, m23));
            }
            VKQ[u] = __builtin_amdgcn_mfma_f32_16x16x16f16(V_A, P_B, VKQ[u], 0, 0, 0);
        }
    };

    // Each warp works on every nwarps-th tile of the CUDA block, the loads for the next tile are issued before the current tile is processed.
    // The last tile is loaded twice: the load is not conditional so that the compiler does not need to wait for all loads.
    const int k_VKQ_max  = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    const int k_VKQ_step = gridDim.y*nbatch;
    int k_VKQ_0 = blockIdx.y*nbatch + threadIdx.y*16;
    if constexpr (prefetch) {
        if (k_VKQ_0 < k_VKQ_max) {
            tile_raw t0;
            tile_raw t1;
            load_tile(k_VKQ_0, t0);
            while (true) {
                load_tile(k_VKQ_0 + k_VKQ_step < k_VKQ_max ? k_VKQ_0 + k_VKQ_step : k_VKQ_0, t1);
                process_tile(t0);
                k_VKQ_0 += k_VKQ_step;
                if (k_VKQ_0 >= k_VKQ_max) {
                    break;
                }

                load_tile(k_VKQ_0 + k_VKQ_step < k_VKQ_max ? k_VKQ_0 + k_VKQ_step : k_VKQ_0, t0);
                process_tile(t1);
                k_VKQ_0 += k_VKQ_step;
                if (k_VKQ_0 >= k_VKQ_max) {
                    break;
                }
            }
        }
    } else {
        for (; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += k_VKQ_step) {
            tile_raw t;
            load_tile(k_VKQ_0, t);
            process_tile(t);
        }
    }

    KQ_sum += __shfl_xor(KQ_sum, 16, 64);
    KQ_sum += __shfl_xor(KQ_sum, 32, 64);

    // Combine the results of the warps:
    __shared__ float KQ_max_s[nwarps][16];
    __shared__ float KQ_sum_s[nwarps][16];
    __shared__ float KQ_meta_s[2][16];
    float (*VKQ_s)[D + 4] = (float (*)[D + 4]) data_s;

    if (g == 0) {
        KQ_max_s[threadIdx.y][jc] = KQ_max;
        KQ_sum_s[threadIdx.y][jc] = KQ_sum;
    }
    __syncthreads();

    float KQ_max_block = KQ_max_s[0][jc];
#pragma unroll
    for (int w = 1; w < nwarps; ++w) {
        KQ_max_block = fmaxf(KQ_max_block, KQ_max_s[w][jc]);
    }
    // Attention sink: only added for the first of all parallel blocks.
    const bool  use_sink = sinks && blockIdx.y == 0 && col_ok;
    const float sink     = use_sink ? ((const float *) sinks)[head0 + jc%ncols2] : -FLT_MAX/2.0f;
    KQ_max_block = fmaxf(KQ_max_block, sink);

    float KQ_sum_block = use_sink ? expf(sink - KQ_max_block) : 0.0f;
#pragma unroll
    for (int w = 0; w < nwarps; ++w) {
        KQ_sum_block += KQ_sum_s[w][jc]*expf(KQ_max_s[w][jc] - KQ_max_block);
    }
    if (threadIdx.y == 0 && g == 0) {
        KQ_meta_s[0][jc] = KQ_max_block;
        KQ_meta_s[1][jc] = KQ_sum_block;
    }

    const float VKQ_scale = expf(KQ_max - KQ_max_block);
#pragma unroll
    for (int w = 0; w < nwarps; ++w) {
        if (int(threadIdx.y) == w) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
#pragma unroll
                for (int u = 0; u < ncalls; u += 4) {
                    float4 * dst_s = (float4 *) &VKQ_s[jc][(4*g + v)*nv_V + u];
                    float4 tmp = make_float4(VKQ_scale*VKQ[u + 0][v], VKQ_scale*VKQ[u + 1][v], VKQ_scale*VKQ[u + 2][v], VKQ_scale*VKQ[u + 3][v]);
                    if (w > 0) {
                        const float4 prev = *dst_s;
                        tmp.x += prev.x;
                        tmp.y += prev.y;
                        tmp.z += prev.z;
                        tmp.w += prev.w;
                    }
                    *dst_s = tmp;
                }
            }
        }
        __syncthreads();
    }

    // Write back results, either final or as partial results for flash_attn_combine_results:
    const int tid = threadIdx.y*64 + threadIdx.x;
#pragma unroll
    for (int i0 = 0; i0 < 16*(D/4); i0 += nwarps*64) {
        const int i    = i0 + tid;
        const int jc_w = i / (D/4);
        const int k    = 4*(i % (D/4));
        if (!col_valid(jc_w)) {
            continue;
        }
        float4 tmp = *(const float4 *) &VKQ_s[jc_w][k];
        if (gridDim.y == 1) {
            const float KQ_sum_inv = 1.0f/KQ_meta_s[1][jc_w];
            tmp.x *= KQ_sum_inv;
            tmp.y *= KQ_sum_inv;
            tmp.z *= KQ_sum_inv;
            tmp.w *= KQ_sum_inv;
        }
        const int j_dst = ((sequence*int(ne01.z) + ic0 + jc_w/ncols2)*ne02 + head0 + jc_w%ncols2)*gridDim.y + blockIdx.y;
        *(float4 *) &dst[j_dst*D + k] = tmp;
    }

    if (gridDim.y != 1 && tid < 16 && col_valid(tid)) {
        const int j_dst = ((sequence*int(ne01.z) + ic0 + tid/ncols2)*ne02 + head0 + tid%ncols2)*gridDim.y + blockIdx.y;
        dst_meta[j_dst] = make_float2(KQ_meta_s[0][tid], KQ_meta_s[1][tid]);
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
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

template <int D, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps    = ggml_cuda_fattn_vec_get_nwarps();
    constexpr int warp_size = 64;
    fattn_kernel_t fattn_kernel = flash_attn_ext_vec<D, ncols1, ncols2, type_K, type_V, use_logit_softcap>;
    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;
    constexpr int min_kv_iter = 8;
    launch_fattn<D, ncols1, ncols2>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, nwarps*16, need_f16_K, need_f16_V, false, warp_size, min_kv_iter);
}

template <int D, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    float logit_softcap;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap == 0.0f) {
        ggml_cuda_flash_attn_ext_vec_case_impl<D, ncols1, ncols2, type_K, type_V, false>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_vec_case_impl<D, ncols1, ncols2, type_K, type_V, true>(ctx, dst);
    }
}

template <int D, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    float max_bias;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    // A block has 16 Q columns: ncols1 Q rows x ncols2 Q heads, ALiBi needs a slope per head.
    // Use the split with the fewest passes over K/V.
    const int ne1       = Q->ne[1];
    const int gqa_ratio = max_bias == 0.0f ? Q->ne[2] / K->ne[2] : 1;
    int ncols2_best  = 1;
    int npasses_best = INT_MAX;
    for (int ncols2 = 1; ncols2 <= 16; ncols2 *= 2) {
        const int npasses = ((ne1 + 16/ncols2 - 1) / (16/ncols2)) * ((gqa_ratio + ncols2 - 1) / ncols2);
        if (npasses < npasses_best) {
            npasses_best = npasses;
            ncols2_best  = ncols2;
        }
    }

    switch (ncols2_best) {
        case  1: ggml_cuda_flash_attn_ext_vec_case_softcap<D, 16,  1, type_K, type_V>(ctx, dst); break;
        case  2: ggml_cuda_flash_attn_ext_vec_case_softcap<D,  8,  2, type_K, type_V>(ctx, dst); break;
        case  4: ggml_cuda_flash_attn_ext_vec_case_softcap<D,  4,  4, type_K, type_V>(ctx, dst); break;
        case  8: ggml_cuda_flash_attn_ext_vec_case_softcap<D,  2,  8, type_K, type_V>(ctx, dst); break;
        case 16: ggml_cuda_flash_attn_ext_vec_case_softcap<D,  1, 16, type_K, type_V>(ctx, dst); break;
        default: GGML_ABORT("fatal error");
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_cuda_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_BF16); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_BF16)

EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q8_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_BF16)
