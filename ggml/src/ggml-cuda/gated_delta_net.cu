#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

// Chunked delta rule for prefill (scalar gate only), chunks of GDN_CHUNK tokens.
// Inside a chunk with cumulative log gate G and state S0 at the chunk start:
//   A[t][s] = beta_t k_t.k_s exp(G_t - G_s) (s < t), T = (I + A)^-1
//   W = T diag(beta exp(G)) K, U = T diag(beta) V, V' = U - W S0
//   O = scale (diag(exp(G)) Q S0 + tril(Q K^T exp(G_t - G_s)) V')
//   S = exp(G_last) S0 + (K exp(G_last - G))^T V'
// gdn_chunk_prep computes W, U, the masked Q K^T and G for all chunks in parallel,
// gdn_chunk_scan walks the chunks in order, each block owns BC state columns.
#define GDN_CHUNK 64
#define GDN_CHUNK_MIN_TOKENS 64

#if defined(AMD_MFMA_AVAILABLE)
typedef float gdn_floatx4 __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ gdn_floatx4 gdn_mfma(const float a, const float b, const gdn_floatx4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
}

// p must be readable, the value is replaced by 0 if !valid
static __device__ __forceinline__ float4 gdn_load4(const float * p, const bool valid) {
    const float4 v = *(const float4 *) p;
    return valid ? v : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

static __device__ __forceinline__ float gdn_get(const float4 v, const int c) {
    return c == 0 ? v.x : (c == 1 ? v.y : (c == 2 ? v.z : v.w));
}
#endif // defined(AMD_MFMA_AVAILABLE)

template <int S>
__global__ void __launch_bounds__(256, 3) gdn_chunk_prep(
        const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
        const float * __restrict__ g, const float * __restrict__ beta,
        float * __restrict__ W, float * __restrict__ U, float * __restrict__ Aqk, float * __restrict__ Gc,
        const int64_t H, const int64_t n_tokens, const int n_chunks,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int64_t sb1, const int64_t sb2, const int64_t sb3,
        const uint3 neqk1_magic, const uint3 rq3_magic) {
#if defined(AMD_MFMA_AVAILABLE)
    constexpr int C      = GDN_CHUNK;
    constexpr int nwaves = 4;

    const int chunk = blockIdx.x;
    const int h     = blockIdx.y;
    const int seq   = blockIdx.z;
    const int tid   = threadIdx.x;
    const int lane  = tid % 64;
    const int wave  = tid / 64;
    const int l16   = lane % 16;
    const int l4    = 4*(lane / 16);

    const int64_t t0      = (int64_t) chunk*C;
    const int     n_valid = (int) min((int64_t) C, n_tokens - t0);

    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(seq, rq3_magic);

    const float * qc = q + iq3*sq3 + t0*sq2 + iq1*sq1;
    const float * kc = k + iq3*sq3 + t0*sq2 + iq1*sq1;
    const float * vc = v + seq*sv3 + t0*sv2 + h*sv1;
    const int64_t gb = seq*sb3 + t0*sb2 + h*sb1;

    const int64_t base = ((int64_t) seq*H + h)*n_chunks + chunk;

    __shared__ float G_s[C];
    __shared__ float beta_s[C];
    __shared__ float xscale_s[C];
    // strictly lower part: A[t][s], upper part with diagonal: T[s][t] (T transposed)
    __shared__ float AT_s[C][C + 4];
    auto T_get = [&](const int r, const int c) {
        return c <= r ? AT_s[c][r] : 0.0f;
    };

    ggml_cuda_pdl_sync();

    if (wave == 0) {
        const bool valid = lane < n_valid;
        float G = valid ? g[gb + lane*sb2] : 0.0f;
#pragma unroll
        for (int offset = 1; offset < 64; offset <<= 1) {
            const float other = __shfl_up(G, offset, 64);
            G += lane >= offset ? other : 0.0f;
        }
        const float b = valid ? beta[gb + lane*sb2] : 0.0f;
        G_s[lane]      = G;
        beta_s[lane]   = b;
        xscale_s[lane] = b*expf(G);
        Gc[base*C + lane] = G;
    }
    __syncthreads();

    // K K^T and Q K^T, wave w computes rows [16m, 16m + 16) for m = w, w + nwaves, ...
    for (int m = wave; m < C/16; m += nwaves) {
        gdn_floatx4 kk[C/16];
        gdn_floatx4 qk[C/16];
#pragma unroll
        for (int nt = 0; nt < C/16; ++nt) {
            kk[nt] = {0.0f, 0.0f, 0.0f, 0.0f};
            qk[nt] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        const int  row_a   = 16*m + lane % 16;
        const bool valid_a = row_a < n_valid;
        const int  ra      = min(row_a, n_valid - 1);
#pragma unroll 2
        for (int k0 = 0; k0 < S; k0 += 16) {
            const int    kk0 = k0 + 4*(lane / 16);
            const float4 ka  = gdn_load4(kc + ra*sq2 + kk0, valid_a);
            const float4 qa  = gdn_load4(qc + ra*sq2 + kk0, valid_a);
#pragma unroll
            for (int nt = 0; nt < C/16; ++nt) {
                if (nt > m) {
                    break;
                }
                const int    row_b = 16*nt + lane % 16;
                const float4 kb    = gdn_load4(kc + min(row_b, n_valid - 1)*sq2 + kk0, row_b < n_valid);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    kk[nt] = gdn_mfma(gdn_get(ka, c), gdn_get(kb, c), kk[nt]);
                    qk[nt] = gdn_mfma(gdn_get(qa, c), gdn_get(kb, c), qk[nt]);
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < C/16; ++nt) {
            if (nt > m) {
                break;
            }
            const int s = 16*nt + lane % 16;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int   t     = 16*m + 4*(lane / 16) + r;
                const float decay = s <= t ? expf(G_s[t] - G_s[s]) : 0.0f;
                if (s < t) {
                    AT_s[t][s] = beta_s[t]*kk[nt][r]*decay;
                }
                Aqk[(base*C + t)*C + s] = qk[nt][r]*decay;
            }
        }
    }
    __syncthreads();

    // T = (I + A)^-1 by 16x16 blocks: invert the diagonal blocks, then T_ij = -T_ii sum_{j <= k < i} A_ik T_kj.
    if (lane < 16) {
        const int i0 = 16*wave;
        float x[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            float sum = r == lane ? 1.0f : 0.0f;
#pragma unroll
            for (int s0 = 0; s0 < r; ++s0) {
                sum -= AT_s[i0 + r][i0 + s0]*x[s0];
            }
            x[r] = sum;
            if (r >= lane) {
                AT_s[i0 + lane][i0 + r] = sum;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int d = 1; d < C/16; ++d) {
        __syncthreads();
        const int i = wave;
        const int j = i - d;
        if (j >= 0) {
            gdn_floatx4 P = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int kb = j; kb < i; ++kb) {
                const float4 a = *(const float4 *) &AT_s[16*i + l16][16*kb + l4];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    P = gdn_mfma(gdn_get(a, c), T_get(16*kb + l4 + c, 16*j + l16), P);
                }
            }
            gdn_floatx4 Tij = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                Tij = gdn_mfma(T_get(16*i + l16, 16*i + l4 + c), P[c], Tij);
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                AT_s[16*j + l16][16*i + l4 + r] = -Tij[r];
            }
        }
    }
    __syncthreads();

    // [W | U] = T [diag(beta exp(G)) K | diag(beta) V], wave w computes the 16-column blocks w, w + 4, ...
    for (int nb = wave; nb < 2*S/16; nb += 4) {
        const bool    is_w = nb < S/16;
        const int     col  = 16*(is_w ? nb : nb - S/16) + l16;
        const float * src  = is_w ? kc : vc;
        const int64_t ss   = is_w ? sq2 : sv2;
        gdn_floatx4 acc[C/16];
#pragma unroll
        for (int i = 0; i < C/16; ++i) {
            acc[i] = {0.0f, 0.0f, 0.0f, 0.0f};
        }
#pragma unroll
        for (int j = 0; j < C/16; ++j) {
            float xb[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int t = 16*j + l4 + c;
                xb[c] = (is_w ? xscale_s[t] : beta_s[t]) * src[min(t, n_valid - 1)*ss + col];
            }
#pragma unroll
            for (int i = j; i < C/16; ++i) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    acc[i] = gdn_mfma(T_get(16*i + l16, 16*j + l4 + c), xb[c], acc[i]);
                }
            }
        }
        float * out = (is_w ? W : U) + base*C*S + col;
#pragma unroll
        for (int i = 0; i < C/16; ++i) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                out[(16*i + l4 + r)*S] = acc[i][r];
            }
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, g, beta, W, U, Aqk, Gc, H, n_tokens, n_chunks, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic);
    NO_DEVICE_CODE;
#endif // defined(AMD_MFMA_AVAILABLE)
}

template <int S, int BC, bool keep_rs_t>
__global__ void __launch_bounds__(256, 1) gdn_chunk_scan(
        const float * __restrict__ q, const float * __restrict__ k,
        const float * __restrict__ W, const float * __restrict__ U, const float * __restrict__ Aqk, const float * __restrict__ Gc,
        const float * __restrict__ curr_state, float * __restrict__ dst, float * __restrict__ state,
        const int64_t H, const int64_t n_tokens, const int n_chunks,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        const float scale, const int64_t state_slot_stride, const int K) {
#if defined(AMD_MFMA_AVAILABLE)
    constexpr int C   = GDN_CHUNK;
    constexpr int NT  = BC/16; // state column tiles per block
    constexpr int NI  = S/64;  // state row tiles per wave
    constexpr int NK  = S/16;
    constexpr int SLD = S + 4;
    constexpr int CLD = C + 4;

    const int h    = blockIdx.x;
    const int seq  = blockIdx.y;
    const int col0 = blockIdx.z*BC;
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int l16  = lane % 16;
    const int l4   = 4*(lane / 16);

    const uint32_t iq1 = fastmodulo(h, neqk1_magic);
    const uint32_t iq3 = fastdiv(seq, rq3_magic);

    const float * qh = q + iq3*sq3 + iq1*sq1;
    const float * kh = k + iq3*sq3 + iq1*sq1;
    const int64_t base0 = ((int64_t) seq*H + h)*n_chunks;

    __shared__ float S_lds[BC*SLD]; // [col][row]
    __shared__ float V_lds[BC*CLD]; // [col][t]
    __shared__ float G_lds[2][C];

    float * attn = dst + (int64_t) seq*n_tokens*H*S + h*S + col0;
    curr_state += ((int64_t) seq*H + h)*S*S + (int64_t) col0*S;
    state      += ((int64_t) seq*H + h)*S*S + (int64_t) col0*S;

    // Operands of a chunk, loaded one chunk ahead. Rows past the end of the sequence load the last row:
    // their W, U and V' are 0 and their outputs are not written.
    float4 w_r[NK];         // W[16*wave + l16][16*kk + l4 ...]
    float4 q_r[NK];         // Q[16*wave + l16][16*kk + l4 ...]
    float  u_r[NT][4];      // U[16*wave + l4 + r][col0 + 16*nt + l16]
    float4 a_r[C/16];       // Aqk[16*wave + l16][16*sb + l4 ...]
    float  k_r[NI][C/4][4]; // K[16*tb + l4 + c][16*(wave + 4*j) + l16]

    auto load_wq = [&](const int chunk) {
        const int64_t t0      = (int64_t) chunk*C;
        const int     n_valid = (int) min((int64_t) C, n_tokens - t0);
        const int64_t base    = base0 + chunk;
        const float * Wt = W + (base*C + 16*wave + l16)*S + l4;
        const float * Qt = qh + (t0 + min(16*wave + l16, n_valid - 1))*sq2 + l4;
#pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
            w_r[kk] = *(const float4 *) (Wt + 16*kk);
            q_r[kk] = *(const float4 *) (Qt + 16*kk);
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                u_r[nt][r] = U[(base*C + 16*wave + l4 + r)*S + col0 + 16*nt + l16];
            }
        }
    };
    auto load_a = [&](const int chunk) {
        const float * At = Aqk + ((base0 + chunk)*C + 16*wave + l16)*C + l4;
#pragma unroll
        for (int sb = 0; sb < C/16; ++sb) {
            a_r[sb] = *(const float4 *) (At + 16*sb);
        }
    };
    auto load_k = [&](const int chunk) {
        const int64_t t0      = (int64_t) chunk*C;
        const int     n_valid = (int) min((int64_t) C, n_tokens - t0);
        const float * Kc = kh + t0*sq2;
#pragma unroll
        for (int j = 0; j < NI; ++j) {
#pragma unroll
            for (int tb = 0; tb < C/16; ++tb) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    k_r[j][tb][c] = Kc[min(16*tb + l4 + c, n_valid - 1)*sq2 + 16*(wave + 4*j) + l16];
                }
            }
        }
    };

    ggml_cuda_pdl_sync();

    load_wq(0);
    load_a(0);
    load_k(0);

    // state tile [16*(wave + 4*j) + l4 + r][16*nt + l16], kept in registers
    gdn_floatx4 st[NI][NT];
#pragma unroll
    for (int j = 0; j < NI; ++j) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int i = 16*(wave + 4*j) + l4;
            const int n = 16*nt + l16;
            const float4 tmp = *(const float4 *) (curr_state + n*S + i);
            st[j][nt] = {tmp.x, tmp.y, tmp.z, tmp.w};
            *(float4 *) &S_lds[n*SLD + i] = tmp;
        }
    }
    if (wave == 0) {
        G_lds[0][lane] = Gc[base0*C + lane];
    }
    __syncthreads();

    // S_p = exp(G_tp) S0 + sum_{t <= tp} K[t]^T exp(G_tp - G_t) V'[t], for the state rows of this wave
    auto state_at = [&](const int tp, const float * G, gdn_floatx4 (&res)[NI][NT]) {
        const float Gp = G[tp];
        const float decay_p = expf(Gp);
#pragma unroll
        for (int j = 0; j < NI; ++j) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                res[j][nt] = st[j][nt]*decay_p;
            }
        }
#pragma unroll
        for (int tb = 0; tb < C/16; ++tb) {
            if (16*tb > tp) {
                break;
            }
            const int    t  = 16*tb + l4;
            const float4 Gt = *(const float4 *) &G[t];
            float d[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                d[c] = t + c <= tp ? expf(Gp - gdn_get(Gt, c)) : 0.0f;
            }
            float4 vb[NT];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                vb[nt] = *(const float4 *) &V_lds[(16*nt + l16)*CLD + t];
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
#pragma unroll
                for (int j = 0; j < NI; ++j) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        res[j][nt] = gdn_mfma(k_r[j][tb][c], gdn_get(vb[nt], c)*d[c], res[j][nt]);
                    }
                }
            }
        }
    };

    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int64_t t0      = (int64_t) chunk*C;
        const int     n_valid = (int) min((int64_t) C, n_tokens - t0);
        const float * G       = G_lds[chunk & 1];
        const bool    next    = chunk + 1 < n_chunks;

        // V' = U - W S0 and Q S0 for rows [16*wave, 16*wave + 16)
        gdn_floatx4 vp[2][NT];
        gdn_floatx4 o[2][NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                vp[e][nt] = {0.0f, 0.0f, 0.0f, 0.0f};
                o[e][nt]  = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
#pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const float4 sb = *(const float4 *) &S_lds[(16*nt + l16)*SLD + 16*kk + l4];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    vp[kk % 2][nt] = gdn_mfma(gdn_get(w_r[kk], c), gdn_get(sb, c), vp[kk % 2][nt]);
                    o[kk % 2][nt]  = gdn_mfma(gdn_get(q_r[kk], c), gdn_get(sb, c), o[kk % 2][nt]);
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const gdn_floatx4 v = vp[0][nt] + vp[1][nt];
            const float4 tmp = make_float4(u_r[nt][0] - v[0], u_r[nt][1] - v[1], u_r[nt][2] - v[2], u_r[nt][3] - v[3]);
            *(float4 *) &V_lds[(16*nt + l16)*CLD + 16*wave + l4] = tmp;
        }
        if (next) {
            load_wq(chunk + 1);
        }
        __syncthreads();

        // O = scale (exp(G) Q S0 + Aqk V')
        {
            gdn_floatx4 out[NT];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                out[nt] = o[0][nt] + o[1][nt];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    out[nt][r] *= expf(G[16*wave + l4 + r]);
                }
            }
#pragma unroll
            for (int sb = 0; sb < C/16; ++sb) {
                if (sb > wave) {
                    break;
                }
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const float4 vb = *(const float4 *) &V_lds[(16*nt + l16)*CLD + 16*sb + l4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        out[nt] = gdn_mfma(gdn_get(a_r[sb], c), gdn_get(vb, c), out[nt]);
                    }
                }
            }
            if (next) {
                load_a(chunk + 1);
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int t = 16*wave + l4 + r;
                if (t < n_valid) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        attn[(t0 + t)*H*S + 16*nt + l16] = out[nt][r]*scale;
                    }
                }
            }
        }

        if constexpr (keep_rs_t) {
            // snapshot slot 0 = state after the last token, slot s = s tokens back
            const int tp_first = max(0, (int) (n_tokens - K - t0));
            for (int tp = tp_first; tp < n_valid; ++tp) {
                gdn_floatx4 snap[NI][NT];
                state_at(tp, G, snap);
                float * dst_s = state + (n_tokens - 1 - (t0 + tp))*state_slot_stride;
#pragma unroll
                for (int j = 0; j < NI; ++j) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        const int i = 16*(wave + 4*j) + l4;
                        const int n = 16*nt + l16;
                        *(float4 *) (dst_s + n*S + i) = make_float4(snap[j][nt][0], snap[j][nt][1], snap[j][nt][2], snap[j][nt][3]);
                    }
                }
            }
        }

        gdn_floatx4 st_new[NI][NT];
        state_at(C - 1, G, st_new);
        if (next) {
            load_k(chunk + 1);
        }
#pragma unroll
        for (int j = 0; j < NI; ++j) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                st[j][nt] = st_new[j][nt];
                const int i = 16*(wave + 4*j) + l4;
                const int n = 16*nt + l16;
                *(float4 *) &S_lds[n*SLD + i] = make_float4(st[j][nt][0], st[j][nt][1], st[j][nt][2], st[j][nt][3]);
            }
        }
        if (wave == 0 && next) {
            G_lds[(chunk + 1) & 1][lane] = Gc[(base0 + chunk + 1)*C + lane];
        }
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int j = 0; j < NI; ++j) {
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int i = 16*(wave + 4*j) + l4;
                const int n = 16*nt + l16;
                *(float4 *) (state + n*S + i) = make_float4(st[j][nt][0], st[j][nt][1], st[j][nt][2], st[j][nt][3]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(q, k, W, U, Aqk, Gc, curr_state, dst, state, H, n_tokens, n_chunks, sq1, sq2, sq3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
    NO_DEVICE_CODE;
#endif // defined(AMD_MFMA_AVAILABLE)
}

template <int S, bool keep_rs_t>
static void launch_gated_delta_net_chunked(
        ggml_backend_cuda_context & ctx,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    constexpr int C  = GDN_CHUNK;
    const int n_chunks = (n_tokens + C - 1) / C;
    const int64_t nblk = n_seqs*H*n_chunks;

    ggml_cuda_pool_alloc<float> W  (ctx.pool(), nblk*C*S);
    ggml_cuda_pool_alloc<float> U  (ctx.pool(), nblk*C*S);
    ggml_cuda_pool_alloc<float> Aqk(ctx.pool(), nblk*C*C);
    ggml_cuda_pool_alloc<float> Gc (ctx.pool(), nblk*C);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    gdn_chunk_prep<S><<<dim3(n_chunks, H, n_seqs), 256, 0, stream>>>(
        q_d, k_d, v_d, g_d, b_d, W.get(), U.get(), Aqk.get(), Gc.get(), H, n_tokens, n_chunks,
        sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic);

    // smallest column split that still fits in one wave of blocks, the scan is latency bound
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    auto launch_scan = [&](auto BC_c) {
        constexpr int BC = decltype(BC_c)::value;
        gdn_chunk_scan<S, BC, keep_rs_t><<<dim3(H, n_seqs, S/BC), 256, 0, stream>>>(
            q_d, k_d, W.get(), U.get(), Aqk.get(), Gc.get(), s_d, dst_d, state_d, H, n_tokens, n_chunks,
            sq1, sq2, sq3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
    };
    if (H*n_seqs*(S/16) <= nsm) {
        launch_scan(std::integral_constant<int, 16>{});
    } else if (H*n_seqs*(S/32) <= nsm) {
        launch_scan(std::integral_constant<int, 32>{});
    } else {
        launch_scan(std::integral_constant<int, 64>{});
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    // chunked path: scalar gate, S_v 64 or 128, 16-byte aligned q/k rows
    const bool aligned = (((uintptr_t) q_d | (uintptr_t) k_d) % 16 == 0) && sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0;
    if (!kda && aligned && (S_v == 64 || S_v == 128) && n_tokens >= GDN_CHUNK_MIN_TOKENS) {
        auto launch = [&](auto S_c, auto keep_c) {
            launch_gated_delta_net_chunked<decltype(S_c)::value, decltype(keep_c)::value>(ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        };
        if (S_v == 64) {
            keep_rs ? launch(std::integral_constant<int, 64>{}, std::true_type{}) : launch(std::integral_constant<int, 64>{}, std::false_type{});
        } else {
            keep_rs ? launch(std::integral_constant<int, 128>{}, std::true_type{}) : launch(std::integral_constant<int, 128>{}, std::false_type{});
        }
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
