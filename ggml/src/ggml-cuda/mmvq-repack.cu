#include "repack.cuh"
#include "unary.cuh"

#include <algorithm>
#include <type_traits>

// GEMV on repacked weights (MMVQ-R): one lane = one row, a wave = one stripe of 64 rows,
// the waves of a block split K. Each lane loads the chunks of its own block (1 KB contiguous per wave
// and load instruction). The q8_1 activation is the same for all lanes and is read with scalar loads.

#define MMVQ_REPACK_MAX_WARPS 8

struct mmvq_repack_args {
    const char       * x;
    const char       * gate;
    const block_q8_1 * y;
    const int32_t    * ids;
    float            * dst;
    const float      * x_bias;
    const float      * gate_bias;
    ggml_glu_op        glu_op;
    float              glu_limit;
    int                nkb;                // blocks per row
    int                nrows;              // rows per matrix
    int64_t            stride_channel_x;   // bytes
    int64_t            stride_sample_x;    // bytes
    int                stride_col_y;       // q8_1 blocks
    int                stride_channel_y;
    int64_t            stride_sample_y;
    int                stride_col_dst;     // floats
    int                stride_channel_dst;
    int64_t            stride_sample_dst;
    uint3              channel_ratio;
    uint3              sample_ratio;
    uint3              nchannels_y;
    int                ids_stride;
    int                kz;                 // blocks per stripe that split K
    float            * partial;            // kz > 1: partial sums [tile][kz][gate][ncols][64]
    int              * counters;           // kz > 1: finished blocks per tile
};

template <ggml_type type, int ncols>
struct mmvq_repack_block;

// For the K-quants a loop step is one block (G = 1). For Q4_0/Q8_0 it is a group of G = 8 blocks,
// whose d values are one 16 byte rest chunk.

// Q4_K chunks: 0 = d, dmin, scales; 1-8 = qs.
template <int ncols>
struct mmvq_repack_block<GGML_TYPE_Q4_K, ncols> {
    static constexpr int nchunk = 9;
    static constexpr int G      = 1;

    static __device__ __forceinline__ void dot(const int4 * v, const int *, const block_q8_1 * const * y, float * acc) {
        const int * q = (const int *) v;
        const float2 dm = __half22float2(*(const half2 *) &q[0]);
        const int scs[2] = {q[1] & 0x3f3f3f3f, (q[3] & 0x0f0f0f0f) | ((q[1] >> 2) & 0x30303030)};
        const int ms[2]  = {q[2] & 0x3f3f3f3f, ((q[3] >> 4) & 0x0f0f0f0f) | ((q[2] >> 2) & 0x30303030)};
        const int * qs = q + 4;
        float sumd[ncols] = {0.0f};
        float summ[ncols] = {0.0f};
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            int vv[8];
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                vv[m] = (qs[(s >> 1)*8 + m] >> (4*(s & 1))) & 0x0f0f0f0f;
            }
            const int scv = (scs[s >> 2] >> (8*(s & 3))) & 0xff;
            const int mv  = (ms[s >> 2]  >> (8*(s & 3))) & 0xff;
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const block_q8_1 * yb = y[j] + s;
                const int * yq = (const int *) yb->qs;
                int dot = 0;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    dot = ggml_cuda_dp4a(vv[m], yq[m], dot);
                }
                const float2 ds = __half22float2(yb->ds);
                sumd[j] += ds.x * (float) (dot*scv);
                summ[j] += ds.y * (float) mv;
            }
        }
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            acc[j] += dm.x*sumd[j] - dm.y*summ[j];
        }
    }
};

// Q8_0: chunks 0-1 = qs, rest = d.
template <int ncols>
struct mmvq_repack_block<GGML_TYPE_Q8_0, ncols> {
    static constexpr int nchunk = 2;
    static constexpr int G      = 8;

    static __device__ __forceinline__ void dot(const int4 * v, const int * rest, const block_q8_1 * const * y, float * acc) {
        const float d = __half2float(__ushort_as_half((uint16_t) rest[0]));
        const int * q = (const int *) v;
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const int * yq = (const int *) y[j]->qs;
            int dot = 0;
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                dot = ggml_cuda_dp4a(q[m], yq[m], dot);
            }
            acc[j] += d * __low2float(y[j]->ds) * (float) dot;
        }
    }
};

// Q4_0: chunk 0 = qs, rest = d.
template <int ncols>
struct mmvq_repack_block<GGML_TYPE_Q4_0, ncols> {
    static constexpr int nchunk = 1;
    static constexpr int G      = 8;

    static __device__ __forceinline__ void dot(const int4 * v, const int * rest, const block_q8_1 * const * y, float * acc) {
        const float d = __half2float(__ushort_as_half((uint16_t) rest[0]));
        const int * q = (const int *) v;
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const int * yq = (const int *) y[j]->qs;
            int dot = 0;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                dot = ggml_cuda_dp4a((q[m] >> 0) & 0x0f0f0f0f, yq[m + 0], dot);
                dot = ggml_cuda_dp4a((q[m] >> 4) & 0x0f0f0f0f, yq[m + 4], dot);
            }
            const float2 ds = __half22float2(y[j]->ds);
            acc[j] += d * (ds.x*dot - 8.0f*ds.y);
        }
    }
};

// Q5_K chunks: 0 = d, dmin, scales; 1-2 = qh; 3-10 = qs.
template <int ncols>
struct mmvq_repack_block<GGML_TYPE_Q5_K, ncols> {
    static constexpr int nchunk = 11;
    static constexpr int G      = 1;

    static __device__ __forceinline__ void dot(const int4 * v, const int *, const block_q8_1 * const * y, float * acc) {
        const int * q = (const int *) v;
        const float2 dm = __half22float2(*(const half2 *) &q[0]);
        const int scs[2] = {q[1] & 0x3f3f3f3f, (q[3] & 0x0f0f0f0f) | ((q[1] >> 2) & 0x30303030)};
        const int ms[2]  = {q[2] & 0x3f3f3f3f, ((q[3] >> 4) & 0x0f0f0f0f) | ((q[2] >> 2) & 0x30303030)};
        const int * qh = q + 4;
        const int * qs = q + 12;
        float sumd[ncols] = {0.0f};
        float summ[ncols] = {0.0f};
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            int vv[8];
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                vv[m] = ((qs[(s >> 1)*8 + m] >> (4*(s & 1))) & 0x0f0f0f0f) | (((qh[m] >> s) & 0x01010101) << 4);
            }
            const int scv = (scs[s >> 2] >> (8*(s & 3))) & 0xff;
            const int mv  = (ms[s >> 2]  >> (8*(s & 3))) & 0xff;
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const block_q8_1 * yb = y[j] + s;
                const int * yq = (const int *) yb->qs;
                int dot = 0;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    dot = ggml_cuda_dp4a(vv[m], yq[m], dot);
                }
                const float2 ds = __half22float2(yb->ds);
                sumd[j] += ds.x * (float) (dot*scv);
                summ[j] += ds.y * (float) mv;
            }
        }
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            acc[j] += dm.x*sumd[j] - dm.y*summ[j];
        }
    }
};

// IQ4_XS: chunk c = qs of sub-block c; rest = d, scales_h, scales_l.
template <int ncols>
struct mmvq_repack_block<GGML_TYPE_IQ4_XS, ncols> {
    static constexpr int nchunk = 8;
    static constexpr int G      = 1; // one block per step is faster than the rest group of 2

    static __device__ __forceinline__ void dot(const int4 * v, const int * rest, const block_q8_1 * const * y, float * acc) {
        const int * q = (const int *) v;
        const float    d        = __half2float(__ushort_as_half((uint16_t) rest[0]));
        const uint32_t scales_h = (uint32_t) rest[0] >> 16;
        const uint32_t scales_l = rest[1];
        float sumf[ncols] = {0.0f};
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            int vv[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int2 t = ggml_cuda_repack_iq4nl_lut(q[4*ib + j]);
                vv[j + 0] = t.x;
                vv[j + 4] = t.y;
            }
            const int ls = ((scales_l >> (4*ib)) & 0x0F) | (((scales_h >> (2*ib)) & 0x03) << 4);
#pragma unroll
            for (int jc = 0; jc < ncols; ++jc) {
                const block_q8_1 * yb = y[jc] + ib;
                const int * yq = (const int *) yb->qs;
                int dot = 0;
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    dot = ggml_cuda_dp4a(vv[m], yq[m], dot);
                }
                sumf[jc] += __low2float(yb->ds) * (float) (dot*(ls - 32));
            }
        }
#pragma unroll
        for (int jc = 0; jc < ncols; ++jc) {
            acc[jc] += d*sumf[jc];
        }
    }
};

// Q6_K chunks: 0-7 = ql, 8-11 = qh, 12 = scales; rest = d.
template <int ncols>
struct mmvq_repack_block<GGML_TYPE_Q6_K, ncols> {
    static constexpr int nchunk = 13;
    static constexpr int G      = 1;

    static __device__ __forceinline__ void dot(const int4 * v, const int * rest, const block_q8_1 * const * y, float * acc) {
        const float d = __half2float(__ushort_as_half((uint16_t) rest[0]));
        const int * q = (const int *) v;
        const int * ql = q;
        const int * qh = q + 32;
        const int8_t * sc = (const int8_t *) (q + 48);
        float sumf[ncols] = {0.0f};
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int sb = 4*n + k;
                int vv[8];
#pragma unroll
                for (int m = 0; m < 8; ++m) {
                    const int lo = (ql[16*n + 8*(k & 1) + m] >> (4*(k >> 1))) & 0x0f0f0f0f;
                    const int hi = ((qh[8*n + m] >> (2*k)) & 0x03030303) << 4;
                    // q - 32 per byte, without borrows between the bytes
                    vv[m] = (((lo | hi) | 0x80808080) - 0x20202020) ^ 0x80808080;
                }
                const int sc0 = sc[8*n + 2*k];
                const int sc1 = sc[8*n + 2*k + 1];
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    const block_q8_1 * yb = y[j] + sb;
                    const int * yq = (const int *) yb->qs;
                    int dot0 = 0;
                    int dot1 = 0;
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        dot0 = ggml_cuda_dp4a(vv[m],     yq[m],     dot0);
                        dot1 = ggml_cuda_dp4a(vv[m + 4], yq[m + 4], dot1);
                    }
                    sumf[j] += __low2float(yb->ds) * (float) (dot0*sc0 + dot1*sc1);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            acc[j] += d*sumf[j];
        }
    }
};

template <ggml_type type, int ncols, bool has_gate>
__launch_bounds__(MMVQ_REPACK_MAX_WARPS*64, 1)
static __global__ void mul_mat_vec_q_repack(const mmvq_repack_args a) {
    constexpr ggml_cuda_repack_layout L = ggml_cuda_repack_get_layout(type);
    constexpr int NC = L.nchunk;
    using blk = mmvq_repack_block<type, ncols>;
    static_assert(blk::nchunk == NC, "layout mismatch");

    const int lane = threadIdx.x;
    const int w    = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int nw   = blockDim.y;

    const int stripe = blockIdx.x / a.kz;
    const int kzi    = blockIdx.x - stripe*a.kz;
    const int s0     = stripe*GGML_CUDA_REPACK_ROWS;
    const int r   = min(GGML_CUDA_REPACK_ROWS, a.nrows - s0);
    const int row = min(lane, r - 1);
    const int nkb = a.nkb;

    // MUL_MAT_ID: one token per blockIdx.z, one used expert per blockIdx.y
    const uint32_t channel_dst = blockIdx.y;
    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;
    uint32_t col_dst;
    if (a.ids) {
        col_dst    = blockIdx.z;
        sample_dst = 0;
        channel_x  = a.ids[channel_dst + col_dst*a.ids_stride];
        channel_y  = fastmodulo(channel_dst, a.nchannels_y);
    } else {
        col_dst    = 0;
        sample_dst = blockIdx.z;
        channel_x  = fastdiv(channel_dst, a.channel_ratio);
        channel_y  = channel_dst;
    }
    const uint32_t sample_x = fastdiv(sample_dst, a.sample_ratio);

    const int64_t xoff = sample_x*a.stride_sample_x + channel_x*a.stride_channel_x + (int64_t) s0*nkb*L.bs;
    const char * sx = a.x + xoff;
    const char * sg = has_gate ? a.gate + xoff : nullptr;
    const block_q8_1 * y = a.y + sample_dst*a.stride_sample_y + channel_y*a.stride_channel_y + col_dst*a.stride_col_y;

    constexpr int G = blk::G;
    const int nu  = (nkb + G - 1) / G; // loop steps of G blocks
    const int wg  = kzi*nw + w;
    const int nwg = a.kz*nw;
    const int u0  = (wg*nu) / nwg;
    const int u1  = ((wg + 1)*nu) / nwg;

    float acc[ncols]  = {0.0f};
    float accg[ncols] = {0.0f};

    // a unit of G blocks; only the last unit of a row can be partial, it has its own code without branches in the full one
    const auto unit = [&](const int u, auto full_tag) {
        constexpr bool full = decltype(full_tag)::value;
        const int kb0 = u*G;
        const int nb  = full ? G : min(G, nkb - kb0);
        int4 v[G][NC];
        int4 vg[G][has_gate ? NC : 1];
        const int64_t pstride = (int64_t) nkb*r*16;
#pragma unroll
        for (int b = 0; b < G; ++b) {
            if (full || b < nb) {
                const int64_t o = ((int64_t) (kb0 + b)*r + row)*16;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    v[b][c] = *(const int4 *) (sx + o + c*pstride);
                    if constexpr (has_gate) {
                        vg[b][c] = *(const int4 *) (sg + o + c*pstride);
                    }
                }
            }
        }
        // rest bytes of each block: one 16 byte load for a whole group, else one load per block
        int rw[G][2] = {};
        int rg[G][2] = {};
        if constexpr (L.rest > 0) {
            static_assert(L.rest == 2 || L.rest == 8, "unsupported rest size");
            const int64_t ro = ggml_cuda_repack_rest_offset(L, kb0, row, r, nkb);
            const auto load_rest = [&](const char * p, int (*w)[2]) {
                if (G > 1 && full) {
                    static_assert(G == 1 || G*L.rest == 16, "rest group must be one chunk");
                    const int4 t = *(const int4 *) (p + ro);
                    const int tv[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
                    for (int b = 0; b < G; ++b) {
                        if constexpr (L.rest == 2) {
                            w[b][0] = (tv[b/2] >> (16*(b % 2))) & 0xFFFF;
                        } else {
                            w[b][0] = tv[2*b + 0];
                            w[b][1] = tv[2*b + 1];
                        }
                    }
                } else {
#pragma unroll
                    for (int b = 0; b < G; ++b) {
                        if (full || b < nb) {
                            if constexpr (L.rest == 2) {
                                w[b][0] = *(const uint16_t *) (p + ro + 2*b);
                            } else {
                                const int2 t = *(const int2 *) (p + ro + 8*b);
                                w[b][0] = t.x;
                                w[b][1] = t.y;
                            }
                        }
                    }
                }
            };
            load_rest(sx, rw);
            if constexpr (has_gate) {
                load_rest(sg, rg);
            }
        }
#pragma unroll
        for (int b = 0; b < G; ++b) {
            if (full || b < nb) {
                const block_q8_1 * yk[ncols];
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    yk[j] = y + j*a.stride_col_y + (kb0 + b)*(ggml_cuda_type_traits<type>::qk/QK8_1);
                }
                blk::dot(v[b], rw[b], yk, acc);
                if constexpr (has_gate) {
                    blk::dot(vg[b], rg[b], yk, accg);
                }
            }
        }
    };

    for (int u = u0; u < u1; ++u) {
        if (G == 1 || (u + 1)*G <= nkb) {
            unit(u, std::true_type{});
        } else {
            unit(u, std::false_type{});
        }
    }

    __shared__ float red[MMVQ_REPACK_MAX_WARPS - 1][has_gate ? 2 : 1][ncols][64];
    if (w > 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            red[w - 1][0][j][lane] = acc[j];
            if constexpr (has_gate) {
                red[w - 1][has_gate ? 1 : 0][j][lane] = accg[j];
            }
        }
    }
    __syncthreads();
    if (w > 0) {
        return;
    }

    float sums[ncols];
    float sumsg[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        sums[j]  = acc[j];
        sumsg[j] = accg[j];
        for (int l = 0; l < nw - 1; ++l) {
            sums[j] += red[l][0][j][lane];
            if constexpr (has_gate) {
                sumsg[j] += red[l][has_gate ? 1 : 0][j][lane];
            }
        }
    }

    if (a.kz > 1) {
        // split K over blocks: the last block of the tile adds the partial sums in a fixed order
        constexpr int ng = has_gate ? 2 : 1;
        const int tile = (blockIdx.z*gridDim.y + blockIdx.y)*(gridDim.x / a.kz) + stripe;
        float * part = a.partial + (int64_t) tile*a.kz*ng*ncols*64;
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            part[(kzi*ng*ncols + j)*64 + lane] = sums[j];
            if constexpr (has_gate) {
                part[(kzi*ng*ncols + ncols + j)*64 + lane] = sumsg[j];
            }
        }
        __threadfence();
        int done = 0;
        if (lane == 0) {
            done = atomicAdd(&a.counters[tile], 1);
        }
        done = __shfl(done, 0, 64);
        if (done != a.kz - 1) {
            return;
        }
        __threadfence();
        const volatile float * vpart = part;
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            sums[j]  = 0.0f;
            sumsg[j] = 0.0f;
            for (int k = 0; k < a.kz; ++k) {
                sums[j] += vpart[(k*ng*ncols + j)*64 + lane];
                if constexpr (has_gate) {
                    sumsg[j] += vpart[(k*ng*ncols + ncols + j)*64 + lane];
                }
            }
        }
        if (lane == 0) {
            a.counters[tile] = 0;
        }
    }

    if (lane >= r) {
        return;
    }

    float * dst = a.dst + sample_dst*a.stride_sample_dst + channel_dst*a.stride_channel_dst + col_dst*a.stride_col_dst + s0 + lane;
    const uint32_t channel_bias = a.ids ? channel_x : channel_dst;
    const int64_t  bias_off     = sample_dst*a.stride_sample_dst + channel_bias*a.stride_channel_dst + s0 + lane;
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        float sum  = sums[j];
        float sumg = sumsg[j];
        if (a.x_bias) {
            sum += a.x_bias[bias_off + j*a.stride_col_dst];
        }
        if constexpr (has_gate) {
            if (a.gate_bias) {
                sumg += a.gate_bias[bias_off + j*a.stride_col_dst];
            }
            switch (a.glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    sum *= ggml_cuda_op_silu_single(sumg);
                    break;
                case GGML_GLU_OP_GEGLU:
                    sum *= ggml_cuda_op_gelu_single(sumg);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    sum = ggml_cuda_op_swiglu_oai_single(sumg, sum);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    sum = ggml_cuda_op_swiglu_clamp_single(sumg, sum, a.glu_limit);
                    break;
                default:
                    sum *= sumg;
                    break;
            }
        }
        dst[j*a.stride_col_dst] = sum;
    }
}

template <ggml_type type, int ncols>
static void mul_mat_vec_q_repack_launch(const mmvq_repack_args & a, const dim3 grid, const int nw, cudaStream_t stream) {
    const dim3 block(64, nw);
    if constexpr (ncols == 1) {
        if (a.gate) {
            mul_mat_vec_q_repack<type, ncols, true><<<grid, block, 0, stream>>>(a);
            return;
        }
    }
    GGML_ASSERT(!a.gate && "gate fusion only for one column");
    mul_mat_vec_q_repack<type, ncols, false><<<grid, block, 0, stream>>>(a);
}

template <ggml_type type>
static void mul_mat_vec_q_repack_switch_ncols(const mmvq_repack_args & a, const int ncols, const dim3 grid, const int nw, cudaStream_t stream) {
    switch (ncols) {
        case 1: mul_mat_vec_q_repack_launch<type, 1>(a, grid, nw, stream); break;
        case 2: mul_mat_vec_q_repack_launch<type, 2>(a, grid, nw, stream); break;
        case 3: mul_mat_vec_q_repack_launch<type, 3>(a, grid, nw, stream); break;
        case 4: mul_mat_vec_q_repack_launch<type, 4>(a, grid, nw, stream); break;
        case 5: mul_mat_vec_q_repack_launch<type, 5>(a, grid, nw, stream); break;
        case 6: mul_mat_vec_q_repack_launch<type, 6>(a, grid, nw, stream); break;
        case 7: mul_mat_vec_q_repack_launch<type, 7>(a, grid, nw, stream); break;
        case 8: mul_mat_vec_q_repack_launch<type, 8>(a, grid, nw, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_vec_q_repack(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_dst, const int stride_col_y, const int stride_col_dst,
        const int nchannels_y, const int nchannels_dst, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_dst, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    GGML_ASSERT(MMVQ_REPACK_MAX_WARPS*64 <= 1024);
    const ggml_type type = src0->type;
    const int64_t nkb = src0->ne[0] / ggml_blck_size(type);

    mmvq_repack_args a = {};
    a.x                  = (const char *) src0->data;
    a.gate               = (const char *) fusion.gate;
    a.y                  = (const block_q8_1 *) vy;
    a.ids                = ids;
    a.dst                = dst;
    a.x_bias             = (const float *) fusion.x_bias;
    a.gate_bias          = (const float *) fusion.gate_bias;
    a.glu_op             = fusion.glu_op;
    a.glu_limit          = fusion.glu_limit;
    a.nkb                = nkb;
    a.nrows              = src0->ne[1];
    a.stride_channel_x   = src0->nb[2];
    a.stride_sample_x    = src0->nb[3];
    a.stride_col_y       = stride_col_y;
    a.stride_channel_y   = stride_channel_y;
    a.stride_sample_y    = stride_sample_y;
    a.stride_col_dst     = stride_col_dst;
    a.stride_channel_dst = stride_channel_dst;
    a.stride_sample_dst  = stride_sample_dst;
    a.channel_ratio      = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / src0->ne[2]);
    a.sample_ratio       = init_fastdiv_values(nsamples_dst / src0->ne[3]);
    a.nchannels_y        = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    a.ids_stride         = ids_stride;

    const int nstripes = (src0->ne[1] + GGML_CUDA_REPACK_ROWS - 1) / GGML_CUDA_REPACK_ROWS;
    // MUL_MAT_ID: one column per block, the tokens are in blockIdx.z
    const int ncols  = ids ? 1 : ncols_dst;
    const int ntiles = nstripes*nchannels_dst*(ids ? ncols_dst : nsamples_dst);

    // waves per block and blocks per stripe: 8 waves per block if K allows it, more blocks only for few stripes
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    // (1024 x 5120 Q5_K: 16 stripes -> 5 blocks of 4 waves per stripe, 11.7 -> 7.2 us)
    const int G  = type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0 ? 8 : 1;
    const int nu = (nkb + G - 1)/G; // loop steps
    int kz = 1;
    if (4*ntiles < nsm) {
        kz = std::max<int>(1, std::min<int>(nu/4, (nsm + ntiles - 1)/ntiles));
    }
    int nw = MMVQ_REPACK_MAX_WARPS;
    while (nw > 1 && kz*nw > nu) {
        nw /= 2;
    }
    a.kz = kz;

    ggml_cuda_pool_alloc<float> partial(ctx.pool());
    if (kz > 1) {
        // the counter ring is only safe on the main stream, see repack_counters_get
        GGML_ASSERT(stream == ctx.stream());
        a.partial  = partial.alloc((size_t) ntiles*kz*(fusion.gate ? 2 : 1)*ncols*64);
        a.counters = ctx.repack_counters_get(ntiles);
    }

    const dim3 grid(nstripes*kz, nchannels_dst, ids ? ncols_dst : nsamples_dst);

    switch (type) {
        case GGML_TYPE_Q4_0: mul_mat_vec_q_repack_switch_ncols<GGML_TYPE_Q4_0>(a, ncols, grid, nw, stream); break;
        case GGML_TYPE_Q8_0: mul_mat_vec_q_repack_switch_ncols<GGML_TYPE_Q8_0>(a, ncols, grid, nw, stream); break;
        case GGML_TYPE_Q4_K: mul_mat_vec_q_repack_switch_ncols<GGML_TYPE_Q4_K>(a, ncols, grid, nw, stream); break;
        case GGML_TYPE_Q5_K: mul_mat_vec_q_repack_switch_ncols<GGML_TYPE_Q5_K>(a, ncols, grid, nw, stream); break;
        case GGML_TYPE_Q6_K: mul_mat_vec_q_repack_switch_ncols<GGML_TYPE_Q6_K>(a, ncols, grid, nw, stream); break;
        case GGML_TYPE_IQ4_XS: mul_mat_vec_q_repack_switch_ncols<GGML_TYPE_IQ4_XS>(a, ncols, grid, nw, stream); break;
        default: GGML_ABORT("unsupported repack type %s", ggml_type_name(type));
    }
    CUDA_CHECK(cudaGetLastError());
}
