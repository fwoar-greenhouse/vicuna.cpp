#include "fattn-cdna.cuh"

#ifndef FATTN_CDNA_NP
#define FATTN_CDNA_NP 1
#endif
#ifndef FATTN_CDNA_NWARPS
#define FATTN_CDNA_NWARPS 8
#endif

static bool ggml_cuda_fattn_cdna_type_ok(const ggml_type type_K, const ggml_type type_V) {
    return (type_K == GGML_TYPE_F16  && type_V == GGML_TYPE_F16)  ||
           (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) ||
           (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) ||
           (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0);
}

bool ggml_cuda_flash_attn_ext_cdna_supported(const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_HIP_FA_CDNA");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!enabled) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    if (!ggml_cuda_fattn_cdna_type_ok(K->type, V->type)) {
        return false;
    }
    if (!mask || sinks || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (mask->ne[2] != 1 || K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0 || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    // Q, f16 K/V and the mask are read with 16 byte loads:
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }
    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio % 2 != 0) {
        return false;
    }
    return Q->ne[1] >= 256;
}

template <int ncols2>
static void ggml_cuda_flash_attn_ext_cdna_switch_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_type type_K = dst->src[1]->type;
    const ggml_type type_V = dst->src[2]->type;
    if (type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) {
        ggml_cuda_flash_attn_ext_cdna_case<256, ncols2, FATTN_CDNA_NWARPS, FATTN_CDNA_NP, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst);
    } else if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0) {
        ggml_cuda_flash_attn_ext_cdna_case<256, ncols2, FATTN_CDNA_NWARPS, FATTN_CDNA_NP, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst);
    } else if (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q4_0) {
        ggml_cuda_flash_attn_ext_cdna_case<256, ncols2, FATTN_CDNA_NWARPS, FATTN_CDNA_NP, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0>(ctx, dst);
    } else if (type_K == GGML_TYPE_Q4_0 && type_V == GGML_TYPE_Q4_0) {
        ggml_cuda_flash_attn_ext_cdna_case<256, ncols2, FATTN_CDNA_NWARPS, FATTN_CDNA_NP, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_flash_attn_ext_cdna(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(gqa_ratio % 2 == 0);
    ggml_cuda_flash_attn_ext_cdna_switch_type<2>(ctx, dst);
}
