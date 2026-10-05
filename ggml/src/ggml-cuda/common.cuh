#pragma once

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>

#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#include "ggml-common.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vendors/hip.h"

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32
#define GGML_CUDA_CC_OFFSET_AMD 0x1000000

static __device__ __forceinline__ void ggml_cuda_syncwarp() {
}

static __device__ __forceinline__ void ggml_cuda_pdl_sync() {
}

static __device__ __forceinline__ void ggml_cuda_pdl_lc() {
}

// ---------------------------------------------------------------------------------------------------------

#define MATRIX_ROW_PADDING 512 // last row of quant. matrices is a multiple of this to avoid out-of-bounds memory accesses

#define GGML_CUDA_MAX_STREAMS 8

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

#define CUDA_CHECK_GEN(err, success, error_fn)                                      \
     do {                                                                           \
        auto err_ = (err);                                                          \
        if (err_ != (success)) {                                                    \
            ggml_cuda_error(#err, __func__, __FILE__, __LINE__, error_fn(err_));    \
        }                                                                           \
    } while (0)

#define CUDA_CHECK(err) CUDA_CHECK_GEN(err, cudaSuccess, cudaGetErrorString)


    static const char * cublas_get_error_str(const cublasStatus_t err) {
        switch (err) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
            default: return "unknown error";
        }
    }

#define CUBLAS_CHECK(err) CUDA_CHECK_GEN(err, CUBLAS_STATUS_SUCCESS, cublas_get_error_str)

#ifdef GGML_USE_NCCL
#define NCCL_CHECK(err) CUDA_CHECK_GEN(err, ncclSuccess, ncclGetErrorString)
#endif // GGML_USE_NCCL


#    define CUDA_SET_SHARED_MEMORY_LIMIT(kernel, nbytes) \
        do {                                             \
            GGML_UNUSED(nbytes);                         \
        } while (0)

#define GGML_CUDA_ASSUME(x)

#if !defined(GGML_HIP_NO_VMM)
#define GGML_USE_VMM
#endif // !defined(GGML_HIP_NO_VMM)

#define FP16_AVAILABLE
#define FAST_FP16_AVAILABLE

#if defined(CDNA)
#define AMD_MFMA_AVAILABLE
#endif // defined(CDNA)

#if !defined(GGML_CUDA_NO_FA)
#define FLASH_ATTN_AVAILABLE
#endif // !defined(GGML_CUDA_NO_FA)

// Checks whether the tensor's base data pointer and higher-dimensional strides are byte-aligned to `alignment` bytes.
static bool ggml_cuda_is_aligned(const ggml_tensor * tensor, const size_t alignment) {
    GGML_ASSERT(tensor != nullptr);
    return (reinterpret_cast<uintptr_t>(tensor->data) % alignment) == 0 &&
           tensor->nb[1] % alignment == 0 &&
           tensor->nb[2] % alignment == 0 &&
           tensor->nb[3] % alignment == 0;
}

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
#if defined(__GFX9__) || defined(__GFX8__)
    return 64;
#else
    return 32;
#endif // defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))
}

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
    return 16;
}


[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {

    printf("%s:%d: ERROR: HIP kernel %s has no device code compatible with HIP arch %d.\n",
           file_name, line, function_name, arch);
    GGML_UNUSED(arch_list);
    __trap();

    GGML_UNUSED(no_device_code); // suppress unused function warning
}

#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))

// The compiler is always able to unroll loops if they contain continue expressions.
// In such cases loop unrolling can still be achieved via recursion:
template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_sum(int x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, offset, width);
        a.y += __shfl_xor_sync(0xffffffff, a.y, offset, width);
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_sum(half2 a) {
#ifdef FP16_AVAILABLE
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        a = __hadd2(a, __shfl_xor_sync(0xffffffff, a, offset, width));
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_all(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __all_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) && x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_any(int x) {
    if (width == ggml_cuda_get_physical_warp_size()) {
        return __any_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) || x;
        }
        return x;
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

// x from lane (lane ^ mask) of the wave64, same as __shfl_xor(x, mask, 64).
// Uses DPP for mask 1/2 and ds_swizzle for mask 4..16 instead of ds_bpermute.
template<int mask>
static __device__ __forceinline__ float ggml_cuda_shfl_xor64(float x) {
    static_assert(mask > 0 && mask < 64 && (mask & (mask - 1)) == 0, "mask must be a power of 2 below 64");
#if defined(CDNA)
    if constexpr (mask == 1) {
        return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0xB1, 0xF, 0xF, false)); // quad_perm [1,0,3,2]
    } else if constexpr (mask == 2) {
        return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(x), 0x4E, 0xF, 0xF, false)); // quad_perm [2,3,0,1]
    } else if constexpr (mask < 32) {
        // ds_swizzle bit mode: offset = and_mask | or_mask << 5 | xor_mask << 10, within 32 lanes
        return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), 0x1F | (mask << 10)));
    } else {
        return __shfl_xor(x, mask, 64);
    }
#else
    return __shfl_xor_sync(0xffffffff, x, mask, 64);
#endif // defined(CDNA)
}

template<typename T, int width = WARP_SIZE>
static __device__ __forceinline__ T warp_prefix_inclusive_sum(T x) {
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const T t = __shfl_up_sync(0xffffffff, x, offset, width);
        if (lane_id >= offset) {
            x += t;
        }
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float2 warp_prefix_inclusive_sum(float2 a) {
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const float t_x = __shfl_up_sync(0xffffffff, a.x, offset, width);
        const float t_y = __shfl_up_sync(0xffffffff, a.y, offset, width);
        if (lane_id >= offset) {
            a.x += t_x;
            a.y += t_y;
        }
    }
    return a;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_prefix_inclusive_sum(half2 a) {
#ifdef FP16_AVAILABLE
    const int lane_id = threadIdx.x % width;
#pragma unroll
    for (int offset = 1; offset < width; offset <<= 1) {
        const half2 t = __shfl_up_sync(0xffffffff, a, offset, width);
        if (lane_id >= offset) {
            a = __hadd2(a, t);
        }
    }
    return a;

#else
    NO_DEVICE_CODE;
    return a;
#endif // FP16_AVAILABLE
}

enum class block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T>
struct block_reduce_policy;

template <typename T, typename... Ts>
inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
inline constexpr bool ggml_cuda_dependent_false_v = false;

template <typename T> struct block_reduce_policy<block_reduce_method::SUM, T> {
    static __device__ T reduce(T val) {
        if constexpr(is_any<T, float, float2, half2, int>) {
            return warp_reduce_sum(val);
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    static __device__ T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return 0.0f;
        } else if constexpr (std::is_same_v<T, float2>) {
            return make_float2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, half2>) {
            return make_half2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, int>) {
            return 0;
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T> struct block_reduce_policy<block_reduce_method::MAX, T> {
    static __device__ T reduce(T val) {
        if constexpr (is_any<T, float, half2>) {
            return warp_reduce_max(val);
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    static __device__ T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return -INFINITY;
        } else if constexpr (std::is_same_v<T, half2>) {
            return make_half2(-INFINITY, -INFINITY);
        } else {
            static_assert(ggml_cuda_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};

template <block_reduce_method reduce_method_t, const unsigned int block_size_template = 0, typename T>
static __device__ T block_reduce(T val, [[maybe_unused]] T * shared_vals) {
    // for multi-warp reductions, callers must not reuse shared_vals until all reads from this invocation have completed
    val                           = block_reduce_policy<reduce_method_t, T>::reduce(val);
    const unsigned int block_size = block_size_template == 0 ? blockDim.x : block_size_template;
    if (block_size > WARP_SIZE) {
        assert((block_size <= 1024) && (block_size % WARP_SIZE) == 0);
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        __syncthreads();
        val = block_reduce_policy<reduce_method_t, T>::sentinel();
        if (lane_id < (static_cast<int>(block_size) / WARP_SIZE)) {
            val = shared_vals[lane_id];
        }
        return block_reduce_policy<reduce_method_t, T>::reduce(val);
    }

    return val;
}

static __device__ __forceinline__ half ggml_cuda_hmax(const half a, const half b) {
#ifdef FP16_AVAILABLE

    return __hmax(a, b);

#else
   NO_DEVICE_CODE;
   GGML_UNUSED(b);
   return a;
#endif // FP16_AVAILABLE
}

static __device__ __forceinline__ half2 ggml_cuda_hmax2(const half2 a, const half2 b) {
    return half2(__hmax(a.x, b.x), __hmax(a.y, b.y));
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ half2 warp_reduce_max(half2 x) {
#pragma unroll
   for (int offset = width/2; offset > 0; offset >>= 1) {
       x = ggml_cuda_hmax2(x, __shfl_xor_sync(0xffffffff, x, offset, width));
   }
   return x;
}

static __device__ __forceinline__ uint32_t __hgt2_mask(const half2 a, const half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float( __low2half(a)) > float( __low2half(b)));
    const uint32_t mask_high = 0xFFFF0000 * (float(__high2half(a)) > float(__high2half(b)));
    return mask_low | mask_high;
}

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(CDNA) || defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(__gfx900__)
    int tmp1;
    int tmp2;
    asm("\n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n \
        v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n \
        v_add3_u32 %0, %1, %2, %0 \n \
        "
        : "+v"(c), "=&v"(tmp1), "=&v"(tmp2)
        : "v"(a), "v"(b)
    );
#else
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
    c += va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2] + va[3] * vb[3];
#endif
    return c;

}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float2 v, const float2 u) {
    acc += v.x*u.x;
    acc += v.y*u.y;
}

#if defined(__gfx906__) || defined(CDNA)
#define V_DOT2_F32_F16_AVAILABLE
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const half2 v, const half2 u) {
#ifdef V_DOT2_F32_F16_AVAILABLE
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
#else
#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(v*u);
    acc += tmp.x + tmp.y;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    acc += tmpv.x * tmpu.x;
    acc += tmpv.y * tmpu.y;
#endif // FAST_FP16_AVAILABLE
#endif // V_DOT2_F32_F16_AVAILABLE
}

static __device__ __forceinline__ void ggml_cuda_mad(half2 & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    acc += v*u;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    float2 tmpacc = __half22float2(acc);
    tmpacc.x += tmpv.x * tmpu.x;
    tmpacc.y += tmpv.y * tmpu.y;
    acc = make_half2(tmpacc.x, tmpacc.y);
#endif // FAST_FP16_AVAILABLE
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
// Important: do not use this function if dst and src both point at registers.
//     Due to the strict aliasing rule the compiler can do incorrect optimizations if src and dst have different types.
//     The function is intended for copies between registers and SRAM/VRAM to make the compiler emit the right instructions.
//     If dst and src point at different address spaces then they are guaranteed to not be aliased.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
        "The intent is for the parameter is only as a workaround if either one of the pointers is not properly aligned. "
        "If you use it to do more bytes per copy than ggml_cuda_max_cpy_bytes() the reads and writes may not be coalesced. "
        "Call ggml_cuda_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

static __device__ __forceinline__ float ggml_cuda_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || (x == 0x7F && x != 0xFF)) { // Convert NaN to 0.0f
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return static_cast<float>(raw / 2);
}

static __device__ __forceinline__ uint8_t ggml_cuda_fp32_to_ue4m3(float x) {
     NO_DEVICE_CODE; // Used only for NVFP4 Scales for Activations, only for Blackwell
}

__device__ __forceinline__ uint8_t ggml_cuda_float_to_fp4_e2m1(float x, float e) {
    const uint8_t sign_bit = (x < 0.0f) << 3;
    float         ax       = fabsf(x) * e;

    // Positive LUT
    static constexpr float pos_lut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

    int   best_i   = 0;
    float best_err = fabsf(ax - pos_lut[0]);

#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const float err = fabsf(ax - pos_lut[i]);
        if (err < best_err) {
            best_err = err;
            best_i   = i;
        }
    }

    return static_cast<uint8_t>(best_i | sign_bit);
}

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

typedef void (*dequantize_kernel_t)(const void * vx, const int64_t ib, const int iqs, float2 & v);

template<typename dst_t>
using dequantize_kq_t = void (*)(const void * vx, const int64_t ib, dst_t * y, const int tid);

static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return powf(base, exph);
}

template <ggml_type type>
struct ggml_cuda_type_traits;

template<>
struct ggml_cuda_type_traits<GGML_TYPE_F16> {
    static constexpr int qk = 1;
    static constexpr int qr = 1;
    static constexpr int bs = sizeof(ggml_half);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q1_0> {
    static constexpr int qk = QK1_0;
    static constexpr int qr = QR1_0;
    static constexpr int qi = QI1_0;
    static constexpr int bs = sizeof(block_q1_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_0> {
    static constexpr int qk = QK2_0;
    static constexpr int qr = QR2_0;
    static constexpr int qi = QI2_0;
    static constexpr int bs = sizeof(block_q2_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;
    static constexpr int qr = QR4_0;
    static constexpr int qi = QI4_0;
    static constexpr int bs = sizeof(block_q4_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_1> {
    static constexpr int qk = QK4_1;
    static constexpr int qr = QR4_1;
    static constexpr int qi = QI4_1;
    static constexpr int bs = sizeof(block_q4_1);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_0> {
    static constexpr int qk = QK5_0;
    static constexpr int qr = QR5_0;
    static constexpr int qi = QI5_0;
    static constexpr int bs = sizeof(block_q5_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_1> {
    static constexpr int qk = QK5_1;
    static constexpr int qr = QR5_1;
    static constexpr int qi = QI5_1;
    static constexpr int bs = sizeof(block_q5_1);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR8_0;
    static constexpr int qi = QI8_0;
    static constexpr int bs = sizeof(block_q8_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_MXFP4> {
    static constexpr int qk = QK_MXFP4;
    static constexpr int qr = QR_MXFP4;
    static constexpr int qi = QI_MXFP4;
    static constexpr int bs = sizeof(block_mxfp4);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_NVFP4> {
    static constexpr int qk = QK_NVFP4;
    static constexpr int qr = QR_NVFP4;
    static constexpr int qi = QI_NVFP4;
    static constexpr int bs = sizeof(block_nvfp4);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_K;
    static constexpr int qi = QI2_K;
    static constexpr int bs = sizeof(block_q2_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q3_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_K;
    static constexpr int qi = QI3_K;
    static constexpr int bs = sizeof(block_q3_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
    static constexpr int bs = sizeof(block_q4_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
    static constexpr int bs = sizeof(block_q5_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
    static constexpr int bs = sizeof(block_q6_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XXS;
    static constexpr int qi = QI2_XXS;
    static constexpr int bs = sizeof(block_iq2_xxs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XS;
    static constexpr int qi = QI2_XS;
    static constexpr int bs = sizeof(block_iq2_xs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_S;
    static constexpr int qi = QI2_S;
    static constexpr int bs = sizeof(block_iq2_s);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_XXS;
    static constexpr int qi = QI3_XXS;
    static constexpr int bs = sizeof(block_iq3_xxs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_S;
    static constexpr int qi = QI1_S;
    static constexpr int bs = sizeof(block_iq1_s);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_M> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_M;
    static constexpr int qi = QI1_M;
    static constexpr int bs = sizeof(block_iq1_m);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_NL> {
    static constexpr int qk = QK4_NL;
    static constexpr int qr = QR4_NL;
    static constexpr int qi = QI4_NL;
    static constexpr int bs = sizeof(block_iq4_nl);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
    static constexpr int bs = sizeof(block_iq4_xs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_S;
    static constexpr int qi = QI3_S;
    static constexpr int bs = sizeof(block_iq3_s);
};

//////////////////////

struct ggml_cuda_device_info {
    int device_count;           // number of (possibly virtual) devices exposed to the rest of ggml
    int physical_device_count;  // number of physical CUDA devices actually present

    struct cuda_device_info {
        int     cc;                             // compute capability
        int     nsm;                            // number of streaming multiprocessors
        size_t  smpb;                           // max. shared memory per block
        size_t  smpbo;                          // max. shared memory per block (with opt-in)
        bool    integrated;                     // Device is integrated as opposed to discrete
        bool    vmm;                            // virtual memory support
        size_t  vmm_granularity;                // granularity of virtual memory
        size_t  total_vram;
        int     warp_size;                      // Number of threads in a dispatch
        bool    supports_cooperative_launch;    // whether cooperative launch is supported
        int     physical_device;                // backing physical CUDA device for this (virtual) device
        int     physical_share_count;           // number of (virtual) devices sharing this device's physical GPU
        int     virtual_index;                  // index of this (virtual) device among those sharing its physical GPU
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};

    std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split = {};
};

const ggml_cuda_device_info & ggml_cuda_info();

void ggml_cuda_set_device(int device);
int ggml_cuda_get_device();

struct ggml_cuda_pool {
    virtual ~ggml_cuda_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_cuda_pool_alloc {
    ggml_cuda_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    ggml_cuda_pool_alloc() = default;

    explicit ggml_cuda_pool_alloc(ggml_cuda_pool & pool) : pool(&pool) {
    }

    ggml_cuda_pool_alloc(ggml_cuda_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_cuda_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_cuda_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc(ggml_cuda_pool_alloc &&) = delete;
    ggml_cuda_pool_alloc& operator=(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc& operator=(ggml_cuda_pool_alloc &&) = delete;
};


// backend interface

struct ggml_tensor_extra_gpu {
    void * data_device[GGML_CUDA_MAX_DEVICES]; // 1 pointer for each device for split tensors
    cudaEvent_t events[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS]; // events for synchronizing multiple GPUs
};


#if (defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS)) || defined(GGML_MUSA_GRAPHS)
#define USE_CUDA_GRAPH
#endif

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    bool warmup_complete = false;
    uint64_t uid = 0;
    int64_t last_used_time = 0;
    struct node_properties {
        ggml_tensor node;
        void *   node_src_data_ptrs[GGML_MAX_SRC];
        int64_t  node_src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
        size_t   node_src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];
    };
    std::vector<node_properties> node_props;

    bool is_enabled() const {
        static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
        return !disable_cuda_graphs_due_to_env;
    }
#endif
};

struct ggml_cuda_concurrent_event {
    std::vector<cudaEvent_t> join_events;
    cudaEvent_t              fork_event = nullptr;

    int                                          n_streams = 0;
    std::unordered_map<const ggml_tensor *, int> stream_mapping;

    // Original order of nodes in this concurrent region (before interleaving)
    // Used to restore grouping for fusion within streams
    std::vector<const ggml_tensor *> original_order;

    const ggml_tensor * join_node;

    ggml_cuda_concurrent_event() = default;

    ggml_cuda_concurrent_event(const ggml_cuda_concurrent_event &) = delete;
    ggml_cuda_concurrent_event & operator=(const ggml_cuda_concurrent_event &) = delete;

    explicit ggml_cuda_concurrent_event(int n_streams) : n_streams(n_streams) {
        join_events.resize(n_streams);

        for (size_t i = 0; i < join_events.size(); ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&join_events[i], cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
    }

    ggml_cuda_concurrent_event(ggml_cuda_concurrent_event && other) noexcept
    : join_events(std::move(other.join_events))
    , fork_event(other.fork_event)
    , n_streams(other.n_streams)
    , stream_mapping(std::move(other.stream_mapping))
    , original_order(std::move(other.original_order))
    , join_node(other.join_node) {
        other.fork_event = nullptr;
    }

    // 1. check if any branches write to overlapping memory ranges (except the join node)
    // 2. check whether all srcs are either within the branch or outside the nodes covered by ggml_cuda_concurrent_event
    // we assume all nodes have the same buffer
    bool is_valid() const {
        std::vector<std::vector<std::pair<int64_t, int64_t>>> write_ranges;
        write_ranges.resize(n_streams);

        // get join_node's memory range to exclude from overlap checking.
        // multiple nodes can use join_node's buffer; we synchronize on the join node.
        const ggml_tensor * join_t     = join_node->view_src ? join_node->view_src : join_node;
        const int64_t       join_start = (int64_t) join_t->data;
        const int64_t       join_end   = join_start + ggml_nbytes(join_t);

        for (const auto & [tensor, stream] : stream_mapping) {
            const ggml_tensor * t = tensor->view_src ? tensor->view_src : tensor;
            const int64_t       t_start = (int64_t) t->data;
            const int64_t       t_end   = t_start + ggml_nbytes(t);

            // skip tensors that overlap with join_node's buffer.
            if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
                continue;
            }

            // concurrent streams begin from 1
            write_ranges[stream - 1].emplace_back(t_start, t_end);
        }

        for (int i = 0; i < n_streams; ++i) {
            // sorts first by start then by end of write range
            std::sort(write_ranges[i].begin(), write_ranges[i].end());
        }

        bool writes_overlap = false;
        bool dependent_srcs = false;
        for (const auto & [tensor, stream] : stream_mapping) {
            const ggml_tensor * t = tensor->view_src ? tensor->view_src : tensor;
            const int64_t       t_start = (int64_t) t->data;
            const int64_t       t_end   = t_start + ggml_nbytes(t);

            // skip tensors that overlap with join_node's buffer
            if ((t_start <= join_start && join_start < t_end) || (join_start <= t_start && t_start < join_end)) {
                continue;
            }

            // check if this buffer's write data overlaps with another stream's
            std::pair<int64_t, int64_t> data_range = std::make_pair(t_start, t_end);
            for (int i = 0; i < n_streams; ++i) {
                if (i == stream - 1) {
                    continue;
                }
                auto it = std::lower_bound(write_ranges[i].begin(), write_ranges[i].end(), data_range);

                if (it != write_ranges[i].end()) {
                    const std::pair<int64_t, int64_t> & other = *it;

                    // std::lower_bound returns the first element where other >= data_range (lexicographically).
                    // This guarantees other.first >= data_range.first.
                    // Therefore, overlap occurs iff other.first < data_range.second
                    // (i.e., the other range starts before this range ends).
                    if (other.first < data_range.second) {
                        GGML_LOG_DEBUG("Writes overlap for %s", tensor->name);
                        writes_overlap = true;
                        break;
                    }
                }
            }

            //check if all srcs are either in branch or don't have a branch
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (!tensor->src[i]) {
                    continue;
                }

                auto it = stream_mapping.find(tensor->src[i]);

                if (it == stream_mapping.end()) {
                    continue;
                }

                if (it->second != stream) {
                    dependent_srcs = true;
                    break;
                }
            }

            if (dependent_srcs || writes_overlap) {
                break;
            }
        }

        return !writes_overlap && !dependent_srcs;
    }

    ~ggml_cuda_concurrent_event() {
        if (fork_event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(fork_event));
        }
        for (cudaEvent_t e : join_events) {
            if (e != nullptr) {
                CUDA_CHECK(cudaEventDestroy(e));
            }
        }
    }
};

struct ggml_cuda_stream_context {
    std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> concurrent_events;

    void reset() {
        concurrent_events.clear();
    }
};

struct ggml_backend_cuda_context {
    int device;
    std::string name;
    cudaEvent_t copy_event = nullptr;

    cudaStream_t streams[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = { { nullptr } };
    cublasHandle_t cublas_handles[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {nullptr};
    void * cublas_workspaces[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS] = {nullptr};
    size_t cublas_workspace_sizes[GGML_CUDA_MAX_DEVICES] = {0};

    int curr_stream_no = 0;

#ifdef USE_CUDA_GRAPH
    // Map from first_node_ptr to cuda_graph - allows multiple graphs per context
    // when the computation is split across CPU/GPU (e.g., with --n-cpu-moe)
    std::unordered_map<const void *, std::unique_ptr<ggml_cuda_graph>> cuda_graphs;

    int64_t last_graph_eviction_sweep = 0;

    ggml_cuda_graph * cuda_graph(const void * first_node_ptr) {
        const int64_t time_now = ggml_time_us();

        // sweep every 5s, evicting cuda graphs unused for >=10s
        if (time_now - last_graph_eviction_sweep >= 5'000'000) {
            last_graph_eviction_sweep = time_now;
            for (auto it = cuda_graphs.begin(); it != cuda_graphs.end(); ) {
                if (time_now - it->second->last_used_time >= 10'000'000) {
                    it = cuda_graphs.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto it = cuda_graphs.find(first_node_ptr);
        if (it == cuda_graphs.end()) {
            it = cuda_graphs.emplace(first_node_ptr, std::make_unique<ggml_cuda_graph>()).first;
        }
        it->second->last_used_time = time_now;
        return it->second.get();
    }

    // Check if any CUDA graph is enabled for this context (used by kernels that need to know
    // if graphs are in use without having access to the specific graph key)
    bool any_cuda_graph_enabled() const {
        for (const auto & [key, graph] : cuda_graphs) {
            if (graph && graph->is_enabled()) {
                return true;
            }
        }
        return false;
    }

    // Check if any CUDA graph has an instance for this context
    bool any_cuda_graph_has_instance() const {
        for (const auto & [key, graph] : cuda_graphs) {
            if (graph && graph->instance != nullptr) {
                return true;
            }
        }
        return false;
    }
#endif // USE_CUDA_GRAPH

    explicit ggml_backend_cuda_context(int device) :
        device(device),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ggml_cuda_stream_context concurrent_stream_context;

    ~ggml_backend_cuda_context();

    cudaStream_t stream(int device, int stream) {
        if (streams[device][stream] == nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&streams[device][stream], cudaStreamNonBlocking));
        }
        return streams[device][stream];
    }

    cudaStream_t stream() { return stream(device, curr_stream_no); }

    ggml_cuda_stream_context & stream_context() { return concurrent_stream_context; }

    cublasHandle_t cublas_handle() {
        if (cublas_handles[device][curr_stream_no] == nullptr) {
            ggml_cuda_set_device(device);
            CUBLAS_CHECK(cublasCreate(&cublas_handles[device][curr_stream_no]));
            CUBLAS_CHECK(cublasSetMathMode(cublas_handles[device][curr_stream_no], CUBLAS_TF32_TENSOR_OP_MATH));
            CUBLAS_CHECK(cublasSetStream(cublas_handles[device][curr_stream_no], stream()));
        }
        return cublas_handles[device][curr_stream_no];
    }

    // pool
    std::unique_ptr<ggml_cuda_pool> pools[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_STREAMS];

    static std::unique_ptr<ggml_cuda_pool> new_pool_for_device(int device, int stream_no);

    ggml_cuda_pool & pool(int device) {
        if (pools[device][curr_stream_no] == nullptr) {
            pools[device][curr_stream_no] = new_pool_for_device(device, curr_stream_no);
        }
        return *pools[device][curr_stream_no];
    }

    ggml_cuda_pool & pool() {
        return pool(device);
    }

    // Counters for the split-K fixup of the repacked GEMV. They are 0 between kernels: the last block of a tile resets
    // its counter. Each launch takes the next range of a ring. A range is reused only after a wrap-around, which is safe
    // because all users run on the main stream (stream()), so the earlier kernel has finished and left its counters at 0.
    // GGML_HIP_REPACK_COUNTERS sets a smaller ring for tests.
    int * repack_counters    = nullptr;
    int   repack_counters_n  = 0;
    int   repack_counter_pos = 0;

    int * repack_counters_get(int n) {
        if (repack_counters == nullptr) {
            const char * e = getenv("GGML_HIP_REPACK_COUNTERS");
            repack_counters_n = e ? atoi(e) : 1 << 16;
            GGML_ASSERT(repack_counters_n > 0);
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaMalloc(&repack_counters, repack_counters_n*sizeof(int)));
            CUDA_CHECK(cudaMemsetAsync(repack_counters, 0, repack_counters_n*sizeof(int), stream()));
        }
        GGML_ASSERT(n <= repack_counters_n);
        if (repack_counter_pos + n > repack_counters_n) {
            repack_counter_pos = 0;
        }
        int * p = repack_counters + repack_counter_pos;
        repack_counter_pos += n;
        return p;
    }

    // q8_1 copies of f32 src1 tensors for MMVQ, kept during one graph evaluation.
    // Later matmuls with the same src1 reuse them until a node writes to the src1 memory.
    struct q8_1_cache_entry {
        const void *     data = nullptr;
        int64_t          ne[GGML_MAX_DIMS] = {0};
        size_t           nb[GGML_MAX_DIMS] = {0};
        size_t           nbytes = 0;
        cudaStream_t     stream = nullptr;
        ggml_cuda_pool * pool   = nullptr;
        void *           q8_1   = nullptr;
        size_t           size   = 0;
        int64_t          last_use = 0;
    };
    static constexpr int Q8_1_CACHE_SIZE = 4;
    q8_1_cache_entry q8_1_cache[Q8_1_CACHE_SIZE];
    bool    q8_1_cache_enabled = false;
    int64_t q8_1_cache_clock   = 0;

    static bool q8_1_cache_match(const q8_1_cache_entry & e, const ggml_tensor * t) {
        if (e.q8_1 == nullptr || e.data != t->data) {
            return false;
        }
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (e.ne[i] != t->ne[i] || e.nb[i] != t->nb[i]) {
                return false;
            }
        }
        return true;
    }

    void q8_1_cache_drop(q8_1_cache_entry & e) {
        if (e.q8_1 != nullptr) {
            e.pool->free(e.q8_1, e.size);
        }
        e = q8_1_cache_entry();
    }

    void q8_1_cache_clear() {
        for (q8_1_cache_entry & e : q8_1_cache) {
            q8_1_cache_drop(e);
        }
    }

    // only tensors in a backend buffer: graph writes to them are seen by q8_1_cache_on_write, pool scratch is not
    bool q8_1_cache_usable(const ggml_tensor * t) const {
        if (!q8_1_cache_enabled || t->buffer == nullptr) {
            return false;
        }
        const char * base = (const char *) ggml_backend_buffer_get_base(t->buffer);
        const char * data = (const char *) t->data;
        return data >= base && data + ggml_nbytes(t) <= base + ggml_backend_buffer_get_size(t->buffer);
    }

    // returns the cached q8_1 data of t, or nullptr
    void * q8_1_cache_find(const ggml_tensor * t) {
        if (!q8_1_cache_usable(t)) {
            return nullptr;
        }
        for (q8_1_cache_entry & e : q8_1_cache) {
            if (q8_1_cache_match(e, t) && e.stream == stream()) {
                e.last_use = ++q8_1_cache_clock;
                return e.q8_1;
            }
        }
        return nullptr;
    }

    // allocates a cache entry for t with nbytes_q8_1 bytes, or returns nullptr if the cache is off
    void * q8_1_cache_alloc(const ggml_tensor * t, size_t nbytes_q8_1) {
        if (!q8_1_cache_usable(t)) {
            return nullptr;
        }
        q8_1_cache_entry * dst = &q8_1_cache[0];
        for (q8_1_cache_entry & e : q8_1_cache) {
            if (q8_1_cache_match(e, t)) {
                dst = &e;
                break;
            }
            if (e.last_use < dst->last_use) {
                dst = &e;
            }
        }
        q8_1_cache_drop(*dst);
        dst->data   = t->data;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            dst->ne[i] = t->ne[i];
            dst->nb[i] = t->nb[i];
        }
        dst->nbytes   = ggml_nbytes(t);
        dst->stream   = stream();
        dst->pool     = &pool();
        dst->q8_1     = dst->pool->alloc(nbytes_q8_1, &dst->size);
        dst->last_use = ++q8_1_cache_clock;
        return dst->q8_1;
    }

    // drops the entries whose src1 memory overlaps the output of node
    void q8_1_cache_on_write(const ggml_tensor * node) {
        if (node->data == nullptr) {
            return;
        }
        const char * w0 = (const char *) node->data;
        const char * w1 = w0 + ggml_nbytes(node);
        for (q8_1_cache_entry & e : q8_1_cache) {
            if (e.q8_1 == nullptr) {
                continue;
            }
            const char * r0 = (const char *) e.data;
            const char * r1 = r0 + e.nbytes;
            if (w0 < r1 && r0 < w1) {
                q8_1_cache_drop(e);
            }
        }
    }
};

struct ggml_cuda_mm_fusion_args_host {
    const ggml_tensor * x_bias = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * gate_bias = nullptr;
    const ggml_tensor * x_scale = nullptr;
    const ggml_tensor * gate_scale = nullptr;
    ggml_glu_op glu_op;
    float glu_limit = 0.0f;
};
struct ggml_cuda_mm_fusion_args_device {
    const void * x_bias = nullptr;
    const void * gate = nullptr;
    const void * gate_bias = nullptr;
    const void * x_scale = nullptr;
    const void * gate_scale = nullptr;
    ggml_glu_op glu_op;
    float glu_limit = 0.0f;
};

struct ggml_cuda_kernel_launch_params {
    dim3 block_nums;
    dim3 block_dims;
    size_t shmem;
    cudaStream_t stream;

    // size_t shmem
    ggml_cuda_kernel_launch_params(const dim3& block_nums_, const dim3& block_dims_, const size_t shmem_, const cudaStream_t stream_)
        : block_nums(block_nums_), block_dims(block_dims_), shmem(shmem_), stream(stream_) {}

    // Some call sites pass ints instead of the required size_t. This 2nd constructor casts int->size_t to avoid these -Wnarrowing warnings.
    ggml_cuda_kernel_launch_params(const dim3& block_nums_, const dim3& block_dims_, const int shmem_, const cudaStream_t stream_)
        : block_nums(block_nums_), block_dims(block_dims_), shmem((size_t)shmem_), stream(stream_) {}
};


// PDL and __restrict__ need to be mutually exclusive, see https://github.com/ggml-org/llama.cpp/pull/24030
# define GGML_CUDA_RESTRICT __restrict__

template<typename Kernel, typename... Args>
static __inline__ void ggml_cuda_kernel_launch(Kernel kernel, const ggml_cuda_kernel_launch_params & launch_params, Args&&... args) {

    kernel<<<launch_params.block_nums, launch_params.block_dims, launch_params.shmem, launch_params.stream>>>(std::forward<Args>(args)... );
    CUDA_CHECK(cudaGetLastError());
}
