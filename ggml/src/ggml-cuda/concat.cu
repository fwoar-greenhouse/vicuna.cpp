#include "concat.cuh"

#include <stdint.h>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

// non-contiguous kernel for a source with nb[1] == element size (a transposed view):
// a 64x64 tile is read along dim 1 and written along dim 0 through shared memory.
#define CONCAT_TILE      64
#define CONCAT_TILE_ROWS  4

template <typename T, int dim>
static __global__ void __launch_bounds__(CONCAT_TILE*CONCAT_TILE_ROWS)
    concat_non_cont_tiled(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t   ne1,
           int64_t   ne2,
           int64_t   ne3,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    __shared__ T tile[CONCAT_TILE][CONCAT_TILE + 1];

    const int64_t i00 = (int64_t) blockIdx.x*CONCAT_TILE;
    const int64_t i10 = (int64_t) blockIdx.y*CONCAT_TILE;

    // grid z is limited to 65535, loop over the rest of dims 2 and 3
    for (int64_t i23 = blockIdx.z; i23 < ne2*ne3; i23 += gridDim.z) {
        const int64_t i2 = i23 % ne2;
        const int64_t i3 = i23 / ne2;

#pragma unroll
        for (int y = threadIdx.y; y < CONCAT_TILE; y += CONCAT_TILE_ROWS) {
            const int64_t i0 = i00 + y;
            const int64_t i1 = i10 + threadIdx.x;
            if (i0 >= ne0 || i1 >= ne1) {
                continue;
            }
            const char * x;
            if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                x = src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00;
            } else if constexpr (dim == 0) {
                x = src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10;
            } else if constexpr (dim == 1) {
                x = src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10;
            } else if constexpr (dim == 2) {
                x = src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10;
            } else {
                x = src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10;
            }
            tile[y][threadIdx.x] = *(const T *) x;
        }

        __syncthreads();

#pragma unroll
        for (int y = threadIdx.y; y < CONCAT_TILE; y += CONCAT_TILE_ROWS) {
            const int64_t i0 = i00 + threadIdx.x;
            const int64_t i1 = i10 + y;
            if (i0 < ne0 && i1 < ne1) {
                *(T *) (dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = tile[threadIdx.x][y];
            }
        }

        __syncthreads();
    }
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {
    auto transposed = [](const ggml_tensor * t) {
        return t->nb[0] != sizeof(T) && t->nb[1] == sizeof(T);
    };
    if (dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_cont_cuda(
                    src0_d + i3*(src0->nb[3] / sizeof(T)),
                    src1_d + i3*(src1->nb[3] / sizeof(T)),
                    dst_d  + i3*( dst->nb[3] / sizeof(T)),
                    ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                    ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else if (dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);

        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
    } else if (!ggml_is_quantized(src0->type) && (transposed(src0) || transposed(src1)) &&
               (dst->ne[1] + CONCAT_TILE - 1)/CONCAT_TILE <= 65535) {
        const dim3 grid_dim((dst->ne[0] + CONCAT_TILE - 1)/CONCAT_TILE, (dst->ne[1] + CONCAT_TILE - 1)/CONCAT_TILE, std::min<int64_t>(dst->ne[2]*dst->ne[3], 65535));
        const dim3 block_dim(CONCAT_TILE, CONCAT_TILE_ROWS, 1);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont_tiled<T, dim><<<grid_dim, block_dim, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0: launch_kernel(std::integral_constant<int, 0>{}); break;
            case 1: launch_kernel(std::integral_constant<int, 1>{}); break;
            case 2: launch_kernel(std::integral_constant<int, 2>{}); break;
            case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
            default: GGML_ABORT("Invalid dim: %d", dim);
        }
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_quantized(src0->type)) {
        if (dim == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));
        } else {
            GGML_ASSERT(ggml_is_contiguous_to_3(src0));
            GGML_ASSERT(ggml_is_contiguous_to_3(src1));
        }
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
                break;
        }
    }
}

// CONCAT(state, x) along dim 0 with few columns + CPY of column windows of the result, one thread per row.
// This is the conv state update of the gated delta net (state = last conv inputs, x = new tokens).
static __global__ void concat_cpy_f32(
        const float * __restrict__ state, const int ne_st, const int st_nb1, const int st_nb2,
        const float * __restrict__ x, const int n_x, const int x_nb0, const int x_nb1, const int x_nb2,
        float * __restrict__ cat, const int cat_nb1, const int cat_nb2,
        const concat_cpy_args cpy, const int nr) {
    const int r = blockIdx.y*blockDim.x + threadIdx.x;
    const int s = blockIdx.x;
    if (r >= nr) {
        return;
    }

    state += s*st_nb2  + r*st_nb1;
    x     += s*x_nb2   + r*x_nb1;
    cat   += s*cat_nb2 + r*cat_nb1;

    for (int j = 0; j < ne_st; ++j) {
        cat[j] = state[j];
    }
    for (int t = 0; t < n_x; ++t) {
        cat[ne_st + t] = x[t*x_nb0];
    }

    for (int i = 0; i < cpy.n; ++i) {
        float * d = cpy.dst[i] + s*cpy.nb1[i] + r*cpy.ne0;
        for (int k = 0; k < cpy.ne0; ++k) {
            const int j = cpy.off[i] + k;
            d[k] = j < ne_st ? state[j] : x[(j - ne_st)*x_nb0];
        }
    }
}

bool ggml_cuda_concat_cpy_supported(const ggml_tensor * cat, const ggml_tensor * const * cpys, int n_cpy) {
    const ggml_tensor * st = cat->src[0];
    const ggml_tensor * x  = cat->src[1];

    if (ggml_get_op_params_i32(cat, 0) != 0 || n_cpy < 1 || n_cpy > CONCAT_CPY_MAX) {
        return false;
    }
    if (cat->type != GGML_TYPE_F32 || st->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32) {
        return false;
    }
    // few columns: the rows are written by one thread each
    if (cat->ne[0] > 40 || cat->ne[3] != 1 || !ggml_is_contiguous(cat) || st->nb[0] != sizeof(float) ||
        st->nb[1] % sizeof(float) || st->nb[2] % sizeof(float) ||
        x->nb[0] % sizeof(float) || x->nb[1] % sizeof(float) || x->nb[2] % sizeof(float)) {
        return false;
    }
    const int64_t nr   = cat->ne[1];
    const int64_t n_s  = cat->ne[2];
    const int64_t ne0  = cpys[0]->src[0]->ne[0];
    for (int i = 0; i < n_cpy; ++i) {
        const ggml_tensor * c   = cpys[i];
        const ggml_tensor * src = c->src[0];
        if (src->view_src != cat || src->ne[0] != ne0 || src->ne[1] != nr || src->ne[2] != n_s || src->ne[3] != 1 ||
            src->nb[0] != sizeof(float) || src->nb[1] != cat->nb[1] || src->nb[2] != cat->nb[2] ||
            src->view_offs % sizeof(float) || src->view_offs/sizeof(float) + ne0 > (size_t) cat->ne[0]) {
            return false;
        }
        if (c->type != GGML_TYPE_F32 || c->nb[0] != sizeof(float) || c->ne[0] != ne0*nr || c->ne[1] != n_s ||
            c->ne[2] != 1 || c->ne[3] != 1 || c->nb[1] % sizeof(float)) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_op_concat_cpy(ggml_backend_cuda_context & ctx, ggml_tensor * cat, ggml_tensor * const * cpys, int n_cpy) {
    const ggml_tensor * st = cat->src[0];
    const ggml_tensor * x  = cat->src[1];

    const int nr  = cat->ne[1];
    const int n_s = cat->ne[2];
    const int fs  = sizeof(float);

    concat_cpy_args args = {};
    args.n   = n_cpy;
    args.ne0 = cpys[0]->src[0]->ne[0];
    for (int i = 0; i < n_cpy; ++i) {
        args.dst[i] = (float *) cpys[i]->data;
        args.off[i] = cpys[i]->src[0]->view_offs / fs;
        args.nb1[i] = cpys[i]->nb[1] / fs;
    }

    const int threads = 128;
    const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
    concat_cpy_f32<<<blocks, threads, 0, ctx.stream()>>>(
        (const float *) st->data, st->ne[0], st->nb[1]/fs, st->nb[2]/fs,
        (const float *) x->data, x->ne[0], x->nb[0]/fs, x->nb[1]/fs, x->nb[2]/fs,
        (float *) cat->data, cat->nb[1]/fs, cat->nb[2]/fs,
        args, nr);
}
