// Round trip tests for the extra (repack) buffer types of the backends:
// set_tensor -> get_tensor must give the same bytes, also for partial writes, reads, views and memset.
// Also checks split-K GEMVs on repacked weights with a small counter ring against the CPU.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static std::mt19937 rng(1234);

static std::vector<uint8_t> rand_bytes(size_t n) {
    std::vector<uint8_t> v(n);
    for (auto & b : v) {
        b = rng() & 0xff;
    }
    return v;
}

static bool check(const char * what, const ggml_tensor * t, const std::vector<uint8_t> & ref) {
    std::vector<uint8_t> out(ggml_nbytes(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size());
    if (memcmp(out.data(), ref.data(), out.size()) != 0) {
        size_t i = 0;
        while (out[i] == ref[i]) {
            i++;
        }
        printf("  FAIL %s %s [%lld, %lld, %lld]: first difference at byte %zu\n", what, ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], i);
        return false;
    }
    return true;
}

static bool test_tensor(ggml_backend_buffer_type_t buft, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    // a view of rows [ne1/2, ne1) of the first matrix
    ggml_tensor * v = ggml_view_2d(ctx, t, ne0, ne1 - ne1/2, t->nb[1], (ne1/2)*t->nb[1]);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    bool ok = true;
    const size_t n = ggml_nbytes(t);

    // whole tensor
    std::vector<uint8_t> ref = rand_bytes(n);
    ggml_backend_tensor_set(t, ref.data(), 0, n);
    ok = check("full", t, ref) && ok;

    // partial writes: whole blocks, then unaligned ranges
    const size_t bs = ggml_type_size(type);
    for (int k = 0; k < 8; k++) {
        size_t off, len;
        if (k < 4) {
            const size_t nb = n / bs;
            off = (rng() % nb) * bs;
            len = (1 + rng() % std::min<size_t>(nb - off/bs, 300)) * bs;
        } else {
            off = rng() % n;
            len = 1 + rng() % std::min<size_t>(n - off, 5000);
        }
        std::vector<uint8_t> part = rand_bytes(len);
        memcpy(ref.data() + off, part.data(), len);
        ggml_backend_tensor_set(t, part.data(), off, len);
    }
    ok = check("partial set", t, ref) && ok;

    // partial reads
    for (int k = 0; k < 8; k++) {
        const size_t off = rng() % n;
        const size_t len = 1 + rng() % std::min<size_t>(n - off, 5000);
        std::vector<uint8_t> out(len);
        ggml_backend_tensor_get(t, out.data(), off, len);
        if (memcmp(out.data(), ref.data() + off, len) != 0) {
            printf("  FAIL partial get %s at %zu+%zu\n", ggml_type_name(type), off, len);
            ok = false;
        }
    }

    // view: read and write through the view
    {
        const size_t voff = (const char *) v->data - (const char *) t->data;
        std::vector<uint8_t> out(ggml_nbytes(v));
        ggml_backend_tensor_get(v, out.data(), 0, out.size());
        if (memcmp(out.data(), ref.data() + voff, out.size()) != 0) {
            printf("  FAIL view get %s\n", ggml_type_name(type));
            ok = false;
        }
        std::vector<uint8_t> part = rand_bytes(ggml_nbytes(v));
        ggml_backend_tensor_set(v, part.data(), 0, part.size());
        memcpy(ref.data() + voff, part.data(), part.size());
        ok = check("view set", t, ref) && ok;
    }

    // memset of a range
    {
        const size_t off = rng() % n;
        const size_t len = 1 + rng() % (n - off);
        ggml_backend_tensor_memset(t, 0x5a, off, len);
        memset(ref.data() + off, 0x5a, len);
        ok = check("memset", t, ref) && ok;
    }

    // memset of the whole tensor, in several chunks for large tensors
    {
        ggml_backend_tensor_memset(t, 0xa5, 0, n);
        memset(ref.data(), 0xa5, n);
        ok = check("memset all", t, ref) && ok;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

// Many GEMVs that split K over blocks, in one graph that runs several times. With a ring of fewer counters than two
// launches need (GGML_HIP_REPACK_COUNTERS, set in main), every launch reuses the counters of the one before, so a
// kernel that does not leave its counters at 0 breaks the next one.
static bool test_split_k_ring(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    const ggml_type type = GGML_TYPE_Q4_K;
    const int64_t   k    = 2048; // 8 blocks per row: 2 blocks per stripe
    const int64_t   m    = 1024; // 16 stripes, few enough for split K
    const int       n_mm = 24;

    std::vector<float> wf(k*m);
    std::vector<float> xf(k*n_mm);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float & v : wf) {
        v = dist(rng);
    }
    for (float & v : xf) {
        v = dist(rng);
    }
    std::vector<uint8_t> wq(ggml_row_size(type, k)*m);
    ggml_quantize_chunk(type, wf.data(), wq.data(), 0, m, k, nullptr);

    auto run = [&](ggml_backend_t backend, ggml_backend_buffer_type_t wbuft, std::vector<float> & out) {
        ggml_init_params params = { ggml_tensor_overhead()*(2*n_mm + 8) + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx_w = ggml_init(params);
        ggml_context * ctx   = ggml_init(params);
        ggml_tensor * w = ggml_new_tensor_2d(ctx_w, type, k, m);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        std::vector<ggml_tensor *> xs, ys;
        for (int i = 0; i < n_mm; i++) {
            xs.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k));
            ys.push_back(ggml_mul_mat(ctx, w, xs.back()));
            ggml_build_forward_expand(gf, ys.back());
        }
        ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, wbuft);
        ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
        for (int i = 0; i < n_mm; i++) {
            ggml_backend_tensor_set(xs[i], xf.data() + i*k, 0, k*sizeof(float));
        }
        out.assign(m*n_mm, 0.0f);
        for (int it = 0; it < 3; it++) {
            for (int i = 0; i < n_mm; i++) {
                ggml_backend_tensor_memset(ys[i], 0, 0, ggml_nbytes(ys[i]));
            }
            ggml_backend_graph_compute(backend, gf);
        }
        for (int i = 0; i < n_mm; i++) {
            ggml_backend_tensor_get(ys[i], out.data() + i*m, 0, m*sizeof(float));
        }
        ggml_backend_buffer_free(buf);
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx);
        ggml_free(ctx_w);
    };

    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    std::vector<float> out_gpu, out_cpu;
    run(gpu, buft, out_gpu);
    run(cpu, ggml_backend_get_default_buffer_type(cpu), out_cpu);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);

    double err = 0.0, ref = 0.0;
    for (size_t i = 0; i < out_cpu.size(); i++) {
        err += (out_gpu[i] - out_cpu[i])*(out_gpu[i] - out_cpu[i]);
        ref += out_cpu[i]*out_cpu[i];
    }
    const double nmse = err/ref;
    const bool ok = std::isfinite(nmse) && nmse < 1e-3;
    printf("  split-K GEMV x %d with a small counter ring: nmse %g %s\n", n_mm, nmse, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    // a ring of 24 counters: each launch of test_split_k_ring needs 16
    setenv("GGML_HIP_REPACK_COUNTERS", "24", 0);
    ggml_backend_load_all();

    int n_tested = 0;
    int n_failed = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        if (!get_extra_bufts || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        for (ggml_backend_buffer_type_t * b = get_extra_bufts(dev); b && *b; ++b) {
            printf("%s: %s\n", ggml_backend_dev_name(dev), ggml_backend_buft_name(*b));
            const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_F16 };
            // the repack buffer keeps matrices with fewer than 256 rows in the GGUF layout
            const int64_t shapes[][3] = {
                { 256,    1, 1}, { 512,   63, 1}, {5120,  256, 1}, {1024,  301, 1}, {2816, 704, 3}, {256, 277, 5},
                {4096, 1000, 1}, {1536, 4097, 1}, {5120, 1500, 1},
            };
            for (ggml_type type : types) {
                for (const auto & s : shapes) {
                    if (s[0] % ggml_blck_size(type) != 0) {
                        continue;
                    }
                    n_tested++;
                    if (!test_tensor(*b, type, s[0], s[1], s[2])) {
                        n_failed++;
                    }
                }
            }
            n_tested++;
            if (!test_split_k_ring(dev, *b)) {
                n_failed++;
            }
        }
    }
    printf("%d/%d tests passed\n", n_tested - n_failed, n_tested);
    if (n_tested == 0) {
        printf("no extra buffer types (set GGML_HIP_REPACK=1)\n");
    }
    return n_failed == 0 ? 0 : 1;
}
