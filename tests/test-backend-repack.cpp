// Round trip tests for the extra (repack) buffer types of the backends:
// set_tensor -> get_tensor must give the same bytes, also for partial writes, reads, views and memset.
// Run with GGML_HIP_REPACK=1 for the ROCm repack buffer type.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
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

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
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
                {4096, 1000, 1}, {1536, 4097, 1},
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
        }
    }
    printf("%d/%d round trip tests passed\n", n_tested - n_failed, n_tested);
    if (n_tested == 0) {
        printf("no extra buffer types (set GGML_HIP_REPACK=1)\n");
    }
    return n_failed == 0 ? 0 : 1;
}
