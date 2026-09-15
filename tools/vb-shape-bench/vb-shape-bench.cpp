// vb-shape-bench: run a single quantized MUL_MAT shape through the CPU backend
// with the weight in the repack buffer type (or the plain type with --no-repack).
// Timing is external: run under a QEMU instruction-count plugin and use two
// runs with different rep counts to isolate the per-matmul cost.
//
// usage: vb-shape-bench TYPE N K M REPS [--no-repack]
//   weight: TYPE[K x N] (K = reduction dim), activation: F32[K x M]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// declared in ggml/src/ggml-cpu/repack.h (internal header)
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s TYPE N K M REPS [--no-repack]\n", argv[0]);
        return 1;
    }
    const char * tname = argv[1];
    const int64_t N = atoll(argv[2]);
    const int64_t K = atoll(argv[3]);
    const int64_t M = atoll(argv[4]);
    const int     reps = atoi(argv[5]);
    const bool    use_repack = !(argc > 6 && strcmp(argv[6], "--no-repack") == 0);

    enum ggml_type type = GGML_TYPE_COUNT;
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        if (strcmp(ggml_type_name((enum ggml_type) t), tname) == 0) {
            type = (enum ggml_type) t;
            break;
        }
    }
    if (type == GGML_TYPE_COUNT) {
        fprintf(stderr, "unknown type %s\n", tname);
        return 1;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    ggml_backend_cpu_set_n_threads(backend, 1);

    struct ggml_init_params ipw = { ggml_tensor_overhead() * 2, NULL, true };
    struct ggml_context * ctx_w = ggml_init(ipw);
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx_w, type, K, N);

    ggml_backend_buffer_type_t buft = use_repack ? ggml_backend_cpu_repack_buffer_type()
                                                 : ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    if (!buf_w) {
        fprintf(stderr, "weight alloc failed\n");
        return 1;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> wf((size_t) K * N);
    for (auto & v : wf) v = dist(rng);
    std::vector<uint8_t> wq(ggml_nbytes(w));
    ggml_quantize_chunk(type, wf.data(), wq.data(), 0, N, K, NULL);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());

    struct ggml_init_params ipg = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true };
    struct ggml_context * ctx_g = ggml_init(ipg);
    struct ggml_tensor * x = ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, K, M);
    ggml_set_input(x);
    struct ggml_tensor * out = ggml_mul_mat(ctx_g, w, x);
    ggml_set_output(out);
    struct ggml_cgraph * gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    std::vector<float> xf((size_t) K * M);
    for (auto & v : xf) v = dist(rng);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));

    for (int r = 0; r < reps; r++) {
        ggml_backend_graph_compute(backend, gf);
    }

    float sink = 0;
    ggml_backend_tensor_get(out, &sink, 0, sizeof(float));
    printf("done type=%s N=%lld K=%lld M=%lld reps=%d repack=%d sink=%f\n",
           tname, (long long) N, (long long) K, (long long) M, reps, use_repack ? 1 : 0, (double) sink);
    return 0;
}
