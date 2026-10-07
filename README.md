# vicuna.cpp

*A vicuna is a small wild relative of the llama.*

vicuna.cpp is a hard fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), specialized for one GPU:
the **AMD Instinct MI100 (gfx908, CDNA1) on ROCm**. The goal is the fastest possible local inference on that card,
first by bringing the HIP backend to parity with upstream's CUDA backend, then by going past it where the MI100 allows.

The fork was taken from upstream at `83209c3d2`. It does not track upstream and is not meant to send changes back.
Binaries, libraries and APIs keep their llama.cpp names (`llama-server`, `libllama`, ...). The name has nothing to do
with the Vicuna model family.

## What is different from upstream

- **Backends:** only CPU, BLAS, RPC and HIP remain. CUDA (as an NVIDIA build), Metal, Vulkan, SYCL, OpenCL,
  WebGPU, VirtGPU, MUSA, CANN, Hexagon, OpenVINO, ET, zDNN and ZenDNN were removed, with their CI, Docker images and docs.
- **gfx908 only:** the HIP backend compiles the `ggml-cuda` sources for gfx908 and nothing else. All NVIDIA, RDNA and
  CDNA2-4 code paths were removed, so every kernel can assume MFMA, wave64 and the MI100's memory system.
  The build refuses other GPU targets.
- **MI100 kernels**, among others:
  - weight repack at load time into a layout that the GEMV can stream at close to full bandwidth (on by default)
  - MFMA flash attention for decode and speculative-verify batches, reading quantized K/V directly
  - a flash-attention kernel for prefill and large verify batches at head size 256, built for CDNA1: it reads
    q8_0/q4_0 K/V directly and runs about 2x faster than converting the cache to f16 first
  - tuned MMVQ/MMQ crossovers and MMQ configurations for 1-32 columns
  - chunked gated delta rule on MFMA for prefill of hybrid models (Qwen3.5/3.8 family)
  - many small-kernel fusions and wave64-aware reductions
  - hipBLAS fallbacks dequantize large weights in chunks of rows, so a big output layer never needs a multi-GiB
    temporary in the memory pool
- **Server:** saving and restoring a slot's state for the prompt cache uses a few large copies instead of thousands of
  small ones, and context checkpoints are shared with the cache instead of copied. With two slots and a unified KV
  cache this cut prompt-cache updates from 1-3.5 s to 0.1-0.6 s. The log reports the time of each update and, with
  `GGML_HIP_POOL_STATS`, the device memory held by each memory pool.
- **Nix flake:** `nix build .#rocm` builds the ROCm package for gfx908 and reports the real git revision.

The full list, with measurements, profiles, prior art and the remaining backlog, is in
[docs/backend/MI100-parity.md](docs/backend/MI100-parity.md).

## Performance

Measured on one MI100 (32 GB) with `-ngl 99 -fa 1`, against tag `mi100-pruned-verified` (the pruned tree before any
optimization). All runs use interleaved A/B measurements; see the parity doc for details and exact commits.

| | before | after |
|---|---:|---:|
| Qwen3.8-27B UD-Q5_K_S, decode (tg128) | 28.2 t/s | ~38 t/s |
| Qwen3.8-27B, decode at 64k context (KV q8_0) | 18.0 t/s | 26.5 t/s (before the weight repack) |
| Gemma 4 31B UD-Q5_K_XL, decode | 22.7 t/s | ~32 t/s |
| Gemma 4 26B-A4B UD-Q5_K_XL (MoE), decode | 94.5 t/s | ~107 t/s |
| Qwen3.8-27B, prefill at depth 0 (pp2048, KV q8_0/q4_0, `-ub 1024`) | 772 t/s | ~965 t/s |
| Qwen3.8-27B, prefill at 48k context (pp2048, KV q8_0/q4_0, `-ub 1024`) | 491.6 t/s | 779.6 t/s |

In production (`llama-server`, two slots, MTP + ngram-mod speculative decoding, KV q8_0/q4_0, 262k context), serving
OpenCode with Qwen3.8-27B:

| | median |
|---|---:|
| decode at 50-100k context | 74 t/s |
| decode at 100-150k context | 62 t/s |
| decode at 150-230k context | 50 t/s |
| cold 67k-token prompt | 860 t/s |
| prefill at 128-192k depth | ~550 t/s |

On the coding eval in `scripts/coding-eval`, the same model on this server passes 34/36 agentic attempts, the same
score as the hosted model on OpenRouter.

## Quick start

Build with Nix:

```sh
nix build .#rocm
./result/bin/llama-server --version
```

or with CMake, using the same ROCm toolchain as the flake (the HIP compiler path comes from the package's CMake flags):

```sh
HIPCC=$(nix eval --json .#packages.x86_64-linux.rocm.cmakeFlags | grep -o 'CMAKE_HIP_COMPILER:STRING=[^"]*' | cut -d= -f2)
nix develop .#rocm -c cmake -B build -G Ninja -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx908 \
    -DCMAKE_HIP_COMPILER="$HIPCC" -DCMAKE_BUILD_TYPE=Release
nix develop .#rocm -c cmake --build build -j
```

With a system ROCm install, see [docs/build.md](docs/build.md#hip).

Run with only the MI100 visible. If the machine has an AMD iGPU, ROCm shows it as a second device, and a gfx908-only
build has no code for it:

```sh
HIP_VISIBLE_DEVICES=0 ./result/bin/llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-Q5_K_S \
    -ngl 999 -c 262144 --flash-attn on --cache-type-k q8_0 --cache-type-v q4_0 \
    --spec-type draft-mtp --jinja
```

Runtime switches specific to this fork:

| variable | default | effect |
|---|---|---|
| `GGML_HIP_REPACK` | on | `0` keeps weights in the GGUF layout instead of the repacked MI100 layout |
| `GGML_HIP_FA_CDNA` | on | `0` disables the CDNA flash-attention kernel for prefill and large verify batches |
| `GGML_HIP_FA_KV_F16` | on | `0` disables the f16 copy of quantized K/V that large prefill batches still use where the CDNA kernel does not apply (head size 256 with an odd GQA ratio or other K/V type pairs) |
| `GGML_HIP_POOL_STATS` | off | `<seconds>`: log the device memory used and each memory pool's size at most this often |

All FlashAttention K/V type pairs are compiled by default (`GGML_CUDA_FA_QUANTS=all`), so mixed caches such as
K q8_0 / V q4_0 run at full speed.

## Verification

Changes are checked on the MI100 with:

- `test-backend-ops` against the CPU backend (full suite, run serially) and `test-backend-repack` for the weight layout
- `llama-perplexity --kl-divergence` against the previous build for every change to numerics
- interleaved `llama-bench` runs on a four-model suite (Qwen3.8-27B, Gemma 4 31B, Gemma 4 26B-A4B and its QAT variant)
- the production `llama-server` command for end-to-end prefill and speculative-decoding numbers
- [scripts/coding-eval](scripts/coding-eval): a coding benchmark run against any OpenAI-compatible endpoint. Twelve
  JavaScript problems in a plain mode and an agentic red/green/refactor mode, graded by hidden tests; the model's
  code is checked statically and run in a capability-restricted sandbox (Deno with no permissions, under bubblewrap)

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [HIP](docs/build.md#hip) | AMD Instinct MI100 (gfx908) |
| [RPC](tools/rpc/README.md) | All |

## Documentation

#### MI100

- [MI100 parity backlog, measurements and profiles](docs/backend/MI100-parity.md)
- [How to build (HIP section)](docs/build.md#hip)

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

This is a private, single-target fork. Changes are committed directly; there is no upstream contribution flow from here.
For the general project, see [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and its
[CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml) - the upstream projects this fork is based on - MIT license
- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
- [acornjs/acorn](https://github.com/acornjs/acorn) - JavaScript parser, vendored in `scripts/coding-eval` - MIT license
