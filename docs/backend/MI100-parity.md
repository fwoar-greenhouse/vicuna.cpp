# MI100 (gfx908) optimization-parity backlog

This fork builds the ROCm backend for AMD Instinct MI100 (gfx908, CDNA1) only.
This file lists kernel paths where a modern NVIDIA GPU gets a faster path than gfx908.

Status: backlog only. Work starts after the pruned tree passes the verification gate (tag `mi100-pruned-verified`).
Each item also needs a profile (rocprofv3) that shows the kernel is below ~60% of its roofline and takes at least 15% of decode or verify time.

Line numbers refer to the upstream base `83209c3d2` and can drift after the gfx908 cleanup.

MI100 peak rates: f16 MFMA ~185 TFLOPS, i8 MFMA ~185 TOPS, bf16 ~92 TFLOPS, f32 MFMA ~46 TFLOPS, 1.23 TB/s HBM2, 120 CUs, 64 KB LDS per CU, wave64.
Because i8 and f16 run at the same rate, MMQ helps by saving dequantization work and memory traffic, not by a higher peak rate.

## Baseline (2026-10-03, srv2, before any optimization)

Model facts (GGUF metadata): arch `qwen35`, 65 layers, full attention every 4th layer (about 16 attention layers, the rest gated DeltaNet), 24 Q heads, 4 KV heads (GQA 6), head dim 256, DeltaNet state 128, inner size 6144.

llama-bench, Qwen3.8-27B UD-Q5_K_S (17.37 GiB), `-ngl 99 -fa 1`, HIP_VISIBLE_DEVICES=0:

| test | t/s |
|---|---:|
| pp2 | 44.6 |
| pp4 | 69.2 |
| pp8 | 119.4 |
| pp16 | 228.3 |
| pp512 | 630.7 |
| tg128 | 28.4 |

- Decode reads about 18.65 GB per token, so tg128 is about 530 GB/s, or 43% of peak (48% of the ~1.1 TB/s that tuned GEMV kernels reach on MI100).
- A verify step at 16 tokens costs about 2.0x a decode step (1.56x the 2-token step).
- With head dim 256, the MFMA FA kernel runs only when ne1 * 6 > 64 (verify width >= 11). Decode and most verify steps use the tile kernel.
- Throughput drifts down about 1% per few minutes of sustained load. Interleave A/B runs.

### Kernel time profile (rocprofv3 --kernel-trace, llama-bench `-r 1`, tag mi100-pruned-verified)

Share of GPU kernel time by kernel family. "decode" = `-p 0 -n 32`, "batch 16" = `-p 16 -n 0`.

| profile | quantized matmul | rms_norm | quantize_q8_1 | flash attn | other notable |
|---|---|---:|---:|---:|---|
| Qwen3.8-27B decode | MMVQ 74.4% | 6.0% | 6.5% | 1.6% | glu 2.5%, GDN 1.7% |
| Qwen3.8-27B batch 16 | MMQ 74.2% | 2.8% | 3.8% | 1.4% | GDN 5.5% |
| Gemma 4 31B decode | MMVQ 78.0% | 9.5% | 4.4% | 5.4% | |
| Gemma 4 31B batch 16 | MMQ 78.2% | 5.3% | 3.0% | 7.4% | |
| Gemma 4 26B-A4B decode | MMVQ 46.0% | 17.6% | 10.3% | 7.4% | top-k 3.5%, binbcast 3.5% |
| Gemma 4 26B-A4B batch 16 | MMQ 62.7% | 4.8% | 2.9% | 4.5% | MMF 13.9% (200 us per call) |

- Qwen decode runs ~1,900 kernels per token. Small kernels take ~4.5 us each regardless of size, about 7.8 ms per token in total.
- MMVQ alone reaches ~785 GB/s on Qwen decode (64% of peak). The rest of the gap to the 43% end-to-end figure is small-kernel time.

## Priority: generic parity first

Goal: bring the gfx908 backend to parity with the CUDA backend across model types before any model-specific tuning.
An item is generic if it helps a whole op family (all quant types, all head sizes) and not one model's shapes.
Every change is measured on the whole benchmark suite below, not on one model.

1. [ ] Decode GEMV bandwidth for all quant types (A12): weight repack at load and a load-first MMVQ (P1, P2). Target ~1.0 TB/s.
2. [ ] Small-batch quantized matmul, 2-16 columns, all types (A12, A4, A6): multi-row MMVQ, then MFMA MMQ with wider J and 32x32 i8 tiles (P1, P5, P8).
3. [x] Flash attention for 1-16 query rows, all head sizes and GQA ratios (A1, A2): split-KV MFMA with 4x4x4 / 16x16x16 shapes, including D=512 (P4). Done for D = 64/128/256/512, see the status below.
3a. [x] Quantized KV cache in the tile and MMA FA kernels (A14): dequantize K/V tiles while loading them into LDS instead of converting the whole cache to f16 first. Done for the MMA kernel; the tile kernel only gets quantized K/V for D % 64 != 0.
3b. [ ] More KV cache types (A15): IQ4_NL and a rotation-aware 3-4 bit codebook type (TurboQuant-style), with FA readers.
4. [ ] MoE MUL_MAT_ID without the host-synchronizing hipBLAS fallback (A3).
5. [ ] Wave64-aware small kernels with DPP reductions (A11, P2).
6. [ ] Flash attention load pipelining (A5).
7. [ ] Gated DeltaNet / linear-attention decode and verify kernel (P1, P3).
8. [ ] Remaining gaps A7-A10, A13.

Model-specific tuning (for example Qwen3.8-27B verify shapes) comes after this list.

### Item 2 status (2026-10-04)

Done: MMVQ/MMQ crossover per type and matrix size (`mmvq_max_ncols`), MMQ register prefetch of the next x tile (two-phase `load_tiles`), MMQ J=16/32 with 4 warps, I=64, 2 blocks per CU.
MMQ at J=16 now reaches ~370-400 GB/s for K-quants and ~550-580 GB/s for Q8_0 on 17408x5120 / 5120x17408. On matrices with >= 4096 rows MMVQ still wins up to 2 columns for Q4_K/Q5_K, 4 for Q6_K/Q8_0 and 5 for Q4_0.

Tried without gain:
- MMVQ for MUL_MAT_ID up to 16 tokens: microbench with random routing favors MMVQ, but real routing reuses experts and MMQ wins (26B-A4B pp16 -11%).
- MMQ I=32 (128 threads, occupancy 4): slower, up to 1.4x for Q8_0.
- L2 prefetch of the next x tile with throwaway loads: slower; the duplicate requests compete with the real loads.
- `sched_group_barrier` to issue all tile loads first: no gain without overlap across iterations.

Design note, needs a weight repack (deferred): a skinny MFMA kernel without LDS for x. Each wave owns 16 rows and loads them straight into the B operand of `v_mfma_i32_16x16x16i8` (y as A, so each lane keeps its own row scale), with y shared through L1. A prototype for Q8_0 reached only ~360 GB/s: with the row-major block layout one load instruction touches 16 rows x 32 bytes, and even x alone stays below ~490 GB/s. A repack that stores, per 16-row group and 32-value block, the 16 rows' bytes in MFMA operand order would turn this into 512-byte contiguous loads per wave. Combined with per-row scales next to the data, this could serve 1-16 columns with one kernel at MMVQ-like bandwidth and MFMA compute.

### Item 3/3a status (2026-10-04)

Done:
- Vector kernel (`fattn-vec.cuh`) rewritten with MFMA. A CUDA block works on one K/V head and 16 Q columns (Q rows x the Q heads that share the K/V head), so K/V is read once per pass instead of once per Q head. Each wave owns tiles of 16 KV rows: KQ and VKQ are `v_mfma_f32_16x16x16f16` with the Q columns as N; K is dequantized into the A operand, V^T is gathered from 4 rows per lane into the A operand, and the KQ accumulator is already the B operand of VKQ. The loads of the next tile are issued before the current tile is processed (register double buffer). All K/V types and pairs, D = 64/128/256/512.
- The vector kernel is used for quantized K/V up to 16 Q rows and for f16 K/V if it needs at most 2 passes over K/V (or instead of the tile kernel up to 16 rows). Short f16 K/V (<= 2048) with a power of 2 GQA ratio and 1-2 Q rows stays on the tile kernel (lower fixed cost, e.g. Gemma 4 SWA layers).
- MMA kernel reads q4_0/q4_1/q5_0/q5_1/q8_0/bf16 K/V directly (3a): each thread dequantizes whole 32-value blocks into the LDS tile. The f16-only instances are unchanged (template flag), so f16 K/V keeps its speed. No f16 copy of K/V: the compute buffer for Qwen3.8-27B with q8_0 KV at 262144 context drops from 1360 to 505 MiB and no longer grows with the context.

FA op time (test-backend-ops perf, base -> new), Qwen3.8-27B shape (D=256, 4 KV heads, GQA 6), q8_0 K/V, 64k KV: 1 row 1067 -> 206 us (~700 GB/s), 4 rows 2102 -> 342 us, 16 rows 3473 -> 849 us; prefill 512 rows at 16k KV 7676 -> 7592 us (q4_0: 7677 -> 6724 us).
llama-bench, interleaved: Qwen3.8-27B q8_0 KV tg64 at depth 65536 18.0 -> 27.3 t/s (30.2 at depth 0), pp16 142.6 -> 194.2 t/s; Gemma 4 31B q8_0 KV at depth 16384 tg64 +7%, pp16 +23%.

Remaining gaps:
- Decode reaches ~700 GB/s on long quantized K/V: one wave per SIMD (~210 VGPRs for D=256), so the per-tile latency is only hidden by the prefetch of one tile. Two waves per SIMD would need < 128 VGPRs.
- The vector kernel has ~20 us fixed cost per call (prologue, LDS combine of 4 warps, partial results and `flash_attn_combine_results`); f16 K/V without GQA at 1.5-2k context is ~5-8% slower than the old scalar kernel.
- D=512 f16 decode stays on the tile kernel (equal speed); the MMA kernel still loads tiles synchronously (A5).

Design note for 3b (new KV types): both kernels convert in the load step only. A new type needs a raw load + unpack to 4-value groups (vector kernel: `ggml_cuda_fattn_vec_load_q`/`unpack_q`, the values become f16 by `v_perm` with 0x64 bytes) and a block loader for the MMA tiles (`flash_attn_ext_f16_load_tile_q`). A codebook type (Lloyd-Max levels for Hadamard-rotated K/V) would replace the `v_perm` + scale step with a 16-entry f16 lookup (per-lane table in registers or LDS), the rest of both kernels stays the same.

### Benchmark suite

All under `~/.cache/huggingface/hub/`, run with `HIP_VISIBLE_DEVICES=0 llama-bench -ngl 99 -fa 1 -p 2,4,8,16,512 -n 128`:

| model | file | type | attention | exercises |
|---|---|---|---|---|
| Qwen3.8-27B | UD-Q5_K_S (17.4 GiB) | dense hybrid, 65 layers, GDN + full attention every 4th layer | D=256, GQA 6 | MMVQ/MMQ, GDN, FA D=256 |
| Gemma 4 31B | UD-Q5_K_XL (20.4 GiB) | dense, 60 layers | SWA D=256 GQA 2; global D=512 GQA 8 | MMVQ/MMQ, FA D=256 and D=512 |
| Gemma 4 26B-A4B | UD-Q5_K_XL (19.8 GiB) | MoE | as Gemma 4 | MUL_MAT_ID, MoE fusions |
| Gemma 4 26B-A4B QAT | UD-Q4_K_XL (13.3 GiB) | MoE | as Gemma 4 | MUL_MAT_ID at Q4 |

Each model also has an MTP draft GGUF (`mtp-*.gguf`) for speculative-decoding tests.

## All gaps

| # | Kernel / path | NVIDIA gets | gfx908 gets | MI100 approach |
|---|---|---|---|---|
| A1 | FA decode and small batch (`fattn.cu:684-697`, `fattn-mma-f16.cuh:1880-1885`) | MMA kernel at ncols=8 for GQA decode | tile kernel (v_dot2) when ne1*gqa <= 16 at D=128 | Allow ncols 8 and 16 on CDNA: `v_mfma_f32_16x16x16f16` with GQA heads on N, or `v_mfma_f32_4x4x4f16` for ncols <= 4. Then re-tune the thresholds. |
| A2 | FA for MLA and large heads 576/512, 512, 320 (`fattn.cu:184-185,694`, `fattn-mma-f16.cuh:1881`) | ncols2 16/32 variants per arch | tile kernel unless ne1*gqa > 128. CDNA configs for ncols 8/16 exist but are not compiled. | Enable ncols=16 for DKQ > 256. Keep nbatch_K2/V2 inside 64 KB LDS. |
| A3 | MoE MUL_MAT_ID prompt processing (`mmq.cu:396-405`, `ggml-cuda.cu:1942-2003,2577-2585`) | always MMQ | sorted hipBLAS path with host syncs, which also disables HIP graphs | Always use MMQ for MUL_MAT_ID on CDNA1, then tune. |
| A4 | MMQ tile shapes (`mmq-config-cdna.cuh`, `mmq.cuh:181-185`, `mma.cuh:1401-1409`) | J up to 128, 256 threads, occupancy 2 | one config: 512 threads, J <= 64, 16x16 i8 tiles; dense ne11 > 128 goes to rocBLAS | J=96/128 configs with `v_mfma_i32_32x32x8i8`, then widen the `ggml_cuda_should_use_mmq` window. |
| A5 | FA load pipelining (`fattn-mma-f16.cuh:377-442,504-525`) | multi-stage cp.async prefetch | nstages=0, so loads and MFMA run in series | Software double buffer (global -> VGPR -> LDS), or `buffer_load ... lds`. |
| A6 | MMF dense f16/bf16 at batch 3-16 (`mmf.cu:174-175`, `mmvf.cu:847,865`) | MMF up to 16 columns | rocBLAS | Tune the MFMA MMF path (`16x16x16f16`, `16x16x8bf16`) and remove the CDNA1 exclusion. |
| A7 | Sparse-mask FA (`fattn.cu:8-151`, `fattn-mma-f16.cuh:2069-2093`) | mask compaction and gather | not available on HIP | Port with 64-bit ballot and popcount. Needs A1/A2 first. |
| A8 | Mamba-2 SSD prefill (`ssm-scan.cu:361-781,840-848`) | chunked SSD with cuBLAS batched GEMM | sequential scan | Use hipBLAS strided-batched GEMM. |
| A9 | Lightning indexer (`lightning-indexer.cu:450-511`) | nvcuda::wmma kernel | vector kernel that assumes 32 lanes | MFMA `16x16x16f16` port, wave64-aware. |
| A10 | ARGSORT on large rows (`common.cuh:114-116`, `ggml-cuda.cu:5577-5586`) | CUB segmented sort | bitonic only; rows above 16384 columns fall back to CPU | hipCUB / rocPRIM segmented radix sort. |
| A11 | 32-lane logical warps on wave64 (`softmax.cu:308,377`, `topk-moe.cu:91,115`) | full warps | half of each wave is idle | Template on `ggml_cuda_get_physical_warp_size()`. |
| A12 | MMVQ/MMVF tuning (`mmvq.cu:105-145,468-598`) | per-arch tables | shares the GCN table | Sweep nwarps 4/8 at ncols=1 for CDNA1. |
| A14 | Quantized KV in tile/MMA FA (`fattn.cu` need_f16_K/V, `fattn-common.cuh` f16 extra data) | same as gfx908 (converts to f16) | Prefill and verify convert the whole visible K and V to f16 in a scratch buffer (`nelements*2` bytes each; ~1 GiB at 262k ctx for Qwen3.8-27B) on every call | Dequantize K/V tiles to f16 in LDS inside the tile and MMA kernels. Saves the scratch memory and the conversion traffic. Beyond CUDA parity. |
| A15 | KV cache types | same set | FA reads only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 in-kernel | Add IQ4_NL and a 3-4 bit Lloyd-Max codebook type for Hadamard-rotated K/V (llama.cpp already rotates quantized KV, `attn_rot_k/v`); optional QJL residual later. Needs a ggml type, CPU reference, SET_ROWS quantize kernel and FA readers. |
| A13 | Multi-GPU | NCCL on by default | RCCL off by default; internal allreduce | Turn on `GGML_HIP_RCCL` for multi-MI100 nodes. |

## Prior art

Collected 2026-10-03. Most numbers are self-reported by the authors.

| # | Source | What it shows | Informs |
|---|---|---|---|
| P1 | [sixvolts/llama-gfx908-rune](https://github.com/sixvolts/llama-gfx908-rune) (llama.cpp fork for MI100, ROCm 6.4.3) | Q8_0 GEMV with compile-time K unroll, all loads issued first, 8-byte loads, 4 rows per block for 2-4 columns (bit-exact). MMQ column loop as straight-line 1/2/4/full variants (runtime bounds spill accumulators), and load/LDS-store split around `sched_barrier`. GDN: 16 lanes per state column, DPP sums, float4 k/q ring, `__launch_bounds__(64,1)` (2 waves/SIMD spills to AGPR and forces `vmcnt(0)`). Decode 34.5 -> 45.6 t/s on a GDN MoE. | 1, 2, 3, 5 |
| P2 | [sixvolts/llamacpp-gfx906-furnace](https://github.com/sixvolts/llamacpp-gfx906-furnace), [reinstinct](https://github.com/sixvolts/reinstinct) (MI50) | Three-plane Q4_K/Q5_K/Q6_K repack at load: matvec goes from ~58% to ~89% of HBM with dp4a. Qwen3.8 27B at 35.3 t/s decode on a 1 TB/s MI50 (vs 26.0 for llama.cpp). DPP/ds_swizzle reductions instead of ds_bpermute. | 1, 2, 5 |
| P3 | [btbtyler09/vllm-gfx908](https://github.com/btbtyler09/vllm-gfx908) docs/mi100_decode_opt, [mi100-llm-testing](https://github.com/btbtyler09/mi100-llm-testing) | Measured GEMV ceiling 1.10-1.17 TB/s (wvSplitK bf16). Graph node floor 1.3-1.8 us. W4 GEMV design rules (10-20 KB in flight per CU, lanes along K, dot4 with Q8_1, DPP, no split-K). Fused HIP GDN decode 5.6 us vs 11.8 us for the FLA Triton kernel. gfx908 has `v_dot4c_i32_i8`, `v_dot2c_f32_f16`, `v_dot8_i32_i4`, DPP row_bcast; no bf16_1k MFMA, `v_pk_fma_f32` or `v_dot2_f32_bf16`. | 1, 3, 5 |
| P4 | [vLLM ROCm paged attention](https://github.com/vllm-project/vllm/blob/main/csrc/rocm/attention.cu) | Split-KV (256-token partitions) plus reduce. `v_mfma_f32_4x4x4f16` kernel for gqa <= 4, 16x16x16 above. Multi-query rows for MTP. Runs on gfx908 with the gate macro added (btbtyler09 fork). | 4, 6 |
| P5 | llama.cpp [#14949](https://github.com/ggml-org/llama.cpp/pull/14949), #23227, #19806, #28576 | MFMA MMQ on gfx908 (1.08-3.39x at batch 32). Per-quant MMVQ->MMQ crossover (Q4_K/Q5_K to MMQ from batch 4, tuned on gfx90a only). | 2 |
| P6 | [LLVM GCNHazardRecognizer](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp), [amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) | gfx908 MFMA C/D live in AGPRs. AGPR read after MFMA write needs 4/10/18 wait states (4x4/16x16/32x32), so small tiles have the cheapest epilogue. | 2, 4 |
| P7 | btbtyler09 GDN Triton notes | Triton on gfx908: `tl.dot` with M or N < 16 miscompiles (pad to 16). `AMDGCN_USE_BUFFER_OPS=1` causes SGPR spills. `num_stages` costs VGPRs. | 3, 4 |
| P8 | [stew675/llama-cpp-rdna-boosts#57](https://github.com/stew675/llama-cpp-rdna-boosts/pull/57) | MMVQ with 1/2/4 rows per block per weight type for 2-8 columns, bit-exact. +22-30% for 27B with MTP (RDNA4). | 2 |

Not usable on gfx908: CK flash attention (MI200+), AITER CK ops, hipBLASLt (partial), upstream vLLM skinny GEMMs (gated to gfx90a+ in source, but work when the gate is added).

Power: decode barely improves above a 200 W cap on MI100 (btbtyler09). Raising the memory clock is untested on MI100.

KV memory for Qwen3.8-27B (16 attention layers x 4 KV heads x 256): 64 KiB per token at f16, so 262k context is ~16 GiB f16, ~8.5 GiB q8_0, ~4.5 GiB q4_0. Mixed K/V types (for example K q8_0, V q4_0) used to fall back to an f16 conversion on every decode call; commit 0e1f695d0 compiles all FA K-V pairs by default.

## Not gaps

- Native FP4: MI100 has no FP4 hardware. MXFP4 and NVFP4 already use the i8 MFMA path.
- Quantized KV: CUDA converts K/V to f16 for the tile and MMA kernels; MI100 now reads quantized K/V in the vector and MMA kernels (3a).
- Already work on HIP: conv2d/conv3d MFMA, rms_norm/rope/GLU/topk-moe fusions, HIP graphs, gated delta net, top-k.
- WMMA flash attention (`GGML_HIP_ROCWMMA_FATTN`): removed upstream. gfx908 has no WMMA.
