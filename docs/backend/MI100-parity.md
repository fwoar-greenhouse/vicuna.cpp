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

### Kernel time profile at HEAD (2026-10-04, after the item 4/5 work)

Same method, commit 9e510ec. "MMQ" includes `mul_mat_q_stream_k_fixup` (Qwen batch 16: 13% of kernel time, 31B: 7%, 26B-A4B: 6%).

| profile | quantized matmul | rms_norm | quantize_q8_1 | flash attn | other notable | kernels per token (24f5984 -> now) |
|---|---|---:|---:|---:|---|---:|
| Qwen3.8-27B decode | MMVQ 76.4% | 5.5% | 4.1% | 1.6% | glu 2.8%, get_rows 2.0%, GDN 1.7% | 1,905 -> 1,709 |
| Qwen3.8-27B batch 16 | MMQ 72.7% | 2.9% | 4.0% | 1.0% | GDN 6.1% | |
| Gemma 4 31B decode | MMVQ 79.2% | 8.8% | 3.4% | 6.5% | cpy 1.6% | 1,450 -> 1,280 |
| Gemma 4 31B batch 16 | MMQ 79.2% | 5.3% | 3.5% | 6.0% | | |
| Gemma 4 26B-A4B decode | MMVQ 49.7% | 16.5% | 7.8% | 9.7% | top-k 3.3%, glu 2.6%, cpy 2.5%, router MMVF 2.4% | 1,285 -> 1,140 |
| Gemma 4 26B-A4B batch 16 | MMQ 70.3% | 6.1% | 3.6% | 4.0% | router MMVF 3.0% (was MMF 15.1%), D2D copies 3.8% | |

- Most small kernels still take 4.2-4.5 us in a HIP graph whatever they do, so the number of launches matters more than the work inside them.
- One-block kernels (rms_norm of a single row) seem to pay for cold instruction cache misses: more code costs time even when it is not executed often (likely cause, not measured with counters). Unrolled epilogues (e.g. a q8_1 output in rms_norm) made the kernel 3-6 us slower in the graph, but not in a microbenchmark.

## Progress (2026-10-04): tag mi100-pruned-verified -> 2b2b4f8

Interleaved llama-bench (2 rounds, `-ngl 99 -fa 1 -r 3`, f16 KV unless noted), Nix flake build vs pre-optimization libs:

| model | pp2 | pp4 | pp8 | pp16 | pp512 | tg128 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B Q5_K_S | +3.6% | +6.2% | +14.0% | +8.4% | -0.7% | 28.2 -> 30.9 (+9.6%) |
| Gemma 4 31B Q5_K_XL | +5.9% | +9.2% | +7.4% | +8.8% | -0.2% | 22.7 -> 25.8 (+13.7%) |
| Gemma 4 26B-A4B Q5_K_XL | +0.2% | +2.4% | +3.5% | +9.2% | +11.4% | 94.5 -> 96.8 (+2.4%) |
| Gemma 4 26B-A4B QAT Q4_K_XL | +3.3% | +4.6% | +3.0% | +9.3% | +17.4% | 110.3 -> 116.4 (+5.5%) |

Qwen3.8-27B, KV q8_0/q8_0 at 64k depth: tg64 18.0 -> 26.5 (+47%), pp16 134.8 -> 192.1 (+42%). FA compute buffer at 262k context: 1360 -> 505 MiB.

Small kernels and MoE (items 4/5), 24f5984 -> 9e510ec, same method:

| model | pp2 | pp4 | pp8 | pp16 | pp512 | tg128 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B Q5_K_S | +1.5% | +0.7% | +0.6% | +0.4% | +0.4% | 31.0 -> 31.4 (+1.3%) |
| Gemma 4 31B Q5_K_XL | +2.2% | +0.8% | +0.5% | +0.4% | +0.7% | 25.9 -> 26.3 (+1.9%) |
| Gemma 4 26B-A4B Q5_K_XL | +7.3% | +32.1% | +17.5% | +13.7% | +1.5..6% | 96.9 -> 104.4 (+7.8%) |
| Gemma 4 26B-A4B QAT Q4_K_XL | +7.1% | +36.0% | +20.2% | +16.0% | +1.8% | 116.1 -> 125.6 (+8.2%) |

## Priority: generic parity first

Goal: bring the gfx908 backend to parity with the CUDA backend across model types before any model-specific tuning.
An item is generic if it helps a whole op family (all quant types, all head sizes) and not one model's shapes.
Every change is measured on the whole benchmark suite below, not on one model.

1. [ ] Decode GEMV bandwidth for all quant types (A12): weight repack at load and a load-first MMVQ (P1, P2). Target ~1.0 TB/s. Done: weight repack on by default (`GGML_HIP_REPACK=0` to disable), see the status below. Open: recover dense pp8/pp16 and QAT pp512.
2. [ ] Small-batch quantized matmul, 2-16 columns, all types (A12, A4, A6): multi-row MMVQ, then MFMA MMQ with wider J and 32x32 i8 tiles (P1, P5, P8).
3. [x] Flash attention for 1-16 query rows, all head sizes and GQA ratios (A1, A2): split-KV MFMA with 4x4x4 / 16x16x16 shapes, including D=512 (P4). Done for D = 64/128/256/512, see the status below.
3a. [x] Quantized KV cache in the tile and MMA FA kernels (A14): dequantize K/V tiles while loading them into LDS instead of converting the whole cache to f16 first. Done for the MMA kernel; the tile kernel only gets quantized K/V for D % 64 != 0.
3b. [ ] More KV cache types (A15): IQ4_NL and a rotation-aware 3-4 bit codebook type (TurboQuant-style), with FA readers.
4. [x] MoE MUL_MAT_ID without the host-synchronizing hipBLAS fallback (A3). Done: always MMQ, see the status below.
5. [ ] Wave64-aware small kernels with DPP reductions (A11, P2). Partly done (rms_norm, topk-moe, q8_1 reuse, rms_norm fusions), see the status below.
6. [ ] Flash attention load pipelining (A5). Tried for the MMA kernel, see the long-prefill status below: not enough registers with in-kernel dequantization.
7. [ ] Gated DeltaNet / linear-attention decode and verify kernel (P1, P3). Prefill (>= 64 tokens) uses the chunked delta rule since 83089a4, see below.
8. [ ] Remaining gaps A7-A10, A13.

Model-specific tuning (for example Qwen3.8-27B verify shapes) comes after this list.

### Item 1 design: weight repack at load (2026-10-04)

Goal: every wave load instruction reads 64 x 16 contiguous bytes that the wave uses fully, with one copy of the weights.

Layout "S64" (same size as the GGUF layout, so row offsets, views at stripe boundaries and MoE expert slices keep their byte offsets):
- Each 2D matrix (each expert of a MUL_MAT_ID stack) is cut into stripes of 64 rows; the last stripe has `ne1 % 64` rows if that is not 0. A stripe starts at the same byte offset as its first row in the GGUF layout and has the same size.
- A block (K-quant superblock) is split into 16-byte chunks plus a small rest. Inside a stripe of `r` rows and `nkb` blocks per row, chunk `c` of block `kb` of row `i` is at `((c*nkb + kb)*r + i)*16` (plane-major). The rest bytes (Q6_K `d`) follow the planes as `[kb/8][row][8]` groups, so one lane reads the rests of 8 blocks with one 16-byte load.
- Per type: Q4_K 144 B = 9 chunks (chunk 0 = d, dmin, packed scales); Q5_K 176 B = 11 chunks (chunk 0 = d, dmin, scales, then 2 qh, 8 qs); Q6_K 210 B = 13 chunks (8 ql, 4 qh, 1 scales) + 2 B d. The bytes inside a chunk are not changed, so the existing dequantization code works on a chunk after the load. Q8_0 / Q4_0 / IQ4_XS: the same with a unit of 8 / 8 / 2 blocks so that the d values fill whole chunks (later).
- get_tensor runs the inverse transform, so the CPU fallback, `llama-quantize`-style reads and the round-trip tests see the GGUF layout.

GEMV reader (MMVQ-R, 1-8 columns): one lane = one row, a wave = one stripe, the waves of a block (and, for small matrices, several blocks) split K. Each lane loads the 11/13 chunks of its block (all loads 1 KB contiguous per wave), so a lane holds a whole superblock and needs no cross-lane reduction. The q8_1 activation is the same for all lanes, so it is read with scalar loads (SGPRs) from the normal q8_1 buffer; no new activation layout is needed. Split-K partial sums are added in a fixed order (LDS inside a block, last-block fixup across blocks), so the result does not depend on timing.

Other readers (all needed before the switch can be default ON, because a refused large-batch MUL_MAT would fall back to the CPU):
- MMQ: `load_tiles` for the repacked types read whole 16-byte chunks with lane = row (64-row MMQ tile = one stripe, one superblock per iteration = 11/13 KB contiguous) and write the usual LDS tile, so `vec_dot` and the MFMA part do not change. Also for MUL_MAT_ID.
- Dequantize to f16/f32 (hipBLAS path for large dense batches), GET_ROWS (one row = gather of its chunks).
- MMVQ fusions (gate + GLU, bias) in MMVQ-R.

Plumbing: a ROCm "extra" buffer type per device (`ROCm0_Repack`), returned by `ggml_backend_dev_get_extra_bufts` unless `GGML_HIP_REPACK=0`. Device memory and allocation as the normal buffer; `set_tensor` uploads to a device scratch and runs a repack kernel per chunk of stripes (partial writes: read-modify-write of the touched stripes), `get_tensor` runs the inverse, `memset_tensor` of whole stripes is a plain memset, `cpy_tensor` only between two repack buffers. A tensor is repacked if it is in this buffer type, not a view, contiguous, and its type has a layout (and `ne0 % 256 == 0` for K-quants); other tensors in the buffer stay in the GGUF layout. `supports_op` refuses any op that reads a repacked tensor without a reader (MUL_MAT / MUL_MAT_ID src0, GET_ROWS src0). The model loader lists the extra GPU buffer types before the default one, so weights that the repack type accepts go there.

Microbenchmark (standalone HIP, 1000-2000 back-to-back launches over 4+ copies of the weights so that they never sit in L2; the baseline is the real `mul_mat_vec_q` from mmvq.cu in the same harness):

| matrix (rows x K) | Q5_K MMVQ | Q5_K S64 | Q6_K MMVQ | Q6_K S64 |
|---|---:|---:|---:|---:|
| 17408 x 5120, 1 col | 93.7 us (654 GB/s) | 69.0 us (888 GB/s) | 106.7 us (685) | 84.4 us (866) |
| 5120 x 17408, 1 col | 94.1 us (651) | 70.3 us (871) | 115.7 us (632) | 85.2 us (859) |
| 21504 x 5376, 1 col | 117.7 us (675) | 88.9 us (894) | 139.4 us (680) | 108.7 us (872) |
| 6144 x 5120, 1 col | 37.8 us (572) | 27.4 us (788) | 39.7 us (650) | 35.0 us (737) |
| 1024 x 5120, 1 col | 11.1 us | 10.0 us | 11.9 us | 12.5 us |
| 17408 x 5120, 2 / 4 / 8 cols | 116.6 / 169.0 / 301.3 us | 74.6 / 86.2 / 192.4 us | 125.8 / 182.0 / 329.0 us | 101.4 / 104.8 / 206.8 us |

- 1 column: +26..37% on the large matrices. A pure read of the same layout (no math) reaches ~950 GB/s and a plain streaming read of a 61 MB buffer ~1000 GB/s per launch on this card, so the S64 GEMV is at ~93% of the access-pattern ceiling.
- 8 columns are VALU/scalar-load bound in this kernel (but still faster than MMVQ); MMQ-R (or the skinny MFMA kernel from item 2, which this layout also serves) takes over from ~5 columns.
- Small matrices need the cross-block split: 1024 x 5120 Q5_K is 14.1 us with 4 waves per stripe and 10.0 us with 4 blocks of 4 waves.

### Item 1 status (2026-10-05)

Done, on by default since this change (`GGML_HIP_REPACK=0` restores the GGUF layout): the `ROCm0_Repack` buffer type with the S64 layout for Q4_0, Q8_0, Q4_K, Q5_K, Q6_K and IQ4_XS (matrices with >= 256 rows), and readers for every op that can read such a weight:
- MMVQ-R (`mmvq-repack.cu`): MUL_MAT up to 4 (Q5_K), 5 (Q4_K, IQ4_XS) or 6 (Q4_0, Q8_0, Q6_K) columns, all columns up to 8 for matrices with < 1024 rows, MUL_MAT_ID up to 8 tokens, gate + GLU and bias fusions. Q4_0/Q8_0 step over groups of 8 blocks (their d values are one 16-byte rest chunk); the full groups have no per-block branches (with them the loads waited: Q4_0 68 -> 57 us). Split K over blocks (last-block fixup in a fixed order) only for fewer than nsm/4 tiles; more blocks with fewer waves were slower in the models.
- MMQ (`mmq-repack.cuh`): tile loaders with lane = row, same shared memory tile as before; MUL_MAT up to the MMQ limit and MUL_MAT_ID above 8 tokens.
- hipBLAS path: dequantize into f16 through shared memory (lane = row decode, 256-byte row stores). This is 14-22% faster than the to_fp16 kernels of the GGUF layout for the K-quants (Q5_K 17408x5120 at 512 columns 1949 -> 1655 us).
- GET_ROWS of 2D weights; stripe-aligned views (start on a stripe, end on a stripe or at the end of the matrix).
- get_tensor returns the GGUF layout (CPU fallback, tests); test-backend-repack checks set -> get bit-exact incl. partial and unaligned writes, views and memset.

test-backend-ops perf, us/run, same case on the default buffer -> repack (17408 x 5120, 1 column): Q5_K 113.6 -> 71.0, Q6_K 109.3 -> 92.8, Q4_K 87.8 -> 58.9, Q8_0 120.9 -> 104.6, Q4_0 71.4 -> 56.9, IQ4_XS 72.2 -> 68.2. MUL_MAT_ID (128 experts, 8 used, 1 token): Q5_K 704 x 2816 21.4 -> 15.1, Q4_0 2816 x 704 22.8 -> 11.8.

llama-bench, interleaved OFF/ON, 2 rounds, `-ngl 99 -fa 1 -r 3`, commit d33d8f8:

| model | pp2 | pp4 | pp8 | pp16 | pp512 | tg128 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B Q5_K_S | +20.7% | +29.4% | -1.7% | -1.5% | +8.9% | 31.63 -> 35.52 (+12.3%) |
| Gemma 4 31B Q5_K_XL | +37.4% | +46.7% | -2.1% | -1.8% | +13.6% | 26.42 -> 31.18 (+18.0%) |
| Gemma 4 26B-A4B Q5_K_XL | +4.2% | +8.3% | +15.5% | +3.4% | +2.9% | 103.51 -> 107.52 (+3.9%) |
| Gemma 4 26B-A4B QAT Q4_K_XL | +8.9% | +15.1% | +27.6% | +3.0% | -1.0% | 124.83 -> 132.44 (+6.1%) |

- Correctness: KLD vs OFF (base ub 512, -c 2048, 16 chunks): Qwen3.8-27B ON ub 512 0.000000 (identical f16 weights in hipBLAS), ON ub 4 0.00218 vs OFF ub 4 0.00196 (other summation order); Gemma 4 31B ON ub 512 0.000000, ON ub 4 0.265 vs OFF ub 4 0.270 (the instruct model is far from wikitext, PPL 225). Greedy outputs diverge after 15-45 tokens at low-margin tokens.
- Load time (mmap, page cache warm): Qwen 1.9-2.3 -> 2.4 s, Gemma 4 31B 2.3 -> 2.9 s, 26B-A4B 1.8 -> 2.2 s, QAT 1.2 -> 1.4 s (upload through a device scratch and the repack kernel, one cudaMalloc per tensor). VRAM: same model buffer size, plus 256 KB of fixup counters per context and a <= 32 MB scratch during set_tensor.
- MoE decode gains less than the op times suggest: the expert GEMVs are 25-35% of the kernel time and the dense K = 2816 matrices of 26B-A4B are as fast as before.

Remaining gaps:
- pp8/pp16 of the dense models are 1.5-2% slower: Q5_K/Q4_K MMQ with the repacked tile loader is 3-4% slower than on the GGUF layout (each warp loads the d/scales and qh chunks again; sharing them through shared memory with an extra barrier was 30% slower). 26B-A4B QAT pp512 -1.0% (Q4_0 MMQ, worst on 2816 x 2112).
- MMVQ-R is 5-20% slower than MMVQ for small matrices that stay in L2 in test-backend-ops (1024 x 5120); not visible end to end.
- Not repacked: Q3_K, IQ4_NL (1.5% of Qwen3.8-27B), other IQ types, matrices with < 256 rows.
- Default on (decided 2026-10-04). Known cost: pp8/pp16 of the Q5_K dense models and QAT pp512 are 1-2% slower than with `GGML_HIP_REPACK=0`; follow-up: extend the repacked GEMV past 8 columns or retune the crossover, and avoid per-warp scale reloads in the repacked MMQ loader.

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

### Items 4/5 status (2026-10-04)

Done (each bit-identical to the code before, except where noted):
- Small f32 matrices (MoE router, 2816x128) with 4-16 columns use MMVF (in chunks of 8 columns above 8) instead of MMF, which ran as 2 blocks: 100-200 us -> 11-33 us per call. 26B-A4B pp4 +26-30%, pp16 +12-14%.
- `rms_norm_f32_vec`: wave64 rms_norm with float4 loads, the row and the weights in registers, 64/256 threads and the reduction of the old kernel mapped to DPP/ds_swizzle steps (same order, same bits). Covers rms_norm, +mul, +mul+add, +scale for ncols % 4 == 0 up to 8192. 5376 columns mul+add 11.7 -> 7.0 us.
- MMVQ keeps the q8_1 copy of src1 during the graph evaluation (up to 4 entries, dropped when a node writes the src1 memory), so Q/K/V, gate/up and the GDN in_proj/z/alpha/beta matmuls quantize once. Decode quantize_q8_1 launches: Qwen 453 -> 257 per token, 26B-A4B 266 -> 181.
- MUL_MAT_ID always uses MMQ (A3): no host sync and HIP graphs stay on for MoE models with <= 64 experts at > 128 tokens. MUL_MAT_ID 32 experts at 256-512 tokens 1.4-2.7x faster than the sorted hipBLAS path.
- topk-moe: DPP/ds_swizzle shuffles, 13.2 -> 11.4 us per call.
- Fusions rms_norm -> scale -> mul (router input) and rms_norm -> mul -> add -> mul by one value (layer output scale): -2 launches per 26B-A4B layer, -1 per 31B layer.

Tried without gain:
- rms_norm (+mul) and GLU writing the q8_1 copy for the next MMVQ directly (bit-identical to quantize_q8_1): removed 2-3 quantize launches per layer, but the one-block rms_norm got 3-6 us slower in the graph (cold I-cache, more code; LDS staging and a non-unrolled loop halved the cost but it stayed slower than norm + quantize). The GLU variant was neutral.
- A 4-values-per-thread quantize_q8_1 (with DPP sums): same kernel time, not kept.
- Stream-k fixup inside `mul_mat_q` (last block of a tile adds the partial sums, counters reset by the kernel): the separate fixup kernels went away, but `mul_mat_q` got slower by the same amount or more (microbenchmark 130 -> 147 us for Q5_K 17408x5120 at 16 columns). The fixup kernel is still 6-13% of batch-16 time and the next target for small batches.

Remaining:
- Launch count: decode still runs 1,140-1,710 kernels per token at ~4.3 us each. Candidates: the three rms_norms of the same input in Gemma 4 MoE layers (not adjacent in the graph, needs a memory-safety check for out-of-order writes), get_rows + moe_weighted_reduction + binbcast in the MoE output, set_rows/cpy pairs, the D2D copies in MoE batch 16 (3.8%).
- The stream-k fixup kernel (above).
- softmax, norm/l2_norm/group_norm and other kernels with 32-lane logical warps (A11).

### Long-context prefill status (2026-10-05)

Workload: Qwen3.8-27B UD-Q5_K_S, llama-server `-ub 1024 -b 2048`, KV q8_0/q4_0. A 75k-token request spent ~80% of its wall time in prefill.

Kernel time profile, llama-bench `-p 2048 -d 32768 -ub 1024 -b 2048`, KV q8_0/q4_0 (rocprofv3 `--kernel-trace`; `--stats` crashes during model load with this build), 16e14ce -> 6c09cd7:

| kernel family | before: share, ms/call | after: share, ms/call |
|---|---|---|
| hipBLAS GEMM | 48.3%, 1.53 | 62.1%, 1.54 |
| gated_delta_net | 17.7%, 5.80 | 3.1%, 0.79 (prep 0.30 + scan 0.49) |
| fattn-mma | 17.1%, 16.86 | 16.9%, 12.91 |
| k_dequant_repack_f16 | 6.0%, 0.24 | 7.5%, 0.23 |
| concat | 3.6%, 1.19 | 0.8%, 0.21 |
| total kernel busy time | 56.7 s | 44.1 s |

Done:
- GATED_DELTA_NET (scalar gate, head size 64/128, >= 64 tokens): chunked delta rule with chunks of 64 tokens, all fp32 with `v_mfma_f32_16x16x4f32`. `gdn_chunk_prep` (all chunks in parallel) computes the cumulative gate, T = (I + A)^-1 by 16x16 blocks, W = T diag(beta exp(G)) K, U = T diag(beta) V and the masked Q K^T; `gdn_chunk_scan` walks the chunks with the state in registers, one block per head and group of 16-64 state columns, and loads the operands of the next chunk during the current one. Rollback snapshots (K > 1) and the fused cache write are supported. NMSE vs CPU <= 1e-9. KDA (vector gate) and smaller batches use the token-by-token kernel. Qwen3.8-27B shape, 1024 tokens: 5505 -> 752 us.
- CONCAT with a transposed source (the conv state `concat(conv_states, transpose(x))` of Qwen3.5/3.8 and Mamba): 64x64 tiles through LDS, 628 -> 141 us.
- FA MMA kernel: for a GQA ratio that is not a power of 2, ncols2 is the largest power of 2 that divides it (GQA 6: 2 instead of 8, no padded Q heads, 32 Q rows per block): +45% f16, +22% q8_0/q4_0 at nb=1024. An extra kernel instance for K/V in {q8_0, q4_0} without the generic dequantization code: +8% (D=256) and +15% (D=512) for these types.

FA op time, test-backend-ops perf, nb=1024, kv=16384 (16e14ce -> 6c09cd7): D=256 GQA 6 f16 15.30 -> 10.66 ms (27 -> 39 TFLOP/s), q8_0/q4_0 15.20 -> 11.59 ms (27 -> 36 TFLOP/s); D=256 GQA 2 q8_0/q4_0 15.88 -> 14.74 ms (35 -> 37 TFLOP/s); D=512 GQA 8 q8_0/q4_0 49.15 -> 42.71 ms (22 -> 26 TFLOP/s); f16 unchanged for power-of-2 GQA.

llama-bench, interleaved 2 rounds, 16e14ce -> 6c09cd7:

| model, KV | test | before | after | |
|---|---|---:|---:|---:|
| Qwen3.8-27B, q8_0/q4_0, `-ub 1024 -b 2048` | pp2048 | 775.8 | 961.7 | +24.0% |
| | pp2048 @ d16384 | 649.8 | 812.7 | +25.1% |
| | pp2048 @ d49152 | 491.6 | 622.8 | +26.7% |
| Gemma 4 31B, f16, `-ub 1024 -b 2048` | pp2048 / @ d16384 | 921.0 / 588.6 | 917.7 / 587.0 | -0.4% / -0.3% |
| Qwen3.8-27B, f16, default | pp512 | 681.5 | 814.2 | +19.5% |

pp2..pp16 and tg128 of all four models within -1.0..+1.5% (noise). llama-server (the production command above, port 8011), 58,593-token prompt: prefill 563 -> 710 t/s (104.0 -> 82.5 s), same output.
KLD (Qwen3.8-27B, KV q8_0/q4_0, -c 4096, 5 chunks, ub 1024) vs before: 0.000333 (base ub 512 vs ub 1024: 0.000318). The FA changes are bit-identical.

Tried without gain:
- FA MMA kernel, register prefetch of the next K tile during VKQ and of the V tile during KQ (A5): f16 +2..5%, but with in-kernel dequantization the kernel (128 VGPR + 128 AGPR at 2 waves/SIMD) spills inside the KV loop and q8_0/q4_0 got 60% slower; V-only prefetch still 14% slower. Not kept.
- FA MMA D=256 with 256 threads (1 wave/SIMD, no spills) or with 2 blocks of 4 waves per CU and nbatch_fa=32: 20-35% slower; the kernel needs 2 waves/SIMD of the same block to hide LDS and global latency.
- FA D=512 with nbatch_fa=32 and the whole row per K/V load (half the barriers): +-2%.
- Skipping the VKQ rescale when no KQ max changed (wave vote): +8% for quantized K/V, -4% for f16 (register allocation), not kept.

Remaining gaps (FA is now ~17% of long prefill, at ~36 of ~185 TFLOP/s):
- Per KV step of 64 rows the D=256 kernel issues per wave 64 MFMAs but ~830 other instructions: 128 `ds_read_u16` + 64 `v_perm` for the V^T operand, ~180 AGPR moves, 4 barriers with synchronous global loads. A V tile stored transposed in LDS (V^T rows contiguous in KV) would turn the V^T reads into `ds_read_b64`.
- Load pipelining needs a register budget that the current kernel does not have: e.g. Q in LDS, a smaller VKQ tile per wave or a kernel built for CDNA (32x32 MFMA for VKQ, P in registers). This is a new kernel, not a tuning step.
- Causal-mask tile skipping matters only at small depth (< 2% of the tiles at 16k).
- gdn_chunk_scan is latency bound (1 block of 4 waves per CU, ~24 us per chunk vs ~9 us of MFMA work); gdn_chunk_prep is latency bound in its K K^T phase.
- Decision for the user: K/V conversion to f16 per call for large batches would make q8_0/q4_0 as fast as f16 (10.7 vs 11.6 ms at 16k) but brings back the f16 copy of the whole visible K/V in the compute buffer (~850 MiB at 262k context).

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
| A3 | MoE MUL_MAT_ID prompt processing (`mmq.cu:396-405`, `ggml-cuda.cu:1942-2003,2577-2585`) | always MMQ | sorted hipBLAS path with host syncs, which also disables HIP graphs | Always use MMQ for MUL_MAT_ID on CDNA1, then tune. Done 2026-10-04 (always MMQ). |
| A4 | MMQ tile shapes (`mmq-config-cdna.cuh`, `mmq.cuh:181-185`, `mma.cuh:1401-1409`) | J up to 128, 256 threads, occupancy 2 | one config: 512 threads, J <= 64, 16x16 i8 tiles; dense ne11 > 128 goes to rocBLAS | J=96/128 configs with `v_mfma_i32_32x32x8i8`, then widen the `ggml_cuda_should_use_mmq` window. |
| A5 | FA load pipelining (`fattn-mma-f16.cuh:377-442,504-525`) | multi-stage cp.async prefetch | nstages=0, so loads and MFMA run in series | Software double buffer (global -> VGPR -> LDS), or `buffer_load ... lds`. |
| A6 | MMF dense f16/bf16 at batch 3-16 (`mmf.cu:174-175`, `mmvf.cu:847,865`) | MMF up to 16 columns | rocBLAS | Tune the MFMA MMF path (`16x16x16f16`, `16x16x8bf16`) and remove the CDNA1 exclusion. |
| A7 | Sparse-mask FA (`fattn.cu:8-151`, `fattn-mma-f16.cuh:2069-2093`) | mask compaction and gather | not available on HIP | Port with 64-bit ballot and popcount. Needs A1/A2 first. |
| A8 | Mamba-2 SSD prefill (`ssm-scan.cu:361-781,840-848`) | chunked SSD with cuBLAS batched GEMM | sequential scan | Use hipBLAS strided-batched GEMM. |
| A9 | Lightning indexer (`lightning-indexer.cu:450-511`) | nvcuda::wmma kernel | vector kernel that assumes 32 lanes | MFMA `16x16x16f16` port, wave64-aware. |
| A10 | ARGSORT on large rows (`common.cuh:114-116`, `ggml-cuda.cu:5577-5586`) | CUB segmented sort | bitonic only; rows above 16384 columns fall back to CPU | hipCUB / rocPRIM segmented radix sort. |
| A11 | 32-lane logical warps on wave64 (`softmax.cu:308,377`, `topk-moe.cu:91,115`) | full warps | half of each wave is idle | Template on `ggml_cuda_get_physical_warp_size()`. rms_norm rewritten for wave64 and topk-moe shuffles moved to DPP (2026-10-04); softmax and the other norms remain. |
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
