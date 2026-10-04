# MI100 (gfx908) optimization-parity backlog

This fork builds the ROCm backend for AMD Instinct MI100 (gfx908, CDNA1) only.
This file lists kernel paths where a modern NVIDIA GPU gets a faster path than gfx908.

Status: backlog only. Work starts after the pruned tree passes the verification gate (tag `mi100-pruned-verified`).
Each item also needs a profile (rocprofv3) that shows the kernel is below ~60% of its roofline and takes at least 15% of decode or verify time.

Line numbers refer to the upstream base `83209c3d2` and can drift after the gfx908 cleanup.

MI100 peak rates: f16 MFMA ~185 TFLOPS, i8 MFMA ~185 TOPS, bf16 ~92 TFLOPS, f32 MFMA ~46 TFLOPS, 1.23 TB/s HBM2, 120 CUs, 64 KB LDS per CU, wave64.
Because i8 and f16 run at the same rate, MMQ helps by saving dequantization work and memory traffic, not by a higher peak rate.

## Priority for the srv2 workload

Workload: dense Qwen3.8-27B UD-Q5_K_S, single user, decode plus MTP speculative verify at 2-16 tokens.

- [ ] 1. Flash attention for 2-16 query rows with GQA (item A1). This is the verify path.
- [ ] 2. Q5_K matmul at verify widths 2-16 (items A12 and A4, small J). Also try load-time weight repacking.
- [ ] 3. MMVQ decode tuning at ncols=1 (item A12). Target: more than 85% of 1.23 TB/s.
- [ ] 4. Gated DeltaNet state update, if the model has those layers.
- [ ] 5. Use the physical warp size (64) in softmax and topk-moe (item A11).
- [ ] 6. Flash attention load pipelining for 128k+ context verify (item A5).

Deferred, because this model does not use them: A2, A3, A7, A8, A9, A10, A13.

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
| A13 | Multi-GPU | NCCL on by default | RCCL off by default; internal allreduce | Turn on `GGML_HIP_RCCL` for multi-MI100 nodes. |

## Not gaps

- Native FP4: MI100 has no FP4 hardware. MXFP4 and NVFP4 already use the i8 MFMA path.
- Quantized KV: both vendors convert K/V to f16 for the tile and MMA kernels.
- Already work on HIP: conv2d/conv3d MFMA, rms_norm/rope/GLU/topk-moe fusions, HIP graphs, gated delta net, top-k.
- WMMA flash attention (`GGML_HIP_ROCWMMA_FATTN`): removed upstream. gfx908 has no WMMA.
