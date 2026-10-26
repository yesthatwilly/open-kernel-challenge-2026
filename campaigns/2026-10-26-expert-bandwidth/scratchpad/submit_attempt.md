# muse_decode_layer · STAMP · submit · campaign-wghyb-pw0-decode

Campaign `2026-10-26-expert-bandwidth`: scored run of the exact bytes of check f59cb301895240e6.

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.edoff-27ef83df.py` = `muse_decode_layer/submission.py` at git `27ef83df`
  (sha256 9aad293f04305aca...; okc content hash 613b6c2d...).
- **Base:** board 1.4956 (dc51d8e8f58e4cb7).

## Why
Check f59cb301895240e6 passed 13/13 at **1.5347** against the board's 1.4956.
- The hot/cold hybrid's wgmma-hot half at T=128 / 256 (H=-16; layer 376.5 vs master 389.0 us and 479.2 vs 558.2 us).
- The tie-break revert (8b098599): v4 at T=2..32 / 128 / 256, v1 at T=1.
- Engine D is in the bytes but off everywhere: it lost every GEMM selector, and every cell keeps its incumbent.
- Rows vs the board run dc51d8e8, each against its own SGLang time:
  - v16 x1.075 / x1.034, v32 x1.204 / x1.118;
  - T <= 32 rows x0.958-x1.003 (noise: v1 full's SGLang time moved 7% too).
- Decode rank 347 -> about 318 of 740 at today's board.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: engine, hybrid and GEMM choice by T and prepare-set flags only; no sync in kernel_fn.
- [x] Nothing derived from `data` is cached across calls.
- [x] Weights: Engine A's lossless records, the wgmma bank's lossless retile and Engine D's (unused) permutation
      records; NVFP4 stays NVFP4, bf16 stays bf16.
- [x] Activations BF16; KV FP8; queries at most FP8 E4M3.
- [x] Attention sees every key.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work inside the timed call; no output caching or replay.

## Result (after the run)
