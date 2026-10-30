# muse_decode_layer · 2026-10-30T2351Z · check · fix-swapab-off

- **Job:** `d4ec9ac0093b4221` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `3676829f` (master 3d4f06b2 + CAMPAIGN_DENSE_SWAPAB = False), 996,744 B

## Goal
Passing judge check of master + the swap-AB containment, the gate for committing it on master so post-close reruns of
master cannot hit the cb182920 sticky illegal access in the [dense-swapab] prepare race. Not chasing a score
(expected ~0.5% below master's 1.6989, since swap-AB is off).

## What changed, and where
- `CAMPAIGN_DENSE_SWAPAB = True -> False`: `_dsa_setup` never runs; qkv_gate / o / unpooler use cuBLAS (incumbent).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: forward = master's with cuBLAS GEMMs in place of the Triton swap-AB.
- [x] Nothing derived from `data` is cached across calls.
- [x] Weights: unchanged; BF16 stays BF16, NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16; KV FP8; queries at most FP8 E4M3.
- [x] Attention sees every key; no top-k, sparse, block-skip, or weight-based skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call; no output caching or replay.

## Risks / what could fail
- Containment, not root cause; other in-prepare copy checks keep the deep-copy pattern.

## Result (after the run)
PASSED, setup clean (no illegal access in the log), 1.684x [1.659, 1.710]; gate met for committing the containment on master.
