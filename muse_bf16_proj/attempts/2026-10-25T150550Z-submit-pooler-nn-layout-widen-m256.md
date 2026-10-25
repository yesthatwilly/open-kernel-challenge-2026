# muse_bf16_proj · 2026-10-25T150550Z · submit · pooler-nn-layout-widen-m256

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T145403Z-pooler-nn-layout-widen-m256.md` (branch `hyp/muse_bf16_proj/pooler-nn-layout-widen-m256`).

- **Job:** `85c91578cc184287` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `d0b968c8`

## Goal
Built on master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, submit ee4f89f2 1.2491x). Master already builds state['pooler_t'] = W_pooler.t().contiguous() ([8192, 4096] bf16, lossless repack) in prepare() and routes pooler/unpooler at NN_GATED_MIN_M=65 <= cfg m <= NN_GATED_MAX_M=128 to torch.mm(x, state[layer + '_t']) in kernel_fn. Widen that gate for the POOLER ONLY to cfg m <= 256: add a new constant NN_POOLER_MAX_M = 256 (keep NN_GATED_MAX_M = 128 for the unpooler) and change the kernel_fn condition so that layer == 'pooler' uses NN_POOLER_MAX_M as its upper bound while layer == 'unpooler' keeps 128. Mirror the same per-layer bound in _ab_component (so the in-process A/B self-check attributes a pooler m in 129..256 cell to 'nn_ok' and can switch it off) and, if cheap, add m=256 pooler to the self-check's captured cells. No new weight copy (pooler_t already exists), no new kernel, no other layer or m changes: qkv_gate and o stay TN at every m (the probe LOST qkv_gate m=256, 0.943x), unpooler m=256 stays TN (probe 1.012x, noise), and every m <= 128 path is byte-identical to master. The gate branches on cfg layer and m only. Evidence: bf16-m128-nn-weight-layout-probe (job 2dc5dce0) ran pooler m=256 through NN at 29.73 us row vs SGLang 30.74 (paired 1.034x, nvjet 128x64_64x8_1x2 NNT 29.5 us); the TN incumbent on the same row read 0.986x (restack 3-cell, nvjet TNT 30.4 us), 0.999x (router-cluster-s2, 28.5 us) and 0.9998x (pooler-unpooler-m128-nn-layout-gated) in three other runs. The review of pooler-unpooler-m128-nn-layout-gated named this widening as its follow-up. The attempt file records the pooler verify_32 nvjet kernel name (must carry the NN suffix, e.g. _NNT) and us from the profile; if the name is still TNT the run is void.

## What changed, and where
Base: worktree base b7d110b7; master e21fdc47 differs only by the claim commit (submission.py identical), so no merge was needed. Edited only muse_bf16_proj/submission.py (+15/-4).
- Added NN_POOLER_MAX_M = 256 (NN_GATED_MAX_M stays 128 for the unpooler).
- kernel_fn: NN branch is now cfg n == pooler_t/unpooler_t width AND ((nn_ok and 65 <= m <= 128) OR (layer == pooler and nn_p256_ok and 128 < m <= 256)). Every m <= 128 path is unchanged.
- Review fix (open question 1, own flag): new state flag nn_p256_ok (default True), inherited by state B, included in _ab_flags, switched off in the second-state-build-failure path, and printed in FINAL FLAGS (nn_ok and nn_p256_ok). _ab_component maps pooler 129..256 to nn_p256_ok, so a miss at the new self-check cell (pooler, 4096, 8192, 256 added to AB_CASES, captured on two states and replayed A,B,A,B) turns off only the m=256 widening and never nn_ok.
- Review: log final flags (printed: "nn_ok=... nn_p256_ok=..."); the attempt must record nn_ok=True, nn_p256_ok=True and the pooler verify_32 kernel name (must be *_NNT) and us. Confounder (dispatch impl vs torch.mm) and the ordering with bf16-verify32-cutlass-coop-256x128-splitk-stage2 are judging notes, not build items; not credited to layout alone.
- Deviation: none.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: gate reads cfg and state flags only; torch.mm, no sync/alloc on values.
- [x] Nothing derived from data cached: no new state besides a constant bool.
- [x] Weights: reuses existing lossless pooler_t; no new copy.
- [x] Activations BF16 unchanged; KV/queries n/a (projection problem).
- [x] Attention: n/a, GEMM only.
- [x] No timing patches, harness access, subprocesses or network added.
- [x] All work inside the timed call; no output caching.
- [x] No device pointers or tensors in module globals added; flag lives in state.

okc_lint: "OK with warnings: muse_bf16_proj/submission.py" (only pre-existing host-sync/dtype warnings in prepare-time code); py_compile clean; __pycache__ removed.

## Risks / what could fail
low
Expected headline effect is small (verify_32 is the weakest load at 1.075x; ~1 us of 94.5 us there is ~+0.2% geomean), so judge on the paired pooler verify_32 ratio, not the headline. Risk: the probe was one run on an older master; cuBLAS NN may pick the same 128x64 tile at m=256 and the 1.034x could be partly drift. Downside is bounded to one row (worst case ~0.97x, the qkv m=256 NN loss magnitude). Shape coverage: the test m=130 unpooler is unaffected (unpooler bound stays 128); no pooler test sits in 129..256, so nearby-shape behaviour equals cuBLAS NN, the same math. Overlap: queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 also targets pooler m=256; if both ever stack, branch order decides, so this keeps its own gate constant NN_POOLER_MAX_M.

## Result (after the run)
- job `85c91578cc184287` (submit), status done, passed True, tests 9/9
- speedup 1.2394x, 95% CI [1.2208, 1.2580], score 3897.794841609372 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2490645809967658, "ci": [1.2303286122818142, 1.267800549711717], "job": "ee4f89f2f62c42d3"}; base result: muse_bf16_proj/hypotheses/done/neutral/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/pooler-nn-layout-widen-m256.result.json`, `muse_bf16_proj/hypotheses/done/neutral/pooler-nn-layout-widen-m256.patch`; branch `hyp/muse_bf16_proj/pooler-nn-layout-widen-m256`
