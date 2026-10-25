# muse_decode_layer · 2026-10-24T233559Z · check · stack-align-par-sliding1

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T233529Z-stack-align-par-sliding1.md` (branch `hyp/muse_decode_layer/stack-align-par-sliding1`).

- **Job:** `d41952184b714e37` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `1774415`

## Goal
Built on master (dense-chain PDL). Stack the two decode checks of 2026-10-24 night that each beat the 1.228 best, applied unchanged from their done/ patches: moe-align-parallel-large-t-port (1.2433, parallel align at T>32) and sliding-decode1-fused-restack-on-pdl (1.2374, fused sliding attention at batch 1). They touch disjoint shapes and code, so the run reads their combined effect.

## What changed, and where
Built in the main thread 2026-10-24 ~23:37Z with user-approved self-review. Worktree on master (dense-chain PDL). Applied moe-align-parallel-large-t-port's done/ patch, then sliding-decode1-fused-restack-on-pdl's change as `git diff c9bcd4b..b85563b` (its harvested .patch was cut against a stale recorded base and carries the dense chain; see DECISIONS). Both clean, no edits. Disjoint: the align port gates _experts at T >= 33; the sliding path gates attention at window != 0, batch == 1, q_len == 1 and feeds the unchanged o GEMM.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: unchanged from the two parents, both passed the judge tonight.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; queries FP8 E4M3 as the parent.
- [x] Every key attended (full 2048 window on the sliding path).
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Risks / what could fail
low
Main-thread stack before the 00:00Z reset, user-approved self-review.

## Result (after the run)
- job `d41952184b714e37` (check), status done, passed True, tests 13/13
- speedup 1.2444x, 95% CI [1.2258, 1.2631], score 18177.362616980445 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/stack-align-par-sliding1.result.json`, `muse_decode_layer/hypotheses/done/neutral/stack-align-par-sliding1.patch`; branch `hyp/muse_decode_layer/stack-align-par-sliding1`
