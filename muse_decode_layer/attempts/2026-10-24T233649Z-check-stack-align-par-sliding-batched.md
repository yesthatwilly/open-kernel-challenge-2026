# muse_decode_layer · 2026-10-24T233649Z · check · stack-align-par-sliding-batched

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T233637Z-stack-align-par-sliding-batched.md` (branch `hyp/muse_decode_layer/stack-align-par-sliding-batched`).

- **Job:** `b94e797333fb4a27` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `73b296a`

## Goal
Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 restack as git diff c9bcd4b..b85563b): moe-align-parallel-large-t-port and sliding-fused-batched-decode-port. One gates T>32 MoE work, the other sliding-window attention, so the run reads their combined effect.

## What changed, and where
Built in the main thread 2026-10-24 ~23:37Z with user-approved self-review. Worktree on master (dense-chain PDL); `git apply -3` of the two parents' changes (the sliding batch-1 restack as `git diff c9bcd4b..b85563b`, the others from their done/ patches, whose recorded bases include the dense chain), both clean, no edits. One parent gates T > 32 MoE work (`_experts` align or the side-stream dispatch), the other sliding-window attention (batch 1 or batch > 1) feeding the unchanged o GEMM, so they do not interact.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: unchanged from the two parents, both passed the judge tonight.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; queries FP8 E4M3 as the parents.
- [x] Every key attended (full 2048 window on the sliding path).
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Risks / what could fail
low
Main-thread stack before the 00:00Z reset, user-approved self-review.

## Result (after the run)
- job `b94e797333fb4a27` (check), status done, passed False, tests 13/13
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: failed** (test:failed)
- files: `muse_decode_layer/hypotheses/done/failed/stack-align-par-sliding-batched.result.json`, `muse_decode_layer/hypotheses/done/failed/stack-align-par-sliding-batched.patch`; branch `hyp/muse_decode_layer/stack-align-par-sliding-batched`
