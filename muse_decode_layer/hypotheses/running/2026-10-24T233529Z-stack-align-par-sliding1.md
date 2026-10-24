---
problem: muse_decode_layer
slug: stack-align-par-sliding1
created: 2026-10-24T233529Z
base: master
area: MoE align (T>32) + sliding batch-1 attention
change: >
  Built on master (dense-chain PDL). Stack the two decode checks of 2026-10-24 night that each
  beat the 1.228 best, applied unchanged from their done/ patches: moe-align-parallel-large-t-port
  (1.2433, parallel align at T>32) and sliding-decode1-fused-restack-on-pdl (1.2374, fused sliding
  attention at batch 1). They touch disjoint shapes and code, so the run reads their combined
  effect.
expected:
  - shapes: "*"
    direction: faster
    why: T>32 align plus sliding decode_1 attention
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: running
stage: dispatched
claimed_at: 2026-10-24T233529Z
base_sha: ba5ea90a00632b3485c30650134f0a6871ea09da
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-align-par-sliding1
worktree: .worktrees/muse_decode_layer/stack-align-par-sliding1
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233559Z
impl_commit: 17744154cf317436e56f9b86e69389853249ebf0
submission_sha256: cddab79bb8be991e2a3292f88bbe122af05bc9b5440e6229aa82bc9ca530d1ce
norm_sha256: 103db1b5154243c5876497fe2a230f7f152a85e9c1fc6d6d01bd668f4f624235
submission_blob: 17801ddbab54e8f62745dee42d9259604e89449f
lint_warnings: 28
attempt_file: 2026-10-24T233559Z-check-stack-align-par-sliding1.md
prepared_at: 2026-10-24T233559Z
job: d41952184b714e37
dispatched_at: 2026-10-24T233616Z
---

# stack-align-par-sliding1

## Change

Built on master (dense-chain PDL). Stack the two decode checks of 2026-10-24 night that each beat the 1.228 best, applied unchanged from their done/ patches: moe-align-parallel-large-t-port (1.2433, parallel align at T>32) and sliding-decode1-fused-restack-on-pdl (1.2374, fused sliding attention at batch 1). They touch disjoint shapes and code, so the run reads their combined effect.

## Expected

- `*` faster: T>32 align plus sliding decode_1 attention

## Risk

Main-thread stack before the 00:00Z reset, user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:37Z with user-approved self-review. Worktree on master (dense-chain PDL). Applied moe-align-parallel-large-t-port's done/ patch, then sliding-decode1-fused-restack-on-pdl's change as `git diff c9bcd4b..b85563b` (its harvested .patch was cut against a stale recorded base and carries the dense chain; see DECISIONS). Both clean, no edits. Disjoint: the align port gates _experts at T >= 33; the sliding path gates attention at window != 0, batch == 1, q_len == 1 and feeds the unchanged o GEMM.

## Rules check

- [x] CUDA-graph capturable: unchanged from the two parents, both passed the judge tonight.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; queries FP8 E4M3 as the parent.
- [x] Every key attended (full 2048 window on the sliding path).
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.
