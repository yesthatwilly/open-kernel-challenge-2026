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
stage: claimed
claimed_at: 2026-10-24T233529Z
base_sha: ba5ea90a00632b3485c30650134f0a6871ea09da
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-align-par-sliding1
worktree: .worktrees/muse_decode_layer/stack-align-par-sliding1
mode: null
restack_of: null
stale_base: null
---

# stack-align-par-sliding1

## Change

Built on master (dense-chain PDL). Stack the two decode checks of 2026-10-24 night that each beat the 1.228 best, applied unchanged from their done/ patches: moe-align-parallel-large-t-port (1.2433, parallel align at T>32) and sliding-decode1-fused-restack-on-pdl (1.2374, fused sliding attention at batch 1). They touch disjoint shapes and code, so the run reads their combined effect.

## Expected

- `*` faster: T>32 align plus sliding decode_1 attention

## Risk

Main-thread stack before the 00:00Z reset, user-approved self-review.
