---
problem: muse_decode_layer
slug: stack-sidestream-sliding1
created: 2026-10-24T233637Z
base: master
area: T>32 MoE dispatch on the side stream + sliding batch-1 fused attention
change: >
  Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night,
  applied unchanged from their done/ patches (the sliding batch-1 restack as git diff
  c9bcd4b..b85563b): moe-dispatch-prep-on-side-stream and sliding-decode1-fused-restack-on-pdl.
  One gates T>32 MoE work, the other sliding-window attention, so the run reads their combined
  effect.
expected:
  - shapes: "*"
    direction: faster
    why: two disjoint sub-noise gains combined
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: running
stage: claimed
claimed_at: 2026-10-24T233637Z
base_sha: f7c9823e181e8f7f0fbb2200b2c38b4ad7be6085
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-sidestream-sliding1
worktree: .worktrees/muse_decode_layer/stack-sidestream-sliding1
mode: null
restack_of: null
stale_base: null
---

# stack-sidestream-sliding1

## Change

Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 restack as git diff c9bcd4b..b85563b): moe-dispatch-prep-on-side-stream and sliding-decode1-fused-restack-on-pdl. One gates T>32 MoE work, the other sliding-window attention, so the run reads their combined effect.

## Expected

- `*` faster: two disjoint sub-noise gains combined

## Risk

Main-thread stack before the 00:00Z reset, user-approved self-review.
