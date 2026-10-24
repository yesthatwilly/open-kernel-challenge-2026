---
problem: muse_decode_layer
slug: stack-align-par-sliding-batched
created: 2026-10-24T233637Z
base: master
area: T>32 parallel align + sliding batch>1 fused attention
change: >
  Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night,
  applied unchanged from their done/ patches (the sliding batch-1 restack as git diff
  c9bcd4b..b85563b): moe-align-parallel-large-t-port and sliding-fused-batched-decode-port. One
  gates T>32 MoE work, the other sliding-window attention, so the run reads their combined effect.
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
claimed_at: 2026-10-24T233638Z
base_sha: 45997c629d0d2ab4312c7688acc660b9616ec817
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-align-par-sliding-batched
worktree: .worktrees/muse_decode_layer/stack-align-par-sliding-batched
mode: null
restack_of: null
stale_base: null
---

# stack-align-par-sliding-batched

## Change

Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 restack as git diff c9bcd4b..b85563b): moe-align-parallel-large-t-port and sliding-fused-batched-decode-port. One gates T>32 MoE work, the other sliding-window attention, so the run reads their combined effect.

## Expected

- `*` faster: two disjoint sub-noise gains combined

## Risk

Main-thread stack before the 00:00Z reset, user-approved self-review.
