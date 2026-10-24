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
stage: implemented
claimed_at: 2026-10-24T233637Z
base_sha: f7c9823e181e8f7f0fbb2200b2c38b4ad7be6085
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-sidestream-sliding1
worktree: .worktrees/muse_decode_layer/stack-sidestream-sliding1
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233648Z
impl_commit: 86326ffcfad9f120326d7df4309a6b373ca52ad4
submission_sha256: 74132f4cca02bcd11713f11921cf212c5d391e975352a8b8712e19277c922ec9
norm_sha256: 3955572fa4dc489b250ad7f18329ced07f068aee16c4185da83caafc015ede01
submission_blob: ad68c3962e7f16fbc2631e288222f7d026192eb0
lint_warnings: 18
---

# stack-sidestream-sliding1

## Change

Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 restack as git diff c9bcd4b..b85563b): moe-dispatch-prep-on-side-stream and sliding-decode1-fused-restack-on-pdl. One gates T>32 MoE work, the other sliding-window attention, so the run reads their combined effect.

## Expected

- `*` faster: two disjoint sub-noise gains combined

## Risk

Main-thread stack before the 00:00Z reset, user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:37Z with user-approved self-review. Worktree on master (dense-chain PDL); `git apply -3` of the two parents' changes (the sliding batch-1 restack as `git diff c9bcd4b..b85563b`, the others from their done/ patches, whose recorded bases include the dense chain), both clean, no edits. One parent gates T > 32 MoE work (`_experts` align or the side-stream dispatch), the other sliding-window attention (batch 1 or batch > 1) feeding the unchanged o GEMM, so they do not interact.

## Rules check

- [x] CUDA-graph capturable: unchanged from the two parents, both passed the judge tonight.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; queries FP8 E4M3 as the parents.
- [x] Every key attended (full 2048 window on the sliding path).
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.
