---
problem: muse_decode_layer
slug: stack-sidestream-sliding-batched
created: 2026-10-24T233638Z
base: master
area: T>32 MoE dispatch on the side stream + sliding batch>1 fused attention
change: >
  Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night,
  applied unchanged from their done/ patches (the sliding batch-1 restack as git diff
  c9bcd4b..b85563b): moe-dispatch-prep-on-side-stream and sliding-fused-batched-decode-port. One
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
stage: implemented
claimed_at: 2026-10-24T233638Z
base_sha: fc2c96a83dc62731f5a5ebd4f4791da8fdeb14ed
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-sidestream-sliding-batched
worktree: .worktrees/muse_decode_layer/stack-sidestream-sliding-batched
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233649Z
impl_commit: 25d71019cfeb117f0e1c0f40fda4ac46d4ea1800
submission_sha256: e23a3d22a0e73566331aee6c9cc6efe861d22e889fb609b52f614c784e84b3f5
norm_sha256: 1a60c51db7a15d6deb5b1f6895c0ac2cf257c888d59bdc52b03462e5fa078383
submission_blob: ad7bee5bbcdcf3ad7d2f24aa5ab58e7ed67db773
lint_warnings: 20
---

# stack-sidestream-sliding-batched

## Change

Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 restack as git diff c9bcd4b..b85563b): moe-dispatch-prep-on-side-stream and sliding-fused-batched-decode-port. One gates T>32 MoE work, the other sliding-window attention, so the run reads their combined effect.

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
