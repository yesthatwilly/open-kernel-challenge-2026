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
stage: implemented
claimed_at: 2026-10-24T233638Z
base_sha: 45997c629d0d2ab4312c7688acc660b9616ec817
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-align-par-sliding-batched
worktree: .worktrees/muse_decode_layer/stack-align-par-sliding-batched
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233649Z
impl_commit: 73b296a8edd9b98db5a21db5d4fa73f8b8617591
submission_sha256: 4bb95888bbec38bfabc90a40fa18a7a42348d3c1edb7f18fe85469c9c00fcd55
norm_sha256: 616c9fd296da0615fe139ad0ae76c96b3a379f65a9262dda83ea270f29d9eb40
submission_blob: f9fc0379411724907de05cd9cdb046eeaf5382e7
lint_warnings: 30
attempt_file: 2026-10-24T233649Z-check-stack-align-par-sliding-batched.md
prepared_at: 2026-10-24T233649Z
---

# stack-align-par-sliding-batched

## Change

Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 restack as git diff c9bcd4b..b85563b): moe-align-parallel-large-t-port and sliding-fused-batched-decode-port. One gates T>32 MoE work, the other sliding-window attention, so the run reads their combined effect.

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
