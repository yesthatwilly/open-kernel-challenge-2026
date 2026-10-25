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
status: done
stage: harvested
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
job: b94e797333fb4a27
dispatched_at: 2026-10-24T233707Z
outcome: failed
verdict_detail: "test:failed"
speedup: null
speedup_ci: null
harvested_at: 2026-10-25T000219Z
best_at_harvest: 1.228328123558382
promotion: null
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

## Result

- job `b94e797333fb4a27` (check), status done, passed False, tests 13/13
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: failed** (test:failed)
- files: `muse_decode_layer/hypotheses/done/failed/stack-align-par-sliding-batched.result.json`, `muse_decode_layer/hypotheses/done/failed/stack-align-par-sliding-batched.patch`; branch `hyp/muse_decode_layer/stack-align-par-sliding-batched`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 132.74 | 133.41 | 0.9952 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.57 | 143.94 | 0.9984 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 217.06 | 190.53 | 1.0524 | gained |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 347.51 | 347.97 | 0.9923 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 246.99 | 221.38 | 1.0476 | gained |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 534.98 | 506.69 | 1.0169 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 186.48 | 196.73 | 0.9871 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 195.92 | 0.0 | None | lost |

gained on sliding decode_16, decode_32; lost on full verify_1. Expected hit: 0/1. Noise ±1.5%.

## Follow-ups

- none

