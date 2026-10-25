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
status: done
stage: harvested
claimed_at: 2026-10-24T233638Z
base_sha: fc2c96a83dc62731f5a5ebd4f4791da8fdeb14ed
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-sidestream-sliding-batched
worktree: .worktrees/muse_decode_layer/stack-sidestream-sliding-batched
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233649Z
impl_commit: 25d71019cfeb117f0e1c0f40fda4ac46d4ea1800
submission_sha256: e23a3d22a0e73566331aee6c9cc6efe861d22e889fb609b52f614c784e84b3f5
norm_sha256: 1a60c51db7a15d6deb5b1f6895c0ac2cf257c888d59bdc52b03462e5fa078383
submission_blob: ad7bee5bbcdcf3ad7d2f24aa5ab58e7ed67db773
lint_warnings: 20
attempt_file: 2026-10-24T233649Z-check-stack-sidestream-sliding-batched.md
prepared_at: 2026-10-24T233649Z
job: 2193fc0dc9be4b3c
dispatched_at: 2026-10-24T233707Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2442805027115935
speedup_ci: [1.2256162951709195, 1.2629447102522673]
harvested_at: 2026-10-25T000220Z
best_at_harvest: 1.228328123558382
promotion: null
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

## Result

- job `2193fc0dc9be4b3c` (check), status done, passed True, tests 13/13
- speedup 1.2443x, 95% CI [1.2256, 1.2629], score 18179.421722597755 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/stack-sidestream-sliding-batched.result.json`, `muse_decode_layer/hypotheses/done/neutral/stack-sidestream-sliding-batched.patch`; branch `hyp/muse_decode_layer/stack-sidestream-sliding-batched`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 132.74 | 133.97 | 0.9951 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.57 | 143.58 | 1.0038 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 217.06 | 195.39 | 1.0551 | gained |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 347.51 | 357.61 | 0.9992 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 246.99 | 236.82 | 1.0411 | gained |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 534.98 | 520.45 | 1.0142 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 186.48 | 188.9 | 0.9925 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 195.92 | 198.4 | 0.9936 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 447.55 | 442.33 | 1.0154 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 599.37 | 588.03 | 1.0193 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 626.79 | 615.65 | 1.0169 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 910.91 | 900.31 | 0.9946 | flat |

gained on sliding decode_16, decode_32. Expected hit: 1/1. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-sidestream-sliding-batched`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

