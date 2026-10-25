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
status: done
stage: harvested
claimed_at: 2026-10-24T233637Z
base_sha: f7c9823e181e8f7f0fbb2200b2c38b4ad7be6085
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-sidestream-sliding1
worktree: .worktrees/muse_decode_layer/stack-sidestream-sliding1
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233648Z
impl_commit: 86326ffcfad9f120326d7df4309a6b373ca52ad4
submission_sha256: 74132f4cca02bcd11713f11921cf212c5d391e975352a8b8712e19277c922ec9
norm_sha256: 3955572fa4dc489b250ad7f18329ced07f068aee16c4185da83caafc015ede01
submission_blob: ad68c3962e7f16fbc2631e288222f7d026192eb0
lint_warnings: 18
attempt_file: 2026-10-24T233648Z-check-stack-sidestream-sliding1.md
prepared_at: 2026-10-24T233648Z
job: 158fed699fdb4d0e
dispatched_at: 2026-10-24T233707Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2351972236258055
speedup_ci: [1.2166692652714184, 1.2537251819801924]
harvested_at: 2026-10-25T000220Z
best_at_harvest: 1.228328123558382
promotion: null
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

## Result

- job `158fed699fdb4d0e` (check), status done, passed True, tests 13/13
- speedup 1.2352x, 95% CI [1.2167, 1.2537], score 18313.107872441804 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/stack-sidestream-sliding1.result.json`, `muse_decode_layer/hypotheses/done/neutral/stack-sidestream-sliding1.patch`; branch `hyp/muse_decode_layer/stack-sidestream-sliding1`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 132.74 | 128.35 | 1.037 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.57 | 142.5 | 0.9968 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 217.06 | 206.3 | 1.0056 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 347.51 | 347.63 | 0.9972 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 246.99 | 236.17 | 1.0052 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 534.98 | 525.52 | 1.0115 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 186.48 | 192.18 | 0.9923 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 195.92 | 199.21 | 0.9855 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 447.55 | 443.86 | 1.0074 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 599.37 | 607.68 | 0.9886 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 626.79 | 656.99 | 1.005 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 910.91 | 928.5 | 1.0059 | flat |

gained on sliding decode_1. Expected hit: 1/1. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-sidestream-sliding1`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

