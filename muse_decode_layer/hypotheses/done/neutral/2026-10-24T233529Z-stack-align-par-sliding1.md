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
status: done
stage: harvested
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
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.2444214530258184
speedup_ci: [1.2257551312304311, 1.2630877748212055]
harvested_at: 2026-10-25T000219Z
best_at_harvest: 1.228328123558382
promotion: null
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

## Result

- job `d41952184b714e37` (check), status done, passed True, tests 13/13
- speedup 1.2444x, 95% CI [1.2258, 1.2631], score 18177.362616980445 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.228328123558382, "ci": [1.2099032017050062, 1.2467530454117577], "job": "81ee6714b7614a99"}; base result: muse_decode_layer/hypotheses/done/gain/small-t-launch-minimal-dense-chain-pdl.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/stack-align-par-sliding1.result.json`, `muse_decode_layer/hypotheses/done/neutral/stack-align-par-sliding1.patch`; branch `hyp/muse_decode_layer/stack-align-par-sliding1`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60 | 46 | 132.74 | 128.18 | 1.039 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60 | 16 | 142.57 | 144.24 | 0.9912 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic | 46 | 217.06 | 215.6 | 1.003 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic | 16 | 347.51 | 359.52 | 0.9957 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic | 46 | 246.99 | 240.82 | 1.001 | flat |
| batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic | 16 | 534.98 | 516.15 | 1.013 | flat |
| batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8 | 46 | 186.48 | 187.44 | 1.0039 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8 | 16 | 195.92 | 203.13 | 0.996 | flat |
| batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 447.55 | 423.32 | 1.0229 | flat |
| batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 599.37 | 579.39 | 1.0151 | flat |
| batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 626.79 | 632.75 | 1.0307 | gained |
| batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 910.91 | 904.91 | 1.014 | flat |

gained on sliding decode_1, verify_32. Expected hit: 1/1. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-align-par-sliding1`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

