---
problem: muse_prefill_layer
slug: stack-tonight-five-prefill
created: 2026-10-24T233200Z
base: master
area: router GEMM + MoE align (both paths) + w2 zero-store + Marlin silu-mul
change: >
  Built on master. stack-tonight-four-prefill (router-hilo-one-gemm-n512,
  moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn,
  moe-align-local-routes-parallel-port) plus silu-mul-triton-local-rows-marlin-turn (sorted
  local-rows silu-mul on the Marlin path, prepare() self-checked), each applied unchanged. On the
  Marlin path the silu-mul walks the parallel align's sorted_ids, which hold local routes only.
  prepare() runs _align_par_setup, then _silu_sorted_setup, then _check_experts_e2e.
expected:
  - shapes: "*"
    direction: faster
    why: the four-stack plus silu-mul on local rows at turn_1/turn_4
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: done
stage: harvested
claimed_at: 2026-10-24T233200Z
base_sha: 3b2b91dc85a1be9e6563f3f0987fd41e4d4c0841
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/stack-tonight-five-prefill
worktree: .worktrees/muse_prefill_layer/stack-tonight-five-prefill
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233216Z
impl_commit: b9ad6f2d2af0db453232b54cc0dbec7d034c51d9
submission_sha256: d42aba8c748c992af59f39c2879c3adda94cb677341926f1f59fb8d69c985dc5
norm_sha256: 4340f5135c3e58122409931d064ba571c954db3fd4f36921dd730a20c3a3d04b
submission_blob: db8a3344398bca79ea5cafeb580f701c055fd475
lint_warnings: 15
attempt_file: 2026-10-24T233216Z-check-stack-tonight-five-prefill.md
prepared_at: 2026-10-24T233216Z
job: c4d14d62f11e4aaf
dispatched_at: 2026-10-24T233227Z
promotion: submitted
promotion_job: 502f2eab95164781
promotion_at: 2026-10-25T000315Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.0705080897445196
speedup_ci: [1.0544504683983518, 1.0865657110906872]
harvested_at: 2026-10-25T000224Z
best_at_harvest: 1.0591912383762452
hand_landed: true
promotion_commit: a81d19c4982ee781eb5faf78329a9ddc63e58206
---

# stack-tonight-five-prefill

## Change

Built on master. stack-tonight-four-prefill (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn, moe-align-local-routes-parallel-port) plus silu-mul-triton-local-rows-marlin-turn (sorted local-rows silu-mul on the Marlin path, prepare() self-checked), each applied unchanged. On the Marlin path the silu-mul walks the parallel align's sorted_ids, which hold local routes only. prepare() runs _align_par_setup, then _silu_sorted_setup, then _check_experts_e2e.

## Expected

- `*` faster: the four-stack plus silu-mul on local rows at turn_1/turn_4

## Risk

Main-thread stack before the 00:00Z reset, user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:33Z with user-approved self-review. Worktree on master; `git apply -3` of the three done/neutral patches, moe-align-local-routes-parallel-port's implementation diff and silu-mul-triton-local-rows-marlin-turn's implementation diff. Two conflicts, both "both sides added" (the two prepare() setup calls before _check_experts_e2e, and the two new code sections before _experts_marlin), resolved by keeping both, align first; no line edited. On the Marlin path the sorted silu-mul consumes the parallel align's sorted_ids/n_padded (local routes only) and still masks on local_ids >= 0.

## Rules check

- [x] CUDA-graph capturable: unchanged from the five parents.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged; router hi+lo split as the rules allow.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Result

- job `c4d14d62f11e4aaf` (check), status done, passed True, tests 5/5
- speedup 1.0705x, 95% CI [1.0545, 1.0866], score 361887.87708503474 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json`, `muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.patch`; branch `hyp/muse_prefill_layer/stack-tonight-five-prefill`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1079.54 | 1101.29 | 1.0068 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1419.09 | 1529.55 | 1.0031 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3334.9 | 3318.71 | 1.012 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4937.98 | 4849.27 | 1.0077 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 15386.32 | 14705.32 | 0.9988 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14275.3 | 14264.64 | 1.0137 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18165.94 | 17746.69 | 1.0196 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 22332.92 | 21733.75 | 1.0189 | flat |

no shape moved outside noise. Expected hit: 0/1. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-tonight-five-prefill`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

