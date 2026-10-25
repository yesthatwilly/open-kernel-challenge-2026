---
problem: muse_prefill_layer
slug: stack-tonight-six-prefill
created: 2026-10-24T233245Z
base: master
area: router GEMM + MoE align (both paths, parallel) + w2 zero-store + Marlin silu-mul
change: >
  Built on master. stack-tonight-five-prefill plus align-local-parallel-first-turn (the parallel
  local-routes align on the Triton path too, self-checked at T=16384, BLOCK_T 256), each applied
  unchanged. With the parallel align on at first_turn, moe-align-32-experts-first-turn's
  moe_align_block_size(LOCAL_EXPERTS) call becomes the fallback branch only.
expected:
  - shapes: "*"
    direction: faster
    why: the five-stack plus no remote token tiles at first_turn
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: done
stage: harvested
claimed_at: 2026-10-24T233245Z
base_sha: c1dfb14c99867d3818e9e99c82fd7d1fa4153ae3
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/stack-tonight-six-prefill
worktree: .worktrees/muse_prefill_layer/stack-tonight-six-prefill
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233306Z
impl_commit: 9ac725846b41a5ef071a020ac68dec3ddebf2835
submission_sha256: 8b0585c9064b33f4be5d7811f94773a1b141ec0e7dac414fa831ba93a4288388
norm_sha256: 3d352ddfccf4b76aa072a333b7fe425d93ef144d4aa53497ec3316abb5b55d98
submission_blob: 6b59d3f29bc81303fae11ff2d74470087b55032c
lint_warnings: 15
attempt_file: 2026-10-24T233306Z-check-stack-tonight-six-prefill.md
prepared_at: 2026-10-24T233306Z
job: 5cb6e68a86ec48f0
dispatched_at: 2026-10-24T233314Z
promotion: null
promotion_job: 3358a99eb27a405a
promotion_at: 2026-10-24T234001Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.0641845512352945
speedup_ci: [1.048221782966765, 1.0801473195038238]
harvested_at: 2026-10-25T000224Z
best_at_harvest: 1.0591912383762452
---

# stack-tonight-six-prefill

## Change

Built on master. stack-tonight-five-prefill plus align-local-parallel-first-turn (the parallel local-routes align on the Triton path too, self-checked at T=16384, BLOCK_T 256), each applied unchanged. With the parallel align on at first_turn, moe-align-32-experts-first-turn's moe_align_block_size(LOCAL_EXPERTS) call becomes the fallback branch only.

## Expected

- `*` faster: the five-stack plus no remote token tiles at first_turn

## Risk

Main-thread stack before the 00:00Z reset, user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:35Z with user-approved self-review: stack-tonight-five-prefill's submission.py, then a 3-way apply of align-local-parallel-first-turn's implementation diff. One real conflict, in _experts_triton's align call (moe-align-32 vs the parallel align), resolved as the spec says: the parallel align when state["align_par_ok"], else moe_align_block_size(local_ids, BLOCK_T, LOCAL_EXPERTS). The extended self-check (T=16384 at BLOCK_T 256) came over cleanly.

## Rules check

- [x] CUDA-graph capturable: unchanged from the parents; prepare()-set gates, shape-only allocations.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged; router hi+lo split as the rules allow.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Result

- job `5cb6e68a86ec48f0` (check), status done, passed True, tests 5/5
- speedup 1.0642x, 95% CI [1.0482, 1.0801], score 364038.26718806016 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/stack-tonight-six-prefill.result.json`, `muse_prefill_layer/hypotheses/done/neutral/stack-tonight-six-prefill.patch`; branch `hyp/muse_prefill_layer/stack-tonight-six-prefill`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1079.54 | 1078.73 | 1.0091 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1419.09 | 1534.39 | 1.0033 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3334.9 | 3377.96 | 1.0129 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4937.98 | 4980.61 | 1.0067 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 15386.32 | 15368.44 | 0.98 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14275.3 | 14429.44 | 0.9931 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18165.94 | 19264.52 | 1.0069 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 22332.92 | 22412.03 | 1.0008 | flat |

no shape moved outside noise. Expected hit: 0/1. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-tonight-six-prefill`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

