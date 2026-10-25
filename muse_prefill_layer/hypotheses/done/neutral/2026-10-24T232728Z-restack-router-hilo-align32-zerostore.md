---
problem: muse_prefill_layer
slug: restack-router-hilo-align32-zerostore
created: 2026-10-24T232728Z
base: master
area: router GEMM + MoE align (T>4096) + w2 zero-store (T>4096)
change: >
  Built on master. Stack the three shape-disjoint prefill checks of 2026-10-24 23:15Z, each
  applied unchanged from its done/ patch: router-hilo-one-gemm-n512 (one N=512 GEMM for the router
  hi|lo halves, f4e3c5f4c306427b 1.0607), moe-align-32-experts-first-turn (moe_align_block_size
  over LOCAL_EXPERTS in _experts_triton, b5bf94b463d34d21 1.0604) and
  w2-nonlocal-zero-store-off-first-turn (prepare() probe turns the w2 zero-store off when the
  weighted sum skips remote slots, 9b731599dfa24278 1.0609). Each alone is within noise of master
  (1.059); together the router change touches every shape and the other two first_turn only, so
  the stack reads their combined effect.
expected:
  - shapes: "*"
    direction: faster
    why: "sum of three sub-noise gains: router one GEMM at every T, smaller align capacity and no zero-stores at first_turn"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: done
stage: harvested
claimed_at: 2026-10-24T232737Z
base_sha: 40aa18e2a31c08252c27c88f94c261e2614a4998
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/restack-router-hilo-align32-zerostore
worktree: .worktrees/muse_prefill_layer/restack-router-hilo-align32-zerostore
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T232747Z
impl_commit: 2188edbec2315da2017111fd5e53d5d02d38277b
submission_sha256: 51d9aae3271de2a6a6f75b1685b671555308bb6d0c6e48bd26720d6709d167a1
norm_sha256: 334f0fbaafcf919ce79ea57db3b84728feb24fd13b67f5317e29ab65ff53d24e
submission_blob: 191990ad36719ff9ae1c2c997b53a68fbd2933a8
lint_warnings: 6
attempt_file: 2026-10-24T232747Z-check-restack-router-hilo-align32-zerostore.md
prepared_at: 2026-10-24T232747Z
job: ead4bc2e7b364add
dispatched_at: 2026-10-24T232756Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.0608200319850172
speedup_ci: [1.044907731505242, 1.0767323324647924]
harvested_at: 2026-10-25T000222Z
best_at_harvest: 1.0591912383762452
promotion: null
---

# restack-router-hilo-align32-zerostore

## Change

Built on master. Stack the three shape-disjoint prefill checks of 2026-10-24 23:15Z, each applied unchanged from its done/ patch: router-hilo-one-gemm-n512 (one N=512 GEMM for the router hi|lo halves, f4e3c5f4c306427b 1.0607), moe-align-32-experts-first-turn (moe_align_block_size over LOCAL_EXPERTS in _experts_triton, b5bf94b463d34d21 1.0604) and w2-nonlocal-zero-store-off-first-turn (prepare() probe turns the w2 zero-store off when the weighted sum skips remote slots, 9b731599dfa24278 1.0609). Each alone is within noise of master (1.059); together the router change touches every shape and the other two first_turn only, so the stack reads their combined effect.

## Expected

- `*` faster: sum of three sub-noise gains: router one GEMM at every T, smaller align capacity and no zero-stores at first_turn

## Risk

Restack built in the main thread before the 00:00Z reset with user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:30Z (user-approved self-review, no new agents). Worktree on master; `git apply -3` of the three done/neutral patches (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn), all clean, no edits. The three touch disjoint lines: prepare()'s router repack and the side-stream router GEMM; the align call in _experts_triton; the prepare() zero-store probe, _check_expert_gemms' w2 setting and the w2 call. Each passed the judge's tests alone tonight.

## Rules check

- [x] CUDA-graph capturable: unchanged from the three parents, each of which passed the judge.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights: NVFP4/BF16 unchanged; router hi+lo split as the rules allow.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Result

- job `ead4bc2e7b364add` (check), status done, passed True, tests 5/5
- speedup 1.0608x, 95% CI [1.0449, 1.0767], score 365192.8586558513 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/restack-router-hilo-align32-zerostore.result.json`, `muse_prefill_layer/hypotheses/done/neutral/restack-router-hilo-align32-zerostore.patch`; branch `hyp/muse_prefill_layer/restack-router-hilo-align32-zerostore`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1079.54 | 1169.39 | 1.003 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1419.09 | 1543.95 | 0.9981 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3334.9 | 3529.67 | 1.0025 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4937.98 | 4821.87 | 1.001 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 15386.32 | 13914.5 | 0.9665 | lost |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14275.3 | 14704.17 | 1.0032 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18165.94 | 18491.3 | 1.0012 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 22332.92 | 21942.14 | 0.9932 | flat |

lost on full first_turn. Expected hit: 0/1. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: restack-router-hilo-align32-zerostore`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

