# muse_prefill_layer · 2026-10-24T232747Z · check · restack-router-hilo-align32-zerostore

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T232728Z-restack-router-hilo-align32-zerostore.md` (branch `hyp/muse_prefill_layer/restack-router-hilo-align32-zerostore`).

- **Job:** `ead4bc2e7b364add` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `2188edb`

## Goal
Built on master. Stack the three shape-disjoint prefill checks of 2026-10-24 23:15Z, each applied unchanged from its done/ patch: router-hilo-one-gemm-n512 (one N=512 GEMM for the router hi|lo halves, f4e3c5f4c306427b 1.0607), moe-align-32-experts-first-turn (moe_align_block_size over LOCAL_EXPERTS in _experts_triton, b5bf94b463d34d21 1.0604) and w2-nonlocal-zero-store-off-first-turn (prepare() probe turns the w2 zero-store off when the weighted sum skips remote slots, 9b731599dfa24278 1.0609). Each alone is within noise of master (1.059); together the router change touches every shape and the other two first_turn only, so the stack reads their combined effect.

## What changed, and where
Built in the main thread 2026-10-24 ~23:30Z (user-approved self-review, no new agents). Worktree on master; `git apply -3` of the three done/neutral patches (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn), all clean, no edits. The three touch disjoint lines: prepare()'s router repack and the side-stream router GEMM; the align call in _experts_triton; the prepare() zero-store probe, _check_expert_gemms' w2 setting and the w2 call. Each passed the judge's tests alone tonight.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: unchanged from the three parents, each of which passed the judge.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights: NVFP4/BF16 unchanged; router hi+lo split as the rules allow.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Risks / what could fail
low
Restack built in the main thread before the 00:00Z reset with user-approved self-review.

## Result (after the run)
- job `ead4bc2e7b364add` (check), status done, passed True, tests 5/5
- speedup 1.0608x, 95% CI [1.0449, 1.0767], score 365192.8586558513 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/restack-router-hilo-align32-zerostore.result.json`, `muse_prefill_layer/hypotheses/done/neutral/restack-router-hilo-align32-zerostore.patch`; branch `hyp/muse_prefill_layer/restack-router-hilo-align32-zerostore`
