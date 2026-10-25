# muse_prefill_layer · 2026-10-24T233216Z · check · stack-tonight-five-prefill

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T233200Z-stack-tonight-five-prefill.md` (branch `hyp/muse_prefill_layer/stack-tonight-five-prefill`).

- **Job:** `c4d14d62f11e4aaf` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `b9ad6f2`

## Goal
Built on master. stack-tonight-four-prefill (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn, moe-align-local-routes-parallel-port) plus silu-mul-triton-local-rows-marlin-turn (sorted local-rows silu-mul on the Marlin path, prepare() self-checked), each applied unchanged. On the Marlin path the silu-mul walks the parallel align's sorted_ids, which hold local routes only. prepare() runs _align_par_setup, then _silu_sorted_setup, then _check_experts_e2e.

## What changed, and where
Built in the main thread 2026-10-24 ~23:33Z with user-approved self-review. Worktree on master; `git apply -3` of the three done/neutral patches, moe-align-local-routes-parallel-port's implementation diff and silu-mul-triton-local-rows-marlin-turn's implementation diff. Two conflicts, both "both sides added" (the two prepare() setup calls before _check_experts_e2e, and the two new code sections before _experts_marlin), resolved by keeping both, align first; no line edited. On the Marlin path the sorted silu-mul consumes the parallel align's sorted_ids/n_padded (local routes only) and still masks on local_ids >= 0.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: unchanged from the five parents.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged; router hi+lo split as the rules allow.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Risks / what could fail
low
Main-thread stack before the 00:00Z reset, user-approved self-review.

## Result (after the run)
- job `c4d14d62f11e4aaf` (check), status done, passed True, tests 5/5
- speedup 1.0705x, 95% CI [1.0545, 1.0866], score 361887.87708503474 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json`, `muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.patch`; branch `hyp/muse_prefill_layer/stack-tonight-five-prefill`
