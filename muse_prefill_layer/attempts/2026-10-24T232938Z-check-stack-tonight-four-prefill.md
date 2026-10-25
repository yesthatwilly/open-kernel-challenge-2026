# muse_prefill_layer · 2026-10-24T232938Z · check · stack-tonight-four-prefill

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T232928Z-stack-tonight-four-prefill.md` (branch `hyp/muse_prefill_layer/stack-tonight-four-prefill`).

- **Job:** `7499d9c71acd4417` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `ad6a52e`

## Goal
Built on master. Stack all four prefill changes built on 2026-10-24 night, each applied unchanged: router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn (as in restack-router-hilo-align32-zerostore) plus moe-align-local-routes-parallel-port (parallel local-routes align on the Marlin path, prepare() self-checked). The four touch disjoint lines (router repack and GEMM; the Triton-path align; the zero-store probe and w2 flag; the Marlin-path align and its setup), so the run reads their combined effect against master and against the two parents.

## What changed, and where
Built in the main thread 2026-10-24 ~23:32Z with user-approved self-review. Worktree on master; `git apply -3` of the three done/neutral patches (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn) and of moe-align-local-routes-parallel-port's implementation diff, all clean, no edits. prepare() order: the zero-store probe and _check_expert_gemms (Triton bank), then the Marlin bank, then _align_par_setup, then _check_experts_e2e, so the end-to-end check covers both changed align paths together.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: unchanged from the four parents; kernel_fn gains only prepare()-set flags and shape-only allocations.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged; router hi+lo split as the rules allow.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added (all checks in prepare()).
- [x] okc_lint: no errors.

## Risks / what could fail
low
Stack built in the main thread before the 00:00Z reset with user-approved self-review.

## Result (after the run)
- job `7499d9c71acd4417` (check), status done, passed True, tests 5/5
- speedup 1.0675x, 95% CI [1.0515, 1.0835], score 362902.3439041988 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/stack-tonight-four-prefill.result.json`, `muse_prefill_layer/hypotheses/done/neutral/stack-tonight-four-prefill.patch`; branch `hyp/muse_prefill_layer/stack-tonight-four-prefill`
