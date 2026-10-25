# muse_prefill_layer · 2026-10-24T233306Z · check · stack-tonight-six-prefill

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T233245Z-stack-tonight-six-prefill.md` (branch `hyp/muse_prefill_layer/stack-tonight-six-prefill`).

- **Job:** `5cb6e68a86ec48f0` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `9ac7258`

## Goal
Built on master. stack-tonight-five-prefill plus align-local-parallel-first-turn (the parallel local-routes align on the Triton path too, self-checked at T=16384, BLOCK_T 256), each applied unchanged. With the parallel align on at first_turn, moe-align-32-experts-first-turn's moe_align_block_size(LOCAL_EXPERTS) call becomes the fallback branch only.

## What changed, and where
Built in the main thread 2026-10-24 ~23:35Z with user-approved self-review: stack-tonight-five-prefill's submission.py, then a 3-way apply of align-local-parallel-first-turn's implementation diff. One real conflict, in _experts_triton's align call (moe-align-32 vs the parallel align), resolved as the spec says: the parallel align when state["align_par_ok"], else moe_align_block_size(local_ids, BLOCK_T, LOCAL_EXPERTS). The extended self-check (T=16384 at BLOCK_T 256) came over cleanly.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: unchanged from the parents; prepare()-set gates, shape-only allocations.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged; router hi+lo split as the rules allow.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Risks / what could fail
medium
Main-thread stack before the 00:00Z reset, user-approved self-review.

## Result (after the run)
- job `5cb6e68a86ec48f0` (check), status done, passed True, tests 5/5
- speedup 1.0642x, 95% CI [1.0482, 1.0801], score 364038.26718806016 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/stack-tonight-six-prefill.result.json`, `muse_prefill_layer/hypotheses/done/neutral/stack-tonight-six-prefill.patch`; branch `hyp/muse_prefill_layer/stack-tonight-six-prefill`
