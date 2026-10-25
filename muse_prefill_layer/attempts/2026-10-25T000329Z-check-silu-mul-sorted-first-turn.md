# muse_prefill_layer · 2026-10-25T000329Z · check · silu-mul-sorted-first-turn

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T233334Z-silu-mul-sorted-first-turn.md` (branch `hyp/muse_prefill_layer/silu-mul-sorted-first-turn`).

- **Job:** `9528eed511b04f33` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `fc7f0a0`

## Goal
Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_triton, when state['silu_sorted_ok'], replace silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1) with _silu_mul_sorted(cache1, cache2, sorted_ids, n_padded, local_ids.view(-1), size_n) over the Triton path's align (BLOCK_T 256); extend _silu_sorted_setup with T=16384 at BLOCK_T 256 so the gate covers it. act_and_mul costs ~297 us at first_turn against a ~84 us local-rows floor. experts-triton-w13-fused-silu-epilogue, the reason the parent excluded this path, is done (neutral), so nothing is confounded.

## What changed, and where
Built in the main thread 2026-10-24 ~23:35Z with user-approved self-review, on silu-mul-triton-local-rows-marlin-turn's implementation commit (claim based it there: parent still running). `_experts_triton`: `_silu_mul_sorted(cache1, cache2, sorted_ids, n_padded, local_ids.view(-1), size_n)` when `state["silu_sorted_ok"]`, else master's silu_and_mul. `_silu_sorted_setup` now also checks T=16384 at the Triton path's BLOCK_T (256); a miss turns the sorted silu OFF on both paths. Equivalence: master's silu_and_mul(expert_ids=local_ids) also skips -1 slots; the w13 Triton GEMM leaves -1-tile rows unwritten and w2 zero-stores those tiles without reading cache2, so skipping them changes no read value.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: one Triton launch, static grid, prepare()-set gate.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.

## Risks / what could fail
medium
Main-thread build before the 00:00Z reset, user-approved self-review.

## Result (after the run)
- job `9528eed511b04f33` (check), status done, passed True, tests 5/5
- speedup 1.0636x, 95% CI [1.0477, 1.0796], score 364222.7153925436 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0705080897445196, "ci": [1.0544504683983518, 1.0865657110906872], "job": "c4d14d62f11e4aaf"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/silu-mul-sorted-first-turn.result.json`, `muse_prefill_layer/hypotheses/done/neutral/silu-mul-sorted-first-turn.patch`; branch `hyp/muse_prefill_layer/silu-mul-sorted-first-turn`
