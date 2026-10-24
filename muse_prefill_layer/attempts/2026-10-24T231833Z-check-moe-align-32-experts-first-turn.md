# muse_prefill_layer · 2026-10-24T231833Z · check · moe-align-32-experts-first-turn

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T184946Z-moe-align-32-experts-first-turn.md` (branch `hyp/muse_prefill_layer/moe-align-32-experts-first-turn`).

- **Job:** `b5bf94b463d34d21` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `1f58c81`

## Goal
Built on master (635cfee). One argument in _experts_triton (T > 4096 only): moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) becomes moe_align_block_size(local_ids, cfg['BLOCK_T'], LOCAL_EXPERTS). This is muse_moe_w4a16 master's deliberate difference from SGLang (its docstring: the ids are already mapped to [-1] U [0, 32), so 32 is the true expert count; the wrapper then sizes sorted_ids and block_experts as numel + 33*(block-1) instead of numel + 257*(block-1), and the per-expert cumsum runs over 32 counters). At BLOCK_T 256 and T=16384 that shrinks sorted_ids' static capacity by ~224*255 = 57K slots, i.e. ~224 token tiles, so the Triton grid (cdiv(EM, BLOCK_T) * N // BLOCK_W) loses ~14.3K w13 and ~7.2K w2 CTAs that today start, read n_padded and exit, one at a time per SM (one 145 KB-smem CTA fits per SM). Remote routes are still bucketed into -1 blocks exactly as today (muse_moe_w4a16's align self-check shows the 32-expert call buckets them too), so ZERO_NONLOCAL_W2 and everything downstream are unchanged. _experts_marlin keeps N_EXPERTS (T <= 4096 untouched). prepare(): _check_experts_e2e already runs _experts_triton at T=8192, which exercises the new call; add one log line with sorted_ids.shape[0] before and after for T=16384.

## What changed, and where
Built in the main thread 2026-10-24 ~23:35Z with self-review (user approved). _experts_triton only (T > 4096): moe_align_block_size(local_ids, BLOCK_T, N_EXPERTS) -> LOCAL_EXPERTS. The Marlin path at line 594 is untouched. Ids are already 0-31 or -1; muse_moe_w4a16 master uses the same call with 32 and passes; fused_post_expert_norm_weighted_sum skips remote slots via slot_expert_ids, so fewer -1 token tiles (and fewer ZERO_NONLOCAL_W2 zero stores) cannot change a read row.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no host sync, allocation on input values or value-dependent control flow added to kernel_fn.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights: NVFP4 and BF16 unchanged; the router hi/lo split is the one the rules allow (hi+lo, fp32 accumulate).
- [x] Activations BF16 as in SGLang; KV FP8; queries unchanged.
- [x] Every key attended (attention untouched).
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added (prepare() only builds a weight repack).
- [x] okc_lint: OK with warnings (pre-existing prepare-only host syncs).

## Risks / what could fail
low
One-argument probe of the dead-CTA mechanism behind experts-triton-persistent-grid-first-turn: if this row moves, the persistent grid should move ~3x as much. Subsumed by that hypothesis and by first-turn-moe-dead-work-stack; the reviewer may prefer to run only the larger ones. Overlap: experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) uses moe_align_block_size at 256 too, and the same one-argument change applies to it.

## Result (after the run)
- job `b5bf94b463d34d21` (check), status done, passed True, tests 5/5
- speedup 1.0604x, 95% CI [1.0445, 1.0763], score 365350.8319343853 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/moe-align-32-experts-first-turn.result.json`, `muse_prefill_layer/hypotheses/done/neutral/moe-align-32-experts-first-turn.patch`; branch `hyp/muse_prefill_layer/moe-align-32-experts-first-turn`
