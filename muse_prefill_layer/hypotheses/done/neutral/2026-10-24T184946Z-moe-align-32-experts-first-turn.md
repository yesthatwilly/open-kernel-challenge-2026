---
problem: muse_prefill_layer
slug: moe-align-32-experts-first-turn
created: 2026-10-24T184946Z
base: master
area: MoE align/dispatch (align over the 32 local experts at T > 4096)
change: >
  Built on master (635cfee). One argument in _experts_triton (T > 4096 only):
  moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) becomes
  moe_align_block_size(local_ids, cfg['BLOCK_T'], LOCAL_EXPERTS). This is muse_moe_w4a16 master's
  deliberate difference from SGLang (its docstring: the ids are already mapped to [-1] U [0, 32),
  so 32 is the true expert count; the wrapper then sizes sorted_ids and block_experts as numel +
  33*(block-1) instead of numel + 257*(block-1), and the per-expert cumsum runs over 32 counters).
  At BLOCK_T 256 and T=16384 that shrinks sorted_ids' static capacity by ~224*255 = 57K slots,
  i.e. ~224 token tiles, so the Triton grid (cdiv(EM, BLOCK_T) * N // BLOCK_W) loses ~14.3K w13
  and ~7.2K w2 CTAs that today start, read n_padded and exit, one at a time per SM (one 145
  KB-smem CTA fits per SM). Remote routes are still bucketed into -1 blocks exactly as today
  (muse_moe_w4a16's align self-check shows the 32-expert call buckets them too), so
  ZERO_NONLOCAL_W2 and everything downstream are unchanged. _experts_marlin keeps N_EXPERTS (T <=
  4096 untouched). prepare(): _check_experts_e2e already runs _experts_triton at T=8192, which
  exercises the new call; add one log line with sorted_ids.shape[0] before and after for T=16384.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "~21K fewer dead CTAs across the two _moe_nvfp4_gemm_kernel calls (est. ~1 us each per SM slot, ~150 us) plus a 32-counter align: about -0.8 to -1.3% on first_turn shapes"
  - shapes: "*load=turn_*"
    direction: neutral
    why: Marlin path unchanged
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: required
status: done
stage: harvested
claimed_at: 2026-10-24T231740Z
base_sha: 32a19128ff06ed728e345e32098cd6f0a13cea32
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/moe-align-32-experts-first-turn
worktree: .worktrees/muse_prefill_layer/moe-align-32-experts-first-turn
mode: check
implemented_at: 2026-10-24T231833Z
impl_commit: 1f58c81d8380134a4c855a96cc0db4966daaf34f
submission_sha256: c841fab1d4e92edea4c0e1c3ecce5eb32c88740c605e8154a8839364643d84d5
norm_sha256: 93c740ab7098bc4e2b58b618c5d2002f9364593414ecf53a4140f6c386224087
submission_blob: 9316ae43a54ccbf24f65d14815c68fa6c81f557a
lint_warnings: 5
attempt_file: 2026-10-24T231833Z-check-moe-align-32-experts-first-turn.md
prepared_at: 2026-10-24T231833Z
job: b5bf94b463d34d21
dispatched_at: 2026-10-24T231843Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.060361346240414
speedup_ci: [1.0444559260468078, 1.0762667664340202]
harvested_at: 2026-10-24T232736Z
best_at_harvest: 1.0591912383762452
promotion: null
---

# moe-align-32-experts-first-turn

## Change

Built on master (635cfee). One argument in _experts_triton (T > 4096 only): moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) becomes moe_align_block_size(local_ids, cfg['BLOCK_T'], LOCAL_EXPERTS). This is muse_moe_w4a16 master's deliberate difference from SGLang (its docstring: the ids are already mapped to [-1] U [0, 32), so 32 is the true expert count; the wrapper then sizes sorted_ids and block_experts as numel + 33*(block-1) instead of numel + 257*(block-1), and the per-expert cumsum runs over 32 counters). At BLOCK_T 256 and T=16384 that shrinks sorted_ids' static capacity by ~224*255 = 57K slots, i.e. ~224 token tiles, so the Triton grid (cdiv(EM, BLOCK_T) * N // BLOCK_W) loses ~14.3K w13 and ~7.2K w2 CTAs that today start, read n_padded and exit, one at a time per SM (one 145 KB-smem CTA fits per SM). Remote routes are still bucketed into -1 blocks exactly as today (muse_moe_w4a16's align self-check shows the 32-expert call buckets them too), so ZERO_NONLOCAL_W2 and everything downstream are unchanged. _experts_marlin keeps N_EXPERTS (T <= 4096 untouched). prepare(): _check_experts_e2e already runs _experts_triton at T=8192, which exercises the new call; add one log line with sorted_ids.shape[0] before and after for T=16384.

## Expected

- `*load=first_turn*` faster: ~21K fewer dead CTAs across the two _moe_nvfp4_gemm_kernel calls (est. ~1 us each per SM slot, ~150 us) plus a 32-counter align: about -0.8 to -1.3% on first_turn shapes
- `*load=turn_*` neutral: Marlin path unchanged

## Risk

One-argument probe of the dead-CTA mechanism behind experts-triton-persistent-grid-first-turn: if this row moves, the persistent grid should move ~3x as much. Subsumed by that hypothesis and by first-turn-moe-dead-work-stack; the reviewer may prefer to run only the larger ones. Overlap: experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) uses moe_align_block_size at 256 too, and the same one-argument change applies to it.

## Implementation notes

Built in the main thread 2026-10-24 ~23:35Z with self-review (user approved). _experts_triton only (T > 4096): moe_align_block_size(local_ids, BLOCK_T, N_EXPERTS) -> LOCAL_EXPERTS. The Marlin path at line 594 is untouched. Ids are already 0-31 or -1; muse_moe_w4a16 master uses the same call with 32 and passes; fused_post_expert_norm_weighted_sum skips remote slots via slot_expert_ids, so fewer -1 token tiles (and fewer ZERO_NONLOCAL_W2 zero stores) cannot change a read row.

## Rules check

- [x] CUDA-graph capturable: no host sync, allocation on input values or value-dependent control flow added to kernel_fn.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights: NVFP4 and BF16 unchanged; the router hi/lo split is the one the rules allow (hi+lo, fp32 accumulate).
- [x] Activations BF16 as in SGLang; KV FP8; queries unchanged.
- [x] Every key attended (attention untouched).
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added (prepare() only builds a weight repack).
- [x] okc_lint: OK with warnings (pre-existing prepare-only host syncs).

## Result

- job `b5bf94b463d34d21` (check), status done, passed True, tests 5/5
- speedup 1.0604x, 95% CI [1.0445, 1.0763], score 365350.8319343853 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/moe-align-32-experts-first-turn.result.json`, `muse_prefill_layer/hypotheses/done/neutral/moe-align-32-experts-first-turn.patch`; branch `hyp/muse_prefill_layer/moe-align-32-experts-first-turn`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1079.54 | 1028.17 | 1.0013 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1419.09 | 1433.99 | 0.9989 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3334.9 | 3408.34 | 0.9996 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4937.98 | 4946.9 | 1.0003 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 15386.32 | 14887.68 | 0.9964 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14275.3 | 13902.23 | 1.0004 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18165.94 | 18111.87 | 1.0121 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 22332.92 | 21983.85 | 0.9992 | flat |

no shape moved outside noise. Expected hit: 0/2. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: moe-align-32-experts-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

