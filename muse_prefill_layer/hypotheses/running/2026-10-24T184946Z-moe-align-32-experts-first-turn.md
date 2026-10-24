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
status: running
stage: claimed
claimed_at: 2026-10-24T231740Z
base_sha: 32a19128ff06ed728e345e32098cd6f0a13cea32
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/moe-align-32-experts-first-turn
worktree: .worktrees/muse_prefill_layer/moe-align-32-experts-first-turn
mode: null
---

# moe-align-32-experts-first-turn

## Change

Built on master (635cfee). One argument in _experts_triton (T > 4096 only): moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) becomes moe_align_block_size(local_ids, cfg['BLOCK_T'], LOCAL_EXPERTS). This is muse_moe_w4a16 master's deliberate difference from SGLang (its docstring: the ids are already mapped to [-1] U [0, 32), so 32 is the true expert count; the wrapper then sizes sorted_ids and block_experts as numel + 33*(block-1) instead of numel + 257*(block-1), and the per-expert cumsum runs over 32 counters). At BLOCK_T 256 and T=16384 that shrinks sorted_ids' static capacity by ~224*255 = 57K slots, i.e. ~224 token tiles, so the Triton grid (cdiv(EM, BLOCK_T) * N // BLOCK_W) loses ~14.3K w13 and ~7.2K w2 CTAs that today start, read n_padded and exit, one at a time per SM (one 145 KB-smem CTA fits per SM). Remote routes are still bucketed into -1 blocks exactly as today (muse_moe_w4a16's align self-check shows the 32-expert call buckets them too), so ZERO_NONLOCAL_W2 and everything downstream are unchanged. _experts_marlin keeps N_EXPERTS (T <= 4096 untouched). prepare(): _check_experts_e2e already runs _experts_triton at T=8192, which exercises the new call; add one log line with sorted_ids.shape[0] before and after for T=16384.

## Expected

- `*load=first_turn*` faster: ~21K fewer dead CTAs across the two _moe_nvfp4_gemm_kernel calls (est. ~1 us each per SM slot, ~150 us) plus a 32-counter align: about -0.8 to -1.3% on first_turn shapes
- `*load=turn_*` neutral: Marlin path unchanged

## Risk

One-argument probe of the dead-CTA mechanism behind experts-triton-persistent-grid-first-turn: if this row moves, the persistent grid should move ~3x as much. Subsumed by that hypothesis and by first-turn-moe-dead-work-stack; the reviewer may prefer to run only the larger ones. Overlap: experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) uses moe_align_block_size at 256 too, and the same one-argument change applies to it.
