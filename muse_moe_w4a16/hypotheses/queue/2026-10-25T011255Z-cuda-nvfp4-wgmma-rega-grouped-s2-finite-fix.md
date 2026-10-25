---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
created: 2026-10-25T011255Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 2 of the wgmma register-A grouped NVFP4 rewrite (make it numerically correct). Stage 1b
  (job b56d17fda2d54be2) finally BUILT and RAN on the judge: TORCH_CUDA_ARCH_LIST=9.0a gave only
  -gencode compute_90a/sm_90a, load_inline 32.8 s, w13+silu 168 regs/0 local, w2 168 regs/8 B
  local, 187-204 KB smem, cuTensorMapEncodeTiled x8 rc 0, and every canary launch (align,
  wg_gather_kernel, wg_kernel<w13+silu>, wg_kernel<w2>, post-norm) completed with device error
  word 0 (no fault, no mbarrier timeout). But the canary output (M=128, slot 0 of every token on
  local expert 0, 7 remote routes, randn activations, REAL retiled weights) was NOT FINITE, so
  setup raised 'canary failed' and PATH CHOSEN = MARLIN: the 1.3303 score is a master rerun. Build
  on stage 1b's code (claim bases a rewrite-on-rewrite on the parent's commit) and change only
  what follows. (1) LOCALIZE in the canary, each step after its own synchronize + flushed print:
  (a) gather: compare the slab/gathered activation buffer with x[sorted token rows] exactly; (b)
  w13+silu: compare cache2 on the LOCAL route rows only against master's Marlin w13 + silu_and_mul
  on the same inputs; print count of non-finite, first non-finite (row, col), col // BN (N-tile)
  and k-stage if derivable, rel-L2 over finite entries, and the 3 worst N-tiles; (c) w2: feed w2
  the MARLIN reference cache2 (so w2 is judged independently of w13) and compare its y on local
  rows against Marlin w2; same prints; (d) report finiteness of y separately on local rows and on
  remote/sentinel rows, since remote rows are never written (Marlin is_ep convention) and must not
  be read by the norm. (2) FIX the likely causes, all cheap and safe: zero the whole dynamic smem
  ring/slab once at CTA start before the first bulk copy (uninitialized smem read through a wrong
  B descriptor/box is the classic NaN source; bf16 garbage is often NaN/Inf), and zero-fill slab
  rows of padding/sentinel tokens explicitly rather than leaving them unwritten; make cache2/y
  zero for rows the post-norm may touch if (1d) shows remote rows are read. (3) BEFORE DISPATCH,
  without a GPU, emulate offline in numpy (as the swap-AB review did to 1e-15): the wgmma m64nNk16
  register-A fragment mapping of the dequantized weight (it differs from mma.sync m16n8k16: warp w
  of the warpgroup owns rows 16w..16w+15), the B shared-memory matrix descriptor (start address >>
  4, LBO, SBO and swizzle mode must match the swizzle the bulk-copy/TMA wrote), the mbarrier
  expect_tx byte count per stage versus the bytes the bulk copies plus TMA actually deliver now
  that the L2::cache_hint operand was dropped, and the e4m3 block-scale / global-scale
  application; fix any mismatch found. (4) Remove the w2 8-byte spill if trivial (not a
  correctness suspect). Keep tiling (BK 128, 4 stages, N buckets 16/32/64/128, grid 132 x 384
  threads, 1 producer + 2 consumer warpgroups), the num_tokens >= 128 gate with its own bound
  constant, the sm_90a build, the in-process guarded first launch and the Marlin fallback. Log
  PATH CHOSEN; a log without PATH CHOSEN = WGMMA is unmeasured. First measurement is an okc check,
  never a submit.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "M1: canary finite, SELF-CHECK PASSED, PATH CHOSEN = WGMMA, 5/5 tests. M2: wg_gather + wg_kernel x2 rows below master Marlin pair + act_and_mul (292.5 + 3.9 us in best profile; 275.0 + 3.7 in stage 1b fallback run). Floor 101.3 us, M3 <= 170 us"
  - shapes: bs=256 *
    direction: faster
    why: M2 below Marlin 391.4 + 6.4 us (best profile; 425.4 + 5.5 in stage 1b run); floor 108.9 us, M3 <= 220 us
  - shapes: bs=1 *
    direction: flat
    why: "control: gate is num_tokens >= 128"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix

## Change

Stage 2 of the wgmma register-A grouped NVFP4 rewrite (make it numerically correct). Stage 1b (job b56d17fda2d54be2) finally BUILT and RAN on the judge: TORCH_CUDA_ARCH_LIST=9.0a gave only -gencode compute_90a/sm_90a, load_inline 32.8 s, w13+silu 168 regs/0 local, w2 168 regs/8 B local, 187-204 KB smem, cuTensorMapEncodeTiled x8 rc 0, and every canary launch (align, wg_gather_kernel, wg_kernel<w13+silu>, wg_kernel<w2>, post-norm) completed with device error word 0 (no fault, no mbarrier timeout). But the canary output (M=128, slot 0 of every token on local expert 0, 7 remote routes, randn activations, REAL retiled weights) was NOT FINITE, so setup raised 'canary failed' and PATH CHOSEN = MARLIN: the 1.3303 score is a master rerun. Build on stage 1b's code (claim bases a rewrite-on-rewrite on the parent's commit) and change only what follows. (1) LOCALIZE in the canary, each step after its own synchronize + flushed print: (a) gather: compare the slab/gathered activation buffer with x[sorted token rows] exactly; (b) w13+silu: compare cache2 on the LOCAL route rows only against master's Marlin w13 + silu_and_mul on the same inputs; print count of non-finite, first non-finite (row, col), col // BN (N-tile) and k-stage if derivable, rel-L2 over finite entries, and the 3 worst N-tiles; (c) w2: feed w2 the MARLIN reference cache2 (so w2 is judged independently of w13) and compare its y on local rows against Marlin w2; same prints; (d) report finiteness of y separately on local rows and on remote/sentinel rows, since remote rows are never written (Marlin is_ep convention) and must not be read by the norm. (2) FIX the likely causes, all cheap and safe: zero the whole dynamic smem ring/slab once at CTA start before the first bulk copy (uninitialized smem read through a wrong B descriptor/box is the classic NaN source; bf16 garbage is often NaN/Inf), and zero-fill slab rows of padding/sentinel tokens explicitly rather than leaving them unwritten; make cache2/y zero for rows the post-norm may touch if (1d) shows remote rows are read. (3) BEFORE DISPATCH, without a GPU, emulate offline in numpy (as the swap-AB review did to 1e-15): the wgmma m64nNk16 register-A fragment mapping of the dequantized weight (it differs from mma.sync m16n8k16: warp w of the warpgroup owns rows 16w..16w+15), the B shared-memory matrix descriptor (start address >> 4, LBO, SBO and swizzle mode must match the swizzle the bulk-copy/TMA wrote), the mbarrier expect_tx byte count per stage versus the bytes the bulk copies plus TMA actually deliver now that the L2::cache_hint operand was dropped, and the e4m3 block-scale / global-scale application; fix any mismatch found. (4) Remove the w2 8-byte spill if trivial (not a correctness suspect). Keep tiling (BK 128, 4 stages, N buckets 16/32/64/128, grid 132 x 384 threads, 1 producer + 2 consumer warpgroups), the num_tokens >= 128 gate with its own bound constant, the sm_90a build, the in-process guarded first launch and the Marlin fallback. Log PATH CHOSEN; a log without PATH CHOSEN = WGMMA is unmeasured. First measurement is an okc check, never a submit.

## Expected

- `bs=128 *` faster: M1: canary finite, SELF-CHECK PASSED, PATH CHOSEN = WGMMA, 5/5 tests. M2: wg_gather + wg_kernel x2 rows below master Marlin pair + act_and_mul (292.5 + 3.9 us in best profile; 275.0 + 3.7 in stage 1b fallback run). Floor 101.3 us, M3 <= 170 us
- `bs=256 *` faster: M2 below Marlin 391.4 + 6.4 us (best profile; 425.4 + 5.5 in stage 1b run); floor 108.9 us, M3 <= 220 us
- `bs=1 *` flat: control: gate is num_tokens >= 128
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control

## Risk

The canary used real weights and randn x, so a non-finite y is a kernel/descriptor/layout bug, not data. If (1) shows y non-finite only on remote rows, the fix is the norm read/zeroing and the GEMMs may already be right. A sticky fault is still unrecoverable in-process; keep -lineinfo and flushed prints.

## Milestones

Rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
