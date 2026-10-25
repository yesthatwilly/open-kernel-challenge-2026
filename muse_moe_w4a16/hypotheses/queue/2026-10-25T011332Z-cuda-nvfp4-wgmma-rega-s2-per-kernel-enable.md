---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-s2-per-kernel-enable
created: 2026-10-25T011332Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 2 variant of the wgmma register-A grouped NVFP4 rewrite, run beside
  cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix so one check measures something even if only one
  kernel is broken. Stage 1b (job b56d17fda2d54be2) built for sm_90a only (32.8 s), launched every
  canary kernel with device error word 0, but the M=128 canary output (real retiled weights, randn
  x) was non-finite, so PATH CHOSEN = MARLIN and nothing was measured. Build on stage 1b's code
  and change: (1) judge and enable the two GEMM kernels INDEPENDENTLY. Canary and full self-check
  run wg_kernel<w13+silu> against master's Marlin w13 + silu_and_mul on local route rows, and
  wg_kernel<w2> fed Marlin's reference cache2 against Marlin w2 on local rows; each gets its own
  finiteness + rel-L2 verdict and its own flag (state['wg_w13_ok'], state['wg_w2_ok']), and its
  own flushed 'PATH CHOSEN w13+silu = WGMMA|MARLIN' / 'PATH CHOSEN w2 = WGMMA|MARLIN' line with
  the reason. (2) kernel_fn at num_tokens >= 128 (own bound constant) runs w13+silu on wgmma if
  wg_w13_ok else Marlin w13 + silu_and_mul, then w2 on wgmma if wg_w2_ok else Marlin w2. Where the
  two paths' intermediate layouts differ (the wgmma path's gathered/sorted cache2 vs Marlin's
  flat-route-index cache2 read through sorted_ids), add one small bridging step (a Triton
  gather/scatter by sorted_ids, or by feeding the wgmma w2 the same sorted_ids Marlin uses) and
  include its row in the replaced-kernel sum; read the stage 1b code to decide which is needed,
  and self-check each mixed combination end to end against master (rel-L2 bound unchanged). (3)
  Zero the dynamic smem ring/slab once at CTA start and zero-fill padding/sentinel slab rows
  (cheap guard against uninitialized-smem NaNs). No tiling, pipeline, dequant or descriptor change
  in this variant (those are the sibling's job), sm_90a build, in-process guarded first launch,
  Marlin fallback and -lineinfo kept. First measurement is an okc check.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "M1: at least one PATH CHOSEN line says WGMMA and 5/5 tests pass. M2: the enabled wgmma kernel row(s) plus any bridge below the Marlin row(s) they replace (best profile Marlin pair 292.5 us + act 3.9; per kernel roughly 2/3 w13, 1/3 w2 by bytes). Floor 101.3 us"
  - shapes: bs=256 *
    direction: faster
    why: same, vs Marlin pair 391.4 + 6.4 us; floor 108.9 us
  - shapes: bs=1 *
    direction: flat
    why: control
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

# cuda-nvfp4-wgmma-rega-s2-per-kernel-enable

## Change

Stage 2 variant of the wgmma register-A grouped NVFP4 rewrite, run beside cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix so one check measures something even if only one kernel is broken. Stage 1b (job b56d17fda2d54be2) built for sm_90a only (32.8 s), launched every canary kernel with device error word 0, but the M=128 canary output (real retiled weights, randn x) was non-finite, so PATH CHOSEN = MARLIN and nothing was measured. Build on stage 1b's code and change: (1) judge and enable the two GEMM kernels INDEPENDENTLY. Canary and full self-check run wg_kernel<w13+silu> against master's Marlin w13 + silu_and_mul on local route rows, and wg_kernel<w2> fed Marlin's reference cache2 against Marlin w2 on local rows; each gets its own finiteness + rel-L2 verdict and its own flag (state['wg_w13_ok'], state['wg_w2_ok']), and its own flushed 'PATH CHOSEN w13+silu = WGMMA|MARLIN' / 'PATH CHOSEN w2 = WGMMA|MARLIN' line with the reason. (2) kernel_fn at num_tokens >= 128 (own bound constant) runs w13+silu on wgmma if wg_w13_ok else Marlin w13 + silu_and_mul, then w2 on wgmma if wg_w2_ok else Marlin w2. Where the two paths' intermediate layouts differ (the wgmma path's gathered/sorted cache2 vs Marlin's flat-route-index cache2 read through sorted_ids), add one small bridging step (a Triton gather/scatter by sorted_ids, or by feeding the wgmma w2 the same sorted_ids Marlin uses) and include its row in the replaced-kernel sum; read the stage 1b code to decide which is needed, and self-check each mixed combination end to end against master (rel-L2 bound unchanged). (3) Zero the dynamic smem ring/slab once at CTA start and zero-fill padding/sentinel slab rows (cheap guard against uninitialized-smem NaNs). No tiling, pipeline, dequant or descriptor change in this variant (those are the sibling's job), sm_90a build, in-process guarded first launch, Marlin fallback and -lineinfo kept. First measurement is an okc check.

## Expected

- `bs=128 *` faster: M1: at least one PATH CHOSEN line says WGMMA and 5/5 tests pass. M2: the enabled wgmma kernel row(s) plus any bridge below the Marlin row(s) they replace (best profile Marlin pair 292.5 us + act 3.9; per kernel roughly 2/3 w13, 1/3 w2 by bytes). Floor 101.3 us
- `bs=256 *` faster: same, vs Marlin pair 391.4 + 6.4 us; floor 108.9 us
- `bs=1 *` flat: control
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control

## Risk

Value is diagnostic + partial measurement: tells which of the two wgmma kernels produced the non-finite canary and times the healthy one on the judge. A bridge launch (~2-4 us) is charged to the rewrite.

## Milestones

Rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
