---
problem: muse_bf16_proj
slug: unpooler-full-m16-swapab16-blockt16
created: 2026-10-25T145403Z
base: master
area: unpooler GEMM (full width n=8192, m=9..16, swap-AB BLOCK_T 16)
change: >
  Built on master (blob 353e3b19). On master the full-width unpooler (n=8192, k=4096, 67.1 MB) at
  9 <= m <= 16 is the only m <= 32 bf16 cell still on cuBLAS: _swapab16_key returns '' for
  unpooler with cfg n != 1024, _unpool_shape_ok needs n == 1024, _swapab_route_ok needs m >= 17,
  so kernel_fn falls through to _bf16_gemm_dispatch_impl (nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT,
  17.6 us kernel, row 18.6 us vs SGLang 18.4, 0.991x paired in the master run). Change, in section
  (6) only: add a key 'unpooler_full': (8192, 4096, 1, 256, 4) to SWAPAB16_CFG (grid (128, 1),
  SPLIT_K 1 so no workspace or tickets, BLOCK_K 256, 4 stages; the same per-layer constants
  master's 17..32 swap-AB uses for the unpooler, SWAPAB_CFG['unpooler']); change _swapab16_key so
  that layer == 'unpooler' returns 'unpooler_slice' when cfg n == 1024 and 'unpooler_full' when
  cfg n == 8192, else ''; in _swapab16_prepare map key 'unpooler_full' to state['unpooler'] (layer
  = 'unpooler' for both unpooler keys) and warm m in 9..16 for it (m_top 16), with the existing
  OutOfResources fallback to 3 stages. _swapab16_route_ok already requires tuple(w.shape) == (n,
  k) for non-slice keys and 1 <= cfg m <= 16; _swapab16_buffers returns (None, None) for the new
  key. The kernel is master's unchanged _bf16_swapab_kernel with BLOCK_T=16 (wgmma m64n16k16),
  exactly as master runs it for pooler/qkv_gate/o at m <= 16. Note the ouns cluster branch is
  checked first in kernel_fn and only takes n == 1024, so it is unaffected. Every other cell is
  byte-identical to master. The attempt file records the unpooler decode_16 kernel name
  (_bf16_swapab_kernel expected) and us, and the prepare log's [swapab16] unpooler_full PTX line
  (wgmma > 0, no spills).
expected:
  - shapes: layer=unpooler load=decode_16 m=16
    direction: faster
    why: unpooler-full-m16-swapab-route (BLOCK_T 32 swap-AB at m=16, job 05782267) read 1.022x paired on this row while nvjet reads 0.991-1.009x; BLOCK_T 16 halves the masked token tile (wgmma n16), which made qkv 18% and o 10-19% faster at m<=16 (bf16-m16-swapab-wgmma-extend)
  - shapes: layer=unpooler load=decode_32 m=32
    direction: flat
    why: "m=32 stays on master's BLOCK_T 32 swap-AB"
  - shapes: layer=unpooler load=verify_1 m=8
    direction: flat
    why: n=1024 slice, ouns cluster path unchanged
  - shapes: layer=unpooler load=decode_1 m=1
    direction: flat
    why: n=1024 slice, ouns cluster path unchanged
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: hold
review_at: 2026-10-25T150337Z
review_companions: null
hold_reason: "Opus review predicts 17.9-18.4 us vs nvjet 17.6 (BLOCK_T 32 run was 18.4 us; its 1.022x was baseline drift; DRAM-bound so BLOCK_T 16 should not help). Best case +0.07% headline: too weak to spend a scored run on."
---

# unpooler-full-m16-swapab16-blockt16

## Change

Built on master (blob 353e3b19). On master the full-width unpooler (n=8192, k=4096, 67.1 MB) at 9 <= m <= 16 is the only m <= 32 bf16 cell still on cuBLAS: _swapab16_key returns '' for unpooler with cfg n != 1024, _unpool_shape_ok needs n == 1024, _swapab_route_ok needs m >= 17, so kernel_fn falls through to _bf16_gemm_dispatch_impl (nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT, 17.6 us kernel, row 18.6 us vs SGLang 18.4, 0.991x paired in the master run). Change, in section (6) only: add a key 'unpooler_full': (8192, 4096, 1, 256, 4) to SWAPAB16_CFG (grid (128, 1), SPLIT_K 1 so no workspace or tickets, BLOCK_K 256, 4 stages; the same per-layer constants master's 17..32 swap-AB uses for the unpooler, SWAPAB_CFG['unpooler']); change _swapab16_key so that layer == 'unpooler' returns 'unpooler_slice' when cfg n == 1024 and 'unpooler_full' when cfg n == 8192, else ''; in _swapab16_prepare map key 'unpooler_full' to state['unpooler'] (layer = 'unpooler' for both unpooler keys) and warm m in 9..16 for it (m_top 16), with the existing OutOfResources fallback to 3 stages. _swapab16_route_ok already requires tuple(w.shape) == (n, k) for non-slice keys and 1 <= cfg m <= 16; _swapab16_buffers returns (None, None) for the new key. The kernel is master's unchanged _bf16_swapab_kernel with BLOCK_T=16 (wgmma m64n16k16), exactly as master runs it for pooler/qkv_gate/o at m <= 16. Note the ouns cluster branch is checked first in kernel_fn and only takes n == 1024, so it is unaffected. Every other cell is byte-identical to master. The attempt file records the unpooler decode_16 kernel name (_bf16_swapab_kernel expected) and us, and the prepare log's [swapab16] unpooler_full PTX line (wgmma > 0, no spills).

## Expected

- `layer=unpooler load=decode_16 m=16` faster: unpooler-full-m16-swapab-route (BLOCK_T 32 swap-AB at m=16, job 05782267) read 1.022x paired on this row while nvjet reads 0.991-1.009x; BLOCK_T 16 halves the masked token tile (wgmma n16), which made qkv 18% and o 10-19% faster at m<=16 (bf16-m16-swapab-wgmma-extend)
- `layer=unpooler load=decode_32 m=32` flat: m=32 stays on master's BLOCK_T 32 swap-AB
- `layer=unpooler load=verify_1 m=8` flat: n=1024 slice, ouns cluster path unchanged
- `layer=unpooler load=decode_1 m=1` flat: n=1024 slice, ouns cluster path unchanged
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Not a duplicate of done/regression unpooler-full-m16-swapab-route: that run used the BLOCK_T 32 kernel on an older master (8b9ad43) and its regression label came from untouched rows drifting (qkv/o/pooler/router lost together); its own target row gained 1.022x. This uses the BLOCK_T 16 variant and is built on current master. Risk medium rather than low: the pooler, the other 67 MB stream, was flat with BLOCK_T 16 (19.2 us at m=16), so the kernel may land at ~17.5-18.4 us, about nvjet's 17.6; the gain on one row is ~1-3% (decode_16 load ~+0.3-0.9%, headline ~+0.1%). Downside is one row. No m=9..16 full-width unpooler test exists (m=5 uses the slice, m=130/1000 stay on cuBLAS), so correctness rests on master's already-proven swap-AB kernel with SPLIT_K 1; prepare should run one fp32 self-check at m=16 and drop the key from swapab16_stages if rel L2 > 1e-2.

## Adversarial review

Reviewed 2026-10-25T150337Z by an adversarial subagent (model opus).

**Verdict: proceed.** Run it alone, without a companion. It changes one row (unpooler decode_16), and the profile's kernel time (_bf16_swapab_kernel vs nvjet's 17.6 us) shows directly whether it worked, but the case for a gain is weak: the BLOCK_T 32 version of this kernel took 18.4 us on this row and 18.1 us at m=32, so a flat or slightly slower row is the likeliest result.

### Justification
Mechanism: at m=16 the full-width unpooler is a pure 67.1 MB weight stream (floor 14.1 us). nvjet reaches 17.6 us, about 3.8 TB/s. The proposed kernel uses master's constants: grid (128,1), SPLIT_K 1, BLOCK_W 64, BLOCK_K 256, 4 stages. That is 40 KB per stage with BLOCK_T 16, so 160 KB of shared memory, still one CTA per SM on 128 of 132 SMs, each CTA streaming 512 KB over 16 K-steps. That is the same occupancy and the same bytes in flight as the BLOCK_T 32 variant. BLOCK_T 16 shrinks the x tile and the wgmma N size, but neither one is the bottleneck for a DRAM-bound stream. The evidence agrees. Prior run unpooler-full-m16-swapab-route (05782267) with BLOCK_T 32: kernel 18.4 us on this row. The same kernel at m=32 on master: 18.1 us. Pooler with BLOCK_T 16 at m=16 (same 67 MB, split-K 2): 19.2 us. All of these are at or above nvjet's 17.6. Weakest link: the 'faster' prediction rests on two things that do not carry over. (1) The 1.022x paired reading in 05782267 is noise. Its SGLang baseline_ns was 18.9 us, against 18.4 in the master run, while the kernel itself was 0.8 us slower than nvjet, and the row was judged flat (1.0129). (2) The qkv 18% and o 10-19% gains came from replacing mma.sync kernels that put the tokens on M with wgmma, not from going from BLOCK_T 32 to 16. My prediction for the kernel: about 17.9-18.4 us against nvjet's 17.6, so the row is flat to slightly slower. The best case is about +2% on one row out of 30, roughly +0.07% on the headline, well inside the ±1.5% noise. The expected shapes and directions do match the gating: only cfg n == 8192 with 9 <= m <= 16 changes, and the only scored row with that shape is unpooler decode_16. Not a duplicate: the do-not-retry list is empty, and the 'do not retry in this form' note on 05782267 covers BLOCK_T 32 on the older 8b9ad43 base.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32: it landed the SWAPAB16 path and the BLOCK_T 16 compile of _bf16_swapab_kernel that this change reuses. It shares the kernel, not timing or memory traffic. There is no interaction, other than that a compile-variant problem in it would show up here too.
- bf16-m32-swapab-wgmma-triton: SWAPAB_CFG['unpooler'] (8192,4096,1,256,4) is copied as a constant into a separate SWAPAB16_CFG key. Retuning that entry later would not change this key. m=17..32 is not touched, so there is no shared-path confound.
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128: the ouns branch runs first in kernel_fn, but it takes only n == 1024, and the NN-layout path takes only m=65..128. Neither matches cfg n=8192 at m=16, so neither can interact. The one adjacent risk is _swapab16_key: if it is written so that an n==1024 m<=8 call no longer returns 'unpooler_slice', the slice fallback after the ouns gate changes. Check the attempt diff for exactly this.
- sglang-gemm-dispatch-parity: it provides the nvjet incumbent that is being replaced. That incumbent's 17.6 us is the bar, and nothing about it varies between runs.

### Interference and A/B needs
No companion is needed. The change swaps exactly one kernel on one scored row. The run's profile gives that row's top kernel and its time, so _bf16_swapab_kernel's time against nvjet's 17.6 us (and against BLOCK_T 32's 18.4 us) answers the hypothesis directly, whatever the headline does. Reverting a landed hypothesis (ab-revert) would only take away the infrastructure this change depends on, and an ab-tune of the stage count or BLOCK_K would be a new hypothesis, not a disambiguator. Both would cost a scored run each, and neither would make a gain or a loss easier to read than the kernel time already does. The headline and the 29 other rows will drift by ±1.5% as before (05782267 was labelled 'regression' purely from untouched rows), so the harvest should judge this run on that row's kernel time, not on the headline.

### Open questions
- Run on a scored submit: is one row with a likely 0-2% kernel delta worth a scored run, when the best case moves the headline by about 0.07%? The deciding question is whether a flat or slower result closes the unpooler-full m<=16 swap-AB line (I think it should), not whether it can be read.
- If the kernel comes in at 18.0 us or more, should the harvest record 'swap-AB Triton cannot beat nvjet on the full-width 67 MB unpooler at m<=32 with one CTA per SM' as a do-not-retry, so that further BLOCK_T or stage variants are not tried? The next lever would be bytes in flight per SM (two CTAs per SM, or a TMA bulk stream), not the tile shape.
- Pooler with BLOCK_T 16 (19.2 us) is slower than the unpooler with BLOCK_T 32 (18.1 us) on the same 67 MB. Is that because of the pooler's split-K 2 reduction, or its shorter rows (k=8192 vs n=8192)? The answer would sharpen the prediction for this run.

