---
problem: muse_attention_decode
slug: full-verify-wgmma-s4-sliding-window
created: 2026-10-25T003705Z
base: master
area: sliding-layer attention (verify, wgmma stream-K kernel)
change: >
  Stage 4 of the full-attention verify rewrite, built on the stage-3 branch
  hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc
  (splitkv_verify_wgmma_kernel, module muse_full_verify_wgmma_v1: stream-K over 64-token tiles,
  one consumer warpgroup per kv head = one wgmma M=64 tile of 8 heads x 8 tokens, fp8 QK and PV,
  in-kernel ticket combine + norm-gate), whatever its score. Sliding verify is 46x per step in
  three loads: 16.7/22.7/28.7 us at verify_1/16/32 (FA3 main 7.6/10.7/14.5 + combine 3.9/4.1/4.3 +
  prepare 1.7 + pre-attn + norm-gate) against floors 0.1/1.9/3.7. Change: route SLIDING verify
  (window != 0, q_len 8, 1 <= batch <= 64, hq 16, hkv 2, head_dim 64, page 1; config-only
  predicate) through the same kernel after master _pre_attention_kernel (ROPE on, PAIRED=False,
  unchanged). Kernel edit, one new int argument window (0 = full, which must compile and behave
  exactly as stage 3): per request key range [lo_b, seq_b) with lo_b = window ? max(0, seq_b - 8 -
  (window - 1)) : 0 computed in-kernel from cache_seqlens; tiles_b = ceil((seq_b - lo_b)/64), tile
  k covers keys lo_b + 64k .. +63 (page 1, so tiles need no alignment); the combine path
  recomputes the same TP prefix. Mask adds the window lower edge: key t valid for block token j
  iff t <= seq_b - 8 + j and t >= seq_b - 8 + j - (window - 1) and t >= lo_b, applied on any tile
  overlapping either edge (matches FA3 window_size=(window - 1, 0) as master passes it, so every
  key in each window is attended). Grid: at sliding verify Ttot is only ~33 tiles per request, so
  use grid = min(2 x SM count, Ttot rounded from config: batch * ceil((window + 7)/64)) so no CTA
  launches empty; workspace sized grid + B. Self-check adds sliding cases (window 2048, lens <
  window, < 8 + window, and 80000, batch 1/2/16/32, kv_scale 1 and 0.25) against an fp32 windowed
  causal reference by rel L2; on failure only the sliding route is disabled (PATH CHOSEN: sliding
  verify FA3 (fallback: <reason>)), else PATH CHOSEN: sliding verify wgmma window. BUILD/SAFETY as
  stage 3 (sm_90a only, __CUDA_ARCH_FEAT_SM90_ALL guard with device error flag, -lineinfo,
  in-process image check + canary + self-check with sync and print after each kernel, never a
  child process). Full shapes keep stage-3 code, sliding decode keeps master. First measurement is
  an okc check, never a scored submit.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: replaced ~14.5 us of a 16.7 us shape (FA3 main 7.6 + combine 3.9 + prepare 1.7 + norm-gate), floor 0.1; one launch of 33 tile-CTAs with in-kernel merge. M1 tests pass; M2 shape <= 13 us; M3 <= 10 us
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: replaced ~16.5 of 22.7 us (FA3 10.7 + combine 4.1 + prepare + norm-gate), floor 1.9. M2 shape <= 17 us
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: replaced ~20 of 28.7 us (FA3 14.5 + combine 4.3 + prepare + norm-gate), floor 3.7. M2 shape <= 22 us
  - shapes: "*layer=full* *layer=sliding load=decode_*"
    direction: flat
    why: outside the new route; stage-3 and master code unchanged
risk: high
depends_on: full-verify-wgmma-s3-warpgroup-per-kvhead-inproc
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on sliding verify
review: required
status: queued
---

# full-verify-wgmma-s4-sliding-window

## Change

Stage 4 of the full-attention verify rewrite, built on the stage-3 branch hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc (splitkv_verify_wgmma_kernel, module muse_full_verify_wgmma_v1: stream-K over 64-token tiles, one consumer warpgroup per kv head = one wgmma M=64 tile of 8 heads x 8 tokens, fp8 QK and PV, in-kernel ticket combine + norm-gate), whatever its score. Sliding verify is 46x per step in three loads: 16.7/22.7/28.7 us at verify_1/16/32 (FA3 main 7.6/10.7/14.5 + combine 3.9/4.1/4.3 + prepare 1.7 + pre-attn + norm-gate) against floors 0.1/1.9/3.7. Change: route SLIDING verify (window != 0, q_len 8, 1 <= batch <= 64, hq 16, hkv 2, head_dim 64, page 1; config-only predicate) through the same kernel after master _pre_attention_kernel (ROPE on, PAIRED=False, unchanged). Kernel edit, one new int argument window (0 = full, which must compile and behave exactly as stage 3): per request key range [lo_b, seq_b) with lo_b = window ? max(0, seq_b - 8 - (window - 1)) : 0 computed in-kernel from cache_seqlens; tiles_b = ceil((seq_b - lo_b)/64), tile k covers keys lo_b + 64k .. +63 (page 1, so tiles need no alignment); the combine path recomputes the same TP prefix. Mask adds the window lower edge: key t valid for block token j iff t <= seq_b - 8 + j and t >= seq_b - 8 + j - (window - 1) and t >= lo_b, applied on any tile overlapping either edge (matches FA3 window_size=(window - 1, 0) as master passes it, so every key in each window is attended). Grid: at sliding verify Ttot is only ~33 tiles per request, so use grid = min(2 x SM count, Ttot rounded from config: batch * ceil((window + 7)/64)) so no CTA launches empty; workspace sized grid + B. Self-check adds sliding cases (window 2048, lens < window, < 8 + window, and 80000, batch 1/2/16/32, kv_scale 1 and 0.25) against an fp32 windowed causal reference by rel L2; on failure only the sliding route is disabled (PATH CHOSEN: sliding verify FA3 (fallback: <reason>)), else PATH CHOSEN: sliding verify wgmma window. BUILD/SAFETY as stage 3 (sm_90a only, __CUDA_ARCH_FEAT_SM90_ALL guard with device error flag, -lineinfo, in-process image check + canary + self-check with sync and print after each kernel, never a child process). Full shapes keep stage-3 code, sliding decode keeps master. First measurement is an okc check, never a scored submit.

## Expected

- `*layer=sliding load=verify_1 *` faster: replaced ~14.5 us of a 16.7 us shape (FA3 main 7.6 + combine 3.9 + prepare 1.7 + norm-gate), floor 0.1; one launch of 33 tile-CTAs with in-kernel merge. M1 tests pass; M2 shape <= 13 us; M3 <= 10 us
- `*layer=sliding load=verify_16*` faster: replaced ~16.5 of 22.7 us (FA3 10.7 + combine 4.1 + prepare + norm-gate), floor 1.9. M2 shape <= 17 us
- `*layer=sliding load=verify_32*` faster: replaced ~20 of 28.7 us (FA3 14.5 + combine 4.3 + prepare + norm-gate), floor 3.7. M2 shape <= 22 us
- `*layer=full* *layer=sliding load=decode_*` flat: outside the new route; stage-3 and master code unchanged

## Risk

High risk: depends on stage 3 reaching its A8 gate; if stage 3 lost on full verify, this still tests whether the kernel wins on the short, latency-bound sliding ranges where FA3 pays prepare + combine launches. Supersedes the queued primary sliding-verify-cuda-streamk-window, which targets the regressed stage-2 kernel.

## Milestones

Rewrite stage 4 of `prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on sliding verify`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
