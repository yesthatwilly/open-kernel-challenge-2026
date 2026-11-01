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
review: hold
status: hold
review_at: 2026-10-25T034754Z
review_companions: null
hold_reason: "adversarial review: Sequencing hold, not a rejection of the design. The stage-3 kernel this builds on never ran in the timed call. It failed its own self-check (case 2: B=6, kv_sca"
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

## Adversarial review

Reviewed 2026-10-25T034754Z by an adversarial subagent (model opus).

**Verdict: hold.** Sequencing hold, not a rejection of the design. The stage-3 kernel this builds on never ran in the timed call. It failed its own self-check (case 2: B=6, kv_scale 0.25, grid 3, rel L2 1.23 for both ticket and B8), so 1.380x is master's paired FA3 re-measured. This stage needs the regime that failed (several tiles per CTA, at batch 16/32), and its self-check covers batch 16/32, so it would fall back too and measure nothing. Fix stage 3's multi-tile bug first (as stage 4), then re-mint this as stage 5.

### Justification
Parent evidence (archive/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc/2c3f7fff72bb4e1d.log). The sm_90a build was good: 128 regs, 0 local bytes, binaryVersion 90, 2 CTAs/SM, 91.7 KB smem. The ticket canary ([70,13], grid 4) passed, and so did case 1 (lens_a, kv_scale 1.0, grid 264), at 4.8e-3. Case 2 (lens_a, kv_scale 0.25, grid 3) returned rel L2 1.23 for both the ticket and the B8 variants, and the log then shows 'PATH CHOSEN: full verify master FA3, paired heads (fallback ...)'. The attribution table agrees: full verify_16 162.61 us and verify_32 301.91 us equal master's incumbent (162.7/301.3), and those 'gains' are only against the stage-2 base. The wgmma kernel has no timing anywhere, so M2/M3 have no evidence behind them. What the passing cases did NOT exercise: lens_a has 242 tiles at grid 264, and the canary has 3 tiles at grid 4, so every passing launch gave each CTA at most 1 tile. Case 2 is the first launch with about 81 tiles per CTA. That covers ring wrap-around and mbarrier phase parity, the double-buffered V^T reuse, segments that cross request boundaries (Q reload), and page-index block rotation past IDX_TILES=16. B8 fails identically, so the fault is in the main kernel, not the ticket merge. Case 2 also changed kv_scale to 0.25, so the cause is confounded between multi-tile and scale, but rel L2 1.23 looks like garbage, not a scale error (a missing or doubled 0.25 gives about 0.75/0.94/3). How this lands on the target shapes: sliding verify_1 has about 33 tiles with grid 33, so 1 tile per CTA, which is the regime that passed. verify_16 (528 tiles, grid 264) and verify_32 (1056 tiles) need 2 and 4 tiles per CTA, which is the regime that failed. The specified self-check includes batch 16/32 and turns off the whole sliding route on any failure, so the run would almost certainly print 'sliding verify FA3 (fallback)' and re-measure master. Design on its own merits: the incumbent numbers (16.7/22.7/28.7 us; FA3 main 7.6/10.7/14.5 + combine 3.9/4.1/4.3 + prepare 1.7) match FINDINGS. The floors are tiny (0.1/1.9/3.7), so the shapes are latency- and launch-bound, and folding prepare + combine + norm-gate into one launch is the right lever. The windowed key range [lo_b, seq_b) with in-kernel lo_b, plus the extra lower-edge mask, correctly reproduces FA3 window_size=(window-1, 0). WEAKEST LINK, even after the fix: at verify_1 the ticket's last CTA LSE-merges 33 partials of 8x16x64 fp32 (32 KB each, about 1 MB) by itself. One SM streams that at roughly 0.1-0.2 TB/s, so 5-10 us of serial tail, which alone threatens M2 (<= 13 us) and rules out M3 (<= 10 us). The batch-1 sibling solves the same problem with a two-level merge. Here the cheaper fix is fewer, longer segments at batch 1 (e.g. grid 8-11, 3-4 tiles each), but that again needs the broken multi-tile path. Expected directions are consistent with TASK.md (sliding verify weight 46 per load). The 'full flat' expectation holds only because full verify is on the fallback.

### Confounders on the current kernel
- full-verify-wgmma-s3-warpgroup-per-kvhead-inproc (parent, not landed): its kernel failed self-check and never ran, so this stage would build on unmeasured and incorrect code. Any result would be read against a parent with no timing.
- sliding-deep-splits-on-cuda-splitkv / sliding-fused-triton-decode1-gate: these own sliding decode only and are untouched. Sliding decode stays on master code, and the predicate's q_len 8 keeps them out, so they do not confound.
- fa3-hdim128-head-pairing + paired-heads-launch-cleanup: sliding verify is unpaired on master (PAIRED=False), so routing it to the new kernel changes no pre-attention layout. Full verify_16/32 stay on the paired FA3 fallback while the stage-3 kernel is broken, so 'full flat' is master's code, not stage 3's.
- fused-preattention-triton: provides the e4m3 q in the unpaired [B*8,16,64] layout that the A operand reads. That layout is the same as for full verify, so it does not confound.

### Interference and A/B needs
No companion would help: the problem is not attribution but that the measured path would be the fallback. The right follow-up is a correctness stage on the parent: 'full-verify-wgmma-s4-multitile-fix', built on the stage-3 branch. Its self-check should separate case 2 into (lens_a, 1.0, grid 3), (lens_a, 0.25, grid 264), and single-request multi-tile cases with N = 2, 4 and 17 tiles per CTA, printing per-case rel L2 and the device error word (wait timeouts set it). That locates the fault (multi-tile ring/phase, V^T reuse, request crossing, meta rotation, or kv_scale) in one okc check. Once stage 3 passes and is timed on full verify_16/32, re-mint this as the next stage. Add a batch-1 segment-length choice (grid of about 8-11 at sliding verify_1) or a two-level merge, so the single-CTA 1 MB merge does not dominate.

### Open questions
- Which variable broke case 2: about 81 tiles per CTA (ring phase / V^T double buffer / request crossing / IDX_TILES rotation) or kv_scale 0.25? Every passing launch had at most 1 tile per CTA.
- Did the device error word (bounded mbarrier wait timeout) get set in case 2? The log does not print it.
- At sliding verify_1, what does the single-CTA merge of 33 x 32 KB partials cost? Stamp it with clock64 in the self-check before committing to grid = Ttot.
- If the orchestrator would rather not add a stage: proceed only if the implementer first reproduces and fixes case 2 on this branch and the self-check passes at batch 16/32. Otherwise retire this run before it spends a check.

