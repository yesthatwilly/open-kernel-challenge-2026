---
problem: muse_attention_decode
slug: full-verify-wgmma-s4-multitile-fix
created: 2026-10-25T035310Z
base: master
area: full-layer attention (verify, wgmma stream-K kernel, multi-tile correctness)
change: >
  Stage 4 (correctness fix) of the full-verify wgmma rewrite, built on the stage-3 commit
  (hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc,
  splitkv_verify_wgmma_kernel<ticket>/<B8>, module muse_full_verify_wgmma_v1) whatever its score;
  rename the module (v2) so it recompiles. Stage 3 built clean (sm_90a, 128/120 regs, 0 local, 2
  CTAs/SM, 91744 B smem) and passed the canary (B=2, grid 4) and case 1 (B=6 lens_a, kv 1.0,
  grid=device) at rel L2 <= 7.9e-3, but failed case 2 (B=6 lens_a, kv_scale 0.25, grid 3) at rel
  L2 1.23, finite, in BOTH ticket and B8 combine, so PATH CHOSEN fell back to master FA3 and the
  kernel was never timed. Every passing launch gave each CTA at most 1 tile; case 2 is the first
  with several (~81) tiles per CTA. (1) BISECT in the prepare self-check, each case after its own
  synchronize + flushed print of rel L2, finiteness and the DEVICE ERROR WORD (bounded mbarrier
  wait timeout sets it; stage 3 never printed it): split case 2 into (lens_a, kv_scale 1.0, grid
  3) and (lens_a, kv_scale 0.25, grid 264); add single-request (B=1, no request crossing) cases
  sized to give exactly 2, 4 and 17 tiles per CTA at kv 1.0 (e.g. grid 1 with 128/256/1088 keys,
  or grid 4 with 8/16/68 tiles); keep the canary, case 1 and original case 2. This one okc check
  must name which variable breaks: multi-tile loop state, request crossing, IDX_TILES page-index
  rotation past 16 tiles, or kv_scale. (2) FIX the most likely cause, the per-CTA multi-tile loop
  state: (a) online-softmax carry across tiles: m and l carried per row in registers, acc (wgmma
  PV accumulator) rescaled by exp2(m_old - m_new) BEFORE the next PV wgmma is issued, after
  wgmma.wait_group drains the previous one, and l rescaled with it; reset m=-inf, l=0, acc=0 only
  at a segment/request boundary, never per tile; (b) the K/V smem ring phase parity: full/empty
  mbarrier phase bit flips each time the slot index wraps the stage count, tracked per slot across
  tiles and segments (not reset per tile or per request), producer and consumer agree on the same
  (iteration / STAGES) & 1; the V^T double buffer index advances per tile, not per segment; (c)
  the FP8 P operand uses FA3 Max_offset across tiles: P = exp2(s*scale_log2 - (m_new - 8)) with
  the same offset folded into the l and acc rescale, so the 2^8 cancels in O/l and in the partial
  merge (LESSONS 2026-10-25); (d) kv_scale is applied once per tile (to s before the max and to
  the PV output), not accumulated into the carry. Q reload on request crossing and page-index
  rotation stay as they are unless the bisect names them. The self-check must still pass all cases
  for the kernel to be chosen (PATH CHOSEN: full verify wgmma v2 vs master FA3 (fallback: <case,
  rel L2, error word>)); when it passes, print a CUDA-event time per kernel at B=16 and 32 vs
  master paired FA3 main+combine+norm-gate. BUILD/SAFETY as stage 3: TORCH_CUDA_ARCH_LIST=9.0a /
  -gencode arch=compute_90a,code=sm_90a only, __CUDA_ARCH_FEAT_SM90_ALL guard, -lineinfo,
  in-process canary, never a child process. Test FP8-P on coherent long-context keys, not only
  random keys. First measurement is an okc check, never a scored submit. Do not touch sliding
  verify or the decode paths (the held s4 sliding-window and verify1-batch1 stages are re-minted
  after this passes).
expected:
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: "M1: every self-check case passes and PATH CHOSEN = wgmma; M2: replaced FA3 main+combine+norm-gate below master 162.0 us shape (FA3 paired); stage-2 A8 ran at 1.44 TB/s, wgmma target >= 2.4 TB/s"
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: "M2: below master 300.8 us shape"
  - shapes: "*load=decode_* *layer=sliding* *layer=full load=verify_1 *"
    direction: flat
    why: outside the 1<batch<=64 full verify gate; master code
risk: high
depends_on: full-verify-wgmma-s3-warpgroup-per-kvhead-inproc
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: "full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)"
review: required
status: queued
---

# full-verify-wgmma-s4-multitile-fix

## Change

Stage 4 (correctness fix) of the full-verify wgmma rewrite, built on the stage-3 commit (hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc, splitkv_verify_wgmma_kernel<ticket>/<B8>, module muse_full_verify_wgmma_v1) whatever its score; rename the module (v2) so it recompiles. Stage 3 built clean (sm_90a, 128/120 regs, 0 local, 2 CTAs/SM, 91744 B smem) and passed the canary (B=2, grid 4) and case 1 (B=6 lens_a, kv 1.0, grid=device) at rel L2 <= 7.9e-3, but failed case 2 (B=6 lens_a, kv_scale 0.25, grid 3) at rel L2 1.23, finite, in BOTH ticket and B8 combine, so PATH CHOSEN fell back to master FA3 and the kernel was never timed. Every passing launch gave each CTA at most 1 tile; case 2 is the first with several (~81) tiles per CTA. (1) BISECT in the prepare self-check, each case after its own synchronize + flushed print of rel L2, finiteness and the DEVICE ERROR WORD (bounded mbarrier wait timeout sets it; stage 3 never printed it): split case 2 into (lens_a, kv_scale 1.0, grid 3) and (lens_a, kv_scale 0.25, grid 264); add single-request (B=1, no request crossing) cases sized to give exactly 2, 4 and 17 tiles per CTA at kv 1.0 (e.g. grid 1 with 128/256/1088 keys, or grid 4 with 8/16/68 tiles); keep the canary, case 1 and original case 2. This one okc check must name which variable breaks: multi-tile loop state, request crossing, IDX_TILES page-index rotation past 16 tiles, or kv_scale. (2) FIX the most likely cause, the per-CTA multi-tile loop state: (a) online-softmax carry across tiles: m and l carried per row in registers, acc (wgmma PV accumulator) rescaled by exp2(m_old - m_new) BEFORE the next PV wgmma is issued, after wgmma.wait_group drains the previous one, and l rescaled with it; reset m=-inf, l=0, acc=0 only at a segment/request boundary, never per tile; (b) the K/V smem ring phase parity: full/empty mbarrier phase bit flips each time the slot index wraps the stage count, tracked per slot across tiles and segments (not reset per tile or per request), producer and consumer agree on the same (iteration / STAGES) & 1; the V^T double buffer index advances per tile, not per segment; (c) the FP8 P operand uses FA3 Max_offset across tiles: P = exp2(s*scale_log2 - (m_new - 8)) with the same offset folded into the l and acc rescale, so the 2^8 cancels in O/l and in the partial merge (LESSONS 2026-10-25); (d) kv_scale is applied once per tile (to s before the max and to the PV output), not accumulated into the carry. Q reload on request crossing and page-index rotation stay as they are unless the bisect names them. The self-check must still pass all cases for the kernel to be chosen (PATH CHOSEN: full verify wgmma v2 vs master FA3 (fallback: <case, rel L2, error word>)); when it passes, print a CUDA-event time per kernel at B=16 and 32 vs master paired FA3 main+combine+norm-gate. BUILD/SAFETY as stage 3: TORCH_CUDA_ARCH_LIST=9.0a / -gencode arch=compute_90a,code=sm_90a only, __CUDA_ARCH_FEAT_SM90_ALL guard, -lineinfo, in-process canary, never a child process. Test FP8-P on coherent long-context keys, not only random keys. First measurement is an okc check, never a scored submit. Do not touch sliding verify or the decode paths (the held s4 sliding-window and verify1-batch1 stages are re-minted after this passes).

## Expected

- `*layer=full load=verify_16*` faster: M1: every self-check case passes and PATH CHOSEN = wgmma; M2: replaced FA3 main+combine+norm-gate below master 162.0 us shape (FA3 paired); stage-2 A8 ran at 1.44 TB/s, wgmma target >= 2.4 TB/s
- `*layer=full load=verify_32*` faster: M2: below master 300.8 us shape
- `*load=decode_* *layer=sliding* *layer=full load=verify_1 *` flat: outside the 1<batch<=64 full verify gate; master code

## Risk

If the bisect shows a fault outside the multi-tile loop state (request crossing, IDX_TILES rotation, kv_scale), the fix attempt may not land but the check still names the cause for stage 5. Held s4 siblings (sliding-window, verify1-batch1-grid-sm) wait on this.

## Milestones

Rewrite stage 4 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + norm-gate (master paired FA3)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
