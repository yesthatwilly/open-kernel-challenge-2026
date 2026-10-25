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
review: proceed
status: queued
review_at: 2026-10-25T040253Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T040253Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound rewrite stage: the bisect cases separate the variables that case 2 confounded, so one check names the faulty variable whether or not the fix lands. However, the four fixes the change lists, (a) to (d), are already implemented in the stage-3 kernel, so the bisect output, not a blind rewrite of the loop, has to drive the fix.

### Justification
Diagnosis: case 2 (B=6, kv 0.25, grid 3, ~81 tiles per CTA) failed at rel L2 1.23, finite, with both combines. Case 1 (grid=device, at most 1 tile per CTA) passed. That places the bug in the main kernel and in something that only happens when a CTA runs several tiles or crosses requests. The harvest supports this, and the bisect plan is the right one: kv 0.25 alone; multi-tile with B=1 at 2/4/17 tiles, so no request crossing and the 17-tile case covers IDX_TILES rotation; plus the original case. WEAKEST LINK: I read splitkv_verify_wgmma_kernel on the s3 branch and the suspects (a) to (d) are already coded as the spec asks. m/l/oacc reset only at segment start (n==0 || lt==0). oacc is rescaled after wg_wait0 of the previous PV, i.e. before the next PV is issued. The full parity is (n/STAGES)&1 and the empty-wait parity ((n-1)/STAGES)&1 is correct for the refill slot. V^T uses buffer n&1 per tile. The FA3 Max_offset (P = 2^(s - (u - 8))) is present, with l and O both carrying the 2^8. k_scale is folded into the per-tile score scale and v_scale is applied in the merge. So the 'most likely cause' named in step (2) is probably not the cause. What I could not rule out by reading: (i) the bounded-wait abort, where flag_s[0] makes every later wait return at once and produces finite garbage of exactly this kind (rel L2 ~1); stage 3 never printed the error word; (ii) the page-index prefetch and request crossing inside a 16-tile meta block (issue_meta uses req_of per tile, which looks right); (iii) start_of/req_of/slot = bx + cb bookkeeping when a CTA spans several requests (the slot mapping is collision-free by my reading); (iv) kv_scale 0.25 with grid=device, which was never run. The milestones are physically plausible. verify is MMA-heavy (64 query rows per kv head = one m64 wgmma), wgmma is the right tensor-core path where mma.sync lost 1.7x (LESSONS 155), and the 2.4 TB/s target is ~1.7x stage 2's 1.44 TB/s. The expected direction (full verify_16/32 faster, everything else flat) matches the gate.

### Confounders on the current kernel
- full-decode-cuda-gather-splitkv / sliding-deep-splits-on-cuda-splitkv (master): the s3 TU shares the pre-attention Triton kernel and the B8 combine lineage with the decode split-KV path. A passing self-check only clears the verify kernel. Any decode-row move in this check is master drift (full verify_1 and sliding verify_1 drifted +2-6% in the 03:41Z batch).
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup (master): the incumbent being replaced is paired FA3 main + combine + norm-gate. M2 must compare the wgmma kernel against those three rows in the same check, not against the headline.
- full-decode1-streamk-s3-two-level-ticket (land candidate, not on master): untouched by this stage, but if it lands first the verify TU must be rebased (both add load_inline modules to the same file). No measurement confound.

### Interference and A/B needs
No companion. This is a rewrite stage whose deliverable is a named cause (M1) and, if the self-check passes, the kernel's own profile rows against FA3 (M2). The B=1 multi-tile cases and the split of case 2 are the in-run controls. A separate ab-revert run would add nothing that the per-case prints cannot. The next stage is the follow-up.

### Open questions
- Did case 2 trip the bounded-wait abort (error word 2)? Finite garbage at rel L2 ~1 is exactly what the abort path produces. If so, the bug is a real barrier deadlock or a too-small WAIT_LIMIT at 81 tiles per CTA, not softmax state.
- BUILDER: make the bisect name the failing segment, not just the failing case. For the failing case, copy the per-slot partials (ws_acc O, ws_ml m/l) to host and compare each slot against a torch fp32 reference of that exact (CTA range, request) segment. Print the first bad slot with its CTA, request, tile count, whether it follows a request crossing, and whether its tile index is past 16. Also print the per-case error word. This turns one check into a line-level diagnosis even if no (2) fix lands.
- BUILDER: do not rewrite items (a) to (d) blindly. They read correct on the s3 branch (reset at lt==0, rescale after wg_wait0, Max_offset, parity). Change the loop only where the bisect points, and keep any speculative edit behind a constexpr so the s3 path can be A/B'd inside the self-check.
- When the self-check passes, choose wgmma regardless of the prepare timing (print both). This is a check, and M2 needs the kernel's profile rows. A timing hard gate would hide it, and prepare under-reads the judge by 20-25%.

