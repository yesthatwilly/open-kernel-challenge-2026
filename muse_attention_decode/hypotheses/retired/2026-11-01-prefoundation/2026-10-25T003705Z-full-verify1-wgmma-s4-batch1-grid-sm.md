---
problem: muse_attention_decode
slug: full-verify1-wgmma-s4-batch1-grid-sm
created: 2026-10-25T003705Z
base: master
area: full-layer attention (batch-1 verify, wgmma stream-K kernel)
change: >
  Stage 4 of the full-attention verify rewrite, built on the stage-3 branch
  hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc
  (splitkv_verify_wgmma_kernel, stream-K 64-token tiles, warpgroup per kv head, fp8 wgmma QK/PV,
  in-kernel ticket combine + norm-gate), whatever its score. Full verify_1 is 30.0 us (FA3 main
  15.0 + combine 7.6 + prepare 2.7 + pre-attn + norm-gate) against a 4.3 us floor, weight 16 in
  the verify_1 load. Change, config only: widen _splitkv_verify_gate from 1 < batch <= 64 to 1 <=
  batch <= 64 so full verify at batch 1 runs master _pre_attention_kernel (PAIRED=False, as master
  batch 1 already is) + the stage-3 kernel; at batch == 1 use grid = SM count (one CTA per SM,
  ~9.5 tiles of 80000 keys each) instead of 2 x SM count, through the existing grid argument;
  batch > 1 unchanged. If stage 3 used the in-kernel ticket, the single request is merged by one
  CTA over 132 partials of 8x16x64 fp32 (~4.3 MB): make that merge two-level for batch 1 (group
  ticket per ceil(sqrt(nseg)) CTAs, the group last merges into a group slot, the last group merges
  the group slots and runs the norm-gate epilogue; both tickets reset to 0; group tickets are a
  module-level int32 zero buffer from import). If stage 3 kept B8, B8 does the merge unchanged.
  Self-check adds batch-1 cases at 80000, 4097, 70 and 9 keys, kv_scale 1 and 0.25, grids 3/7/SM
  count, two replays for ticket reset. BUILD/SAFETY as stage 3 (sm_90a only, feature-macro guard +
  device error flag, -lineinfo, in-process image check + canary + self-check, sync and print after
  each kernel, never a child process). Fallback: batch 1 returns to master FA3 (PATH CHOSEN: full
  verify_1 FA3 (fallback: <reason>)). Every other shape keeps stage-3/master code. First
  measurement is an okc check, never a scored submit.
expected:
  - shapes: "*layer=full load=verify_1 *"
    direction: faster
    why: "replaced ~27 us of a 30.0 us shape, floor 4.3: 20.5 MB at the stage-3 gate rate (>= 2.4 TB/s) is ~8.5 us plus a ~2 us two-level merge. M1 tests pass, shape < 30; M2 <= 17 us; M3 <= 13 us"
  - shapes: "*layer=full load=verify_16* *layer=full load=verify_32*"
    direction: flat
    why: batch > 1 keeps stage-3 grid and merge
  - shapes: "*load=decode_* *layer=sliding*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: full-verify-wgmma-s3-warpgroup-per-kvhead-inproc
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: prepare_varlen_num_blocks + unpaired FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on full verify batch 1
review: hold
status: hold
review_at: 2026-10-25T034754Z
review_companions: null
hold_reason: "adversarial review: Hold by this hypothesis's own condition ('worth running only once stage 3 shows A8 at or near its gate'). Stage 3's wgmma kernel failed its self-check (case 2, "
---

# full-verify1-wgmma-s4-batch1-grid-sm

## Change

Stage 4 of the full-attention verify rewrite, built on the stage-3 branch hyp/muse_attention_decode/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc (splitkv_verify_wgmma_kernel, stream-K 64-token tiles, warpgroup per kv head, fp8 wgmma QK/PV, in-kernel ticket combine + norm-gate), whatever its score. Full verify_1 is 30.0 us (FA3 main 15.0 + combine 7.6 + prepare 2.7 + pre-attn + norm-gate) against a 4.3 us floor, weight 16 in the verify_1 load. Change, config only: widen _splitkv_verify_gate from 1 < batch <= 64 to 1 <= batch <= 64 so full verify at batch 1 runs master _pre_attention_kernel (PAIRED=False, as master batch 1 already is) + the stage-3 kernel; at batch == 1 use grid = SM count (one CTA per SM, ~9.5 tiles of 80000 keys each) instead of 2 x SM count, through the existing grid argument; batch > 1 unchanged. If stage 3 used the in-kernel ticket, the single request is merged by one CTA over 132 partials of 8x16x64 fp32 (~4.3 MB): make that merge two-level for batch 1 (group ticket per ceil(sqrt(nseg)) CTAs, the group last merges into a group slot, the last group merges the group slots and runs the norm-gate epilogue; both tickets reset to 0; group tickets are a module-level int32 zero buffer from import). If stage 3 kept B8, B8 does the merge unchanged. Self-check adds batch-1 cases at 80000, 4097, 70 and 9 keys, kv_scale 1 and 0.25, grids 3/7/SM count, two replays for ticket reset. BUILD/SAFETY as stage 3 (sm_90a only, feature-macro guard + device error flag, -lineinfo, in-process image check + canary + self-check, sync and print after each kernel, never a child process). Fallback: batch 1 returns to master FA3 (PATH CHOSEN: full verify_1 FA3 (fallback: <reason>)). Every other shape keeps stage-3/master code. First measurement is an okc check, never a scored submit.

## Expected

- `*layer=full load=verify_1 *` faster: replaced ~27 us of a 30.0 us shape, floor 4.3: 20.5 MB at the stage-3 gate rate (>= 2.4 TB/s) is ~8.5 us plus a ~2 us two-level merge. M1 tests pass, shape < 30; M2 <= 17 us; M3 <= 13 us
- `*layer=full load=verify_16* *layer=full load=verify_32*` flat: batch > 1 keeps stage-3 grid and merge
- `*load=decode_* *layer=sliding*` flat: outside the gate

## Risk

Supersedes the queued primary full-verify1-cuda-streamk-batch1, which stacks on the regressed stage-2 A8 (1.44 TB/s). Worth running only once stage 3 shows A8 at or near its gate; if stage 3 is DEAD, retire this with it.

## Milestones

Rewrite stage 4 of `prepare_varlen_num_blocks + unpaired FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate on full verify batch 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T034754Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold by this hypothesis's own condition ('worth running only once stage 3 shows A8 at or near its gate'). Stage 3's wgmma kernel failed its self-check (case 2, rel L2 1.23 for ticket and B8) and never ran, so there is no gate evidence. Batch-1 full verify at grid = SM count puts about 9.5 tiles in each CTA, which is exactly the multi-tile regime that failed, so this run would fall back to master FA3 and measure nothing. Fix stage 3 first, then re-mint.

### Justification
Parent facts (archive/full-verify-wgmma-s3-warpgroup-per-kvhead-inproc/2c3f7fff72bb4e1d.log). Build and occupancy were good: 128 regs, 0 local, sm_90a, 2 CTAs/SM. The canary and case 1 passed, but in both every CTA had at most 1 tile (3 tiles over grid 4; 242 tiles over grid 264). Case 2 (lens_a, kv_scale 0.25, grid 3, about 81 tiles per CTA) failed at rel L2 1.23 in both ticket and B8 modes, so the fault is in the main kernel's multi-tile machinery (ring and mbarrier phase wrap, V^T double-buffer reuse, Q reload across requests, page-index rotation past 16 tiles), or else in kv_scale handling; the two are confounded. Fallback: 'full verify master FA3, paired heads'. The 1.380x is master re-measured: verify_16/32 at 162.61/301.91 us equal the incumbent's 162.7/301.3. This hypothesis's speed estimate rests on 'the stage-3 gate rate (>= 2.4 TB/s)', which was never measured; the only measured CUDA verify rate is stage 2's A8 at 1.44 TB/s. At batch 1 with 80000 keys (1250 tiles) and grid 132, each CTA runs about 9.5 tiles, and the specified self-check (80000/4097 keys, grids 3/7/SM) will hit the failing regime. Request crossing cannot occur at batch 1, so this would be a partial bisect if the bug is request crossing; it would still fail if the bug is ring phase, V^T reuse or kv_scale. The design itself is sound on its merits. Full verify_1 is 30.0 us (FA3 main 15.0 + combine 7.6 + prepare 2.7) against a 4.3 us floor, weight 16, and FINDINGS confirms it. One launch at grid = SM count, with a two-level ticket merge replacing a single-CTA 4.3 MB merge, is the right shape. sqrt-grouping (about 12 groups of 11) brings the serial merge down to about 11 x 32 KB per level. WEAKEST LINK (after the fix): 20.5 MB in about 8.5 us needs 2.4 TB/s from one request on 132 CTAs at 1 CTA/SM. That is half the resident CTAs the kernel was tuned for (2 per SM), so the bytes in flight per SM halve. A 4-stage ring of 16 KB tiles gives about 64 KB per SM, which is marginal for 2.4 TB/s at around 1 us of latency. grid = 2 x SM, with 4.7 tiles per CTA, may be faster at batch 1. Make the grid a prepare-time sweep (SM, 2 x SM) rather than fixing SM count.

### Confounders on the current kernel
- full-verify-wgmma-s3-warpgroup-per-kvhead-inproc (parent): it fell back, so the 'stage-3 kernel' this widens has no timing and fails correctness in the regime batch 1 needs.
- paired-heads-launch-cleanup-ab-tune-batch-gate: master runs full verify batch 1 UNPAIRED (paired requires batch > 1), so widening the gate to batch 1 keeps the q layout. On fallback, batch 1 must stay on master's unpaired FA3 path (as the spec says), not on a paired one.
- full-decode-cuda-gather-splitkv / full-decode1-streamk-s3-two-level-ticket (not landed): the same two-level ticket idea for decode batch 1 reached 1.432x without promotion. Reusing its group-ticket code is fine, but its buffers must be separate per module.

### Interference and A/B needs
No companion: the result is read from the replaced kernel's own profile row, and the obstacle is that the run would measure the fallback, which no A/B fixes. The prerequisite is a stage-3 correctness fix (the same one proposed for the sibling full-verify-wgmma-s4-sliding-window). Its self-check bisects case 2: (lens_a, 1.0, grid 3) vs (lens_a, 0.25, grid 264); a single request at N = 2, 4 and 17 tiles per CTA; and the device error word printed. Once that passes and full verify_16/32 show the kernel's own time, re-mint this unchanged, with the grid as a prepare-time sweep of {SM, 2 x SM} at batch 1.

### Open questions
- Is the stage-3 failure from multiple tiles per CTA or from kv_scale 0.25? A batch-1 run only exercises the first, without request crossing.
- At batch 1, does grid = SM count (1 CTA/SM, about 64 KB in flight per SM) actually beat 2 x SM? Sweep both in prepare().
- What does the two-level merge's tail actually cost? Stamp it with clock64 in the self-check; the spec's ~2 us is an estimate.

