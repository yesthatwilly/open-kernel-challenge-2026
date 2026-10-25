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
review: required
status: queued
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
