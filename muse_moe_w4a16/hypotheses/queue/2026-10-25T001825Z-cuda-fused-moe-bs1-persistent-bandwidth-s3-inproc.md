---
problem: muse_moe_w4a16
slug: cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc
created: 2026-10-25T001825Z
base: master
area: whole-layer fusion
change: >
  Stage 3 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at
  muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec
  done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto
  current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs
  master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41
  in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART B of the held
  cuda-fused-moe-bs1-persistent-nospin-bandwidth (split per its review), on top of stage 2
  cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc (Part A, slow mode removed): raise
  bandwidth from ~1.5 TB/s toward 3 TB/s with a deeper cp.async ring and phase-1 K-split when R is
  1-2, overlapping the ~27-op/k16 dequant with the next stage's loads. Keep dequant, weight
  layout, routing and numerics unchanged. Keep stage 2's one-wave sizing and tight poll, plain
  cp.async.cg and -lineinfo; first launches in-process (never in a child process: the rules
  statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS
  2026-10-25, corrected)) with the sm_90a image check and a synthetic canary followed by
  torch.cuda.synchronize() and a print. Run only after stage 2's log shows which phase stalled and
  the replay distribution is unimodal. The first measurement of this hypothesis must be an `okc
  check`, never a scored submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: bs=1 *
    direction: faster
    why: "stage 2 fast mode ~40-47 us at ~1.5 TB/s, floor 11.4 us (~55 MB). M2: below stage 2's bs=1 row; M3 <= 18 us (~3 TB/s) via deeper cp.async ring + phase-1 K-split at R 1-2"
  - shapes: bs=8 *
    direction: flat
    why: "control: master"
  - shapes: bs=16 *
    direction: flat
    why: "control: master"
  - shapes: bs=32 *
    direction: flat
    why: "control: master"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
  - shapes: bs=256 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: _align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1
review: required
status: queued
---

# cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc

## Change

Stage 3 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41 in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART B of the held cuda-fused-moe-bs1-persistent-nospin-bandwidth (split per its review), on top of stage 2 cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc (Part A, slow mode removed): raise bandwidth from ~1.5 TB/s toward 3 TB/s with a deeper cp.async ring and phase-1 K-split when R is 1-2, overlapping the ~27-op/k16 dequant with the next stage's loads. Keep dequant, weight layout, routing and numerics unchanged. Keep stage 2's one-wave sizing and tight poll, plain cp.async.cg and -lineinfo; first launches in-process (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)) with the sm_90a image check and a synthetic canary followed by torch.cuda.synchronize() and a print. Run only after stage 2's log shows which phase stalled and the replay distribution is unimodal. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `bs=1 *` faster: stage 2 fast mode ~40-47 us at ~1.5 TB/s, floor 11.4 us (~55 MB). M2: below stage 2's bs=1 row; M3 <= 18 us (~3 TB/s) via deeper cp.async ring + phase-1 K-split at R 1-2
- `bs=8 *` flat: control: master
- `bs=16 *` flat: control: master
- `bs=32 *` flat: control: master
- `bs=128 *` flat: control: untouched
- `bs=256 *` flat: control: untouched

## Risk

split from cuda-fused-moe-bs1-persistent-nospin-bandwidth (Part B), held for a child-process self-check; stage 3 after cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc

## Milestones

Rewrite stage 3 of `_align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
