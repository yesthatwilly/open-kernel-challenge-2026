---
problem: muse_moe_w4a16
slug: cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc
created: 2026-10-25T001824Z
base: master
area: whole-layer fusion
change: >
  Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at
  muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec
  done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto
  current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs
  master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41
  in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART A ONLY (Part B, the
  bandwidth work, is split out to stage 3): kill the slow mode first: (1) in the self-check only,
  record per-CTA entry globaltimer and %smid, and globaltimer at phase-1 start/end and phase-2
  start, and the phase-2 spin-iteration count, into a debug buffer and print min/median/max per
  phase; (2) replace __nanosleep(32) in the phase-2 ld.acquire spin on the w13 chunk counters with
  a tight acquire poll, bounded (trap-free: after a large bound set an error flag and fall back);
  (3) remove the second-wave dependency: pad/split phase-1 w13 work so it is exactly one wave
  (reviewer: prefer ceil(128R/132) whole tiles per CTA over a K-split, which would need a pre-silu
  reduction) and order phase-2 (route, k-split) items so each is taken by CTAs whose phase-1
  chunks finish first. Self-check additionally runs 100 graph replays with a >= 2x L2-size buffer
  written between replays (cold L2, as the judge does) and asserts the replay distribution is
  unimodal (p90/p10 < 1.3). Run the self-check's first launches IN-PROCESS (never in a child
  process: the rules statically reject submissions that start processes and okc_lint flags
  multiprocessing (LESSONS 2026-10-25, corrected); stage 1 already ran this kernel in-process on
  the judge): confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() ==
  (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes
  binaryVersion/ptxVersion for moe_bs1_kernel; anything else turns the path OFF), then a tiny
  synthetic canary launch of moe_bs1_kernel (R=1), each followed by torch.cuda.synchronize() and a
  print so a fault names its kernel; then keep plain cp.async.cg (no createpolicy), build with
  -lineinfo. No bandwidth change in this stage. The first measurement of this hypothesis must be
  an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: bs=1 *
    direction: faster
    why: "incumbent replaced kernels 51.2 us (scored 48.4), floor 11.4 us (~55 MB). M1: passes and replays unimodal; M2: removing the slow mode alone lands ~42 us scored (~40-47 us fast mode). Read against master at dispatch. M3 (<= 18 us) is stage 3's bandwidth work"
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
depends_on: cuda-fused-moe-bs1-persistent-w13-silu-w2
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1
review: required
status: queued
---

# cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc

## Change

Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41 in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART A ONLY (Part B, the bandwidth work, is split out to stage 3): kill the slow mode first: (1) in the self-check only, record per-CTA entry globaltimer and %smid, and globaltimer at phase-1 start/end and phase-2 start, and the phase-2 spin-iteration count, into a debug buffer and print min/median/max per phase; (2) replace __nanosleep(32) in the phase-2 ld.acquire spin on the w13 chunk counters with a tight acquire poll, bounded (trap-free: after a large bound set an error flag and fall back); (3) remove the second-wave dependency: pad/split phase-1 w13 work so it is exactly one wave (reviewer: prefer ceil(128R/132) whole tiles per CTA over a K-split, which would need a pre-silu reduction) and order phase-2 (route, k-split) items so each is taken by CTAs whose phase-1 chunks finish first. Self-check additionally runs 100 graph replays with a >= 2x L2-size buffer written between replays (cold L2, as the judge does) and asserts the replay distribution is unimodal (p90/p10 < 1.3). Run the self-check's first launches IN-PROCESS (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected); stage 1 already ran this kernel in-process on the judge): confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for moe_bs1_kernel; anything else turns the path OFF), then a tiny synthetic canary launch of moe_bs1_kernel (R=1), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; then keep plain cp.async.cg (no createpolicy), build with -lineinfo. No bandwidth change in this stage. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `bs=1 *` faster: incumbent replaced kernels 51.2 us (scored 48.4), floor 11.4 us (~55 MB). M1: passes and replays unimodal; M2: removing the slow mode alone lands ~42 us scored (~40-47 us fast mode). Read against master at dispatch. M3 (<= 18 us) is stage 3's bandwidth work
- `bs=8 *` flat: control: master
- `bs=16 *` flat: control: master
- `bs=32 *` flat: control: master
- `bs=128 *` flat: control: untouched
- `bs=256 *` flat: control: untouched

## Risk

supersedes cuda-fused-moe-bs1-persistent-nospin-bandwidth, held for a child-process self-check; Part A only (stage 2), Part B split to stage 3 per review

## Milestones

Rewrite stage 2 of `_align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
