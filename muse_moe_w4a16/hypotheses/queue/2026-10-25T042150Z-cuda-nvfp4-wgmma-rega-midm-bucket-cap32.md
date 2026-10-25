---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-midm-bucket-cap32
created: 2026-10-25T042150Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Mid-M branch of the wgmma register-A grouped NVFP4 rewrite, using ONLY the token buckets stage 2
  proved correct. Stage 2 (job 583a73418fbf4d01) showed the N16 and N32 bucket instantiations
  correct for both kernels (w13+silu rel 6.3e-3/6.5e-3, w2 3.4e-3/3.5e-3) and only N64/N128
  broken; s3 (running) fixes those for T >= 128. This hypothesis does not touch N64/N128 and does
  not change T >= 128. Base: the stage-2 commit (rewrite-on-rewrite), then rebase onto current
  master (9080a71, which includes w2-marlin-atomic-add-small-m) so every fallback and untouched
  shape is master byte for byte; rename the load_inline module so it recompiles. Build exactly as
  stage 2: TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo,
  wgmma bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process,
  Marlin fallback. Change: (1) add a second gate with its own bound constants WG_MIDM_MIN_TOKENS =
  8 and WG_MIDM_MAX_TOKENS = 32 (verify_1 bs=8, decode_16, decode_32; bs=1, the bs=4 test and the
  bs=33 test stay master Marlin); (2) in the device-side work-item builder (wg_gather / bucket
  split), cap every item at 32 rows: an expert with cnt routes becomes ceil(cnt/32) items of <= 32
  rows, each assigned the N16 bucket if its rows <= 16 else N32 - computed on device from the
  align counts, no host branching on values, buffers sized from T (capacity T*8 + 32*(32-1) rows)
  so it is graph-capturable; an expert split into several items re-reads its weights (from L2)
  once per item; (3) the w13+silu and w2 kernels, tiling (BK 128, 4 stages, grid 132 x 384, 1
  producer + 2 consumer warpgroups), dequant and epilogues are unchanged; the post-expert
  norm+combine stays master's _post_norm_combine_kernel (T < 32) /
  fused_post_expert_norm_weighted_sum (T = 32) fed with y in master's layout. Hard gate in
  prepare(): after the correctness self-check passes, time the new path against master's path on
  the same synthetic routings (skewed and random, cold L2 via a >L2 buffer read, CUDA graph
  replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T bucket; enable a bucket only if
  its p50 beats master by >= 3% and assert the enabled choice equals the printed winner (LESSONS
  2026-10-25 selector bug). Print one PATH CHOSEN line per T bucket with both timings. The
  self-check must FORCE each instantiation it can use (one-expert, all-local, skewed and random
  routings at each T) and judge rel-L2 against an fp32 dequant reference with master's own error
  printed beside it (bound max(2x master's error, 1e-2)), not against Marlin alone. A log without
  PATH CHOSEN = WGMMA for a bucket means that bucket is unmeasured. First measurement is an okc
  check, never a submit. T buckets for the gate: 8, 16, 32. Motivation: in the best profile
  Marlin's pair is 94.6-114 us at bs=8 (floor 37.3), 170-176 at bs=16 (floor 52.5), 156-207 at
  bs=32 (floor 70.8) - 2.5-3x off the floor, and stage-2 prepare timing put the wgmma pair at
  0.57x Marlin at M=128/256; the persistent grid streams each touched expert once regardless of
  route count, which Marlin's m-block scheme lacks.
expected:
  - shapes: bs=8 *
    direction: faster
    why: Marlin pair ~95-114 + act 2.0 us vs floor 37.3; M1 PATH CHOSEN = WGMMA at T=8 with tests 5/5; M2 wgmma pair below Marlin pair; M3 <= 56 us
  - shapes: bs=16 *
    direction: faster
    why: Marlin pair ~170-176 + 2.2 us vs floor 52.5; M3 <= 79 us
  - shapes: bs=32 *
    direction: faster
    why: "Marlin pair ~156-207 + 2.3 us vs floor 70.8; decode_32 is master's weakest load (~1.03x); M3 <= 106 us"
  - shapes: bs=1 *
    direction: flat
    why: "control: below the 8-token gate, master Marlin with w2 atomic"
  - shapes: bs=128 *
    direction: flat
    why: "control: T >= 128 is master (s3 owns it)"
  - shapes: bs=256 *
    direction: flat
    why: control
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 8 <= num_tokens <= 32
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-midm-bucket-cap32

## Change

Mid-M branch of the wgmma register-A grouped NVFP4 rewrite, using ONLY the token buckets stage 2 proved correct. Stage 2 (job 583a73418fbf4d01) showed the N16 and N32 bucket instantiations correct for both kernels (w13+silu rel 6.3e-3/6.5e-3, w2 3.4e-3/3.5e-3) and only N64/N128 broken; s3 (running) fixes those for T >= 128. This hypothesis does not touch N64/N128 and does not change T >= 128. Base: the stage-2 commit (rewrite-on-rewrite), then rebase onto current master (9080a71, which includes w2-marlin-atomic-add-small-m) so every fallback and untouched shape is master byte for byte; rename the load_inline module so it recompiles. Build exactly as stage 2: TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo, wgmma bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process, Marlin fallback. Change: (1) add a second gate with its own bound constants WG_MIDM_MIN_TOKENS = 8 and WG_MIDM_MAX_TOKENS = 32 (verify_1 bs=8, decode_16, decode_32; bs=1, the bs=4 test and the bs=33 test stay master Marlin); (2) in the device-side work-item builder (wg_gather / bucket split), cap every item at 32 rows: an expert with cnt routes becomes ceil(cnt/32) items of <= 32 rows, each assigned the N16 bucket if its rows <= 16 else N32 - computed on device from the align counts, no host branching on values, buffers sized from T (capacity T*8 + 32*(32-1) rows) so it is graph-capturable; an expert split into several items re-reads its weights (from L2) once per item; (3) the w13+silu and w2 kernels, tiling (BK 128, 4 stages, grid 132 x 384, 1 producer + 2 consumer warpgroups), dequant and epilogues are unchanged; the post-expert norm+combine stays master's _post_norm_combine_kernel (T < 32) / fused_post_expert_norm_weighted_sum (T = 32) fed with y in master's layout. Hard gate in prepare(): after the correctness self-check passes, time the new path against master's path on the same synthetic routings (skewed and random, cold L2 via a >L2 buffer read, CUDA graph replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T bucket; enable a bucket only if its p50 beats master by >= 3% and assert the enabled choice equals the printed winner (LESSONS 2026-10-25 selector bug). Print one PATH CHOSEN line per T bucket with both timings. The self-check must FORCE each instantiation it can use (one-expert, all-local, skewed and random routings at each T) and judge rel-L2 against an fp32 dequant reference with master's own error printed beside it (bound max(2x master's error, 1e-2)), not against Marlin alone. A log without PATH CHOSEN = WGMMA for a bucket means that bucket is unmeasured. First measurement is an okc check, never a submit. T buckets for the gate: 8, 16, 32. Motivation: in the best profile Marlin's pair is 94.6-114 us at bs=8 (floor 37.3), 170-176 at bs=16 (floor 52.5), 156-207 at bs=32 (floor 70.8) - 2.5-3x off the floor, and stage-2 prepare timing put the wgmma pair at 0.57x Marlin at M=128/256; the persistent grid streams each touched expert once regardless of route count, which Marlin's m-block scheme lacks.

## Expected

- `bs=8 *` faster: Marlin pair ~95-114 + act 2.0 us vs floor 37.3; M1 PATH CHOSEN = WGMMA at T=8 with tests 5/5; M2 wgmma pair below Marlin pair; M3 <= 56 us
- `bs=16 *` faster: Marlin pair ~170-176 + 2.2 us vs floor 52.5; M3 <= 79 us
- `bs=32 *` faster: Marlin pair ~156-207 + 2.3 us vs floor 70.8; decode_32 is master's weakest load (~1.03x); M3 <= 106 us
- `bs=1 *` flat: control: below the 8-token gate, master Marlin with w2 atomic
- `bs=128 *` flat: control: T >= 128 is master (s3 owns it)
- `bs=256 *` flat: control

## Risk

Kill: if every T bucket's prepare gate keeps Marlin, or the judged bs=8/16/32 rows are not below master's Marlin pair by more than the ~20% Marlin profile noise, stop the mid-M wgmma line. Risk: at 7-13 touched experts the 132-CTA persistent grid may be under-filled (bs=8 w2: ~7 experts x 32 N-tiles = 224 items, 1.7 waves); the gate protects the headline. Distinct from the held cuda-nvfp4-wgmma-rega-s2-fix-gate-from-8 (widened into broken buckets) and from running s3 (T >= 128, fixes N64/N128).

## Milestones

Rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 8 <= num_tokens <= 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
