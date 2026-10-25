---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-w2-only-midm
created: 2026-10-25T042150Z
base: master
area: w2 marlin
change: >
  Per-kernel arm of the mid-M wgmma line: replace ONLY the w2 Marlin call at 8 <= T <= 32 with
  stage 2's wg_kernel<w2>, fed by master's Marlin w13 output after silu_and_mul (cache2) through
  wg_gather. Stage 2's diagnostic already ran exactly this pairing ('w2 fed Marlin's h') and
  passed at N16/N32/N64 (rel 3.4-3.5e-3), so this is the arm most likely to run correctly. Base:
  the stage-2 commit (rewrite-on-rewrite), then rebase onto current master (9080a71, which
  includes w2-marlin-atomic-add-small-m) so every fallback and untouched shape is master byte for
  byte; rename the load_inline module so it recompiles. Build exactly as stage 2:
  TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo, wgmma
  bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process,
  Marlin fallback. Change: own bound constants WG_W2_MIDM_MIN = 8, WG_W2_MIDM_MAX = 32; w13
  Marlin, silu_and_mul, align and the post-norm combine stay master byte for byte (including
  master's w2 use_atomic_add path for T <= 32, which becomes the fallback); work items capped at
  32 rows on device (ceil(cnt/32) items per expert, N16 if <= 16 rows else N32), y written in
  master's [T, 8, H] layout with remote route rows zeroed so the combine is unchanged. Hard gate
  in prepare(): after the correctness self-check passes, time the new path against master's path
  on the same synthetic routings (skewed and random, cold L2 via a >L2 buffer read, CUDA graph
  replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T bucket; enable a bucket only if
  its p50 beats master by >= 3% and assert the enabled choice equals the printed winner (LESSONS
  2026-10-25 selector bug). Print one PATH CHOSEN line per T bucket with both timings. The
  self-check must FORCE each instantiation it can use (one-expert, all-local, skewed and random
  routings at each T) and judge rel-L2 against an fp32 dequant reference with master's own error
  printed beside it (bound max(2x master's error, 1e-2)), not against Marlin alone. A log without
  PATH CHOSEN = WGMMA for a bucket means that bucket is unmeasured. First measurement is an okc
  check, never a submit. T buckets: 8, 16, 32. Compare against w2's own Marlin row (profile the
  two Marlin launches separately once in prepare and print each, since the judge profile merges
  them).
expected:
  - shapes: bs=8 *
    direction: faster
    why: "w2 is ~1/3 of the Marlin pair bytes (8.4 of 25.2 MB per expert): roughly 30-38 us of the ~95-114 us pair; target w2 <= 20 us"
  - shapes: bs=16 *
    direction: faster
    why: w2 share ~55 us of ~170; target <= 30 us
  - shapes: bs=32 *
    direction: faster
    why: w2 share ~55-70 us of ~156-207; target <= 40 us
  - shapes: bs=1 *
    direction: flat
    why: control
  - shapes: bs=128 *
    direction: flat
    why: control
  - shapes: bs=256 *
    direction: flat
    why: control
risk: medium
depends_on: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: Marlin w2 (second moe_wna16_marlin_gemm) at 8 <= num_tokens <= 32
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-w2-only-midm

## Change

Per-kernel arm of the mid-M wgmma line: replace ONLY the w2 Marlin call at 8 <= T <= 32 with stage 2's wg_kernel<w2>, fed by master's Marlin w13 output after silu_and_mul (cache2) through wg_gather. Stage 2's diagnostic already ran exactly this pairing ('w2 fed Marlin's h') and passed at N16/N32/N64 (rel 3.4-3.5e-3), so this is the arm most likely to run correctly. Base: the stage-2 commit (rewrite-on-rewrite), then rebase onto current master (9080a71, which includes w2-marlin-atomic-add-small-m) so every fallback and untouched shape is master byte for byte; rename the load_inline module so it recompiles. Build exactly as stage 2: TORCH_CUDA_ARCH_LIST=9.0a (only -gencode compute_90a/sm_90a), -std=c++20 -O3 -lineinfo, wgmma bodies under __CUDA_ARCH_FEAT_SM90_ALL, in-process guarded first launch, no child process, Marlin fallback. Change: own bound constants WG_W2_MIDM_MIN = 8, WG_W2_MIDM_MAX = 32; w13 Marlin, silu_and_mul, align and the post-norm combine stay master byte for byte (including master's w2 use_atomic_add path for T <= 32, which becomes the fallback); work items capped at 32 rows on device (ceil(cnt/32) items per expert, N16 if <= 16 rows else N32), y written in master's [T, 8, H] layout with remote route rows zeroed so the combine is unchanged. Hard gate in prepare(): after the correctness self-check passes, time the new path against master's path on the same synthetic routings (skewed and random, cold L2 via a >L2 buffer read, CUDA graph replays, interleaved A/B, >= 50 reps, p10/p50/p90 printed) per T bucket; enable a bucket only if its p50 beats master by >= 3% and assert the enabled choice equals the printed winner (LESSONS 2026-10-25 selector bug). Print one PATH CHOSEN line per T bucket with both timings. The self-check must FORCE each instantiation it can use (one-expert, all-local, skewed and random routings at each T) and judge rel-L2 against an fp32 dequant reference with master's own error printed beside it (bound max(2x master's error, 1e-2)), not against Marlin alone. A log without PATH CHOSEN = WGMMA for a bucket means that bucket is unmeasured. First measurement is an okc check, never a submit. T buckets: 8, 16, 32. Compare against w2's own Marlin row (profile the two Marlin launches separately once in prepare and print each, since the judge profile merges them).

## Expected

- `bs=8 *` faster: w2 is ~1/3 of the Marlin pair bytes (8.4 of 25.2 MB per expert): roughly 30-38 us of the ~95-114 us pair; target w2 <= 20 us
- `bs=16 *` faster: w2 share ~55 us of ~170; target <= 30 us
- `bs=32 *` faster: w2 share ~55-70 us of ~156-207; target <= 40 us
- `bs=1 *` flat: control
- `bs=128 *` flat: control
- `bs=256 *` flat: control

## Risk

A/B arm for cuda-nvfp4-wgmma-rega-midm-bucket-cap32: separates the w2 kernel's contribution from w13's. Kill: prepare gate keeps Marlin at all three T, or judged rows flat within noise. If the reviewer judges one check enough, run this one first (higher chance of PATH CHOSEN = WGMMA).

## Milestones

Rewrite stage 3 of `Marlin w2 (second moe_wna16_marlin_gemm) at 8 <= num_tokens <= 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
