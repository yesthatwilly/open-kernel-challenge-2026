---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix
created: 2026-10-25T035329Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 3 of the wgmma register-A grouped NVFP4 rewrite, built on the stage-2 commit
  (cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix, claim bases a rewrite-on-rewrite on the parent
  commit); rename the module so it recompiles. Stage 2 (job 583a73418fbf4d01) diagnosed the bug:
  build sm_90a clean, w13+silu 168 regs/0 spill, w2 168 regs/8 B spill, cuTensorMapEncodeTiled x8
  rc 0, device error word 0, gather clean (0/644 bad rows, pads zero, 0 remote writes); the 16-
  and 32-row token buckets are correct (w13 rel 6.3e-3/6.5e-3, w2 3.4e-3/3.5e-3), but w13 is all
  NaN/inf at the 64- and 128-row buckets and w2 at 128 only; whole rows are non-finite in every
  N-tile, i.e. a per-bucket (M-tile instantiation) error. Change ONLY what differs between the
  16/32 and the 64/128 instantiations. First, in the prepare canary, per kernel and per bucket
  (one-expert routings that force M=64 and M=128 into a single bucket, w13 and w2 separately, w2
  fed Marlin h), after its own sync + flushed print: first non-finite (row, col), whether the
  whole row is NaN vs scattered inf, and for the failing bucket print the template parameters
  actually instantiated (wgmma n-shape, accumulator register count per thread, B descriptor
  LBO/SBO/swizzle, expect_tx bytes per stage, smem offsets of every A/B/slab carve-out vs total
  dyn smem). Then fix, checking each item against the passing 16/32 path and an offline numpy
  emulation before dispatch: (1) the wgmma instruction shape at wider N: m64n64k16/m64n128k16 need
  32/64 fp32 accumulator registers per thread; the fragment-to-(row,col) mapping and the epilogue
  loop must use the bucket N, not a hard-coded 16/32 extent; (2) the B shared-memory matrix
  descriptor at wider N: leading/stride byte offsets and swizzle mode must match what TMA wrote
  for the wider box (128B swizzle with SBO = 8 rows x 128 B; LBO between core-matrix columns), and
  the start-address advance per k16 step; (3) the TMA box / expect_tx bytes per stage must equal
  the real bytes landed for 64/128 rows (a short expect_tx releases the barrier before the data
  lands; a long one hangs, but the error word was 0, so look for short); (4) the smem carve-out
  beyond 32 rows: the slab/A region for 64/128 rows must not overlap a B ring stage (w13
  8192+187392 B, w2 8192+203776 B; print offsets and assert no overlap), and the second consumer
  warpgroup row base (rows 64..127) must point into its own slab rows; (5) the w2 8 B spill:
  confirm it is not the 128-row accumulator (spilled accumulators are fine numerically, but check
  the index). Keep the both-kernels-pass gate: WGMMA is chosen only if every bucket passes for w13
  and w2 and the end-to-end canary is finite at rel L2 within the existing bound; else PATH CHOSEN
  = MARLIN with the failing (kernel, bucket, cause) printed. When chosen, print cold-L2 CUDA-event
  times vs Marlin at M=128 and 256 (stage 2 warm eager figures were 515.9 vs 911.3 us at M=128 and
  728.0 vs 1258.5 at M=256; the judge reads 20-25% higher than prepare). Gate unchanged
  (num_tokens >= 128); Marlin below. Build as stage 2 (sm_90a only, -lineinfo, in-process, never a
  child process). First measurement is an okc check, never a scored submit.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "M1: every bucket finite and within bound, PATH CHOSEN = WGMMA, tests pass; M2: wgmma pair below Marlin pair + act_and_mul (prepare 515.9 vs 911.3 us warm); floor 101.3 us, M3 <= 170 us"
  - shapes: bs=256 *
    direction: faster
    why: "M2: below Marlin (prepare 728.0 vs 1258.5 us warm); floor 108.9 us, M3 <= 220 us"
  - shapes: bs=1 * bs=8 * bs=16 * bs=32 *
    direction: flat
    why: below the num_tokens >= 128 gate; Marlin unchanged
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix

## Change

Stage 3 of the wgmma register-A grouped NVFP4 rewrite, built on the stage-2 commit (cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix, claim bases a rewrite-on-rewrite on the parent commit); rename the module so it recompiles. Stage 2 (job 583a73418fbf4d01) diagnosed the bug: build sm_90a clean, w13+silu 168 regs/0 spill, w2 168 regs/8 B spill, cuTensorMapEncodeTiled x8 rc 0, device error word 0, gather clean (0/644 bad rows, pads zero, 0 remote writes); the 16- and 32-row token buckets are correct (w13 rel 6.3e-3/6.5e-3, w2 3.4e-3/3.5e-3), but w13 is all NaN/inf at the 64- and 128-row buckets and w2 at 128 only; whole rows are non-finite in every N-tile, i.e. a per-bucket (M-tile instantiation) error. Change ONLY what differs between the 16/32 and the 64/128 instantiations. First, in the prepare canary, per kernel and per bucket (one-expert routings that force M=64 and M=128 into a single bucket, w13 and w2 separately, w2 fed Marlin h), after its own sync + flushed print: first non-finite (row, col), whether the whole row is NaN vs scattered inf, and for the failing bucket print the template parameters actually instantiated (wgmma n-shape, accumulator register count per thread, B descriptor LBO/SBO/swizzle, expect_tx bytes per stage, smem offsets of every A/B/slab carve-out vs total dyn smem). Then fix, checking each item against the passing 16/32 path and an offline numpy emulation before dispatch: (1) the wgmma instruction shape at wider N: m64n64k16/m64n128k16 need 32/64 fp32 accumulator registers per thread; the fragment-to-(row,col) mapping and the epilogue loop must use the bucket N, not a hard-coded 16/32 extent; (2) the B shared-memory matrix descriptor at wider N: leading/stride byte offsets and swizzle mode must match what TMA wrote for the wider box (128B swizzle with SBO = 8 rows x 128 B; LBO between core-matrix columns), and the start-address advance per k16 step; (3) the TMA box / expect_tx bytes per stage must equal the real bytes landed for 64/128 rows (a short expect_tx releases the barrier before the data lands; a long one hangs, but the error word was 0, so look for short); (4) the smem carve-out beyond 32 rows: the slab/A region for 64/128 rows must not overlap a B ring stage (w13 8192+187392 B, w2 8192+203776 B; print offsets and assert no overlap), and the second consumer warpgroup row base (rows 64..127) must point into its own slab rows; (5) the w2 8 B spill: confirm it is not the 128-row accumulator (spilled accumulators are fine numerically, but check the index). Keep the both-kernels-pass gate: WGMMA is chosen only if every bucket passes for w13 and w2 and the end-to-end canary is finite at rel L2 within the existing bound; else PATH CHOSEN = MARLIN with the failing (kernel, bucket, cause) printed. When chosen, print cold-L2 CUDA-event times vs Marlin at M=128 and 256 (stage 2 warm eager figures were 515.9 vs 911.3 us at M=128 and 728.0 vs 1258.5 at M=256; the judge reads 20-25% higher than prepare). Gate unchanged (num_tokens >= 128); Marlin below. Build as stage 2 (sm_90a only, -lineinfo, in-process, never a child process). First measurement is an okc check, never a scored submit.

## Expected

- `bs=128 *` faster: M1: every bucket finite and within bound, PATH CHOSEN = WGMMA, tests pass; M2: wgmma pair below Marlin pair + act_and_mul (prepare 515.9 vs 911.3 us warm); floor 101.3 us, M3 <= 170 us
- `bs=256 *` faster: M2: below Marlin (prepare 728.0 vs 1258.5 us warm); floor 108.9 us, M3 <= 220 us
- `bs=1 * bs=8 * bs=16 * bs=32 *` flat: below the num_tokens >= 128 gate; Marlin unchanged

## Risk

The cause is inferred from bucket behaviour, not a named line; several suspects may need fixing. Prepare timings are warm and optimistic (the log warns a race releasing early would also look fast), so a pass with an implausibly fast time should be read with the finiteness/rel-L2 evidence.

## Milestones

Rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
