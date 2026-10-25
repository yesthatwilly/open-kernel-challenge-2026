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
review: proceed
status: running
review_at: 2026-10-25T040253Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T040259Z
base_sha: 570f5e4ac5465931183addaf7a472e01668ced15
base_blob: 1ae7e9cac6dc078a0e71c7a6b0b77ec48b0ed01f
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix
mode: null
restack_of: null
stale_base: true
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

## Adversarial review

Reviewed 2026-10-25T040253Z by an adversarial subagent (model opus).

**Verdict: proceed.** Diagnosis is right at the bucket level and the upside is large (prepare 0.57x of Marlin at M=128/256). Reading the s2 kernel rules out two of the five listed suspects (expect_tx and smem carve-out are bucket-independent). It also adds a stronger one: register-A operands are overwritten by the next k-step's dequant while the previous wgmma group may still be reading them (wait_group 1), which is undefined behaviour that shows up per <NB,G> instantiation.

### Justification
Stage 2's per-bucket table localises the bug. N16/N32 are correct, w13 fails at N64 and N128, w2 only at N128, and whole rows go non-finite in every N-tile. That is a per-instantiation fault, and the hypothesis targets exactly the NB-dependent code. From the s2 source (consume_item<NB,G>, producer loop, wg_set_maps): suspect (3) is ruled out, because tx = WST + 2*(16<<bucket)*128 equals the two TMA boxes (64 bf16 x rows) plus the weight record exactly. Suspect (4) is ruled out too: the slab is a fixed 2 x SLAB_HALF (128 rows x 128 B) per stage for every bucket, there is no per-bucket carve-out, and the second consumer warpgroup's base is a weight-row offset (cw*512) independent of the bucket. Suspect (2) is unlikely, because make_desc is identical for every N (K-major SW128, SBO 1024) and is already proven at N16/N32. Suspect (1) reads correct: the asm is generated per NB with na = NB/2 operands, and the epilogue loops use NB/8. WEAKEST LINK, and a missing suspect: in consume_item the loop dequantizes the next stage into a[8][4], issues 8 RS wgmmas, commits, then wgmma_wait<1>. That leaves the just-issued group in flight while the next iteration's dequant8 rewrites a[] (the same physical registers, since the kb loop is not unrolled). The PTX ISA makes touching A-fragment registers of an in-flight wgmma undefined. Whether ptxas reuses those registers depends on allocation, which differs per <NB, G>; that would explain w13 failing at 64 while w2 passes at 64. Wider N also makes each wgmma longer, which widens the race window. A transient register holding bf16x2_bcast(scale * 2^119) read as A would give inf, which silu turns into whole-row NaN. The milestones are plausible if fixed: the floor is 101/109 us, and Marlin's pair is 911/1259 us warm.

### Confounders on the current kernel
- align-parallel-per-expert-verify (master, BEST): it produces the sorted_ids/expert blocks that wg_gather and the bucket split consume at T > 32. A gain at bs=128/256 includes its align, which stays master in both arms. No confound for the stage's own milestones (kernel rows vs the Marlin pair).
- marlin-block-m-two-smaller-on-postnorm (master): it changes Marlin's block_size_m at T >= 128. The incumbent to beat is that tuned Marlin pair plus silu_and_mul, so compare to the profile Marlin rows of this batch, not to stage 2's prepare figures.
- w2-marlin-atomic-add-small-m (land candidate): bs=1 only. Disjoint from the num_tokens >= 128 gate.

### Interference and A/B needs
No companion. This is a rewrite correctness stage, and the both-kernels-pass gate makes the fallback a free in-run control (PATH CHOSEN = MARLIN reproduces master). The per-bucket canary already forces every bucket via one-expert routings. The extra discriminator below belongs inside the canary, not in a second run.

### Open questions
- BUILDER (first fix to try, cheap and decisive): eliminate the register-A WAR hazard. Either wgmma_wait<0>() before writing a[] for the next k-step (a quick test), or double-buffer a as a[2][8][4] with the kb loop unrolled by 2 so the in-flight group's registers are never rewritten. Also grep the ptxas output for 'serialized' / C75xx. Re-run the per-bucket canary for both variants inside the same prepare so the log shows whether N64/N128 turn finite.
- BUILDER: skip suspects (3) expect_tx and (4) slab overlap, or demote them to one print. The s2 code shows tx and the slab layout are identical across buckets. Print the instantiated NA and tx once per bucket as evidence.
- The s2 log says 'the fp32-dequant reference also reads inf on those'. Confirm the reference itself is finite on the skewed/all-one-expert routings (Marlin passes the judge). If the reference is inf, the comparison metric is broken, not just the kernel.
- When chosen, verify the cold-L2 time is plausible against bytes: a race that releases early would also look fast. The gate should also assert the profile kernel name in one probe call (LESSONS 164).

