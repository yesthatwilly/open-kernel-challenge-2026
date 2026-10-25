---
problem: muse_bf16_proj
slug: bf16-lossless-exp4-occupancy-dequant-once-stage2
created: 2026-10-25T001220Z
base: master
area: lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel)
change: >
  Build on the stage-1 code of bf16-lossless-exp4-weight-stream-rewrite (job 542f2b169c934706;
  codec 12.17 bits/elem, ratio 0.761, bit-exact; load_inline + per-route self-check + fallback all
  kept). Base is bf16 master = bf16-m32-swapab-wgmma-triton (the 1.2215 restack is NOT landed,
  pending the user; do not build on it). Three kernel changes plus a routing cut, all inside
  _lossless_swapab_kernel<NT> and its launcher: (1) OCCUPANCY: stage 1 ran one 128-thread CTA/SM
  (115-165 KB smem, 8 stages) = 4 warps/SM, one warp per SMSP serially doing cp.async waits,
  integer dequant, ldmatrix of x and dependent mma.sync chains. Reach 8-16 warps/SM: either cut
  the ring to 3-4 stages (~13 KB compressed/stage) so 2-3 CTAs fit per SM, or go to 8 warps/CTA
  with 2 warps per 16-row tile splitting each K-stage (reduce the pair in smem at the end). Keep
  regs <=128 so 2+ CTAs fit; print cudaOccupancyMaxActiveBlocksPerMultiprocessor and achieved
  warps/SM per NT in prepare. (2) DEQUANT ONCE PER k16 A FRAGMENT: decode each weight fragment
  exactly once into registers and reuse it for all NT/8 token (B) fragments; check the stage-1
  code for re-decoding per n-fragment (time grew 40.6->45.2->51.4 us on pooler for NT 8/16/32,
  which says it is not amortized). Software-pipeline: decode stage s+1 fragments while stage s
  mma.sync issue. (3) SCOPE: route only pooler (m 1/8/16/32), unpooler-full (m 16/32) and qkv_gate
  (m 1/8/16/32) to the lossless kernel; route o (all m) and the unpooler slice (m 1/8) back to
  master's Triton path (they are latency-bound 12.8/6.4 MB streams and cannot win). Router and all
  m>=128 untouched. (4) In prepare, print a cold-L2 timing per route (flush a >L2 buffer before
  the event) next to the warm one, since qkv/o profile-vs-row gaps mislead. Stay on mma.sync (no
  wgmma), so the sm_90 vs sm_90a load_inline bug does not apply; if wgmma is introduced, set
  TORCH_CUDA_ARCH_LIST=9.0a around load_inline. Codec, weight compression at load time, and output
  are unchanged; results must stay bit-exact with stage 1's self-check.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: "M-KILL/M2: stage-1 kernel 40.6 us vs incumbent Triton 19.0 us; raw floor 14.0, compressed floor 10.7. Must reach <=19.0 us (>=2.7 TB/s compressed) or the line is DEAD. M3 target 15.5 us (~3.3 TB/s compressed)."
  - shapes: layer=pooler load=verify_1 m=8
    direction: faster
    why: "M2: stage1 41.0 vs incumbent 19.3 us; floor 14.0 raw / 10.7 compressed; M3 <=15.5"
  - shapes: layer=pooler load=decode_16 m=16
    direction: faster
    why: "M2: stage1 45.2 vs incumbent 19.5 us; floor 14.1/10.7; dequant-once should flatten the NT growth"
  - shapes: layer=pooler load=decode_32 m=32
    direction: faster
    why: "M2: stage1 51.4 vs incumbent 20.5 us; floor 14.1/10.8"
  - shapes: layer=unpooler load=decode_16 m=16
    direction: faster
    why: "M2: stage1 43.4 vs incumbent nvjet 18.5 us; floor 14.1/10.7"
  - shapes: layer=unpooler load=decode_32 m=32
    direction: faster
    why: "M2: stage1 45.1 vs incumbent 18.9 us; floor 14.1/10.8"
  - shapes: layer=qkv_gate load=decode_1 m=1
    direction: faster
    why: "M2 on the cold-L2 row: stage1 row 30.19 vs incumbent row 12.91 (kernel 11.8); floor 7.9 raw / 6.0 compressed; L2-resident weight, judge on the row not the profile"
  - shapes: layer=qkv_gate load=verify_1 m=8
    direction: faster
    why: "M2: stage1 row 31.59 vs incumbent row 13.18 (kernel 12.0); floor 7.9/6.0"
  - shapes: layer=qkv_gate load=decode_16 m=16
    direction: faster
    why: "M2: stage1 row 33.86 vs incumbent row 13.39 (kernel 12.3); floor 7.9/6.0"
  - shapes: layer=qkv_gate load=decode_32 m=32
    direction: faster
    why: "M2: stage1 row 38.70 vs incumbent row 13.69 (kernel 12.1); floor 8.0/6.1"
  - shapes: layer=o *
    direction: neutral
    why: routed back to master Triton; stage-1 regression (~6.5 -> 15-16.6 us rows) must disappear
  - shapes: layer=unpooler load=decode_1 m=1
    direction: neutral
    why: slice routed back to master (incumbent row 5.94, floor 1.75)
  - shapes: layer=unpooler load=verify_1 m=8
    direction: neutral
    why: slice routed back to master (incumbent row 6.13, floor 1.76)
risk: high
depends_on: bf16-lossless-exp4-weight-stream-rewrite
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: Triton swap-AB m<=32 kernels (bf16-m32-swapab-wgmma-triton) on pooler m 1-32, unpooler-full m 16/32, qkv_gate m 1-32
review: proceed
status: done
review_at: 2026-10-25T001525Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T001525Z
base_sha: 7895e56373c79d74ba78b7c0e7146755b118ce2c
base_blob: a34e409b452e9a6e5f7ee3612486d30a989832f2
branch: hyp/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2
worktree: .worktrees/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T002149Z
impl_commit: 39098a3d90b3988b8d51bd2e81cc65c40b81d89c
submission_sha256: d11133a85dae43fccde181ba07ec3cf62693e14085f3ae9d1ee3f44ab6524c0c
norm_sha256: ee2393a43ede21faa24e34a23cdd9e03ec8dabb7460fe48bc478ace3e2395f3e
submission_blob: 12be4cd61f2a429f08b6a1d38499e6af8845bf86
lint_warnings: 11
attempt_file: 2026-10-25T002149Z-check-bf16-lossless-exp4-occupancy-dequant-once-stage2.md
prepared_at: 2026-10-25T002150Z
job: f1024212dd2f49f3
dispatched_at: 2026-10-25T002202Z
outcome: neutral
verdict_detail: beats-base-not-best
speedup: 0.8869580788126001
speedup_ci: [0.8736537076304112, 0.9002624499947891]
harvested_at: 2026-10-25T010412Z
best_at_harvest: 1.2215006063849365
promotion: null
---

# bf16-lossless-exp4-occupancy-dequant-once-stage2

## Change

Build on the stage-1 code of bf16-lossless-exp4-weight-stream-rewrite (job 542f2b169c934706; codec 12.17 bits/elem, ratio 0.761, bit-exact; load_inline + per-route self-check + fallback all kept). Base is bf16 master = bf16-m32-swapab-wgmma-triton (the 1.2215 restack is NOT landed, pending the user; do not build on it). Three kernel changes plus a routing cut, all inside _lossless_swapab_kernel<NT> and its launcher: (1) OCCUPANCY: stage 1 ran one 128-thread CTA/SM (115-165 KB smem, 8 stages) = 4 warps/SM, one warp per SMSP serially doing cp.async waits, integer dequant, ldmatrix of x and dependent mma.sync chains. Reach 8-16 warps/SM: either cut the ring to 3-4 stages (~13 KB compressed/stage) so 2-3 CTAs fit per SM, or go to 8 warps/CTA with 2 warps per 16-row tile splitting each K-stage (reduce the pair in smem at the end). Keep regs <=128 so 2+ CTAs fit; print cudaOccupancyMaxActiveBlocksPerMultiprocessor and achieved warps/SM per NT in prepare. (2) DEQUANT ONCE PER k16 A FRAGMENT: decode each weight fragment exactly once into registers and reuse it for all NT/8 token (B) fragments; check the stage-1 code for re-decoding per n-fragment (time grew 40.6->45.2->51.4 us on pooler for NT 8/16/32, which says it is not amortized). Software-pipeline: decode stage s+1 fragments while stage s mma.sync issue. (3) SCOPE: route only pooler (m 1/8/16/32), unpooler-full (m 16/32) and qkv_gate (m 1/8/16/32) to the lossless kernel; route o (all m) and the unpooler slice (m 1/8) back to master's Triton path (they are latency-bound 12.8/6.4 MB streams and cannot win). Router and all m>=128 untouched. (4) In prepare, print a cold-L2 timing per route (flush a >L2 buffer before the event) next to the warm one, since qkv/o profile-vs-row gaps mislead. Stay on mma.sync (no wgmma), so the sm_90 vs sm_90a load_inline bug does not apply; if wgmma is introduced, set TORCH_CUDA_ARCH_LIST=9.0a around load_inline. Codec, weight compression at load time, and output are unchanged; results must stay bit-exact with stage 1's self-check.

## Expected

- `layer=pooler load=decode_1 m=1` faster: M-KILL/M2: stage-1 kernel 40.6 us vs incumbent Triton 19.0 us; raw floor 14.0, compressed floor 10.7. Must reach <=19.0 us (>=2.7 TB/s compressed) or the line is DEAD. M3 target 15.5 us (~3.3 TB/s compressed).
- `layer=pooler load=verify_1 m=8` faster: M2: stage1 41.0 vs incumbent 19.3 us; floor 14.0 raw / 10.7 compressed; M3 <=15.5
- `layer=pooler load=decode_16 m=16` faster: M2: stage1 45.2 vs incumbent 19.5 us; floor 14.1/10.7; dequant-once should flatten the NT growth
- `layer=pooler load=decode_32 m=32` faster: M2: stage1 51.4 vs incumbent 20.5 us; floor 14.1/10.8
- `layer=unpooler load=decode_16 m=16` faster: M2: stage1 43.4 vs incumbent nvjet 18.5 us; floor 14.1/10.7
- `layer=unpooler load=decode_32 m=32` faster: M2: stage1 45.1 vs incumbent 18.9 us; floor 14.1/10.8
- `layer=qkv_gate load=decode_1 m=1` faster: M2 on the cold-L2 row: stage1 row 30.19 vs incumbent row 12.91 (kernel 11.8); floor 7.9 raw / 6.0 compressed; L2-resident weight, judge on the row not the profile
- `layer=qkv_gate load=verify_1 m=8` faster: M2: stage1 row 31.59 vs incumbent row 13.18 (kernel 12.0); floor 7.9/6.0
- `layer=qkv_gate load=decode_16 m=16` faster: M2: stage1 row 33.86 vs incumbent row 13.39 (kernel 12.3); floor 7.9/6.0
- `layer=qkv_gate load=decode_32 m=32` faster: M2: stage1 row 38.70 vs incumbent row 13.69 (kernel 12.1); floor 8.0/6.1
- `layer=o *` neutral: routed back to master Triton; stage-1 regression (~6.5 -> 15-16.6 us rows) must disappear
- `layer=unpooler load=decode_1 m=1` neutral: slice routed back to master (incumbent row 5.94, floor 1.75)
- `layer=unpooler load=verify_1 m=8` neutral: slice routed back to master (incumbent row 6.13, floor 1.76)

## Risk

Milestones (per target, judged on rows and the lossless kernel time vs incumbent): M1 = builds, self-check bit-exact on the 3 kept routes, o/slice on master, prepare prints >=8 warps/SM achieved. M-KILL = pooler m=1 kernel <=19.0 us; if pooler m=1 is still above 19 us the line is DEAD (codec cannot be streamed fast enough with mma.sync) - do not mint stage 3. M2 = at or below incumbent on each target: pooler m1/8/16/32 <=19.0/19.3/19.5/20.5; unpooler-full m16/32 <=18.5/18.9; qkv_gate rows m1/8/16/32 <=12.91/13.18/13.39/13.69. M3 = below the raw-byte floor where only compression can get: pooler and unpooler-full <=14 us (stretch 15.5 first), qkv_gate rows <=7.9. Kernel time should be ~flat in NT (8/16/32) if dequant is amortized; if it still grows >=4 us from m=1 to m=32, (2) did not land. Risks: 2-3 CTAs/SM with a short ring may expose DRAM latency again; 8-warp split-K adds an smem reduce; qkv rows are cold-L2 and the profile understates them. Incumbent numbers are from check job 1602a4e947e64090; master is still bf16-m32-swapab-wgmma-triton (restack 1.2215 not landed, pending the user). Source: muse_bf16_proj/hypotheses/REWRITE_READ_2026-10-25.md section 3.

## Milestones

Rewrite stage 2 of `Triton swap-AB m<=32 kernels (bf16-m32-swapab-wgmma-triton) on pooler m 1-32, unpooler-full m 16/32, qkv_gate m 1-32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T001525Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-2 rewrite with a clear kill gate, so proceed. Two parts of the plan need fixing before the build: sub-change (2) is already in the stage-1 code, and option (a) cannot raise warps/SM unless the grid also grows past 132 CTAs. Without those fixes the M-KILL result would not be interpretable.

### Justification
The mechanism is plausible. Stage 1 streams 1.05-1.19 TB/s compressed with one 128-thread CTA per SM. That is one warp per SMSP serialising the cp.async waits, the PRMT/LOP3 dequant, ldmatrix and dependent mma.sync. The gemv rewrite had no dequant and ran 8 warps/SM; it reached ~3.5 TB/s. So latency hiding is the credible bottleneck, and the integer op count (~5 ops per bf16 pair, ~18 int ops/clk/SM) is far below the INT32 rate. The M-KILL bar is physically reachable: the pooler at <=19 us needs ~2.7 TB/s of compressed DRAM traffic (51.1 MB), which is below the ~3.5 TB/s the incumbent already pulls. M3 at 15.5 us needs ~3.3 TB/s, which is aggressive but under 4.8. The expected directions are consistent with TASK.md: every shape has weight 62. Pooler and unpooler-full are 67 MB DRAM streams with 14.0-14.1 floors, and only compression can get below them. qkv_gate is cold-L2 on the judge, so judging it on the row is correct. Sending o and the unpooler slice back to master as 'neutral' is right: stage 1 lost 2-2.8x there, and those streams are latency-bound.

The argument has three defects.

(1) WEAKEST LINK, the occupancy plan. Stage 1 launches one wave of exactly pooler 64x2=128, unpooler-full 128x1=128 and qkv 36x3=108 CTAs, all at or below 132 SMs. Option (a) cuts the ring to 3-4 stages so 2-3 CTAs FIT per SM. On its own that changes nothing: with 128 or 108 CTAs each SM still gets at most one CTA and still runs 4 warps. qkv even leaves 24 SMs idle. Option (a) works only if the grid also grows to >=264 CTAs: pooler split 4 (256), unpooler-full split 2 (256), qkv split 7 (252). The extra split-K adds partial-store and ticket-reduce traffic, which the gemv rewrite showed scales with m. Option (b), 8 warps per CTA at the same grid, raises warps/SM directly and leaves the reduce shape unchanged; it is the safer route. Separately, cudaOccupancyMaxActiveBlocksPerMultiprocessor reports what FITS, not what is ACHIEVED. The 'achieved warps/SM' print must be computed as min(grid, occ*132)/132 * warps_per_CTA. Otherwise M1 ('>=8 warps/SM achieved') can be met on paper while the SMs still run 4 warps.

(2) Sub-change (2) is already in stage 1. In _lossless_swapab_kernel the loop runs `decode_step(G, kp*2+t, ...)` once per k16 step, then `for f < NT/8: mma_bf16(acc[f], a, ...)`. The A fragment is decoded once and reused across all B fragments. So the 40.6->45.2->51.4 us growth with NT does NOT come from repeated decoding. The likely causes are: NT/8 ldmatrix.x4 and NT/8 dependent MMAs per k16 on a single warp per SMSP; the x tile read from L2 into smem by every CTA (BLOCK_N=64, so at m=32 the pooler moves ~33.5 MB of x from L2 against 51 MB of weight from DRAM); and the 164.9 KB ring at NT=32. Only the software-pipelining half of (2) is new. The milestone line 'if it still grows >=4 us from m=1 to m=32, (2) did not land' is therefore misattributed. With 8 warps the NT growth should shrink through latency hiding; whatever growth remains points at x traffic/ldmatrix, not decode.

(3) Bit-exactness. The codec round trip stays bit-exact. GEMM outputs will NOT be bit-identical to stage 1 if option (b) splits K across two warps and reduces the pair in smem, or if split-K changes, because the fp32 summation order changes. The self-check has to stay tolerance-based against the reference, not a bit-compare with stage-1 outputs.

### Confounders on the current kernel
- bf16-m32-swapab-wgmma-triton (the landed incumbent this replaces). It is the comparison point, not a confounder. However, o m=32 and unpooler m=8 return to its paths and to the m<=16 shape-gated kernels, so those rows must route to exactly master's kernels for the 'neutral' expectation to hold. Confirm the lossless gate sits ahead of _pooler_shape_ok / _qkv_route_ok / _unpool_shape_ok / _swapab_route_ok in kernel_fn, and that it rejects o and the n==1024 unpooler slice so they reach master's gates unchanged.
- bf16-shape-gated-triton-winners-combo. It owns pooler/qkv/unpooler-slice/o at m<=16. Stage 2 takes pooler and qkv away from it but leaves o and the slice on it, so no interaction beyond correct ordering of the routing gates. If any lossless route is DISABLED by the self-check, its fallback must land on these kernels, not on _bf16_gemm_dispatch_impl. Otherwise a 'loss' is really a fallback to cuBLAS.
- router-triton-hilo-fused-control-splitk-all-m / router-hilo-at-m8 / sglang-gemm-dispatch-parity. Different layers and every timed replay starts cold-L2, so there is no shared memory traffic or launch sequence. They add only noise on router rows, which the per-shape milestones do not read.

### Interference and A/B needs
No companions. This is a kind:rewrite judged per shape on the replaced kernels' own time (pooler and unpooler-full from profile top_kernels, qkv from the cold-L2 row) against fixed incumbent numbers from job 1602a4e947e64090. It is not judged on the headline, so no landed hypothesis can fake or hide a per-kernel gain. The only attribution question inside the stage is occupancy versus software-pipelined dequant. The prepare() prints (fitted vs achieved warps/SM per NT, warm and cold timing per route) answer it if they are implemented as noted, and the next stage is the follow-up. An ab-revert of bf16-m32-swapab-wgmma-triton would only move the yardstick. A control that isolates occupancy (stage-1 kernel at 8 warps, no pipelining) would spend a check to split a result the prints can already split.

### Open questions
- Which occupancy option will be built? If (a), what new split per layer makes the grid >=2x132 CTAs, and has the extra reduce cost been budgeted against the 19 us kill gate (the pooler at split 4 writes 4 x m x 4096 fp32 partials)?
- Since stage 1 already decodes A once per k16 step, what concretely does (2) change: register double-buffering of the next k16's decoded A plus the next group's Grp load ahead of the MMAs? What register cost does that add against the <=128-reg target at NT=32 (stage 1 was already at 115 regs)?
- x L2 re-read: at NT=32 each pooler CTA (BLOCK_N=64) pulls m x K/split x 2 B of x, ~33.5 MB total versus 51 MB of compressed weight. Should stage 2 or 3 move to BLOCK_N=128 (8 warps, one 16-row tile each), which halves x traffic, instead of 2 warps per tile splitting K, which keeps it?
- qkv_gate at split 3 uses 108 of 132 SMs. Is its split being raised regardless of the occupancy option?
- Is the M-KILL gate read from the pooler m=1 profile kernel time (DRAM-bound, profile and row agree within ~1 us) and not from the warm prepare() print? If it is the print, that should be stated.


## Implementation notes

All edits are in the worktree's `muse_bf16_proj/submission.py`, section (6), which holds `_LL_CUDA`, the launcher and the `_ll_*` helpers. The diff is 161+/78-. kernel_fn, the routers and the m>=128 paths are unchanged.

- **(1) Occupancy: option (b), as the review asked.** Option (a) was rejected because it raises warps/SM only if the grid grows past 132 CTAs, which adds split-K reduce traffic.
  - `THREADS` goes from 128 to 256 (8 warps per CTA). Warp w owns 16-row tile `w >> 1` and 64-K group `w & 1` of every 128-K stage, so the two warps of a tile split each K-stage.
  - After the mainloop the ring is reused for the pair reduce: K-half 1 writes its accumulators to smem (bank-conflict-free, `[...]*32 + lane`, at most 8 KB), and K-half 0 adds them in fixed order (half 0 + half 1). Then K-half 0 runs the unchanged epilogue (direct bf16 store, or ws partial plus ticket reduce).
  - The grid is unchanged: pooler 64x2=128, unpooler-full 128x1=128, qkv 36x3=108. So 8 warps/SM are actually resident on the busy SMs (1 CTA/SM, one wave).
  - The ring stays at 8 stages, and the x-tile issue loop is now guarded: at NT=8, `NT*16 = 128 < 256` threads.
  - `__launch_bounds__(256, 1)`. Deviation: the spec's "regs <=128 so 2+ CTAs fit" applies to option (a) only. Under (b) occupancy is limited by smem (1 CTA/SM), and 256 threads x up to 255 regs still fits the register file. The register count is printed.
  - prepare prints, per NT, regs, spills, smem and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (what fits). Per NT x route it also prints the **achieved** warps/SM as `min(grid, occ*SMs)/SMs*8` and the number of busy SMs, per the review's formula.
  - qkv keeps split 3 (108 CTAs, 24 SMs idle). Raising it would mean a second wave or option (a); that is left open for stage 3.
- **(2) Dequant once per k16: ALREADY IN STAGE 1. Not re-implemented.** Stage 1 already calls `decode_step` once per k16 step and reuses it for all NT/8 B fragments, so the NT growth is not caused by re-decoding. Per the review, it more likely comes from ldmatrix/MMA serialisation on one warp per SMSP and from x L2 re-reads.
  - Only the software-pipelining half was added:
    - (i) Each warp loads the NEXT stage's group from smem into registers (`Gn`) before issuing this stage's MMAs. The wait became `cp_wait<STAGES-3>` so that stage i+1 has landed, and the prologue loads stage 0 after `cp_wait<STAGES-2>`. `issue(i+7)` writes stage i-1's buffer, so it does not race with that load.
    - (ii) Inside a group, k16 step s+1's A fragment is decoded into `a[(s+1)&1]` before step s's MMAs issue.
  - The milestone line "(2) did not land if the NT growth is >=4 us" is misattributed, per the review. Any remaining NT growth points at x traffic or ldmatrix, not decode.
- **(3) Scope.**
  - `LL_CFG` drops `o` and `unpooler_slice`, and `LL_WEIGHTS` drops `o`, so o is no longer encoded.
  - `_ll_route` returns `"unpooler_slice"` for any unpooler n != 8192. With no LL_CFG entry, `_ll_route_ok` returns False, so o (all m) and the slice reach master's `_unpool_shape_ok`, `_o_skinny_applies` and `_swapab_route_ok` gates exactly as on master.
  - If a kept route's self-check fails, it falls back to master's shape-gated Triton kernels (the same gates), not straight to `_bf16_gemm_dispatch_impl`.
- **(4) Cold-L2 timing and the kill-gate reading.** `_ll_time_route` now prints, for m in 1/8/16/32:
  - a warm-L2 time (20 back-to-back launches);
  - a cold-L2 time: a 256 MB buffer (larger than the 50 MB L2) is touched with `add_(1)` before each timed launch, and the median of 15 is taken;
  - TB/s compressed and raw-equivalent for both.
  - pooler m=1 also prints `KILL-GATE pooler m=1 cold-L2 X us vs gate 19.0 us: PASS/MISS`. This is timing only and routes nothing. The judged kill gate is still the pooler m=1 **profile** kernel time, and the line says so. The flush buffer is freed at the end of prepare.
- **LESSONS compliance.**
  - No child processes.
  - The self-check syncs and prints after each launch (`launch 1 ok` / `launch 2 ok` per route and m), and after each weight's decode kernel (`decode kernel ok`), so a fault is attributed.
  - Build adds `-lineinfo`. The extension name is bumped to `muse_bf16_lossless_v2`.
  - The CUDA is still mma.sync, ldmatrix and cp.async.cg only (no wgmma or setmaxnreg), so `TORCH_CUDA_ARCH_LIST=9.0a` is not needed and is not set.
- **Bit-exactness.** The codec round trip is still checked bit-exact through the kernel decoder. GEMM outputs will differ from stage 1 in the last bits because the fp32 summation order changed (pair reduce). The self-check stays tolerance-based against fp64 (rel L2 <= 1e-3, worst row <= 5e-3, one-hot row exact, launch 1 == launch 2), as the review requires.
- **Fallback.** All master fallbacks and `PATH CHOSEN` lines are kept.
- **Uncertainties.**
  - The register count at NT=32 with the `Gn` prefetch (about 18 extra registers) is unverified. It is printed, and spills would show as local bytes.
  - Per SMSP there are now 2 warps rather than 4, since 8 warps sit across 4 SMSPs. That may still be too little latency hiding for 2.7 TB/s compressed.
  - The x L2 re-read (BLOCK_N=64) is unchanged.
  - Nothing could be compiled or run on this node.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2 diff`:

- [x] CUDA-graph capturable: kernel_fn is unchanged. `_ll_forward` makes one launch with no sync, no allocation beyond `torch.empty` for out (same as stage 1), and branches only on cfg/shape/strides/prepare constants. The new syncs and prints are in prepare-only helpers (`_ll_check_route`, `_ll_time_route`, `_lossless_prepare`).
- [x] Nothing derived from `data` is cached across calls: no new state keys except the prepare-only flush buffer, which is deleted.
- [x] Weights are repacked losslessly in prepare() only. The codec is unchanged and bit-exact round-trip gated. BF16 stays BF16 (losslessly smaller), and the router is untouched.
- [x] Activations are BF16 where SGLang keeps BF16: x is consumed as bf16, fp32 accumulation, bf16 output. n/a for KV cache and queries: not an attention problem.
- [x] n/a: attention key coverage. This problem has no attention, and nothing is skipped: every K element of every weight row is streamed.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import, subprocesses or network. The diff adds no imports. load_inline is unchanged apart from the name and `-lineinfo`, and it runs in-process with no child process of our own.
- [x] All work happens inside the timed call, with no output caching or replay: the pair reduce and the split-K reduce run inside the kernel each launch.
- [x] The first-line marker `# okc: problem=muse_bf16_proj` is kept, with no triton.autotune and no host sync in kernel_fn.

Lint output:

```
warn  dtype        L1189: sm8 = sm.to(torch.uint8)  (allowed only where SGLang does the same)
warn  dtype        L1190: c8 = c.to(torch.uint8)  (allowed only where SGLang does the same)
warn  dtype        L1202: hdr_t = hdr.view(rt, 16, kg).permute(0, 2, 1).to(torch.uint8).contiguous().view(-1)  (allowed only where SGLang does the same)
warn  host-sync    L1264: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1267: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1308: e1.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1318: c1.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1342: attrs = _LL_EXT.lossless_attrs().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1343: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1386: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1428: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings
```

The dtype warnings are the compressed payload packing to uint8 (lossless codec bytes, not a weight precision change; the same in stage 1). The host-sync warnings are all in prepare(). py_compile OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-24 by an Opus subagent, against stage 1 (7895e56) and master (e5e9a7c).

**Verdict: dispatch.** No fixes were needed. Lint is OK with warnings (prepare-only syncs and codec uint8 packing), py_compile passes, and __pycache__ was removed.

What was checked:
- **Pair reduce.** It is race-free:
  - `cp_wait<0>` and `__syncthreads` run before the ring is reused.
  - hk==1 writes at most 8 KB (NT=32: 2048 floats), then `__syncthreads`, then hk==0 adds.
  - Every barrier is CTA-uniform: `nsteps` and `split` are per-CTA, and the `return` is uniform.
  - hk==0 warps cover all 64 rows for the direct store and the ws partials.
  - The ticket reduce is unchanged.
- **Next-stage prefetch.** A Python emulation of the cp.async group, buffer and stage schedule ran for nsteps 21/22/32 (STAGES=8).
  - `cp_wait<STAGES-3>` guarantees stage i+1 has landed before `load_grp(Gn)` reads it.
  - `issue(i+7)` always writes buffer (i-1)%8, never buffer i or i+1, and it runs after the barrier that ends iteration i-1's reads.
  - The prologue `cp_wait<STAGES-2>` plus a barrier covers stage 0.
- **A pipeline.** The double buffer `a[2][4]` decodes step s+1 into the other slot before step s's MMAs. Indices are compile-time after unrolling, so there is no local-memory indexing.
- **Codec and indexing.** The codec and `decode_step` are unchanged, so the round trip stays bit-exact, and it is still gated in prepare.
  - The warp chunk offset `wt*3072 + hk*1536`, the header offset `(wt*2+hk)*16` and the x slice `kc = hk*2+kp` all match the issue layout.
  - The W issue is exactly 3 chunks per thread at 256 threads.
  - The x issue is guarded at NT=8.
- **Routing.**
  - Outside section (6) and the lossless gate in kernel_fn, the file differs from master only by additions.
  - o and `unpooler_slice` have no LL_CFG entry, so they reach master's `_unpool_shape_ok`, `_o_skinny_applies` and `_swapab_route_ok` exactly as on master.
  - A kept route that fails its self-check falls back to the same master gates.
  - The test shapes (m 7 pooler, m 33 o, m 5 and 130 unpooler) route correctly.
- **Kill-gate print.** It only prints PASS or MISS and routes nothing. The flush buffer is prepare-only and freed afterwards.
- **Compile.** The CUDA is mma.sync, ldmatrix and cp.async only. `__launch_bounds__(256,1)` caps registers at 255, so the launch cannot fail for lack of registers. No constructs looked doubtful when read.
- **Rules.** kernel_fn is unchanged apart from its call graph: one launch and an output `torch.empty`. There are no new imports or processes, nothing derived from `data` is cached, and the weights stay bf16, repacked losslessly in prepare.

Residual risk: the kernel was never compiled, so build or register-pressure surprises at NT=32 stay possible. Any such failure lands on master with a PATH CHOSEN line. On performance, 2 warps per SMSP may still not hide enough latency to reach the 19 us kill gate.

## Result

- job `f1024212dd2f49f3` (check), status done, passed True, tests 9/9
- speedup 0.8870x, 95% CI [0.8737, 0.9003], score 5446.480634650908 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/regression/bf16-lossless-exp4-weight-stream-rewrite.result.json
- **outcome: neutral** (beats-base-not-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-lossless-exp4-occupancy-dequant-once-stage2.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-lossless-exp4-occupancy-dequant-once-stage2.patch`; branch `hyp/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 30.19 | 22.58 | 1.3371 | gained |
| layer=o load=decode_1 m=1 | 62 | 15.46 | 6.46 | 2.3649 | gained |
| layer=pooler load=decode_1 m=1 | 62 | 41.49 | 31.23 | 1.3486 | gained |
| layer=unpooler load=decode_1 m=1 | 62 | 10.98 | 6.09 | 1.7953 | gained |
| layer=router load=decode_1 m=1 | 62 | 6.26 | 5.92 | 1.0195 | flat |
| layer=qkv_gate load=decode_16 m=16 | 62 | 33.86 | 25.65 | 1.284 | gained |
| layer=o load=decode_16 m=16 | 62 | 15.29 | 6.66 | 2.4856 | gained |
| layer=pooler load=decode_16 m=16 | 62 | 44.82 | 34.6 | 1.2846 | gained |
| layer=unpooler load=decode_16 m=16 | 62 | 41.58 | 32.15 | 1.3112 | gained |
| layer=router load=decode_16 m=16 | 62 | 5.85 | 5.82 | 0.9973 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 38.7 | 28.77 | 1.336 | gained |
| layer=o load=decode_32 m=32 | 62 | 16.58 | 6.32 | 2.69 | gained |
| layer=pooler load=decode_32 m=32 | 62 | 51.05 | 38.22 | 1.3465 | gained |
| layer=unpooler load=decode_32 m=32 | 62 | 44.83 | 34.42 | 1.3316 | gained |
| layer=router load=decode_32 m=32 | 62 | 6.59 | 6.47 | 1.0125 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 31.59 | 23.43 | 1.3319 | gained |
| layer=o load=verify_1 m=8 | 62 | 15.56 | 6.65 | 2.4535 | gained |
| layer=pooler load=verify_1 m=8 | 62 | 42.11 | 31.08 | 1.3693 | gained |
| layer=unpooler load=verify_1 m=8 | 62 | 11.49 | 5.91 | 1.939 | gained |
| layer=router load=verify_1 m=8 | 62 | 6.22 | 5.83 | 1.032 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 17.24 | 16.98 | 0.9988 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.5 | 7.59 | 1.0019 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 25.57 | 26.04 | 0.9768 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 20.1 | 20.45 | 0.9886 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.96 | 7.57 | 1.0204 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.57 | 19.93 | 1.0138 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.56 | 9.73 | 0.9913 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 31.12 | 31.22 | 0.9766 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.74 | 24.2 | 1.0078 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.58 | 9.54 | 1.0171 | flat |

gained on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_1, decode_16, decode_32, verify_1; pooler decode_1, decode_16, decode_32, verify_1; unpooler decode_1, decode_16, decode_32, verify_1; router verify_1. Expected hit: 13/13. Noise ±1.5%.

## Follow-ups

- rewrite stage 2 of `Triton swap-AB m<=32 kernels (bf16-m32-swapab-wgmma-triton) on pooler m 1-32, unpooler-full m 16/32, qkv_gate m 1-32`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_bf16_proj "<what to fix or tune>" --kind rewrite --depends-on bf16-lossless-exp4-occupancy-dequant-once-stage2 --stage 3`; a slow or failed first cut is the expected start, not a do-not-retry.
- beats-base-not-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-lossless-exp4-occupancy-dequant-once-stage2`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

