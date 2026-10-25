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
status: running
review_at: 2026-10-25T001525Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T001525Z
base_sha: 7895e56373c79d74ba78b7c0e7146755b118ce2c
base_blob: a34e409b452e9a6e5f7ee3612486d30a989832f2
branch: hyp/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2
worktree: .worktrees/muse_bf16_proj/bf16-lossless-exp4-occupancy-dequant-once-stage2
mode: null
restack_of: null
stale_base: null
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

