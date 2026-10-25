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
review: required
status: queued
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
