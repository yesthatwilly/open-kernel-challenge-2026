---
problem: muse_prefill_layer
slug: experts-cuda-wgmma-s2-setmaxnreg-pingpong-first-turn
created: 2026-10-25T042410Z
base: master
area: "w13 + w2 expert GEMMs at T > 4096 (CUDA sm_90a wgmma W4A16 grouped GEMM, stage 2: setmaxnreg register split + overlapped register-A dequant)"
change: >
  Stage 2 of the CUDA wgmma expert line, on the stage-1b branch
  (experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc, job 6d047cdeefd94357, built on master
  a81d19c). Stage 1b proved the sm_90a build runs but _nvfp4_wgmma_grouped_kernel took 14.7-16.2
  ms vs master's Triton 8.1-9.2 ms, with 168 regs, a 680 B local spill, 1 CTA/SM, 288 threads (1
  producer warp + 2 cooperative consumer warpgroups on m64n256k16 RS wgmma). Change ONLY
  _nvfp4_wgmma_grouped_kernel's thread layout and mainloop; build, canary, one-hot/e2e checks,
  master's 32-expert align and state['zero_nonlocal_w2'] handling stay. (1) Register split: 384
  threads = 1 producer warpgroup + 2 consumer warpgroups, __launch_bounds__(384, 1); the producer
  warpgroup runs setmaxnreg.dec.sync.aligned.u32 40 and one elected thread issues the
  cp.async.bulk weight/scale copies and the activation-tile loads into the existing 5-stage ring;
  consumers run setmaxnreg.inc.sync.aligned.u32 232. Target 0 B local (the m64n256 fp32
  accumulator alone is 128 regs/thread). (2) Overlap the register-A dequant with the tensor core
  inside each consumer warpgroup: double-buffer the A fragments (2 x 4 regs per k16 step) so the
  dequant of k16 step j+1 (E2M1 nibbles -> bf16x2 via the existing bit trick, times the e4m3 block
  scale) is issued after wgmma.commit_group of step j and before wgmma.wait_group 1, as CUTLASS's
  mixed-input RS mainloop does; today each step dequantizes then waits for its own wgmma,
  serializing ALU and MMA. (3) Offset the two consumer warpgroups by half a BLOCK_K via a
  named-barrier handoff so their dequant phases do not coincide. Tile shape (BLOCK_W 128 = 2 x m64
  rows, BLOCK_T 256 tokens) and the persistent grid of 132 are unchanged. Build rules (LESSONS
  2026-10-25): torch load_inline with os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' set just before
  the build (restored after) and only -gencode arch=compute_90a,code=sm_90a plus -std=c++20 -O3
  -lineinfo in extra_cuda_cflags; all wgmma/setmaxnreg asm under #if
  defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets a device error word and returns; print the
  nvcc flags, compile seconds, and regs / local bytes / smem / binaryVersion per kernel.
  In-process only (never a child process: the rules reject submissions that start processes).
  Prepare canary: one tiny launch per kernel, each followed by torch.cuda.synchronize() and a
  flushed print, then a finiteness check that names the first non-finite tensor; self-checks must
  force every tile/bucket shape (an all-one-expert routing and a skewed one, not only random
  routings) and compare against an fp32 dequant reference with a bound calibrated on the
  incumbent's own error (not a fixed 1e-2 against another approximate kernel). Hard gate:
  interleaved CUDA-graph replays with a cold-L2 flush (>=7 alternations) of the new kernels vs the
  incumbent at the scored T; ON only at <= 0.92x; print both times and PATH CHOSEN and assert the
  chosen one is the faster. Any failure prints one OFF: <reason> line and master runs unchanged.
  The first measurement must be an okc check, never a scored submit. Gate shape: T=16384 soft
  routing, CUDA w13 and w2 each vs master's Triton call, gated per GEMM.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: replaced kernels 8.1-9.2 ms (Triton incumbent) / 14.7-16.2 ms (stage 1b) -> <= 7.5 ms, floor 1670 us; first_turn shapes -4 to -8% if gated ON
  - shapes: "*load=turn_1*"
    direction: unchanged
    why: Marlin path untouched
  - shapes: "*load=turn_4*"
    direction: unchanged
    why: Marlin path untouched
risk: high
depends_on: experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _moe_nvfp4_gemm_kernel x2 (Triton w13, w2) at T > 4096 (first_turn)
review: required
status: queued
---

# experts-cuda-wgmma-s2-setmaxnreg-pingpong-first-turn

## Change

Stage 2 of the CUDA wgmma expert line, on the stage-1b branch (experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc, job 6d047cdeefd94357, built on master a81d19c). Stage 1b proved the sm_90a build runs but _nvfp4_wgmma_grouped_kernel took 14.7-16.2 ms vs master's Triton 8.1-9.2 ms, with 168 regs, a 680 B local spill, 1 CTA/SM, 288 threads (1 producer warp + 2 cooperative consumer warpgroups on m64n256k16 RS wgmma). Change ONLY _nvfp4_wgmma_grouped_kernel's thread layout and mainloop; build, canary, one-hot/e2e checks, master's 32-expert align and state['zero_nonlocal_w2'] handling stay. (1) Register split: 384 threads = 1 producer warpgroup + 2 consumer warpgroups, __launch_bounds__(384, 1); the producer warpgroup runs setmaxnreg.dec.sync.aligned.u32 40 and one elected thread issues the cp.async.bulk weight/scale copies and the activation-tile loads into the existing 5-stage ring; consumers run setmaxnreg.inc.sync.aligned.u32 232. Target 0 B local (the m64n256 fp32 accumulator alone is 128 regs/thread). (2) Overlap the register-A dequant with the tensor core inside each consumer warpgroup: double-buffer the A fragments (2 x 4 regs per k16 step) so the dequant of k16 step j+1 (E2M1 nibbles -> bf16x2 via the existing bit trick, times the e4m3 block scale) is issued after wgmma.commit_group of step j and before wgmma.wait_group 1, as CUTLASS's mixed-input RS mainloop does; today each step dequantizes then waits for its own wgmma, serializing ALU and MMA. (3) Offset the two consumer warpgroups by half a BLOCK_K via a named-barrier handoff so their dequant phases do not coincide. Tile shape (BLOCK_W 128 = 2 x m64 rows, BLOCK_T 256 tokens) and the persistent grid of 132 are unchanged. Build rules (LESSONS 2026-10-25): torch load_inline with os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' set just before the build (restored after) and only -gencode arch=compute_90a,code=sm_90a plus -std=c++20 -O3 -lineinfo in extra_cuda_cflags; all wgmma/setmaxnreg asm under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets a device error word and returns; print the nvcc flags, compile seconds, and regs / local bytes / smem / binaryVersion per kernel. In-process only (never a child process: the rules reject submissions that start processes). Prepare canary: one tiny launch per kernel, each followed by torch.cuda.synchronize() and a flushed print, then a finiteness check that names the first non-finite tensor; self-checks must force every tile/bucket shape (an all-one-expert routing and a skewed one, not only random routings) and compare against an fp32 dequant reference with a bound calibrated on the incumbent's own error (not a fixed 1e-2 against another approximate kernel). Hard gate: interleaved CUDA-graph replays with a cold-L2 flush (>=7 alternations) of the new kernels vs the incumbent at the scored T; ON only at <= 0.92x; print both times and PATH CHOSEN and assert the chosen one is the faster. Any failure prints one OFF: <reason> line and master runs unchanged. The first measurement must be an okc check, never a scored submit. Gate shape: T=16384 soft routing, CUDA w13 and w2 each vs master's Triton call, gated per GEMM.

## Expected

- `*load=first_turn*` faster: replaced kernels 8.1-9.2 ms (Triton incumbent) / 14.7-16.2 ms (stage 1b) -> <= 7.5 ms, floor 1670 us; first_turn shapes -4 to -8% if gated ON
- `*load=turn_1*` unchanged: Marlin path untouched
- `*load=turn_4*` unchanged: Marlin path untouched

## Risk

Milestones on the replaced rows: M0 0 B local after setmaxnreg; M1 <= 10 ms per first_turn shape (>= 1.5x faster than stage 1b); M2 <= 0.92x Triton in the prepare A/B so it ships. Kill: a spill remains with setmaxnreg, or M1 missed -> the RS-cooperative design is dead and the line goes to a different decomposition. A run whose canary fails measures master; check the log for ON and PATH CHOSEN T=16384: cuda before reading rows.

## Milestones

Rewrite stage 2 of `_moe_nvfp4_gemm_kernel x2 (Triton w13, w2) at T > 4096 (first_turn)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
