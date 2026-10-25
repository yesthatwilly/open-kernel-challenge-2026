---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc
created: 2026-10-25T001824Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  RE-MEASURE (stage 1b): the previous run never executed the new kernel; load_inline failed
  because cpp_extension appended its own -gencode arch=compute_90,code=sm_90 beside -arch=sm_90a
  and ptxas cannot assemble wgmma.mma_async/fence/commit_group/wait_group or setmaxnreg.inc/dec
  for plain sm_90, so it fell back to Marlin and measured master. Apply the archived kernel
  verbatim (muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-verify.patch,
  spec in done/neutral/2026-10-24T172038Z-cuda-nvfp4-wgmma-rega-grouped-verify.md), rebased onto
  current master (align-parallel-per-expert-verify), and make ONLY the build fix: (1) set
  os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' immediately before load_inline (restore the previous
  value after), and pass only -gencode arch=compute_90a,code=sm_90a in extra_cuda_cflags; (2) wrap
  every wgmma and setmaxnreg body in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) ... #else set a device
  error flag and return; #endif so any stray compute_90 pass still assembles and turns the path
  OFF instead of trapping; (3) add -lineinfo. Also an IN-PROCESS GUARDED FIRST LAUNCH (never in a
  child process: the rules statically reject submissions that start processes and okc_lint flags
  multiprocessing (LESSONS 2026-10-25, corrected)): in setup, confirm the sm_90a image actually
  loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only
  compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for each new kernel;
  anything else turns the path OFF); read the error flag back after the canary (non-zero -> OFF);
  then a tiny synthetic canary launch of the gather, w13 and w2 kernels on synthetic weights
  (smallest gated M), each followed by torch.cuda.synchronize() and a print so a fault names its
  kernel; only then the full setup self-check in-process, enabling the path only if it prints
  PASSED. No tiling, pipeline, dequant, gate (num_tokens >= 128, own bound constant) or layout
  change. Log the build line and PATH CHOSEN; a run whose log does not say PATH CHOSEN = wgmma is
  unmeasured, not neutral. The first measurement of this hypothesis must be an `okc check`, never
  a scored submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: bs=128 *
    direction: faster
    why: "verify_16: incumbent Marlin w13+w2 316.1 us + act_and_mul 3.9 us vs floor 101.3 us; M2 = replaced kernels below incumbent (~320 us), M3 <= 170 us (~2.9 TB/s). Marlin rows move +/-20% run to run, so judge on the replaced-kernel row, not the headline"
  - shapes: bs=256 *
    direction: faster
    why: "verify_32: incumbent 490.2 + 6.4 us (403.1 in the fallback rerun) vs floor 108.9 us; M3 <= 220 us via wgmma register-A, one weight stream per expert chunk"
  - shapes: bs=32 *
    direction: flat
    why: "control: num_tokens < 128 runs master"
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-verify
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc

## Change

RE-MEASURE (stage 1b): the previous run never executed the new kernel; load_inline failed because cpp_extension appended its own -gencode arch=compute_90,code=sm_90 beside -arch=sm_90a and ptxas cannot assemble wgmma.mma_async/fence/commit_group/wait_group or setmaxnreg.inc/dec for plain sm_90, so it fell back to Marlin and measured master. Apply the archived kernel verbatim (muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-verify.patch, spec in done/neutral/2026-10-24T172038Z-cuda-nvfp4-wgmma-rega-grouped-verify.md), rebased onto current master (align-parallel-per-expert-verify), and make ONLY the build fix: (1) set os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' immediately before load_inline (restore the previous value after), and pass only -gencode arch=compute_90a,code=sm_90a in extra_cuda_cflags; (2) wrap every wgmma and setmaxnreg body in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) ... #else set a device error flag and return; #endif so any stray compute_90 pass still assembles and turns the path OFF instead of trapping; (3) add -lineinfo. Also an IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): in setup, confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for each new kernel; anything else turns the path OFF); read the error flag back after the canary (non-zero -> OFF); then a tiny synthetic canary launch of the gather, w13 and w2 kernels on synthetic weights (smallest gated M), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; only then the full setup self-check in-process, enabling the path only if it prints PASSED. No tiling, pipeline, dequant, gate (num_tokens >= 128, own bound constant) or layout change. Log the build line and PATH CHOSEN; a run whose log does not say PATH CHOSEN = wgmma is unmeasured, not neutral.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `bs=128 *` faster: verify_16: incumbent Marlin w13+w2 316.1 us + act_and_mul 3.9 us vs floor 101.3 us; M2 = replaced kernels below incumbent (~320 us), M3 <= 170 us (~2.9 TB/s). Marlin rows move +/-20% run to run, so judge on the replaced-kernel row, not the headline
- `bs=256 *` faster: verify_32: incumbent 490.2 + 6.4 us (403.1 in the fallback rerun) vs floor 108.9 us; M3 <= 220 us via wgmma register-A, one weight stream per expert chunk
- `bs=32 *` flat: control: num_tokens < 128 runs master
- `bs=16 *` flat: control
- `bs=8 *` flat: control
- `bs=1 *` flat: control: untouched

## Risk

supersedes cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild, held for a child-process self-check

## Milestones

Rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
