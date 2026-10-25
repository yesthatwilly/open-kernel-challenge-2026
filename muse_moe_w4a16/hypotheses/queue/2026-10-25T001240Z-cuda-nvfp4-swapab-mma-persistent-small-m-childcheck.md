---
problem: muse_moe_w4a16
slug: cuda-nvfp4-swapab-mma-persistent-small-m-childcheck
created: 2026-10-25T001240Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  FIX (stage 1b, make it run): the previous run built (51.2 s, 80-128 regs, 0 local) but its first
  CUDA launch in _cuda_self_check hit illegal instruction 715 (sticky), so the in-process fallback
  could not save it and prepare() crashed (tests 0/0). Apply the archived kernel
  (muse_moe_w4a16/hypotheses/done/failed/cuda-nvfp4-swapab-mma-persistent-small-m.patch; spec
  done/failed/2026-10-24T165851Z-cuda-nvfp4-swapab-mma-persistent-small-m.md) rebased onto current
  master (align-parallel-per-expert-verify), with exactly three changes and nothing else: (1) move
  the self-check into a spawned child process (multiprocessing spawn context, same load_inline
  build dir) that first runs a canary of w13+silu at M=1, NT=1 and then w2 at M=1, NT=1 as
  SEPARATE launches, printing a line before and after each, then the full per-M self-check; the
  parent enables the CUDA path only if the child exits 0 and prints PASSED, else PATH CHOSEN =
  MARLIN with the child's last line logged; the parent never launches the kernel before that; (2)
  replace cp_async16_ef (createpolicy.fractional.L2::evict_first + cp.async.cg ... L2::cache_hint)
  with the bs=1 kernel's proven plain cp.async.cg.shared.global 16 B form and delete createpolicy;
  (3) add -lineinfo to extra_cuda_cflags. Keep ldmatrix.x4 and the cp.async.ca src_bytes zero-fill
  as is (next suspects if it still traps), and keep tiling, dequant, layout and the bs<=32 gate
  unchanged.
expected:
  - shapes: bs=1 *
    direction: faster
    why: incumbent Marlin + silu 44.6 + 1.6 us, floor 11.4, M3 <= 18 us; M1 = runs with PATH CHOSEN = CUDA and passes all tests, M2 = replaced kernels below incumbent
  - shapes: bs=8 *
    direction: faster
    why: 105.8 + 2.0 us vs floor 37.3, M3 <= 56
  - shapes: bs=16 *
    direction: faster
    why: 116.0 + 2.2 us vs floor 52.5, M3 <= 79
  - shapes: bs=32 *
    direction: faster
    why: 187.6 + 2.3 us vs floor 70.8, M3 <= 106
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
  - shapes: bs=256 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: cuda-nvfp4-swapab-mma-persistent-small-m
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at bs<=32
review: required
status: queued
---

# cuda-nvfp4-swapab-mma-persistent-small-m-childcheck

## Change

FIX (stage 1b, make it run): the previous run built (51.2 s, 80-128 regs, 0 local) but its first CUDA launch in _cuda_self_check hit illegal instruction 715 (sticky), so the in-process fallback could not save it and prepare() crashed (tests 0/0). Apply the archived kernel (muse_moe_w4a16/hypotheses/done/failed/cuda-nvfp4-swapab-mma-persistent-small-m.patch; spec done/failed/2026-10-24T165851Z-cuda-nvfp4-swapab-mma-persistent-small-m.md) rebased onto current master (align-parallel-per-expert-verify), with exactly three changes and nothing else: (1) move the self-check into a spawned child process (multiprocessing spawn context, same load_inline build dir) that first runs a canary of w13+silu at M=1, NT=1 and then w2 at M=1, NT=1 as SEPARATE launches, printing a line before and after each, then the full per-M self-check; the parent enables the CUDA path only if the child exits 0 and prints PASSED, else PATH CHOSEN = MARLIN with the child's last line logged; the parent never launches the kernel before that; (2) replace cp_async16_ef (createpolicy.fractional.L2::evict_first + cp.async.cg ... L2::cache_hint) with the bs=1 kernel's proven plain cp.async.cg.shared.global 16 B form and delete createpolicy; (3) add -lineinfo to extra_cuda_cflags. Keep ldmatrix.x4 and the cp.async.ca src_bytes zero-fill as is (next suspects if it still traps), and keep tiling, dequant, layout and the bs<=32 gate unchanged.

## Expected

- `bs=1 *` faster: incumbent Marlin + silu 44.6 + 1.6 us, floor 11.4, M3 <= 18 us; M1 = runs with PATH CHOSEN = CUDA and passes all tests, M2 = replaced kernels below incumbent
- `bs=8 *` faster: 105.8 + 2.0 us vs floor 37.3, M3 <= 56
- `bs=16 *` faster: 116.0 + 2.2 us vs floor 52.5, M3 <= 79
- `bs=32 *` faster: 187.6 + 2.3 us vs floor 70.8, M3 <= 106
- `bs=128 *` flat: control: untouched
- `bs=256 *` flat: control: untouched

## Risk

Stage 1b recorded as stage 1 (hyp.py takes an integer stage). Primary milestone is M1 (it runs); perf unknown, closest evidence is the bs=1 fused kernel at ~1.5 TB/s. If the child still traps, the log names the instantiation; next stage swaps ldmatrix.x4 for 32-bit ld.shared B-fragment loads or fixes the zfill path. cuda-fused-moe-midm-single-launch-routed stays held until this harvests with PATH CHOSEN = CUDA and rows reaching M2. Same helper set as decode's moe-chain-cuda-swapab-expert-once-large-t, which died with the same 715.

## Milestones

Rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at bs<=32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
