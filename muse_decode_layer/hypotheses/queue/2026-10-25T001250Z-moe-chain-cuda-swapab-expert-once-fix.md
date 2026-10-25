---
problem: muse_decode_layer
slug: moe-chain-cuda-swapab-expert-once-fix
created: 2026-10-25T001250Z
base: master
area: w13 marlin + silu-mul + w2 marlin (T>32)
change: >
  Stage 1b FIX (numbered stage 2 only because stage is an int; the original stage plan's stage 2
  still comes after this) of failed moe-chain-cuda-swapab-expert-once-large-t (job
  1bba1bab70da4f61: built, then a sticky 715 CUDA_ERROR_ILLEGAL_INSTRUCTION in the first prepare
  self-check launch killed the worker, 0/0 tests; REWRITE_READ_2026-10-25 section 5, LESSONS
  2026-10-25). Take its patch (done/failed/moe-chain-cuda-swapab-expert-once-large-t.patch)
  re-applied on current master (stack-align-par-sidestream-sliding1: parallel align at T>32 and
  MoE dispatch on the side stream; keep the same gate tokens > 32 with its OWN bound constant,
  LESSONS stack note, and launch the three new kernels on the same side stream master uses for MoE
  dispatch). Design unchanged; only these fixes: (1) FAULT-SAFE SELF-CHECK: run the compile-cached
  TU's first launches and the whole T=40/128/256 self-check in a torch.multiprocessing spawn child
  (or subprocess importing the same load_inline build dir, same module name so no recompile),
  which returns OK plus max errors or the error text; the parent enables the path only on OK and
  otherwise keeps master Marlin; the parent never launches the new kernels before the child
  reports OK. Inside the child call torch.cuda.synchronize() and print after each of
  moe_route_list, moe_w13_silu_swapab and moe_w2_swapab so a fault names its kernel; also a 1-CTA
  micro self-test of dequant8 + one mma tile before the full launch. (2) Replace
  createpolicy.fractional.L2::evict_first + cp.async.cg...L2::cache_hint with plain
  cp.async.cg.shared.global [d],[s],16 (the form dense-chain and splitkv run on the judge). (3) In
  dequant8 replace the inline fma.rn.bf16x2 with a -0 addend by __hmul2 on __nv_bfloat162 (scale
  fold unchanged). (4) Build flags: add -lineinfo; set TORCH_CUDA_ARCH_LIST=9.0a / only -gencode
  arch=compute_90a,code=sm_90a so a single sm_90a cubin is built (hygiene; no wgmma). Keep 221 KB
  dyn smem at 256 threads but print cudaFuncGetAttributes and the max-dynamic-smem attribute
  result in the child. Print [moe-cuda-large-t] ON|OFF with child status, faulting kernel name if
  any, max row rel, and per-kernel CUDA-event times at T=128/256.
expected:
  - shapes: "*load=verify_16 *"
    direction: faster
    why: "Marlin w13+w2+act_and_mul 254/271 us (sliding/full) at ~1.1-1.2 TB/s vs MoE floor ~101 us; expert-once swap-AB streams each local expert once per <=64-route chunk: target replaced sum <= Marlin (M1/M2), ~150 us (M3 ~1.5x floor)"
  - shapes: "*load=verify_32 *"
    direction: faster
    why: Marlin 443/427 us vs floor ~109 us; target <= ~200 us
  - shapes: "*load=decode_*"
    direction: flat
    why: "T<=32 outside gate: control"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "T=8 outside gate: control"
risk: high
depends_on: moe-chain-cuda-swapab-expert-once-large-t
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: moe_align + Marlin w13 + act_and_mul + Marlin w2 at tokens > 32
review: required
status: queued
---

# moe-chain-cuda-swapab-expert-once-fix

## Change

Stage 1b FIX (numbered stage 2 only because stage is an int; the original stage plan's stage 2 still comes after this) of failed moe-chain-cuda-swapab-expert-once-large-t (job 1bba1bab70da4f61: built, then a sticky 715 CUDA_ERROR_ILLEGAL_INSTRUCTION in the first prepare self-check launch killed the worker, 0/0 tests; REWRITE_READ_2026-10-25 section 5, LESSONS 2026-10-25). Take its patch (done/failed/moe-chain-cuda-swapab-expert-once-large-t.patch) re-applied on current master (stack-align-par-sidestream-sliding1: parallel align at T>32 and MoE dispatch on the side stream; keep the same gate tokens > 32 with its OWN bound constant, LESSONS stack note, and launch the three new kernels on the same side stream master uses for MoE dispatch). Design unchanged; only these fixes: (1) FAULT-SAFE SELF-CHECK: run the compile-cached TU's first launches and the whole T=40/128/256 self-check in a torch.multiprocessing spawn child (or subprocess importing the same load_inline build dir, same module name so no recompile), which returns OK plus max errors or the error text; the parent enables the path only on OK and otherwise keeps master Marlin; the parent never launches the new kernels before the child reports OK. Inside the child call torch.cuda.synchronize() and print after each of moe_route_list, moe_w13_silu_swapab and moe_w2_swapab so a fault names its kernel; also a 1-CTA micro self-test of dequant8 + one mma tile before the full launch. (2) Replace createpolicy.fractional.L2::evict_first + cp.async.cg...L2::cache_hint with plain cp.async.cg.shared.global [d],[s],16 (the form dense-chain and splitkv run on the judge). (3) In dequant8 replace the inline fma.rn.bf16x2 with a -0 addend by __hmul2 on __nv_bfloat162 (scale fold unchanged). (4) Build flags: add -lineinfo; set TORCH_CUDA_ARCH_LIST=9.0a / only -gencode arch=compute_90a,code=sm_90a so a single sm_90a cubin is built (hygiene; no wgmma). Keep 221 KB dyn smem at 256 threads but print cudaFuncGetAttributes and the max-dynamic-smem attribute result in the child. Print [moe-cuda-large-t] ON|OFF with child status, faulting kernel name if any, max row rel, and per-kernel CUDA-event times at T=128/256.

## Expected

- `*load=verify_16 *` faster: Marlin w13+w2+act_and_mul 254/271 us (sliding/full) at ~1.1-1.2 TB/s vs MoE floor ~101 us; expert-once swap-AB streams each local expert once per <=64-route chunk: target replaced sum <= Marlin (M1/M2), ~150 us (M3 ~1.5x floor)
- `*load=verify_32 *` faster: Marlin 443/427 us vs floor ~109 us; target <= ~200 us
- `*load=decode_*` flat: T<=32 outside gate: control
- `*load=verify_1 *` flat: T=8 outside gate: control

## Risk

High: the fault opcode was never pinned; if the child still faults with plain cp.async and __hmul2, the -lineinfo line and the per-kernel sync name it for the next fix, and the run still completes on Marlin (a fall-back run is a valid UNMEASURED result, not a regression). Judge on the replaced kernels' time in the verify_16/32 profiles. The same fix applies to moe cuda-nvfp4-swapab-mma-persistent-small-m.

## Milestones

Rewrite stage 2 of `moe_align + Marlin w13 + act_and_mul + Marlin w2 at tokens > 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
