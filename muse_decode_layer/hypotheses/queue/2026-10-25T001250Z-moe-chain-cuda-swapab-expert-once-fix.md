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
review: hold
status: hold
review_at: 2026-10-25T001517Z
review_companions: null
hold_reason: "adversarial review: Hold for a rules problem only: fix (1), the central fix, runs the self-check in a torch.multiprocessing spawn child or subprocess, and TASK.md says a submission"
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

## Adversarial review

Reviewed 2026-10-25T001517Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold for a rules problem only: fix (1), the central fix, runs the self-check in a torch.multiprocessing spawn child or subprocess, and TASK.md says a submission that 'starts processes' is rejected (this workspace treats subprocesses as a violation, e.g. the cuobjdump check dropped in muse_attention_decode full-decode-cuda-fp8-qk-mma). Re-mint with fixes (2)-(4) and the per-kernel in-process sync/print diagnostics, no child process, run as `okc check` rather than submit, and the rewrite is otherwise sound to proceed.

### Justification
The mechanism is plausible and the profile supports it. In FINDINGS (master f1f2f35), Marlin is 271.6/326.1 us at verify_16 sliding/full and 443.2/449.7 us at verify_32. That is 63%/51% and 70%/47% of those shapes, against whole-layer floors of 126.6/188.7/143.8/267.2 us. The MoE weight stream for 32 local NVFP4 experts at T=128/256 is near-dense, so Marlin, which re-streams experts per block_m tile, runs well below 4.8 TB/s. Expert-once swap-AB, with tokens on the N side of m16n8k16 and each expert's weight streamed once per <=64-route chunk, is the right shape of fix on H200 without wgmma. The expected directions fit the weights: verify_16/32 faster, decode_*/verify_1 (T<=32 and T=8) flat because the gate excludes them. The ~150/~200 us targets (about 1.5x the MoE floor) are ambitious but physically reachable. The fix diagnosis follows REWRITE_READ_2026-10-25 section 5. The two never-judge-proven PTX forms (createpolicy.fractional + cp.async L2::cache_hint with a 64-bit policy, and the fma.rn.bf16x2 dequant with a -0 addend) are swapped for forms that dense-chain and splitkv already run on the judge. That is the right first move. The muse_moe_w4a16 sibling died the same way with no route-list kernel, which points at the shared GEMM body. WEAKEST LINK: the faulting opcode was never pinned. If the trap is a .sync.aligned collective (ldmatrix/mma.sync) run by a non-converged warp, from the producer/consumer meta lambdas or the per-thread `rows[i] = p < n ? ...` paths around the item boundaries, neither swap fixes it, and the run dies again. Without the child process, nothing can recover from a sticky 715 in-process, so a second fault costs the whole run (0/0 tests) and gives only the last-printed kernel name as the next clue.

### Confounders on the current kernel
- stack-align-par-sidestream-sliding1: master's T>32 path runs the parallel moe_align plus the expert_map gather on the side stream (kernel_fn lines ~1829-1846), overlapping the pooler GEMM and pre_expert_norm. The rewrite replaces moe_align at T>32 with moe_route_list. If moe_route_list stays on the side stream as specified, the incumbent comparison is fair. If it lands on the main stream, part of any verify_16/32 delta is lost overlap rather than GEMM time. Read the profile's per-kernel times, not the headline.
- marlin-block-m-cap-16: this is the incumbent Marlin configuration at T=128/256 (block_m clamped to 16), so it sets the bar the replaced sum must beat. A different cap would move the bar but not the new kernels' own time. Not a confounder for the rewrite's milestones, which are judged on the replaced kernels' own time.
- post-expert-norm-triton-small-t: gated to small T. At T>32 both master and the rewrite end in SGLang's fused_post_expert_norm_weighted_sum (the patch keeps that call), so the numerical and launch path after w2 is the same. No confound, provided the new w2 output keeps Marlin's y layout and dtype ([M*8, 4096] bf16, slot order matching local_ids).

### Interference and A/B needs
No companions. This is a rewrite stage judged on the replaced kernels' per-kernel time in the verify_16/32 profiles against Marlin's 254-450 us and the floor. The gate (tokens > 32) leaves decode_* and verify_1 as built-in controls, and the incumbent kernels sit in master's own profile. An ab-revert of stack-align-par-sidestream-sliding1 would only re-measure the align overlap, not the GEMM rewrite. The open question is correctness and survival (does it fault?), and only the next stage answers that.

### Open questions
- Without a child process, is there a rules-clean way to survive a sticky 715? I don't believe so in-process. Either accept losing a check per fault, or first run the 1-CTA micro self-test on its own as a tiny diagnostic check, with the full launch disabled, to clear dequant8 and the mma tile cheaply.
- Are the ldmatrix/mma.sync calls reached by every lane of each warp at the chunk and item boundaries (nn_n < 8*NF_MAX, the bx >= total early return, the kk >= nk tail of load_meta)? A non-converged .sync.aligned collective also traps as an illegal instruction, and neither swap would fix that.
- Does moe_route_list launch on state['side_stream'] where master's _expert_dispatch runs now, and does the w13 launch wait on it through the existing main_stream.wait_stream(side_stream)?
- 221184 B dynamic smem at 256 threads leaves one CTA per SM. Does the 132-CTA persistent grid still keep enough bytes in flight to beat ~1.2 TB/s at T=128, where many local experts get few routes?

