---
problem: muse_decode_layer
slug: moe-chain-cuda-swapab-expert-once-fix-inproc
created: 2026-10-25T001824Z
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
  dispatch). Design unchanged; only these fixes: (1) IN-PROCESS GUARDED FIRST LAUNCH (never in a
  child process: the rules statically reject submissions that start processes and okc_lint flags
  multiprocessing (LESSONS 2026-10-25, corrected)): in prepare(), after the build, confirm the
  sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc
  -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes
  binaryVersion/ptxVersion for moe_route_list, moe_w13_silu_swapab and moe_w2_swapab; anything
  else turns the path OFF); then a 1-CTA micro canary of dequant8 + one mma tile on synthetic
  weights, then a tiny synthetic canary launch of moe_route_list, moe_w13_silu_swapab and
  moe_w2_swapab (separate launches, smallest T above the gate), each followed by
  torch.cuda.synchronize() and a print so a fault names its kernel; only then the whole
  T=40/128/256 self-check in-process, enabling the path only if everything passes and otherwise
  keeping master Marlin. (2) Replace createpolicy.fractional.L2::evict_first +
  cp.async.cg...L2::cache_hint with plain cp.async.cg.shared.global [d],[s],16 (the form
  dense-chain and splitkv run on the judge). (3) In dequant8 replace the inline fma.rn.bf16x2 with
  a -0 addend by __hmul2 on __nv_bfloat162 (scale fold unchanged). (4) Build flags: add -lineinfo;
  set TORCH_CUDA_ARCH_LIST=9.0a / only -gencode arch=compute_90a,code=sm_90a so a single sm_90a
  cubin is built (hygiene; no wgmma). Keep 221 KB dyn smem at 256 threads but print
  cudaFuncGetAttributes and the max-dynamic-smem attribute result before the canary. Print
  [moe-cuda-large-t] ON|OFF with self-check status, faulting kernel name if any, max row rel, and
  per-kernel CUDA-event times at T=128/256. The first measurement of this hypothesis must be an
  `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).
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
review: proceed
status: queued
review_at: 2026-10-25T002143Z
review_companions: null
---

# moe-chain-cuda-swapab-expert-once-fix-inproc

## Change

Stage 1b FIX (numbered stage 2 only because stage is an int; the original stage plan's stage 2 still comes after this) of failed moe-chain-cuda-swapab-expert-once-large-t (job 1bba1bab70da4f61: built, then a sticky 715 CUDA_ERROR_ILLEGAL_INSTRUCTION in the first prepare self-check launch killed the worker, 0/0 tests; REWRITE_READ_2026-10-25 section 5, LESSONS 2026-10-25). Take its patch (done/failed/moe-chain-cuda-swapab-expert-once-large-t.patch) re-applied on current master (stack-align-par-sidestream-sliding1: parallel align at T>32 and MoE dispatch on the side stream; keep the same gate tokens > 32 with its OWN bound constant, LESSONS stack note, and launch the three new kernels on the same side stream master uses for MoE dispatch). Design unchanged; only these fixes: (1) IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): in prepare(), after the build, confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for moe_route_list, moe_w13_silu_swapab and moe_w2_swapab; anything else turns the path OFF); then a 1-CTA micro canary of dequant8 + one mma tile on synthetic weights, then a tiny synthetic canary launch of moe_route_list, moe_w13_silu_swapab and moe_w2_swapab (separate launches, smallest T above the gate), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; only then the whole T=40/128/256 self-check in-process, enabling the path only if everything passes and otherwise keeping master Marlin. (2) Replace createpolicy.fractional.L2::evict_first + cp.async.cg...L2::cache_hint with plain cp.async.cg.shared.global [d],[s],16 (the form dense-chain and splitkv run on the judge). (3) In dequant8 replace the inline fma.rn.bf16x2 with a -0 addend by __hmul2 on __nv_bfloat162 (scale fold unchanged). (4) Build flags: add -lineinfo; set TORCH_CUDA_ARCH_LIST=9.0a / only -gencode arch=compute_90a,code=sm_90a so a single sm_90a cubin is built (hygiene; no wgmma). Keep 221 KB dyn smem at 256 threads but print cudaFuncGetAttributes and the max-dynamic-smem attribute result before the canary. Print [moe-cuda-large-t] ON|OFF with self-check status, faulting kernel name if any, max row rel, and per-kernel CUDA-event times at T=128/256.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `*load=verify_16 *` faster: Marlin w13+w2+act_and_mul 254/271 us (sliding/full) at ~1.1-1.2 TB/s vs MoE floor ~101 us; expert-once swap-AB streams each local expert once per <=64-route chunk: target replaced sum <= Marlin (M1/M2), ~150 us (M3 ~1.5x floor)
- `*load=verify_32 *` faster: Marlin 443/427 us vs floor ~109 us; target <= ~200 us
- `*load=decode_*` flat: T<=32 outside gate: control
- `*load=verify_1 *` flat: T=8 outside gate: control

## Risk

supersedes moe-chain-cuda-swapab-expert-once-fix, held for a child-process self-check

## Milestones

Rewrite stage 2 of `moe_align + Marlin w13 + act_and_mul + Marlin w2 at tokens > 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002143Z by an adversarial subagent (model opus).

**Verdict: proceed.** This rewrite stage only fixes faults, and the target is real: Marlin takes 254-443 us at verify_16/32 against a MoE floor of about 101-109 us. The fixes don't identify the 715 root cause, so treat this run as a diagnostic okc check whose main job is to name the faulting kernel. That only works if the logging is unbuffered and announces each launch before it starts.

### Justification
Mechanism is plausible. FINDINGS shows Marlin at 271.6/326.1 us (sliding/full verify_16) and 443.2/449.7 us (verify_32), 47-70% of layer time, running at about 1.1-1.2 TB/s against a weight-stream floor of roughly 101-109 us. A persistent 132-CTA swap-AB mma.sync kernel that streams each touched local expert once per chunk of up to 64 routes can reasonably reach about 2 TB/s on H100. The 'decode_* / verify_1 flat' controls are consistent because the tokens > 32 gate only admits T=128/256. Rewrite milestones M1-M3 can be read from profile top_kernels (moe_w13_silu_swapab plus moe_w2_swapab plus moe_route_list against Marlin x2 plus act_and_mul plus align). Weakest link: neither of the 'proven-form' swaps clearly removes the fault. (3) is probably a SASS no-op, because cuda_bf16.hpp implements __hmul2 for bf16 on sm_90 as fma.rn.bf16x2 with a -0.0 addend, i.e. the same instruction as before. (2) createpolicy/cache_hint is standard CUTLASS PTX on sm_80+ and an unlikely trap source. ptxas accepted every form, so an encoding problem is less likely than a trap inside a .sync.aligned collective (ldmatrix/mma inside `if (f < me_nf)`), a stray trap/assert, or a bad smem configuration. The run will probably fault again. Its value is the canary ladder that attributes the fault, which the sibling moe failure never got. -lineinfo will not name the instruction in a judge log without compute-sanitizer. Two design points check out: chunk capacity (routes//64 + 33 >= worst case sum ceil(c_e/64) <= routes/64 + 32), and the bf16 rounding of hg/hu before silu, which matches Marlin's bf16 w13 output.

### Confounders on the current kernel
- stack-align-par-sidestream-sliding1: at T>32 the new moe_route_list replaces master's parallel align on the side stream. The incumbent is therefore parallel-align + Marlin, which is the right baseline. Stream placement matters, though: the new kernels must run on state['side_stream'] inside the existing wait_stream fence (submission.py ~1830-1844). Otherwise a gain or loss may come partly from lost or gained overlap rather than from the GEMMs. Read kernel times, not the headline.
- marlin-block-m-cap-16: the incumbent Marlin times (254-449 us) are measured with block_m capped at 16 at T=128/256. A different cap would move the incumbent, not this kernel. Comparing against today's master is correct and needs no A/B.
- post-expert-norm-triton-small-t: gated to small T, so at T>32 the SGLang fused_post_expert_norm_weighted_sum consumes y. It is not a confounder, as long as y's layout ([T,8,K], remote rows never read, slot_expert_ids=local_ids) is unchanged.

### Interference and A/B needs
No companions. This is a rewrite stage whose first measurement must be an okc check. Its main outcomes are 'faults with the kernel named' or 'passes, and the kernel times can be read against Marlin per shape', and a revert or tune of any landed item disambiguates neither. The decode_*/verify_1 shapes act as built-in controls because they fall outside the gate. If it passes, the next stage is the follow-up.

### Open questions
- Prints must use flush=True (or sys.stdout.flush()), and each kernel needs a 'launching <kernel>' line before its launch as well as an 'ok' line after its synchronize. A sticky 715 kills the worker, and buffered stdout would lose exactly the line that names the kernel.
- Order the canaries so each step adds one suspect form: cvt e4m3x2 alone, then dequant8 (hmul2) alone, then ldmatrix.x4 plus one mma.sync tile, then cp.async.cg at the 221 KB smem configuration, then the route list, then w13, then w2. A fault then identifies a form, not just a kernel.
- Is the __hmul2 change actually different SASS from the old fma.rn.bf16x2 with a -0 addend? If not, add a cuobjdump/SASS note to the attempt file so the next stage does not count it as an eliminated suspect.
- If the canaries pass but the full launch faults, the remaining suspects are an unconverged .sync.aligned collective (me_nf or item-end not truly CTA-uniform across persistent items) and the 221 KB smem configuration. A next stage could test 2 stages and about 120 KB smem.

