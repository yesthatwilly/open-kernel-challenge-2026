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
review: hold
status: hold
review_at: 2026-10-25T001525Z
review_companions: null
hold_reason: "adversarial review: Hold. Change (1) moves the self-check into a multiprocessing spawn child. TASK.md and the technical rules statically reject any submission that starts processes"
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

## Adversarial review

Reviewed 2026-10-25T001525Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold. Change (1) moves the self-check into a multiprocessing spawn child. TASK.md and the technical rules statically reject any submission that starts processes, and okc_lint flags `multiprocessing` at level error, so this run would be refused or rejected before it could say anything about the 715 trap. Re-mint it without the child process: keep changes (2) and (3) and add in-process canary launches, each followed by a synchronize and a print.

### Justification
A: The stage-1 design is still sound and was reviewed as proceed last time. FINDINGS puts Marlin at 87-95% of every small-M shape: 44.7 us at bs=1, 114.0 at bs=8, 176.1 at bs=16 and 207.5 at bs=32. The floors are 11.4, 37.3, 52.5 and 70.8 us, so there is 2-3x of headroom on loads weighted 60 each. The previous run compiled cleanly (80-128 registers, no local memory, smem 88-136 KB) and failed only at its first launch with a sticky 715. The milestone for this stage is M1 (it runs). Fix (2) is a reasonable bisection: replace the createpolicy.fractional.L2::evict_first plus cp.async.cg...L2::cache_hint pair with the plain cp.async.cg form that the working bs=1 kernel uses. The decode moe-chain kernel shared this helper set and died with the same 715, while the bs=1 kernel (plain cp.async.cg, no ldmatrix, no zero-fill cp.async.ca) ran. Fix (3), -lineinfo, is harmless. The `expected` rows simply carry the parent's targets and are consistent with the TASK floors and weights. Bs=128 and 256 stay untouched as in-run controls. WEAKEST LINK: the instrument for the fix (1) is illegal under the rules (see rules_flags). Without it the run reverts to the parent's failure mode, a sticky crash in prepare with tests 0/0. Then only the last printed line tells you which launch trapped, and only if every canary is synchronized in-process. A second weakness: three suspects (cache_hint/createpolicy, ldmatrix.x4, zero-fill cp.async.ca) are tested one at a time, so one run may only remove one of them.

### Confounders on the current kernel
- align-parallel-per-expert-verify (new base since the parent): it changes only the num_tokens > 32 else-branch, which this hypothesis leaves byte for byte, so it does not confound the small-M rows. Rebase conflicts are limited to the prepare() warm-up and self-check region, so check that the _align_small warm-up at BSM 16/32 survives the rebase.
- triton-map-align-on-block-m-restack: the new kernel feeds _align_small BSM = 8/16/32 instead of Marlin's 8/8/16, which changes that kernel's specialization and its 2.7-4.2 us row. That is attributable in top_kernels, not a confound for the replaced-kernel milestone.
- post-expert-norm-triton-small-m: the consumer of y (cache3) is unchanged and the new kernel writes y at the same flat route index. No interaction unless the w2 epilogue's indexing is wrong, which the self-check catches.
- marlin-block-m-two-smaller-on-postnorm and sglang-marlin-path-parity: they touch only num_tokens >= 128, or they are the incumbent and the reference being replaced. No confound.

### Interference and A/B needs
No companions. This is a rewrite stage whose milestone is 'it runs and passes'. Each bs<=32 row is read directly against the Marlin pair plus act_and_mul in top_kernels, and bs>=128 is the in-run control. No landed hypothesis shares the replaced launch sequence in a way a revert would isolate. The useful extra information is diagnostic: which instruction traps. A same-day split would answer that better than an A/B companion: one check with the plain cp.async.cg swap (this run), and optionally a sibling check that also swaps ldmatrix.x4 for 32-bit ld.shared B-fragment loads and the zero-fill cp.async.ca for an explicit predicated zero store. Those are stage-1b siblings, not companions in the A/B sense.

### Open questions
- Mode: this is a make-it-run stage. Use okc check, not submit, unless the quota policy says otherwise. The parent spent a submit on tests 0/0.
- The 715 surfaced at the torch.cuda.synchronize at submission.py L1004, after setup. Confirm it is the first kernel launch and not a launch inside the retile or align self-check. Add a torch.cuda.synchronize() + print immediately before the first CUDA kernel launch, so that an earlier async error is attributed correctly.
- Removing createpolicy also removes the evict_first hint the design relied on so it would not pollute L2 with weights that are streamed once. That is fine for M1, but record it, because it can matter for M2/M3 at bs=16/32, where activations and partials compete for L2.
- If the plain cp.async.cg swap does not clear the trap, check two more candidates. (a) ldmatrix.x4 addresses: every lane's row address must be 16 B aligned inside the swizzled slab, and lanes 8-31 must point at the right sub-matrices for NT=1, where there may be fewer than 32 rows. (b) The cp.async.ca 16-byte form with src_bytes = 0: the source pointer must still be a valid global address, so clamp sentinel rows to row 0 rather than computing an out-of-range pointer from route >= M*8.
- Also check that load_inline does not append TORCH_CUDA_ARCH_LIST gencode flags alongside -arch=sm_90a that would produce a mismatched cubin. Print torch.cuda.get_device_capability() and the effective nvcc flags in the log.

