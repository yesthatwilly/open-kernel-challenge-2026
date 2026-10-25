---
problem: muse_moe_w4a16
slug: cuda-nvfp4-swapab-mma-persistent-small-m-inproc
created: 2026-10-25T001824Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  FIX (stage 1b, make it run): the previous run built (51.2 s, 80-128 regs, 0 local) but its first
  CUDA launch in _cuda_self_check hit illegal instruction 715 (sticky), so the in-process fallback
  could not save it and prepare() crashed (tests 0/0). Apply the archived kernel
  (muse_moe_w4a16/hypotheses/done/failed/cuda-nvfp4-swapab-mma-persistent-small-m.patch; spec
  done/failed/2026-10-24T165851Z-cuda-nvfp4-swapab-mma-persistent-small-m.md) rebased onto current
  master (align-parallel-per-expert-verify), with exactly three changes and nothing else: (1)
  IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject
  submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25,
  corrected)): torch.cuda.synchronize() + print immediately before the first CUDA kernel launch
  (so an earlier async error is attributed correctly), confirm the sm_90a image actually loaded
  (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only
  compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for both kernels;
  anything else turns the path OFF), then a canary of w13+silu at M=1, NT=1 and then w2 at M=1,
  NT=1 as SEPARATE launches, each followed by torch.cuda.synchronize() and a print, then the full
  per-M self-check in-process; enable the CUDA path only if it prints PASSED, else PATH CHOSEN =
  MARLIN with the reason logged; (2) replace cp_async16_ef
  (createpolicy.fractional.L2::evict_first + cp.async.cg ... L2::cache_hint) with the bs=1
  kernel's proven plain cp.async.cg.shared.global 16 B form and delete createpolicy; (3) add
  -lineinfo to extra_cuda_cflags. Keep ldmatrix.x4 and the cp.async.ca src_bytes zero-fill as is
  (next suspects if it still traps), and keep tiling, dequant, layout and the bs<=32 gate
  unchanged. The first measurement of this hypothesis must be an `okc check`, never a scored
  submit (a sticky fault cannot be recovered in-process).
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
review: proceed
status: running
review_at: 2026-10-25T002203Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T002203Z
base_sha: f8b3f7230f2656a4143b6a87bb6b8dc1be0b5e49
base_blob: 7d31b758c60017e045f6e0f75d2e437931ea34b0
branch: hyp/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m-inproc
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m-inproc
mode: null
restack_of: null
stale_base: true
implemented_at: 2026-10-25T003249Z
impl_commit: 98d0dd04651b5f30e34b1aff6cf4233906880375
submission_sha256: 7fcc10488ad03889e2a90a0a719352967ce356cbe71a7f10d034eacdb37fcb61
norm_sha256: c38f444c80bdae160f500f98ce4f8d792c61ad89161de7183964f653065d1829
submission_blob: 90abfd1b6a054e344505bc60431e382b3f5fe552
lint_warnings: 24
---

# cuda-nvfp4-swapab-mma-persistent-small-m-inproc

## Change

FIX (stage 1b, make it run): the previous run built (51.2 s, 80-128 regs, 0 local) but its first CUDA launch in _cuda_self_check hit illegal instruction 715 (sticky), so the in-process fallback could not save it and prepare() crashed (tests 0/0). Apply the archived kernel (muse_moe_w4a16/hypotheses/done/failed/cuda-nvfp4-swapab-mma-persistent-small-m.patch; spec done/failed/2026-10-24T165851Z-cuda-nvfp4-swapab-mma-persistent-small-m.md) rebased onto current master (align-parallel-per-expert-verify), with exactly three changes and nothing else: (1) IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): torch.cuda.synchronize() + print immediately before the first CUDA kernel launch (so an earlier async error is attributed correctly), confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for both kernels; anything else turns the path OFF), then a canary of w13+silu at M=1, NT=1 and then w2 at M=1, NT=1 as SEPARATE launches, each followed by torch.cuda.synchronize() and a print, then the full per-M self-check in-process; enable the CUDA path only if it prints PASSED, else PATH CHOSEN = MARLIN with the reason logged; (2) replace cp_async16_ef (createpolicy.fractional.L2::evict_first + cp.async.cg ... L2::cache_hint) with the bs=1 kernel's proven plain cp.async.cg.shared.global 16 B form and delete createpolicy; (3) add -lineinfo to extra_cuda_cflags. Keep ldmatrix.x4 and the cp.async.ca src_bytes zero-fill as is (next suspects if it still traps), and keep tiling, dequant, layout and the bs<=32 gate unchanged.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `bs=1 *` faster: incumbent Marlin + silu 44.6 + 1.6 us, floor 11.4, M3 <= 18 us; M1 = runs with PATH CHOSEN = CUDA and passes all tests, M2 = replaced kernels below incumbent
- `bs=8 *` faster: 105.8 + 2.0 us vs floor 37.3, M3 <= 56
- `bs=16 *` faster: 116.0 + 2.2 us vs floor 52.5, M3 <= 79
- `bs=32 *` faster: 187.6 + 2.3 us vs floor 70.8, M3 <= 106
- `bs=128 *` flat: control: untouched
- `bs=256 *` flat: control: untouched

## Risk

supersedes cuda-nvfp4-swapab-mma-persistent-small-m-childcheck, held for a child-process self-check

## Milestones

Rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at bs<=32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002203Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as an okc check. This is a sound stage-1b make-it-run step for a rewrite whose design was already reviewed and emulated exactly offline. Its real value is localizing the sticky 715 trap, and it has no rules problem, but the removal of createpolicy is a guess, not a diagnosed cause.

### Justification
The design is unchanged from the reviewed stage 1. The target is the Marlin pair, which takes 88-95% of every bs<=32 shape at roughly 1.2-2.2 TB/s against floors of 11.4, 37.3, 52.5 and 70.8 us, so the headroom is real and the floors are physically reachable. The weight is dequantized in registers with an exact bit trick feeding mma.sync m16n8k16 (swap-AB). A persistent grid of 132 runs a flat cp.async ring with about 54 KB in flight per SM, and split-K uses tickets that never spin. The offline numpy emulation confirmed the fragment, nibble, swizzle and split-K mapping to about 1e-15. The archived log shows the build is healthy: 51.2 s compile, 80-128 regs, no local memory, smem 88-136 KB with attributes set. It also shows the H200 (sm_90) trapped with CUDA_ERROR_ILLEGAL_INSTRUCTION on the first launch, and the sticky error killed prepare.

I re-read the archived kernel for the usual causes of 715. Every .aligned instruction (ldmatrix.sync.aligned, mma.sync.aligned, bar.sync) sits under CTA-uniform control flow: the bx>=total return, the loop over N, the uniform e<0 and last_s continues, and `if (tid < S_BYTES/16)` guards only a cp.async, which is not .aligned. So warp divergence is not an obvious culprit. cvt.rn.f16x2.e4m3x2 (sm_89+), fma.rn.bf16x2, cp.async.ca with src_bytes and createpolicy.fractional are all legal on sm_90.

WEAKEST LINK: change (2), replacing the cache_hint and createpolicy pair with plain cp.async.cg, is a plausible but undiagnosed guess. If it is not the cause, the run traps again and costs a check. Its outcome is still readable, though. The synchronize+print canaries separate an earlier async error from a trap in w13+silu or in w2, and -lineinfo is in place for any later in-process diagnosis.

The expected directions agree with TASK: four faster small-M loads and bs=128/256 untouched as in-run controls. The incumbent figures in the spec (44.6/105.8/116.0/187.6 us) are stale against the current FINDINGS profile, where the Marlin rows are 44.7, 114.0, 176.1 and 207.5 us at bs 1, 8, 16 and 32. Note that the bs=16 Marlin row (176.1 us) is larger than that shape's scored time (148.4 us), so profile rows are not on the scored-time scale. Judge M2 kernel row against kernel row from master's profile (job 1947e3ec2abb4bdf), never against the headline or the spec's numbers.

### Confounders on the current kernel
- align-parallel-per-expert-verify (current master): it touches only num_tokens > 32, which this rewrite leaves to Marlin byte for byte. There is no interaction on the target shapes, but the rebase must keep master's restructured _align_self_check (two-smaller bsm list at >=128) and add the patch's bsm_of=_cuda_bsm warm-up and check at BSM 16/32 on top without dropping either. A merge slip here would show up as an align failure or as JIT on capture, not as a kernel effect.
- triton-map-align-on-block-m-restack / triton-map-align-small-m-port: _align_small gets new BSM values (16 at M 9-16, 32 at M 17-32), which changes its Triton specialization and its 2.7-4.4 us row. The effect is small and attributable; read the _align_small row separately from the replaced-kernel rows.
- post-expert-norm-triton-small-m: it consumes cache3 unchanged, and the w2 epilogue writes y at the flat route index with remote and sentinel rows unwritten, exactly as Marlin is_ep does. This is not a confound unless the epilogue indexing is wrong, and the self-check against Marlin would catch that.
- marlin-block-m-two-smaller-on-postnorm: affects only bs>=128, which acts as the in-run control here. No interaction.
- sglang-marlin-path-parity: this is the incumbent being replaced and the self-check's reference. The Marlin bank is kept for M>32 and for the fallback, so it is not a confound.

### Interference and A/B needs
No companions. This is kind: rewrite, and its outcome is a correctness and runs-at-all milestone (M1), judged on its own kernel rows in top_kernels. No landed hypothesis shares the replaced launches in a way a revert or tune would isolate, and bs>=128 is the built-in control. The one tempting companion is a parallel bisect variant (for example also swapping the cp.async.ca src_bytes zero-fill for explicit zero stores, or ldmatrix.x4 for x2), which would halve the round trips if this run traps again. It is not warranted yet, for two reasons: the canary logging in this run already says which kernel traps, and the next suspect is better chosen from that evidence. If it traps again, mint stage 1c from the log rather than guessing two suspects at once now.

### Open questions
- Gating-criterion false negative: torch's cpp_extension drops its own -gencode flags when any user cflag contains 'arch', so with -arch=sm_90a only sm_90a is built. However, cudaFuncGetAttributes binaryVersion reports 90 for both sm_90 and sm_90a, and the nvcc flag list is not observable at run time beyond what the code passes. Implement the 'anything else turns the path OFF' check as capability == (9,0) plus binaryVersion == 90 only. Do not gate on something you cannot observe, or a correct kernel gets disabled and the run measures master.
- The sticky-error reality: a second 715 still crashes prepare whatever the 'PASSED else MARLIN' logic says. The fallback only helps for non-sticky failures, such as a numerical self-check mismatch. Make sure every print is flushed (print(..., flush=True)) before each canary launch, so the log tail shows exactly which launch died.
- If it traps again, the next suspects in order are: (a) cp.async.ca with src_bytes=0 on the sentinel path (replace with st.shared zeros, or point it at a valid row and multiply by zero); (b) the ldmatrix.x4 address or swizzle (check 16 B alignment of slab rows and SLAB_ROW); (c) the e4m3x2 cvt inline asm with .reg .b16 declared inside a braces scope that ptxas might duplicate; (d) a mismatch between the dynamic smem request and the cudaFuncSetAttribute call for the instantiation actually launched (set and launch must use the same NT and GATED template). A W-only canary (GATED=false, NT=1, SK=1, one item) would be the minimal repro to mint as stage 1c.
- Harvest: read the log tail first. 'PATH CHOSEN = MARLIN' or a missing 'SELF-CHECK PASSED' means the run measured master. Compare replaced-kernel rows with master's profile (job 1947e3ec2abb4bdf) Marlin and act_and_mul rows per shape, not with the stale figures in the spec.
- Base drift: two in-flight hypotheses (stack-onewarp-hoisted-dispatchcombine and -from17) change _align_small at bs<=8 and the small-M post-norm combine. If either is promoted before this is built, re-run the rebase so the CUDA path's align warm-up and self-check cover whichever align kernel master uses at BSM 8/16/32.


## Implementation notes

Worktree `.worktrees/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m-inproc`, only `muse_moe_w4a16/submission.py`. Not committed, no okc run.

**Rebase onto current master (align-parallel-per-expert-verify, blob 108d36e).** The worktree was on stage 1's implementation (f8b3f72, built on d9a9375), so I 3-way merged stage 1's diff onto master's file. There were 4 conflicts, resolved by hand:
- `_align_self_check`: master's version (its `_align_parallel` cases at 33/57/120/128/256/257, including the two-smaller bsm at >=128, and the `[align-parallel-per-expert-verify]` summary), plus stage 1's `bsm_of` / `tag0` arguments and `return len(hard)`. With `bsm_of` set (the CUDA path's call, BSM 8/16/32) only the tokens 1/8/16/32 `_align_small` cases run and the >32 summary is skipped. Stage 1's "(info) capacity" relabel is kept.
- `_moe` Marlin branch: master's `_align_small` / `_align_parallel` dispatch (on the `topk_ids` argument) inside stage 1's `use_cuda` split. prepare()'s `_align_parallel` warm-up merged cleanly next to stage 1's `_cuda_setup` call.

**Spec change (2):** `cp_async16_ef` (createpolicy.fractional.L2::evict_first + `cp.async.cg ... L2::cache_hint`) is replaced by `cp_async16_cg`, the plain `cp.async.cg.shared.global [..], [..], 16`. The `createpolicy` asm and the `policy` variable are deleted. ldmatrix.x4, the `cp.async.ca` src_bytes zero-fill, tiling, dequant, layout and the bs<=32 gate are unchanged.

**Spec change (3):** `CUDA_CFLAGS = ["-O3", "-arch=sm_90a", "-lineinfo"]`, printed before the build together with TORCH_CUDA_ARCH_LIST. The module is renamed `muse_moe_w4a16_swapab_v2`.

**Spec change (1), in-process guarded first launch:**
- Every CUDA-path print now goes through `_cuda_log` with `flush=True`.
- Before the build, `torch.cuda.get_device_capability() == (9, 0)` is checked and logged.
- `w4a16_init` now also returns `binaryVersion` and `ptxVersion` for all 6 instantiations. Both are logged, and the path is gated on `binaryVersion == 90` only.
- `torch.cuda.synchronize()` plus a log line runs immediately before the first CUDA-path launch.
- Canary (`_cuda_canary`): w13+silu at M=1, NT=1, one local route, then w2 at M=1, NT=1. They are separate launches, each preceded by a flushed `> launch ...` line and followed by `torch.cuda.synchronize()` and a flushed `< ok ...` line. A non-finite output turns the path off.
- In the full per-M self-check, `_moe` takes a prepare-only `trace(what, done)` callback. It wraps every w13+silu and w2 launch in a flushed begin line, a synchronize and a flushed end line, labelled `M=<m> <case> w13+silu|w2 NT= SK=`. Each Marlin reference call is followed by a synchronize, so a Marlin-side async error is not blamed on the CUDA launches. kernel_fn never passes `trace`.
- PATH CHOSEN = CUDA only after SELF-CHECK PASSED. Otherwise it is MARLIN, and the raised reason or the FAIL lines are logged.

**Review fixes applied:**
- Gate on capability plus binaryVersion only. This follows the review over the spec, which also asked to gate on the -gencode list. That list is not observable at run time, so I log it but do not gate on it.
- Every print before a canary is flushed.
- master's restructured `_align_self_check` is kept and the `bsm_of=_cuda_bsm` warm-up and check are added on top.
- Judge M2 against master's profile rows (job 1947e3ec2abb4bdf). That is a harvest note.

**Deviation (beyond "exactly three changes"), for the caller's localization goal:** both canary kernels are one template, so they execute the same instruction forms, and the canary alone cannot separate the review's next suspects. I added `probe_kernel` / `w4a16_probe`, one 32-thread block per instruction form, each launched separately with a synchronize and flushed log lines before the canary:
- 0: plain cp.async.cg 16
- 1: cp.async.ca 16 with src_bytes 16/0 (zero-fill)
- 2: ldmatrix.x4
- 3: cvt e4m3x2 + fma bf16x2 + mma.sync, the forms already cleared by the bs=1 kernel

Each probe is a strict subset of what w4a16_kernel executes, so a probe trap means the real kernel would have trapped too. Probe values (copy equality, zero-fill, the ldmatrix lane/register mapping, finiteness) are logged as info and do not gate the path.

**Harvest reading of the log tail:**
- The last `> launch probe N` line with no `< ok` names the faulting instruction form.
- The last `> launch canary ...` or `> launch M=.. ..` line names the faulting kernel and launch.
- `PATH CHOSEN = MARLIN` or a missing `SELF-CHECK PASSED` means the run measured master.

No sm_90a-only PTX is used, so TORCH_CUDA_ARCH_LIST and the `__CUDA_ARCH_FEAT_SM90_ALL` guard do not apply. The kernel has no mbarrier and no spin waits: split-K tickets are atomicAdd with no wait. That satisfies the bounded-wait requirement trivially.

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m-inproc diff` and `git diff master`.

- [x] CUDA-graph capturable. kernel_fn → `_moe(..., use_cuda)` has no `.item()`, `.cpu()`, synchronize or tensor prints. The new syncs and prints are in `_cuda_launch_*`, `_cuda_probe`, `_cuda_canary` and `_cuda_self_check`, all reached only from prepare(). `trace` defaults to None and kernel_fn does not pass it. Branches depend on shape and the prepare-time flag only, and allocations are shape-sized.
- [x] Nothing derived from `data` is cached across calls. The state holds the retiled weights, workspace and tickets, all built in prepare from weights or shapes.
- [x] Weights: lossless retiling in prepare() only (round-trip asserted). NVFP4 codes and E4M3 scales are kept as bytes, and the global scale ×2^7 is exact. The new probe buffers are synthetic uint8 test data, not weights.
- [x] Activations stay BF16 in and out, with fp32 accumulation and the fp32 split-K workspace (unchanged from stage 1).
- [x] n/a: attention. This is an MoE problem with no keys or windows.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import, network, or subprocess/multiprocessing/fork/spawn (grep is empty). load_inline is used in prepare only.
- [x] All work happens inside the timed call, and there is no output caching or replay. The self-check, probes and canary run in prepare on synthetic inputs only.
- [x] Unsupported shapes (M > 32, no expert_map) and any build, image, canary or self-check failure fall back to master's Marlin path without raising. The exception is a sticky device fault, which cannot be recovered in-process. So the first measurement must be `okc check`.

okc_lint output (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`):
```
25 warnings, all of the form "host-sync ... (outside kernel_fn's call graph; fine in prepare())" or
"dtype ... uint8 (allowed only where SGLang does the same)" for the retiled NVFP4 byte banks and the
probe's synthetic uint8 source.
OK with warnings: muse_moe_w4a16/submission.py
```
An earlier draft called the synchronize helper directly from `_moe`, and lint rejected it (`error host-sync`). It is now reached only through the prepare-only callback.

py_compile: OK. __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only on the kernel, no GPU). **Verdict: dispatch** (as an `okc check`, per the spec). No fixes applied.

- Merge: the non-CUDA code matches master blob 108d36e. `_moe`'s else branch is master's kernel_fn body unchanged (the two-smaller pick at >=128, the `_align_small` / `_align_parallel` dispatch, both Marlin calls, silu_and_mul), and so is the post-norm tail. prepare() keeps master's warm-ups and `_align_self_check(device, count)`. `_align_self_check` is master's version plus `bsm_of` / `tag0` / `return len(hard)`, and the `(info) capacity` relabel is required: capacity always differs from SGLang (master's log shows 135/135 capacity-only), so without it the CUDA gate would never pass. With `bsm_of=None`, master's `[align-parallel-per-expert-verify]` line now always reports 0 capacity-only. That is a log-only change.
- Gates that can pass: capability (9,0) and binaryVersion == 90 only, both observable on an H200. The probes don't gate. The self-check bound of 1e-2 per row against Marlin and against the fp32 dequant sits well above the expected bf16-rounding gap (about 2-3e-3). blocks_cap = 32, the tickets and the workspace are big enough for M <= 32 at BSM 8/16/32 (each expert has at most one m-block because top-k ids are distinct).
- Fallback: every build, image, align, canary, retile-assert and self-check failure is caught inside `_cuda_setup`, which logs `PATH CHOSEN = MARLIN`. The exception is a sticky 715, which cannot be recovered in-process (accepted by the spec).
- Rules: kernel_fn -> `_moe(..., use_cuda)` has no sync, `.item()` or print. `trace` is None there, the allocations depend on shape only, and the launches use the current stream, so it is graph-capturable. Nothing is derived from inputs. The NVFP4 bytes are retiled losslessly (asserted) and gs x 2^7 is exact. Activations stay BF16. There is no subprocess, network or reference import, and `os.environ` is read only for a log line. okc_lint: OK with warnings (prepare-only syncs, the uint8 NVFP4 byte banks). py_compile: OK.
- Scope: the probe_kernel (4 one-instruction launches) goes beyond "exactly three changes". It is diagnostic only, runs only in prepare and doesn't gate, so it is acceptable.
- Note: the `_build_cuda` comment says that cpp_extension adds no -gencode beside `-arch=sm_90a`. The sibling wgmma rebuilds observed that it does add compute_90/sm_90. That is harmless here because no sm_90a-only instruction is used.

Residual risk: the 715 cause is undiagnosed (dropping createpolicy is a guess). A second sticky trap still kills prepare, but the flushed probe and canary lines will name the instruction or launch.
