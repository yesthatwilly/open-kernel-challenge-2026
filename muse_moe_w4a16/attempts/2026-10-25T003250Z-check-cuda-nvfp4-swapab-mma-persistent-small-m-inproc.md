# muse_moe_w4a16 · 2026-10-25T003250Z · check · cuda-nvfp4-swapab-mma-persistent-small-m-inproc

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T001824Z-cuda-nvfp4-swapab-mma-persistent-small-m-inproc.md` (branch `hyp/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m-inproc`).

- **Job:** `fecc63ef567b4dc2` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `98d0dd0`

## Goal
FIX (stage 1b, make it run): the previous run built (51.2 s, 80-128 regs, 0 local) but its first CUDA launch in _cuda_self_check hit illegal instruction 715 (sticky), so the in-process fallback could not save it and prepare() crashed (tests 0/0). Apply the archived kernel (muse_moe_w4a16/hypotheses/done/failed/cuda-nvfp4-swapab-mma-persistent-small-m.patch; spec done/failed/2026-10-24T165851Z-cuda-nvfp4-swapab-mma-persistent-small-m.md) rebased onto current master (align-parallel-per-expert-verify), with exactly three changes and nothing else: (1) IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): torch.cuda.synchronize() + print immediately before the first CUDA kernel launch (so an earlier async error is attributed correctly), confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for both kernels; anything else turns the path OFF), then a canary of w13+silu at M=1, NT=1 and then w2 at M=1, NT=1 as SEPARATE launches, each followed by torch.cuda.synchronize() and a print, then the full per-M self-check in-process; enable the CUDA path only if it prints PASSED, else PATH CHOSEN = MARLIN with the reason logged; (2) replace cp_async16_ef (createpolicy.fractional.L2::evict_first + cp.async.cg ... L2::cache_hint) with the bs=1 kernel's proven plain cp.async.cg.shared.global 16 B form and delete createpolicy; (3) add -lineinfo to extra_cuda_cflags. Keep ldmatrix.x4 and the cp.async.ca src_bytes zero-fill as is (next suspects if it still traps), and keep tiling, dequant, layout and the bs<=32 gate unchanged. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
high
supersedes cuda-nvfp4-swapab-mma-persistent-small-m-childcheck, held for a child-process self-check

## Result (after the run)
- job `fecc63ef567b4dc2` (check), status done, passed True, tests 5/5
- speedup 0.9250x, 95% CI [0.9111, 0.9389], score 14539.136720437127 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: SGLang (no base result)
- **outcome: regression** (regression)
- files: `muse_moe_w4a16/hypotheses/done/regression/cuda-nvfp4-swapab-mma-persistent-small-m-inproc.result.json`, `muse_moe_w4a16/hypotheses/done/regression/cuda-nvfp4-swapab-mma-persistent-small-m-inproc.patch`; branch `hyp/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m-inproc`
