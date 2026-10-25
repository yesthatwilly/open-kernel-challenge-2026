# muse_moe_w4a16 · 2026-10-25T003418Z · check · cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T001824Z-cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc.md` (branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc`).

- **Job:** `b56d17fda2d54be2` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `c89b01f`

## Goal
RE-MEASURE (stage 1b): the previous run never executed the new kernel; load_inline failed because cpp_extension appended its own -gencode arch=compute_90,code=sm_90 beside -arch=sm_90a and ptxas cannot assemble wgmma.mma_async/fence/commit_group/wait_group or setmaxnreg.inc/dec for plain sm_90, so it fell back to Marlin and measured master. Apply the archived kernel verbatim (muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-verify.patch, spec in done/neutral/2026-10-24T172038Z-cuda-nvfp4-wgmma-rega-grouped-verify.md), rebased onto current master (align-parallel-per-expert-verify), and make ONLY the build fix: (1) set os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' immediately before load_inline (restore the previous value after), and pass only -gencode arch=compute_90a,code=sm_90a in extra_cuda_cflags; (2) wrap every wgmma and setmaxnreg body in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) ... #else set a device error flag and return; #endif so any stray compute_90 pass still assembles and turns the path OFF instead of trapping; (3) add -lineinfo. Also an IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): in setup, confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for each new kernel; anything else turns the path OFF); read the error flag back after the canary (non-zero -> OFF); then a tiny synthetic canary launch of the gather, w13 and w2 kernels on synthetic weights (smallest gated M), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; only then the full setup self-check in-process, enabling the path only if it prints PASSED. No tiling, pipeline, dequant, gate (num_tokens >= 128, own bound constant) or layout change. Log the build line and PATH CHOSEN; a run whose log does not say PATH CHOSEN = wgmma is unmeasured, not neutral. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## What changed, and where
Worktree file: `.worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc/muse_moe_w4a16/submission.py` (uncommitted; diff vs HEAD f50b333 +331/-51).

- **Rebase onto current master.** The worktree was at the archived stage-1 impl (f50b333, base 8440ae5). Ported master's
  align-parallel-per-expert-verify delta (8440ae5..1a5da3d): `_align_parallel_kernel`/`_align_parallel`, its
  prepare warm-up, the extended `_align_self_check`, and `_moe_marlin` now uses `_align_parallel` above 32 tokens. The
  Marlin path (all M < 128, M > 512, and fallback) is master's code; `git diff 1a5da3d` shows only the wgmma additions
  and the kernel_fn -> `_moe_marlin` split.
- **Build fix (spec 1, 3).** `_wg_build`: TORCH_CUDA_ARCH_LIST set to `9.0a` around load_inline and restored in
  `finally`; `extra_cuda_cflags = ["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"]` (the old
  `-arch=sm_90a` dropped); new module name `muse_moe_w4a16_wgmma_rega_v2`. Logs torch/CUDA version, the device
  capability, and the arch flags from `cpp_extension._get_cuda_arch_flags()` under that env. Review open question
  ("does this torch accept 9.0a?"): if `_get_cuda_arch_flags` raises, the list is restored to its default, the
  build proceeds with the guards, and error word 2 decides (logged).
- **Guards (spec 2).** `wgmma_fence/commit/wait`, each generated `Wg<N>::mma`, and the whole `wg_kernel` body
  (which holds both setmaxnreg) are under `#if defined(__CUDA_ARCH_FEAT_SM90_ALL)`; the `#else` of the kernel sets
  error word 2 (`atomicExch(err, 2)`) and returns. mbarrier waits were already bounded (2^26 cycles -> error word 1);
  only the comment was fixed (it said 2^31).
- **Review fix applied (review overrides the spec's "ONLY the build fix"):** dropped the never-run createpolicy /
  L2::cache_hint path. `bulk_g2s` is now plain `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`
  (no policy operand); both `createpolicy.fractional` instructions and the per-item policy pick are gone
  (`Item.multi` is still computed, now unused). No tiling, pipeline, dequant, gate or layout change.
- **Image check (review fix):** `wg_init` also returns `binaryVersion`/`ptxVersion`; they are logged and gated only
  on binaryVersion == 90 because, as the review says, they cannot tell sm_90 from sm_90a. Error word 2 after the canary
  is the real proof. Device capability != (9, 0) turns the path off before the build.
- **In-process guarded first launch:** new `_wg_canary` runs before the full self-check: M = 128 (smallest gated M),
  slot 0 of every token on local expert 0 and 7 remote routes, synthetic seeded activations. Deviation: it uses the
  real retiled bank (no separate synthetic weight bank, which would add ~0.9 GiB or a second bank layout); the inputs
  are synthetic. Each launch (align, gather, w13, w2, norm) is preceded/followed by a sync + flushed print through
  `_wg_tracer` (a closure passed only from prepare; kernel_fn passes nothing). Then error word and finiteness are
  checked; anything nonzero -> path off. The full self-check's eager calls use the same tracer, and the graph
  capture and each of the 3 replays are bracketed by flushed prints with a sync after each replay.
- **Compile-time budget (review fix):** `WG_SETUP_BUDGET_S = 200`; if the build (or build + canary) has used more
  than that, the path stays off. Build seconds and total setup seconds are logged.
- Logs: `PATH CHOSEN = WGMMA|MARLIN for 128 <= num_tokens <= 512; MARLIN for num_tokens < 128 and > 512` (flushed).
- Not fixable in-process: a sticky fault (715/700) in the canary still kills the run (prepare's next sync re-raises),
  but the log will name the kernel. The first measurement must be an `okc check`.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc diff`:
- [x] CUDA-graph capturable: kernel_fn branches only on shape, the prepare-time `wgmma_ok` flag and the bank key; no
      `.item()/.cpu()/synchronize()` reachable from kernel_fn (the tracer that syncs is never passed from kernel_fn,
      and okc_lint's call graph confirms no host-sync error); the error word is read only in prepare.
- [x] Nothing derived from `data` is cached: the state scratches (wg_xs, wg_hs, wg_err) and tensor maps are made in
      prepare; per call they are overwritten, nothing is read across calls.
- [x] Weights: the wgmma bank is the same lossless NVFP4 retiling as stage 1 (round trips asserted); E4M3 scales
      unchanged, global scales fp32 (x 2^7, exact). No precision change in this diff.
- [x] Activations BF16 where SGLang keeps BF16 (gathered x, h_sorted, cache3 all bf16; fp32 accumulate).
- [x] n/a: attention (MoE problem, no attention or key skipping).
- [x] No timing patches, harness frames/I/O, module edits, reference import, subprocesses or network: no
      subprocess/multiprocessing/fork anywhere (lint PROCESS rule clean); the only environment touch is
      TORCH_CUDA_ARCH_LIST around load_inline, restored after.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] First line marker `# okc: problem=muse_moe_w4a16` kept.

okc_lint (28 host-sync warnings, all outside kernel_fn's call graph, omitted):
```
warn  dtype        L1220: q13 = torch.empty((count, rec13), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L1221: q2 = torch.empty((count, rec2), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
OK with warnings: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc/muse_moe_w4a16/submission.py
```
py_compile: OK; __pycache__ removed.

## Risks / what could fail
high
supersedes cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild, held for a child-process self-check

## Result (after the run)
- job `b56d17fda2d54be2` (check), status done, passed True, tests 5/5
- speedup 1.3303x, 95% CI [1.3103, 1.3502], score 10109.749971131336 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-verify.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc.patch`; branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc`
