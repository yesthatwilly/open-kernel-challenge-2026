# muse_moe_w4a16 · 2026-10-25T003453Z · check · cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T001824Z-cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc.md` (branch `hyp/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc`).

- **Job:** `7a6919420b724ee7` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `ea3464f`

## Goal
Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.patch, spec done/neutral/2026-10-24T165851Z-cuda-fused-moe-bs1-persistent-w13-silu-w2.md), rebased onto current master (align-parallel-per-expert-verify); num_tokens == 1 only, every other M runs master. Stage 1 profiled at 37.7 us (-26% vs 51.2 us) but the 100 timed replays were bimodal (41 in 40-47 us, 59 in 64-79 us; median 66.1 us) so the score was flat. PART A ONLY (Part B, the bandwidth work, is split out to stage 3): kill the slow mode first: (1) in the self-check only, record per-CTA entry globaltimer and %smid, and globaltimer at phase-1 start/end and phase-2 start, and the phase-2 spin-iteration count, into a debug buffer and print min/median/max per phase; (2) replace __nanosleep(32) in the phase-2 ld.acquire spin on the w13 chunk counters with a tight acquire poll, bounded (trap-free: after a large bound set an error flag and fall back); (3) remove the second-wave dependency: pad/split phase-1 w13 work so it is exactly one wave (reviewer: prefer ceil(128R/132) whole tiles per CTA over a K-split, which would need a pre-silu reduction) and order phase-2 (route, k-split) items so each is taken by CTAs whose phase-1 chunks finish first. Self-check additionally runs 100 graph replays with a >= 2x L2-size buffer written between replays (cold L2, as the judge does) and asserts the replay distribution is unimodal (p90/p10 < 1.3). Run the self-check's first launches IN-PROCESS (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected); stage 1 already ran this kernel in-process on the judge): confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for moe_bs1_kernel; anything else turns the path OFF), then a tiny synthetic canary launch of moe_bs1_kernel (R=1), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; then keep plain cp.async.cg (no createpolicy), build with -lineinfo. No bandwidth change in this stage. The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## What changed, and where
Worktree `.worktrees/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc`, file `muse_moe_w4a16/submission.py` only, uncommitted. Built on stage 1's shipped code (ROWS=128, STAGES=8, SPLIT2=4), as the review requires.

- **Rebase onto current master.** The worktree branched from stage 1's implementation (0efab7b), which predates master's `align-parallel-per-expert-verify` (1a5da3d). I ported that commit's submission.py diff with `git apply --3way`, which applied cleanly and was then unstaged. `git diff master -- submission.py` is now additions only, apart from the old `kernel_fn` def/docstring lines, so every M != 1 runs current master byte for byte.
- **(1) Per-CTA diagnostic log.** `moe_bs1_kernel` is now `template <bool DBG>`. The timed path launches `<false>`: `_fused_forward` passes an empty int64 `dbg_off` tensor, and the host picks the instantiation on `numel` (a shape branch, fixed at capture). Each `if (DBG ...)` is compiled out of the timed kernel. `<true>` writes 12 u64 per CTA: entry `%globaltimer`, `%smid`, setup done, the end of the last phase-1 item, the first phase-2 wait's start and end, total spin iterations, total wait `clock64` cycles, phase-1 and phase-2 item counts, exit time, and whether the CTA ran the combine.
- **(2) Tight poll.** `__nanosleep(32)` is removed from the phase-2 `ld.acquire` spin. The bound is SPIN_MAX = 1<<21 tight polls, about 1 s.
  - **Applies the caller/review fix:** the sticky flag no longer poisons later calls. A bound hit sets a new per-launch `C_ABORT`, which ends the remaining waits of that launch only and is reset by the exit ticket. `C_ERR` is now a sticky counter that only reports; the kernel never reads it. Stage 1's `ld_acquire(ctrl + C_ERR)` early-out, which made every later launch break its waits after one bound hit, is gone.
  - Nothing on the timed path reads a flag or switches paths. The prepare-time decision is made once, and loudly.
- **(3) One-wave phase 1 / phase-2 ordering: NO CODE CHANGE (the review wins).**
  - At decode_1's likely R=2 the shipped kernel has 64R = 128 phase-1 items on 132 CTAs, which is already one wave.
  - Static round-robin already hands the first phase-2 items, starting at item n1, to CTAs n1 % G and later. Those are exactly the CTAs with the fewest phase-1 items, which finish first.
  - The (r, s, n) phase-2 order is already monotone in the readiness of the chunk it waits on, because the phase-1 tile index rises with (r, s).
  - Contiguous blocks of ceil(64R/G) tiles per CTA would make every chunk depend on the last wave, which is worse.
  - The dynamic-ticket item queue the review suggests is left for stage 3.
- **Self-check (in-process, no child process):**
  - **Arch gate.** The build sets `TORCH_CUDA_ARCH_LIST=9.0a` around `load_inline` and restores it after. `-arch=sm_90a` was dropped, so the only gencode is torch's, and the flags from `_get_cuda_arch_flags()` are printed. The gate requires all of the following, and anything else gives `PATH CHOSEN = master`:
    - capability == (9, 0);
    - gencode tokens limited to {compute_90a, sm_90a};
    - `cudaFuncGetAttributes` binaryVersion and ptxVersion == 90 for both instantiations;
    - a new `arch_probe_kernel` in the same module reporting `__CUDA_ARCH_FEAT_SM90_ALL` and `__CUDA_ARCH__` 900.
  - **Deviation:** if the private `_get_cuda_arch_flags` helper is missing, the probe kernel is the evidence (the line logs "unknown").
  - **Build flags:** `-O3 -lineinfo`; the extension is renamed `..._v2`.
  - **Canaries.** The first launches are an R=1 synthetic canary of `<false>`, then of `<true>`. Each is printed before, synchronized, printed after, and compared with master.
  - **Stage-1 checks kept.** The 50 eager launches, the 2 fp32-dequant checks and the 10 graph replays are unchanged, with a synchronize after every launch. Round 0 prints around each launch; later rounds print one line per round, to protect the 16 kB log tail.
  - **Failure handling.** A correctness failure or a bound hit gives a loud fallback in prepare.
- **Diagnostic (`_fused_diag`, REPORT ONLY, after PASSED).** Each replay is preceded by a `fill_` of a buffer of 2 x L2 + 16 MB (cold L2).
  - **Main timing.** 100 interleaved replays of the timed fused graph and of master's bs=1 graph (control) at R=2. Each prints p10/p50/p90/min/max, p90/p10 and a mode split (the widest step > 8% between consecutive sorted samples).
  - **UNIMODAL line.** `UNIMODAL yes/NO` against p90/p10 < 1.3. **It never changes the path** (applies the review/caller fix).
  - **Other R.** 50 replays each at R = 1, 3 and 8.
  - **Logged run.** 100 replays of the DBG graph with the per-CTA log copied out after each replay. Replays are split by their own mode. For each mode it prints medians and maxima of: entry skew, the late-entry SMs, setup, the last phase-1 end, the first-wait maximum and median, spins, kernel span, and replay time minus span (time outside the kernel). For each mode's median replay it prints per-CTA min/med/max of entry, setup, phase-1 end, wait start and end, exit, spins, wait cycles and item counts, plus which SM ran the combine. It also prints a globaltimer resolution bound.
  - **Exceptions.** An exception inside the diagnostic is printed and the path is kept.
  - **Final decision.** A spin-bound hit during the diagnostic (a stall of about 1 s, a hang, never a slow mode) still gives a loud prepare-time `PATH CHOSEN = master`. The last line is `PATH CHOSEN = fused for num_tokens == 1 ...`.
- **Reading the log.** Large entry skew or a large "time - span" in the slow mode means a late CTA or launch, not the kernel's dataflow; the stage-3 fix would then be grid < SMs or a work-stealing ticket. A large phase-2 first-wait with a late phase-1 end means a phase-1 straggler. If the self-check replays are unimodal, the judge's slow mode was not reproduced, which is the review's open question 2.
- **Unverifiable here.** There is no nvcc or GPU on this node, so CUDA compile and run risks remain:
  - the function-pointer ternary launch of the template;
  - `reinterpret_cast<const void*>` of the kernel templates;
  - torch accepting "9.0a" in `TORCH_CUDA_ARCH_LIST`;
  - the ptxVersion that `cudaFuncGetAttributes` reports for an image with no PTX embedded (if it is not 90, the arch gate turns the path OFF and the run measures master).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc diff`:

- [x] CUDA-graph capturable: `kernel_fn` and `_fused_forward` have no `.item()`, `.cpu()`, `synchronize()` or tensor prints. The only branches are on shape, config, and the prepare-time `state["fused"]`. The DBG/non-DBG choice is on the `numel` of a prepare-time tensor. All syncs and `.tolist()` calls are in `prepare()`/`_fused_setup`/`_fused_diag`, and lint confirms they are outside kernel_fn's call graph.
- [x] Nothing derived from `data` is cached across calls. The scratch buffers and counters are prepare-time, and the kernel resets every counter but the report-only C_ERR at exit. C_ERR is never read on the timed path.
- [x] Weights: unchanged from stage 1. The repack is a lossless NVFP4 fragment shuffle in `prepare()` with a round-trip assert, and global scales are multiplied by 2^7 exactly.
- [x] Activations BF16 where SGLang keeps BF16: unchanged (h and y are bf16, accumulation fp32). n/a for KV cache and queries: this is MoE.
- [x] n/a: no attention in this problem.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import or network. **No subprocess, multiprocessing, os.fork or spawn** (grep is empty). The `os` import only sets and restores `TORCH_CUDA_ARCH_LIST` around `load_inline` in prepare.
- [x] All work happens inside the timed call, with no output caching or replay. The diagnostic graphs and replays run only in `prepare()` on synthetic inputs, and their results are only printed.
- [x] The first measurement must be an `okc check`, never a scored submit (per spec and LESSONS 2026-10-25).

okc_lint output (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`): 33 × `warn host-sync ... (outside kernel_fn's call graph; fine in prepare())`. These are the synchronize, `.tolist()` and randperm calls in `_align_self_check`, `_fused_case`, `_fp32_moe`, `_arch_gate`, `_fused_setup`, `_fused_diag` and `prepare`. Final line: `OK with warnings: submission.py`. `py_compile` passed, and `__pycache__` was removed.

## Risks / what could fail
high
supersedes cuda-fused-moe-bs1-persistent-nospin-bandwidth, held for a child-process self-check; Part A only (stage 2), Part B split to stage 3 per review

## Result (after the run)
- job `7a6919420b724ee7` (check), status done, passed True, tests 5/5
- speedup 1.3257x, 95% CI [1.3058, 1.3683], score 10144.69657248685 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc.patch`; branch `hyp/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc`
