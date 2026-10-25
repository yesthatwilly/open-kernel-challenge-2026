# muse_bf16_proj · 2026-10-25T040754Z · check · pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T035343Z-pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild.md` (branch `hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild`).

- **Job:** `571ea532ad7a410d` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `b3e80b4`

## Goal
Stage 2 (build fix and rerun) of pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite, built on its commit (8ba8624, claim bases a rewrite-on-rewrite on the parent commit); no design change. Stage 1 (job cd39d36e) never compiled: the nvcc compile of the .cu failed with "#error C++20 or later compatible compiler is required to use PyTorch", then "[bulk] DISABLED: no extension; master paths", so its 1.183 is master plus batch drift and the design is unmeasured. The host main.o compile carried -std=c++20, but the .cu was given only -gencode=arch=compute_90a,code=sm_90a. Change: add "-std=c++20" to that load_inline call extra_cuda_cflags (keep -gencode arch=compute_90a,code=sm_90a, -O3, -lineinfo; keep TORCH_CUDA_ARCH_LIST=9.0a), and rename the module so it rebuilds. Wrap the load_inline in the existing try/except so that on a build failure it prints the LAST ~60 LINES OF THE NVCC STDERR (the exception text / build log tail, not only the first #error) before "[bulk] DISABLED", so a second build error is named in one check. After the build, before any other work, print ptxas regs/spill/smem per kernel and assert the PATH CHOSEN line names the bulk kernel for each (layer, m) it gates; keep the existing self-check (rel L2 vs master) and fallback unchanged. Everything else (repack, ring, gates, other layers) is stage 1 code byte for byte. First measurement is an okc check, never a scored submit.

## What changed, and where
Edited only `muse_bf16_proj/submission.py` in the worktree (+43/-7).

- **Review fix applied (the review wins over the spec's diagnosis).** Stage 1 passed `extra_cuda_cflags=["-O3","-lineinfo","-std=c++17"]`; the "-gencode only" log line was torch's arch-flag diagnostic, not the nvcc command. I REPLACED `-std=c++17` with `-std=c++20` (new constant `BULK_NVCC_FLAGS = ["-O3", "-lineinfo", "-std=c++20"]`), not appended. No second `-gencode` (torch emits it from TORCH_CUDA_ARCH_LIST=9.0a, kept and restored). Host cflags: no `extra_cflags` is passed, so torch's default applies; the stage-1 log's main.o line carries `-std=c++20`, so the host side was never the problem. The flags list is now printed before the build.
- Module renamed `okc_bf16_bulk_rowown_s2` so it rebuilds.
- Build failure now prints the last 60 lines of the exception text (torch embeds the ninja/nvcc output in it) between `build FAILED` and `end of build log`, before "[bulk] DISABLED: no extension".
- The per-instance regs / local (spill) bytes / static + dynamic smem lines (`bulk_info`, labelled `ptxas-equivalent`) now print right after the build, before repack and self-check.
- After the kill-gate section, a profiler probe calls `kernel_fn` itself (so the real routing is exercised) for pooler m in {1,8,16,17,32} and unpooler m in {9,16,17,32}, and requires a `bulk_kernel` launch (LESSONS 164). A miss disables that layer before the PATH CHOSEN line; a profiler exception only prints.
- Read of the stage-1 log (archive/pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite/cd39d36e31814087.log): the only error is the `#error C++20 ...` from the .cu compile. I read the CUDA source for further compile errors (asm operand syntax, template/if-constexpr use, SLOT % 128 and KSW geometry for all NT, launch/attr code) and found none, so no kernel code changed. A second error would still surface in the 60-line tail.
- **LESSONS line ~165 needs correcting after the run (not edited by me):** the cause was an explicit `-std=c++17` override in extra_cuda_cflags, not a missing `-std=c++20` / "only -gencode". The same applies to the 03:41Z harvest note.

## Rules check (each must be true before the run)
- [x] Only `muse_bf16_proj/submission.py` changed (git diff --stat: 1 file), marker line kept.
- [x] No subprocess / fork / multiprocessing added (grep clean); torch.profiler runs in-process in prepare().
- [x] No timing patches, harness frames, network or reference import in the diff.
- [x] No host sync, allocation on input values or data-dependent control flow in kernel_fn; kernel_fn untouched (the probe lives in prepare()).
- [x] CUDA-graph capturable: timed path unchanged from stage 1.
- [x] Nothing derived from inputs cached; probe inputs are synthetic.
- [x] Precision: unchanged (bf16 in, fp32 accumulate, bf16 out; rel L2 self-check vs fp64).
- [x] sm_90a: no 90a-only PTX in this kernel (mma.sync/ldmatrix/cp.async.bulk); TORCH_CUDA_ARCH_LIST=9.0a set around load_inline and restored; arch flags printed.
- [x] Fallback: build error, self-check miss or probe miss leaves the route off; PATH CHOSEN printed per layer.
- [x] Build log tail / bounded waits: mbarrier waits bounded (0.5 s) as in stage 1.

okc_lint: `OK with warnings` (0 errors; 11 host-sync warnings, all in prepare()-only helpers). `py_compile` clean, `__pycache__` removed.

## Risks / what could fail
medium
Base is the stage-1 commit on master 0a2ef67, not the unlanded restack. Read untouched small-m rows against batch drift (+2-6% in the 03:41Z harvest); only rows whose kernel changed count. The design itself is the risk: every earlier stream (Triton, TMA tiles, CUDA gemv) stopped near 3.5 TB/s.

## Result (after the run)
- job `571ea532ad7a410d` (check), status done, passed True, tests 9/9
- speedup 1.0331x, 95% CI [1.0176, 1.0486], score 4676.137949309303 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "1e90292c7b954d96"}; base result: muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild.result.json`, `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild.patch`; branch `hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild`
