---
problem: muse_bf16_proj
slug: pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild
created: 2026-10-25T035343Z
base: master
area: pooler GEMM + unpooler GEMM (m<=32 DRAM weight streams, bulk-TMA row-owner kernel)
change: >
  Stage 2 (build fix and rerun) of pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite, built
  on its commit (8ba8624, claim bases a rewrite-on-rewrite on the parent commit); no design
  change. Stage 1 (job cd39d36e) never compiled: the nvcc compile of the .cu failed with "#error
  C++20 or later compatible compiler is required to use PyTorch", then "[bulk] DISABLED: no
  extension; master paths", so its 1.183 is master plus batch drift and the design is unmeasured.
  The host main.o compile carried -std=c++20, but the .cu was given only
  -gencode=arch=compute_90a,code=sm_90a. Change: add "-std=c++20" to that load_inline call
  extra_cuda_cflags (keep -gencode arch=compute_90a,code=sm_90a, -O3, -lineinfo; keep
  TORCH_CUDA_ARCH_LIST=9.0a), and rename the module so it rebuilds. Wrap the load_inline in the
  existing try/except so that on a build failure it prints the LAST ~60 LINES OF THE NVCC STDERR
  (the exception text / build log tail, not only the first #error) before "[bulk] DISABLED", so a
  second build error is named in one check. After the build, before any other work, print ptxas
  regs/spill/smem per kernel and assert the PATH CHOSEN line names the bulk kernel for each
  (layer, m) it gates; keep the existing self-check (rel L2 vs master) and fallback unchanged.
  Everything else (repack, ring, gates, other layers) is stage 1 code byte for byte. First
  measurement is an okc check, never a scored submit.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: incumbent 19.0 us (_bf16_splitk_kernel ~3.5 TB/s); floor 14.0; M1 builds and PATH CHOSEN = bulk; M2 below 19.0; M3 <= 16.5
  - shapes: layer=pooler load=verify_1 m=8
    direction: faster
    why: incumbent 19.14; floor 14.0
  - shapes: layer=pooler load=decode_16 m=16
    direction: faster
    why: incumbent 19.85; floor 14.1
  - shapes: layer=pooler load=decode_32 m=32
    direction: faster
    why: incumbent swap-AB 20.0 row; floor 14.1
  - shapes: layer=unpooler load=decode_16 m=16
    direction: faster
    why: incumbent nvjet 18.5; floor 14.1
  - shapes: layer=unpooler load=decode_32 m=32
    direction: faster
    why: incumbent swap-AB 18.7; floor 14.1
  - shapes: layer=qkv_gate * layer=o * layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: pooler m<=32 (_bf16_splitk_kernel m<=16, _bf16_swapab_kernel m=32) and unpooler-full m=16/32 (nvjet 64x16, _bf16_swapab_kernel)
review: proceed
status: running
review_at: 2026-10-25T040253Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T040259Z
base_sha: 8ba8624df9248fb986052e4a67ab8e9a96a9770d
base_blob: d2ef0372de1f177e26ce1a1a4f578a993b4fa98c
branch: hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild
worktree: .worktrees/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T040754Z
impl_commit: b3e80b459792ae1a8d986ced4268006f90e59c7d
submission_sha256: 399d29b58238eb4e5ab5800bcfd66fb62401347a0bc5ac740f757558233d8587
norm_sha256: 10962952539a6deb4eaeb60f91383cab497206cad634271f6a26e62555b5b2a2
submission_blob: 624198bade1b9eb179d6baa90e3125ca2378f3e7
lint_warnings: 10
attempt_file: 2026-10-25T040754Z-check-pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild.md
prepared_at: 2026-10-25T040754Z
---

# pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild

## Change

Stage 2 (build fix and rerun) of pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite, built on its commit (8ba8624, claim bases a rewrite-on-rewrite on the parent commit); no design change. Stage 1 (job cd39d36e) never compiled: the nvcc compile of the .cu failed with "#error C++20 or later compatible compiler is required to use PyTorch", then "[bulk] DISABLED: no extension; master paths", so its 1.183 is master plus batch drift and the design is unmeasured. The host main.o compile carried -std=c++20, but the .cu was given only -gencode=arch=compute_90a,code=sm_90a. Change: add "-std=c++20" to that load_inline call extra_cuda_cflags (keep -gencode arch=compute_90a,code=sm_90a, -O3, -lineinfo; keep TORCH_CUDA_ARCH_LIST=9.0a), and rename the module so it rebuilds. Wrap the load_inline in the existing try/except so that on a build failure it prints the LAST ~60 LINES OF THE NVCC STDERR (the exception text / build log tail, not only the first #error) before "[bulk] DISABLED", so a second build error is named in one check. After the build, before any other work, print ptxas regs/spill/smem per kernel and assert the PATH CHOSEN line names the bulk kernel for each (layer, m) it gates; keep the existing self-check (rel L2 vs master) and fallback unchanged. Everything else (repack, ring, gates, other layers) is stage 1 code byte for byte. First measurement is an okc check, never a scored submit.

## Expected

- `layer=pooler load=decode_1 m=1` faster: incumbent 19.0 us (_bf16_splitk_kernel ~3.5 TB/s); floor 14.0; M1 builds and PATH CHOSEN = bulk; M2 below 19.0; M3 <= 16.5
- `layer=pooler load=verify_1 m=8` faster: incumbent 19.14; floor 14.0
- `layer=pooler load=decode_16 m=16` faster: incumbent 19.85; floor 14.1
- `layer=pooler load=decode_32 m=32` faster: incumbent swap-AB 20.0 row; floor 14.1
- `layer=unpooler load=decode_16 m=16` faster: incumbent nvjet 18.5; floor 14.1
- `layer=unpooler load=decode_32 m=32` faster: incumbent swap-AB 18.7; floor 14.1
- `layer=qkv_gate * layer=o * layer=router *` flat: untouched

## Risk

Base is the stage-1 commit on master 0a2ef67, not the unlanded restack. Read untouched small-m rows against batch drift (+2-6% in the 03:41Z harvest); only rows whose kernel changed count. The design itself is the risk: every earlier stream (Triton, TMA tiles, CUDA gemv) stopped near 3.5 TB/s.

## Milestones

Rewrite stage 2 of `pooler m<=32 (_bf16_splitk_kernel m<=16, _bf16_swapab_kernel m=32) and unpooler-full m=16/32 (nvjet 64x16, _bf16_swapab_kernel)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T040253Z by an adversarial subagent (model opus).

**Verdict: proceed.** Rerunning the unmeasured design is the right call, but the stated cause of the build failure is wrong. Stage 1 explicitly passed -std=c++17 in extra_cuda_cflags; the -gencode flag the harvest saw came from torch's arch-flag diagnostic line. Torch then omits its own C++20 default, and the torch headers reject C++17. The builder must REPLACE (or drop) -std=c++17, not add -std=c++20 next to it.

### Justification
Evidence: on the stage-1 commit 8ba8624, _bulk build calls load_inline(..., extra_cuda_cflags=['-O3', '-lineinfo', '-std=c++17']), and the attempt file records the same flags. Log line 5, '[bulk] arch flags passed to nvcc: [-gencode=arch=compute_90a,code=sm_90a]', is the code's own print of torch's arch flags, not the full nvcc command. The host main.o got -std=c++20 because torch adds its default -std only when the caller gives none. For the .cu the caller gave c++17, and this image's torch requires C++20, hence the #error. So the 03:41Z harvest note and LESSONS 165 ('nvcc flags were only -gencode') misdiagnose the cause, even though the remedy 'pass -std=c++20' is close. Appending -std=c++20 while leaving -std=c++17 risks nvcc rejecting a redefined -std or keeping the wrong one. Delete the c++17 entry and either pass only -std=c++20 or pass no -std and let torch choose; the main.o line proves torch's choice is c++20. Base: 8ba8624^ has submission blob a8bb42a, identical to master's (8b9ad43), so the stage runs on current master as claimed. Design weakest link (unchanged from stage 1): every earlier stream stopped near 3.5 TB/s. pooler m<=16 at 19.0 us is already 67.1 MB / 19 us = 3.5 TB/s, and the 14.0 us floor implies ~4.8 TB/s, so M3 (16.5 us = 4.1 TB/s) needs ~85% of peak from a row-owner bulk stream. M1 (builds, PATH CHOSEN = bulk) and M2 are readable from one check; M3 is a stretch.

### Confounders on the current kernel
- bf16-m32-swapab-wgmma-triton (master): the m=32 pooler/unpooler incumbent is _bf16_swapab_kernel. Read those rows against this batch's swapab rows, since batch drift moved untouched small-m rows +2-6% together at 03:41Z.
- router-triton-hilo-fused-control-splitk-all-m / bf16-shape-gated-triton-winners-combo (master): they own the m<=16 pooler path (_bf16_splitk_kernel / _pooler_skinny) the bulk kernel replaces. Untouched layers (qkv_gate, o, router) are the in-run drift control.
- bf16-restack-m16-swapab-router-swapab-m32 (BEST.json, unlanded): not this run's base. Identify the base from the kernel names in the profile (LESSONS 166), never from attribution base_result.

### Interference and A/B needs
No companion. The untouched layers in the same check are the drift control, and the replaced kernels are judged on their own profile rows. A second build-only run would duplicate the prepare print.

### Open questions
- BUILDER: remove '-std=c++17' from extra_cuda_cflags and use ['-O3', '-lineinfo', '-std=c++20'] (or omit -std entirely). Do not append c++20 beside c++17. Keep TORCH_CUDA_ARCH_LIST=9.0a, and do not add a second -gencode since torch already emits it. Print the full nvcc command or flags list, and the last ~60 lines of the build log on failure (the current except keeps only str(e)[-600:]).
- After it builds, assert in a one-call profiler probe that the launched kernel name is the bulk kernel for each gated (layer, m) (LESSONS 164), not just the PATH CHOSEN line.
- Correct LESSONS 165 and the 03:41Z harvest note after the run: the cause was an explicit -std=c++17 override, not a missing flag.


## Implementation notes

Edited only `muse_bf16_proj/submission.py` in the worktree (+43/-7).

- **Review fix applied (the review wins over the spec's diagnosis).** Stage 1 passed `extra_cuda_cflags=["-O3","-lineinfo","-std=c++17"]`; the "-gencode only" log line was torch's arch-flag diagnostic, not the nvcc command. I REPLACED `-std=c++17` with `-std=c++20` (new constant `BULK_NVCC_FLAGS = ["-O3", "-lineinfo", "-std=c++20"]`), not appended. No second `-gencode` (torch emits it from TORCH_CUDA_ARCH_LIST=9.0a, kept and restored). Host cflags: no `extra_cflags` is passed, so torch's default applies; the stage-1 log's main.o line carries `-std=c++20`, so the host side was never the problem. The flags list is now printed before the build.
- Module renamed `okc_bf16_bulk_rowown_s2` so it rebuilds.
- Build failure now prints the last 60 lines of the exception text (torch embeds the ninja/nvcc output in it) between `build FAILED` and `end of build log`, before "[bulk] DISABLED: no extension".
- The per-instance regs / local (spill) bytes / static + dynamic smem lines (`bulk_info`, labelled `ptxas-equivalent`) now print right after the build, before repack and self-check.
- After the kill-gate section, a profiler probe calls `kernel_fn` itself (so the real routing is exercised) for pooler m in {1,8,16,17,32} and unpooler m in {9,16,17,32}, and requires a `bulk_kernel` launch (LESSONS 164). A miss disables that layer before the PATH CHOSEN line; a profiler exception only prints.
- Read of the stage-1 log (archive/pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite/cd39d36e31814087.log): the only error is the `#error C++20 ...` from the .cu compile. I read the CUDA source for further compile errors (asm operand syntax, template/if-constexpr use, SLOT % 128 and KSW geometry for all NT, launch/attr code) and found none, so no kernel code changed. A second error would still surface in the 60-line tail.
- **LESSONS line ~165 needs correcting after the run (not edited by me):** the cause was an explicit `-std=c++17` override in extra_cuda_cflags, not a missing `-std=c++20` / "only -gencode". The same applies to the 03:41Z harvest note.

## Rules check

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

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only emulation; no GPU on this node).

**Verdict: dispatch-with-fixes.**

Fixes (worktree `muse_bf16_proj/submission.py`, +1 line in `_bulk_forward`, probe body replaced):
- Replaced the torch.profiler kernel-name probe in `_bulk_prepare` with a plain host-side route flag: `_bulk_forward` sets `state["bulk_route_hit"] = (layer, m)` (shape only, never data values; Python-only, so it costs nothing under graph replay), and the probe clears it, calls `kernel_fn` once on synthetic x and requires the flag to equal `(layer, m)`. Reason: the profiler spawns nothing (Kineto/CUPTI is in-process), but its likely failure mode in a container is not an exception. CUPTI init failure or missing permission gives a warning and an empty event list, so `hit=False` would have silently DISABLED a self-checked route, which wastes the run. It also attaches CUPTI in the judge's process before the judge's own profiling pass, and a static checker could read it as timing instrumentation. A kernel_fn exception in the probe now disables that layer (the routed call itself failed) instead of being ignored.

Checks that passed:
- Build flags: `-std=c++17` is replaced, not appended (`["-O3","-lineinfo","-std=c++20"]`); module renamed `_s2`; TORCH_CUDA_ARCH_LIST=9.0a set and restored; the build failure prints the last 60 lines and returns None, leading to `DISABLED` and `PATH CHOSEN master`.
- Read the CUDA source for a second compile error under c++20/sm_90a and found none: PTX forms are valid on sm_90 (mbarrier init/arrive.expect_tx with `_` sink/try_wait.parity `.shared::cta`, `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`, `fence.mbarrier_init.release.cluster`, `%%globaltimer`); `if constexpr` discards are template-dependent; `static_assert`s hold for every instantiation (SLOT%128==0 for R=32/64 x NT=8/16/32; R*CKK*2==16384); `cudaFuncSetAttribute`/`cudaFuncGetAttributes` template overloads take the kernel specialization; there is no C++20-deprecated volatile compound assignment (`pr` is not volatile). Max dynamic smem is 199,936 B (pooler NT=32), under 227 KB.
- Indexing: the A ldmatrix x4 fragment order, B x2/x4 token mapping, XOR swizzle (physical = logical ^ (row&7)), and repack offsets all match the kernel's chunk addressing; bulk copy src/dst are 16 B aligned (x checked %16, XS=528/272); epilogue writes only tok < m.
- Self-check gate is passable: emulated bf16 output rounding gives rel L2 of 1.66e-3 against the 3e-3 gate; the power-of-two probe row is exact. Probe m sets: pooler {1,8,9,16,17,32}, unpooler {9,16,17,32}.
- Rules: timed path unchanged apart from the shape-only flag write; no host sync, alloc on values or data caching; BF16 weights repacked losslessly (bit-exact verified); no processes/network/harness access. okc_lint OK with warnings (prepare-only host syncs); py_compile clean.

Residual risk: no compile error is visible on reading, but stage 1 never compiled. The first build, mbarrier/bulk-copy runtime behaviour and the 0.5 s bounded waits are still unexercised on hardware.
