---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc
created: 2026-10-25T001824Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  RE-MEASURE (stage 1b): the previous run never executed the new kernel; load_inline failed
  because cpp_extension appended its own -gencode arch=compute_90,code=sm_90 beside -arch=sm_90a
  and ptxas cannot assemble wgmma.mma_async/fence/commit_group/wait_group or setmaxnreg.inc/dec
  for plain sm_90, so it fell back to Marlin and measured master. Apply the archived kernel
  verbatim (muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-verify.patch,
  spec in done/neutral/2026-10-24T172038Z-cuda-nvfp4-wgmma-rega-grouped-verify.md), rebased onto
  current master (align-parallel-per-expert-verify), and make ONLY the build fix: (1) set
  os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' immediately before load_inline (restore the previous
  value after), and pass only -gencode arch=compute_90a,code=sm_90a in extra_cuda_cflags; (2) wrap
  every wgmma and setmaxnreg body in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) ... #else set a device
  error flag and return; #endif so any stray compute_90 pass still assembles and turns the path
  OFF instead of trapping; (3) add -lineinfo. Also an IN-PROCESS GUARDED FIRST LAUNCH (never in a
  child process: the rules statically reject submissions that start processes and okc_lint flags
  multiprocessing (LESSONS 2026-10-25, corrected)): in setup, confirm the sm_90a image actually
  loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only
  compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for each new kernel;
  anything else turns the path OFF); read the error flag back after the canary (non-zero -> OFF);
  then a tiny synthetic canary launch of the gather, w13 and w2 kernels on synthetic weights
  (smallest gated M), each followed by torch.cuda.synchronize() and a print so a fault names its
  kernel; only then the full setup self-check in-process, enabling the path only if it prints
  PASSED. No tiling, pipeline, dequant, gate (num_tokens >= 128, own bound constant) or layout
  change. Log the build line and PATH CHOSEN; a run whose log does not say PATH CHOSEN = wgmma is
  unmeasured, not neutral. The first measurement of this hypothesis must be an `okc check`, never
  a scored submit (a sticky fault cannot be recovered in-process).
expected:
  - shapes: bs=128 *
    direction: faster
    why: "verify_16: incumbent Marlin w13+w2 316.1 us + act_and_mul 3.9 us vs floor 101.3 us; M2 = replaced kernels below incumbent (~320 us), M3 <= 170 us (~2.9 TB/s). Marlin rows move +/-20% run to run, so judge on the replaced-kernel row, not the headline"
  - shapes: bs=256 *
    direction: faster
    why: "verify_32: incumbent 490.2 + 6.4 us (403.1 in the fallback rerun) vs floor 108.9 us; M3 <= 220 us via wgmma register-A, one weight stream per expert chunk"
  - shapes: bs=32 *
    direction: flat
    why: "control: num_tokens < 128 runs master"
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-verify
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128
review: proceed
status: running
review_at: 2026-10-25T002121Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T002121Z
base_sha: f50b333a602cc590f2785a1fdd2171af25fe94bb
base_blob: 5c34fabe0f05fb53014dd85b66864cd3b38bc673
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T003418Z
impl_commit: c89b01f6d4eeee26bad681a66af3e272ffc51f07
submission_sha256: e0877f3e365fcbabe7eee80f4343f64b2eacb77ad952034ad3785b27b45ef821
norm_sha256: c1b2762ac7d1bc7b239c59ceb492b584c0a35be528a0f37fc3849e6f31505214
submission_blob: f09390b45e1e1af24f0ccc3645988b81f97b2bf8
lint_warnings: 30
attempt_file: 2026-10-25T003418Z-check-cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc.md
prepared_at: 2026-10-25T003418Z
---

# cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc

## Change

RE-MEASURE (stage 1b): the previous run never executed the new kernel; load_inline failed because cpp_extension appended its own -gencode arch=compute_90,code=sm_90 beside -arch=sm_90a and ptxas cannot assemble wgmma.mma_async/fence/commit_group/wait_group or setmaxnreg.inc/dec for plain sm_90, so it fell back to Marlin and measured master. Apply the archived kernel verbatim (muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-verify.patch, spec in done/neutral/2026-10-24T172038Z-cuda-nvfp4-wgmma-rega-grouped-verify.md), rebased onto current master (align-parallel-per-expert-verify), and make ONLY the build fix: (1) set os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' immediately before load_inline (restore the previous value after), and pass only -gencode arch=compute_90a,code=sm_90a in extra_cuda_cflags; (2) wrap every wgmma and setmaxnreg body in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) ... #else set a device error flag and return; #endif so any stray compute_90 pass still assembles and turns the path OFF instead of trapping; (3) add -lineinfo. Also an IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): in setup, confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for each new kernel; anything else turns the path OFF); read the error flag back after the canary (non-zero -> OFF); then a tiny synthetic canary launch of the gather, w13 and w2 kernels on synthetic weights (smallest gated M), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; only then the full setup self-check in-process, enabling the path only if it prints PASSED. No tiling, pipeline, dequant, gate (num_tokens >= 128, own bound constant) or layout change. Log the build line and PATH CHOSEN; a run whose log does not say PATH CHOSEN = wgmma is unmeasured, not neutral.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `bs=128 *` faster: verify_16: incumbent Marlin w13+w2 316.1 us + act_and_mul 3.9 us vs floor 101.3 us; M2 = replaced kernels below incumbent (~320 us), M3 <= 170 us (~2.9 TB/s). Marlin rows move +/-20% run to run, so judge on the replaced-kernel row, not the headline
- `bs=256 *` faster: verify_32: incumbent 490.2 + 6.4 us (403.1 in the fallback rerun) vs floor 108.9 us; M3 <= 220 us via wgmma register-A, one weight stream per expert chunk
- `bs=32 *` flat: control: num_tokens < 128 runs master
- `bs=16 *` flat: control
- `bs=8 *` flat: control
- `bs=1 *` flat: control: untouched

## Risk

supersedes cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild, held for a child-process self-check

## Milestones

Rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002121Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a stage-1b rewrite check (never a submit). The sm_90a build fix is the documented one, and the in-process canary removes the child-process rules problem that held the sibling. The biggest remaining risk is that the kernel is carried over verbatim with its never-executed createpolicy / bulk-copy L2::cache_hint path, the same helper family under suspicion for the swap-AB 715 trap.

### Justification
Design (kind: rewrite, judged on replaced-kernel rows): sound for H200 at these shapes. FINDINGS puts master's Marlin at 292.5 us (bs=128) and 391.4 us (bs=256), 95% of each shape, against floors of 101.3 / 108.9 us, so about 1.5 / 1.1 TB/s effective. At bs=256, the busiest GPU's 350-450 local routes x 2 GEMMs is tens of us of MMA even at wgmma peak, so an sm80 mma.sync engine plus dequant cannot approach the floor, and wgmma with register-A dequantized weight (swap-AB) is the right tensor-core path. The plan for bytes in flight (>=48 KB of weight per SM, bulk copy plus TMA, 1 producer and 2 consumer warpgroups) is the right order for ~3 TB/s. The register split 40x128 + 232x256 = 64512 fits the launch_bounds(384,1) pool. The archived kernel already bounds its mbarrier waits with a timeout and an error flag, so a protocol bug ends as a flag rather than a 15-minute hang. Expected directions match TASK.md: bs>=128 faster, bs<128 gated off and flat as in-run controls. The M2/M3 targets (<=320 / <=170 us at bs=128, <=497 / <=220 us at bs=256) are physically reachable, well above the floors. The build diagnosis matches REWRITE_READ section 3: cpp_extension appended compute_90, and the fix is TORCH_CUDA_ARCH_LIST=9.0a with only compute_90a/sm_90a plus the __CUDA_ARCH_FEAT_SM90_ALL guard. WEAKEST LINK: no instruction of this kernel has ever executed on the judge, and 'verbatim' keeps createpolicy.fractional.L2::evict_first/evict_normal and the cp.async.bulk ... L2::cache_hint form (patch lines ~160, ~447). The swap-AB sibling died with illegal instruction 715 using that helper family, and a 715 is sticky, so the in-process canary can only name the faulting kernel. It cannot fall back. Dropping the cache-hint operand (plain cp.async.bulk ... complete_tx::bytes) costs little performance at stage 1 and removes the prime suspect. I recommend it even though the spec says ONLY the build fix. Second: the 'binaryVersion/ptxVersion' check cannot tell sm_90 from sm_90a (both report 90). Only the gencode-flag log and the device error flag after the canary actually prove the sm_90a image ran, so the spec should not rely on cudaFuncGetAttributes for that. Third: compile time for 2 families x 4 N buckets of a wgmma template is unmeasured against the 15-minute run and 300 s prepare budget. Log the compile seconds and keep the fallback if it overruns.

### Confounders on the current kernel
- align-parallel-per-expert-verify (current master): the wgmma path bypasses master's _align_parallel_kernel at bs>=128 and runs the expert_map gather + moe_align_block_size(local_ids, 16, 32) instead (2-3 launches). The replaced-kernel accounting must add that align delta, a few us against Marlin's ~300-400 us, so it does not confound a milestone read, but the headline delta at bs=128/256 includes it.
- marlin-block-m-two-smaller-on-postnorm / triton-map-align-on-block-m-restack: these set the incumbent Marlin row (block_size_m pick at bs>=128) that the new kernel is judged against. The wgmma path ignores block_size_m (fixed 16-row unit), so it is unaffected; compare against this run's own base Marlin rows from the latest master profile, remembering the +/-20% run-to-run Marlin noise (333.5 vs 316.1 and 403.1 vs 490.2 us between identical-code runs).
- post-expert-norm-triton-small-m: no interaction. bs>=128 uses SGLang's fused_post_expert_norm_weighted_sum on both paths, and the new w2 epilogue writes cache3 in the layout it expects.
- sglang-marlin-path-parity: defines the Marlin bank that stays live for bs<128 and serves as the self-check oracle. The wgmma bank is a second ~0.9 GiB copy; memory pressure only, no timing confound.

### Interference and A/B needs
No companions. This is a rewrite judged on its own replaced-kernel rows (gather + w13 + w2 vs Marlin x2 + act_and_mul) per shape from top_kernels, and bs=1/8/16/32 are gated off as in-run controls. No landed hypothesis acts on the replaced kernels' path at bs>=128 except the align bypass, which is a few us and visible as its own rows in the profile. Reverting or tuning a landed change would not make a stage-1 read more interpretable. The only interpretability condition is in the spec already: a run whose log lacks PATH CHOSEN = wgmma is unmeasured, not neutral. The next stage is the follow-up.

### Open questions
- Will the implementer drop the createpolicy / .L2::cache_hint operand on cp.async.bulk (the prime 715 suspect shared with the swap-AB sibling), or keep it verbatim and accept that a trap costs this check?
- Does the judge's torch version accept '9.0a' in TORCH_CUDA_ARCH_LIST (arch-suffix support in _get_cuda_arch_flags)? If not, the fallback should be passing an empty arch list plus explicit -gencode in extra_cuda_cflags; log the final nvcc command line.
- Compile time of 8 wgmma instantiations on the judge: does it fit inside the 15-minute run with prepare's self-check?
- Since binaryVersion is 90 for both sm_90 and sm_90a, the image check should key on the device error flag after the canary (set in the #else branch) rather than cudaFuncGetAttributes.


## Implementation notes

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

## Rules check

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

## Pre-flight audit

Audited 2026-10-24 by an opus subagent (read-only plus a numpy emulation; no GPU on this node).

**Verdict: dispatch-with-fixes.** Must be an `okc check`, never a submit.

Verified:
- Master port: the worktree file contains master (1a5da3d, blob 108d36e) verbatim apart from the wgmma additions and the kernel_fn -> `_moe_marlin` split (`data[...]` -> arguments, the return tuple). `_align_parallel_kernel`, its warm-up, `_align_self_check` and block-m two-smaller are identical, so the non-wgmma path is master's.
- Retiling vs device read: a numpy emulation of `_wg_tile_expert` against consume_item's loads (cw*512 + lane*16 word s, nibble 4kl + 3 - i, scale byte 2s + rh) and the wgmma A fragment (i = rh + 2kh) gives 0 mismatches for both w13 (gate/up interleave) and w2. The stmatrix .trans epilogue maps and the exactness of the dequant (bf16 subnormal 2^-127 kept by fma.rn.bf16x2; 448 * 2^119 < bf16 max) check out by hand, so the self-check is not structurally doomed.
- Illegal-instruction surface: every wgmma (fence, commit, wait, mma_async) and both setmaxnreg sit under `__CUDA_ARCH_FEAT_SM90_ALL` (the #else sets error word 2). No createpolicy or L2::cache_hint remains. The rest are sm_90 / sm_89 instructions: mbarrier expect_tx/try_wait, cp.async.bulk(.tensor), stmatrix, cvt e4m3x2, fma bf16x2. The setmaxnreg.inc hang is gated on numRegs*384 >= 64512, and mbarrier waits are bounded.
- Fallback: a build error, a 9.0a rejection, the attrs, smem, regs or binaryVersion gate, tensor-map encode, canary, self-check, error word or budget each land on Marlin with a flushed PATH CHOSEN line. A sticky 715/700 is still fatal, which is why this must be a check.
- Rules: no subprocess or multiprocessing (load_inline's nvcc is sanctioned); TORCH_CUDA_ARCH_LIST is restored in `finally`. kernel_fn branches only on shape and the prepare flag. Its wgmma call graph has no sync, .item or value-dependent allocation. Tensor maps are kernel params over prepare-time scratch. wg_xs/wg_hs are overwritten wherever a stored output depends on them (stale rows feed only unstored N columns). The NVFP4 bank is a lossless retiling with gs * 128 exact.
- Shapes: M = 128, 256 and the 257 test are gated in. Capacity at M <= 512: sorted_ids <= 4576 <= 8192 rows and block_experts <= 286 <= 384 (MAXB = THREADS).

Fixes applied:
1. The setup budget is now measured from prepare()'s start (`t_prepare` passed to `_wg_setup`), not from `_wg_setup`'s start. Reason: master's prepare work (Marlin repack, about 36 Triton align compiles, align self-check) and the unbudgeted wgmma self-check both sat outside the 200 s window. A slow build could then push prepare() past its 300 s limit and turn a neutral fallback into a failed run.

okc_lint: OK with warnings (only the uint8 NVFP4 bank allocations in prepare); py_compile OK; __pycache__ removed.

Residual risk: no instruction of this kernel has ever executed. A sticky 715/700 in the canary still kills the run (the log names the kernel), and ptxas may serialize the wgmma pipeline (A-register reuse), which costs only speed.
