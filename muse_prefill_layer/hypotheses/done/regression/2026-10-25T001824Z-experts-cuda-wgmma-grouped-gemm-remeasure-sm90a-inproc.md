---
problem: muse_prefill_layer
slug: experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc
created: 2026-10-25T001824Z
base: master
area: "w13 + w2 expert GEMMs at T > 4096 (CUDA sm_90a wgmma W4A16 grouped GEMM, stage 1b: re-measure of experts-cuda-wgmma-mixed-input-grouped-gemm with the build fix, on the new master)"
change: >
  BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored), not on
  the old branch: start from master's submission.py and hand-port the CUDA expert engine from
  done/neutral/experts-cuda-wgmma-mixed-input-grouped-gemm.patch (branch
  hyp/muse_prefill_layer/experts-cuda-wgmma-mixed-input-grouped-gemm:
  _CUDA_EXPERTS_CUDA/_CUDA_EXPERTS_CPP, _nvfp4_wgmma_grouped_kernel, nvfp4_wgmma_gemm, _cuda_bank,
  _setup_cuda_experts, _check_cuda_onehot, _check_cuda_e2e, _use_cuda_experts, the PATH CHOSEN
  log). The kernel design is UNCHANGED (persistent 288-thread CTAs, 2 cooperative consumer WGs on
  m64n256k16 RS wgmma, 1 producer warp, 5-stage ring, 1-D cp.async.bulk weight blocks in
  A-fragment order, BLOCK_T 256). That run (job 1821bb0e) never ran its kernel: ptxas rejected
  wgmma on .target sm_90 because torch's cpp_extension added its own compute_90 gencode beside
  -arch=sm_90a, so it measured master (REWRITE_READ_2026-10-25 #1). Changes vs that patch, and
  only these: (1) BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' immediately before
  load_inline (restore the old value after) and pass only -gencode arch=compute_90a,code=sm_90a;
  wrap every wgmma/fence/commit_group/wait_group (and any setmaxnreg) asm in #if
  defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that sets the error word and returns, so a
  stray plain-sm_90 image still compiles and the self-check turns the path OFF; log the arch flags
  nvcc received and the compile seconds. (2) IN-PROCESS GUARDED FIRST LAUNCH (never in a child
  process: the rules statically reject submissions that start processes and okc_lint flags
  multiprocessing (LESSONS 2026-10-25, corrected)): in prepare(), confirm the sm_90a image
  actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags
  showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for
  _nvfp4_wgmma_grouped_kernel; anything else turns the path OFF); the wgmma code sits under #if
  defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets a device error flag and returns (no __trap),
  and the flag is read back after the canary (non-zero -> OFF); then a tiny synthetic canary
  launch of _nvfp4_wgmma_grouped_kernel on synthetic weights (one tile, w13 then w2), each
  followed by torch.cuda.synchronize() and a print so a fault names its kernel; only then the
  in-process one-hot and e2e self-checks; any failure prints OFF: <reason> and master's Triton
  path runs. (3) MASTER'S w2 ZERO-STORE FLAG: the w2 call's zero_nonlocal argument must be
  state['zero_nonlocal_w2'] exactly as master's _experts_triton passes it (OFF at first_turn now,
  since master's probe found the weighted sum skips remote slots); the CUDA kernel's -1-block
  items must skip both MMA and stores when it is False, instead of always zero-storing (that would
  add back ~940 MB of stores master removed). The NaN-prefilled non-local w2 comparison in
  _check_cuda_e2e already tolerates either setting; run it with master's flag value. (4) MASTER'S
  32-EXPERT ALIGN: route the CUDA kernel inside master's _experts_triton, which already calls
  moe_align_block_size(local_ids, 256, LOCAL_EXPERTS) on ids that are 0-31 or -1; the kernel's
  expert index is therefore 0-31 into the 32-expert bank, and n_padded/block_experts come from
  that align (fewer padded tiles than the old 257-expert align); _cuda_bank must index the same 32
  local experts as state['bank']. Do NOT touch the Marlin path at T <= 4096 (master's
  _align_parallel local-routes sorted_ids and _silu_mul_sorted_kernel stay as they are), the
  router, the silu_and_mul between the GEMMs, or the post-expert weighted sum. Fallback stays: any
  build, image, error-flag, canary, one-hot or e2e failure prints exactly one '[cuda expert gemm]
  OFF: <reason>' and master runs unchanged. Harvest: grep the log for the build line, '[cuda
  expert gemm] ON' and 'PATH CHOSEN T=16384: cuda', and the profile for an
  _nvfp4_wgmma_grouped_kernel row (2 calls) before reading anything else. The first measurement of
  this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be
  recovered in-process).
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "Replaced: _moe_nvfp4_gemm_kernel x2, 8083-9174 us on the new master (c4d14d62) across the four first_turn shapes, ~5x the 1.67 ms math floor (~2.0-2.3 ms once BLOCK_T-256 padding is counted). Milestones as stage 1: M0 log ON + PATH CHOSEN cuda + 5/5 tests; M1 w13+w2 under 6.0 ms; M2 under 3.5 ms (~470 TFLOP/s). At M2 the first_turn load gains ~1.5x and the headline ~1.15x. Kernel-row noise is about +-6%, so judge the row, not the headline."
  - shapes: "*load=turn_1*"
    direction: neutral
    why: Marlin path untouched (768.6/735.9 us on master); T <= 4096 never routes to CUDA.
  - shapes: "*load=turn_4*"
    direction: neutral
    why: Marlin path untouched (2148.2/2115.3 us on master).
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: _moe_nvfp4_gemm_kernel x2 (w13, w2) at T > 4096 (first_turn)
review: proceed
status: done
review_at: 2026-10-25T002131Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T002131Z
base_sha: d7ccd530009419f0e32f5bd4bdae7f0fdad273a3
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc
worktree: .worktrees/muse_prefill_layer/experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T003406Z
impl_commit: 0ca54a36be47eb3c075d62faf9b4ff7369491667
submission_sha256: 29e30e2184a52c4c93a1a68b2b4c701d393ee221d0b9145a1ec9ac62d5941110
norm_sha256: 59a1e5f9157ec54310476843316c32370929dd2005a745307aa00f1ff6822b1a
submission_blob: 113e7ac977c8ebb579d91c34400f9530c4ae6881
lint_warnings: 24
attempt_file: 2026-10-25T003406Z-check-experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc.md
prepared_at: 2026-10-25T003406Z
job: 6d047cdeefd94357
dispatched_at: 2026-10-25T003418Z
outcome: regression
verdict_detail: regression
speedup: 0.9400661628181909
speedup_ci: [0.925965170375918, 0.9541671552604637]
harvested_at: 2026-10-25T010420Z
best_at_harvest: 1.074386115190733
promotion: null
---

# experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc

## Change

BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored), not on the old branch: start from master's submission.py and hand-port the CUDA expert engine from done/neutral/experts-cuda-wgmma-mixed-input-grouped-gemm.patch (branch hyp/muse_prefill_layer/experts-cuda-wgmma-mixed-input-grouped-gemm: _CUDA_EXPERTS_CUDA/_CUDA_EXPERTS_CPP, _nvfp4_wgmma_grouped_kernel, nvfp4_wgmma_gemm, _cuda_bank, _setup_cuda_experts, _check_cuda_onehot, _check_cuda_e2e, _use_cuda_experts, the PATH CHOSEN log). The kernel design is UNCHANGED (persistent 288-thread CTAs, 2 cooperative consumer WGs on m64n256k16 RS wgmma, 1 producer warp, 5-stage ring, 1-D cp.async.bulk weight blocks in A-fragment order, BLOCK_T 256). That run (job 1821bb0e) never ran its kernel: ptxas rejected wgmma on .target sm_90 because torch's cpp_extension added its own compute_90 gencode beside -arch=sm_90a, so it measured master (REWRITE_READ_2026-10-25 #1). Changes vs that patch, and only these: (1) BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' immediately before load_inline (restore the old value after) and pass only -gencode arch=compute_90a,code=sm_90a; wrap every wgmma/fence/commit_group/wait_group (and any setmaxnreg) asm in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that sets the error word and returns, so a stray plain-sm_90 image still compiles and the self-check turns the path OFF; log the arch flags nvcc received and the compile seconds. (2) IN-PROCESS GUARDED FIRST LAUNCH (never in a child process: the rules statically reject submissions that start processes and okc_lint flags multiprocessing (LESSONS 2026-10-25, corrected)): in prepare(), confirm the sm_90a image actually loaded (log torch.cuda.get_device_capability() == (9, 0), the nvcc -gencode flags showing only compute_90a/sm_90a, and cudaFuncGetAttributes binaryVersion/ptxVersion for _nvfp4_wgmma_grouped_kernel; anything else turns the path OFF); the wgmma code sits under #if defined(__CUDA_ARCH_FEAT_SM90_ALL) whose #else sets a device error flag and returns (no __trap), and the flag is read back after the canary (non-zero -> OFF); then a tiny synthetic canary launch of _nvfp4_wgmma_grouped_kernel on synthetic weights (one tile, w13 then w2), each followed by torch.cuda.synchronize() and a print so a fault names its kernel; only then the in-process one-hot and e2e self-checks; any failure prints OFF: <reason> and master's Triton path runs. (3) MASTER'S w2 ZERO-STORE FLAG: the w2 call's zero_nonlocal argument must be state['zero_nonlocal_w2'] exactly as master's _experts_triton passes it (OFF at first_turn now, since master's probe found the weighted sum skips remote slots); the CUDA kernel's -1-block items must skip both MMA and stores when it is False, instead of always zero-storing (that would add back ~940 MB of stores master removed). The NaN-prefilled non-local w2 comparison in _check_cuda_e2e already tolerates either setting; run it with master's flag value. (4) MASTER'S 32-EXPERT ALIGN: route the CUDA kernel inside master's _experts_triton, which already calls moe_align_block_size(local_ids, 256, LOCAL_EXPERTS) on ids that are 0-31 or -1; the kernel's expert index is therefore 0-31 into the 32-expert bank, and n_padded/block_experts come from that align (fewer padded tiles than the old 257-expert align); _cuda_bank must index the same 32 local experts as state['bank']. Do NOT touch the Marlin path at T <= 4096 (master's _align_parallel local-routes sorted_ids and _silu_mul_sorted_kernel stay as they are), the router, the silu_and_mul between the GEMMs, or the post-expert weighted sum. Fallback stays: any build, image, error-flag, canary, one-hot or e2e failure prints exactly one '[cuda expert gemm] OFF: <reason>' and master runs unchanged. Harvest: grep the log for the build line, '[cuda expert gemm] ON' and 'PATH CHOSEN T=16384: cuda', and the profile for an _nvfp4_wgmma_grouped_kernel row (2 calls) before reading anything else.
 The first measurement of this hypothesis must be an `okc check`, never a scored submit (a sticky fault cannot be recovered in-process).

## Expected

- `*load=first_turn*` faster: Replaced: _moe_nvfp4_gemm_kernel x2, 8083-9174 us on the new master (c4d14d62) across the four first_turn shapes, ~5x the 1.67 ms math floor (~2.0-2.3 ms once BLOCK_T-256 padding is counted). Milestones as stage 1: M0 log ON + PATH CHOSEN cuda + 5/5 tests; M1 w13+w2 under 6.0 ms; M2 under 3.5 ms (~470 TFLOP/s). At M2 the first_turn load gains ~1.5x and the headline ~1.15x. Kernel-row noise is about +-6%, so judge the row, not the headline.
- `*load=turn_1*` neutral: Marlin path untouched (768.6/735.9 us on master); T <= 4096 never routes to CUDA.
- `*load=turn_4*` neutral: Marlin path untouched (2148.2/2115.3 us on master).

## Risk

supersedes experts-cuda-wgmma-grouped-gemm-remeasure-sm90a, held for a child-process self-check

## Milestones

Rewrite stage 1 of `_moe_nvfp4_gemm_kernel x2 (w13, w2) at T > 4096 (first_turn)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T002131Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound rewrite stage-1 re-measure: the build fix addresses exactly why job 1821bb0e measured master, the target (_moe_nvfp4_gemm_kernel x2 at 8.1-9.2 ms, ~5x the 1.67 ms floor) is large and M2 (~470 TFLOP/s) is physically reachable, and the in-process guard removes the child-process rules problem; judge it on the _nvfp4_wgmma_grouped_kernel row, not the headline.

### Justification
Floor arithmetic checks out: ~16384 local routes x (w13 N=8192 + w2 N=4096) x K=4096 x 2 = ~1.65 TFLOP -> 1.67 ms at 989 TFLOP/s; the Triton incumbent runs at ~200 TFLOP/s (8.1-9.2 ms, 37-57% of every first_turn shape, and first_turn sliding carries weight 138), so even M1 (<6.0 ms) moves the first_turn load ~15%. The design (persistent 288-thread CTAs, 2 consumer WGs on m64n256k16 RS wgmma with weights dequantized into register A fragments, 1 producer warp, 5-stage mbarrier ring, 1-D cp.async.bulk weight blocks pre-permuted in A-fragment order, cp.async 16 B activation gather into the 128B-swizzled K-major B layout) is the standard Hopper mixed-input shape and needs no TMA gather. Turn shapes are untouched by construction (T <= 4096 stays Marlin), so 'neutral' there is right. Weakest link: the kernel has NEVER executed (1821bb0e died in ptxas), so there is zero evidence of correctness or speed; and structurally, the 128 weight-row x 256 token tile re-reads each activation row N/128 times (64x for w13), i.e. ~114 FLOP per smem-filled byte, which needs ~8.6 TB/s of L2->SM activation traffic at peak. That caps this tile shape around 55-70% of peak (L2-bound gather by a single producer warp issuing ~64 cp.async per lane per stage), so M2 (~470 TFLOP/s) is reachable but the ~1.5x-of-floor milestone 3 probably is not without a wider weight tile (e.g. 2 CTAs in a cluster multicasting activations, or 256 weight rows per CTA). Secondary: none of the 5 pre-timing tests reaches T > 4096, so M0's '5/5 tests' says nothing about the CUDA path; its only correctness evidence is prepare's one-hot/e2e self-checks plus the judge's per-call checks on the first_turn timed shapes.

### Confounders on the current kernel
- experts-hybrid-marlin-turn-triton-first-turn: defines the T > 4096 gate the CUDA path sits behind; not a confounder for the kernel row, but it guarantees turn shapes cannot move, so any turn_1/turn_4 delta is noise.
- stack-tonight-five-prefill (w2-nonlocal-zero-store-off-first-turn): the incumbent w2 row is already measured without the ~940 MB zero-store; if the CUDA kernel ignored state['zero_nonlocal_w2'] and zero-stored -1 blocks, a loss would be the port, not the design. Change (3) handles it; verify in the log which value the CUDA call received.
- stack-tonight-five-prefill (moe-align-32-experts-first-turn): the incumbent row is timed on the 32-expert align's padded tile count; the CUDA kernel must consume the same sorted_ids/block_experts/n_padded, otherwise its tile count (and row time) is not comparable. Whether this align emits any -1 blocks at all is unclear; the one-hot check covers -1 explicitly either way.
- experts-triton-w4a16-first-turn-blockt256: fixed BLOCK_T 256 for the align padding both engines share; the CUDA kernel's TILE_T 256 matches, so no interaction beyond identical padding.

### Interference and A/B needs
No companion. This is kind: rewrite, judged on the replaced kernels' own row (_nvfp4_wgmma_grouped_kernel, 2 calls) against the incumbent _moe_nvfp4_gemm_kernel's 8083-9174 us on c4d14d62 and against the floor, not on the headline. The landed hypotheses it interacts with only set the inputs (align layout, zero-store flag) that the change explicitly reuses, so reverting or retuning them would change the incumbent and the candidate equally. The PATH CHOSEN / ON / OFF log lines are the internal control: an OFF run is a re-measure of master and is self-identifying. The next stage is the follow-up.

### Open questions
- Does master's moe_align_block_size(local_ids, 256, 32) emit any -1 block_experts at all, or does it drop -1 slots entirely? If none are emitted, change (3) is moot for the kernel and the -1 branch is only exercised by the one-hot check.
- Do cudaFuncGetAttributes binaryVersion/ptxVersion actually distinguish sm_90 from sm_90a (both report 90)? The __CUDA_ARCH_FEAT_SM90_ALL error-word readback after the canary is the decisive check; the attribute log is only diagnostic.
- With 2 consumer WGs each holding a 64x256 fp32 accumulator (128 regs/thread) plus two A-fragment buffers and dequant temporaries, do ptxas registers fit without spills under 288 threads/CTA (setmaxnreg rebalancing)? Log n_regs/local bytes from the attribute query.
- For stage 2: is the activation gather (one producer warp, 2048 x 16 B cp.async per stage, 64x re-read for w13) the limiter? Compare the kernel row against ~4 TB/s of L2 activation traffic; if bound there, widen the weight tile (cluster multicast of the B slab or 256 weight rows per CTA) before tuning dequant.
- silu-mul-sorted-first-turn is running and first-turn-moe-dead-work-stack / experts-triton-persistent-grid-first-turn are queued on the same T > 4096 path; if any lands first, re-read the incumbent row from the new master before scoring milestones.


## Implementation notes

Built on master d7ccd53 (the worktree's submission.py). Hunks 1-5 of `done/neutral/experts-cuda-wgmma-mixed-input-grouped-gemm.patch` were applied with `git apply -C1` (the CUDA source and helpers, `_cuda_fragment_maps`/`_cuda_bank_one`/`_cuda_bank`, `_check_cuda_onehot`, `_rel_err`, `_check_cuda_e2e`, `_setup_cuda_experts`, and its call at the end of `prepare()`). The `_experts`/`_experts_triton` hunks were ported by hand onto master's versions. The kernel design is unchanged: persistent 288-thread CTAs, 2 consumer WGs on m64n256k16 RS wgmma, 1 producer warp, 5 stages, 1-D cp.async.bulk weight blocks, plain `cp.async.cg` 16 B activation gather, and every mbarrier wait bounded by about 2 s of clock64.

Changes against the parent patch:
1. **Build (sm_90a only).** `_build_cuda_experts` now runs in `prepare()` (via `_setup_cuda_experts`), not at import. It sets `TORCH_CUDA_ARCH_LIST=9.0a` around `load_inline` and restores the old value in `finally`. It logs torch's own arch flags (`_get_cuda_arch_flags()`), the extra cflags, the compile seconds, and the `-gencode`/`-arch` tokens read back from the build.ninja that load_inline wrote (what nvcc received). If any of those tokens is not 90a, the path is turned OFF. cflags are `-O3 -lineinfo`, plus `-gencode=arch=compute_90a,code=sm_90a` only when torch's own flags carry a non-90a entry.
   - **Deviation.** The spec says "pass only -gencode arch=compute_90a,code=sm_90a". I did not add our own gencode when torch's flags are already 90a-only. Torch's JIT path appends its arch flags regardless of user flags, so adding ours too would duplicate the gencode.
   - The extension was renamed `muse_prefill_nvfp4_wgmma_sm90a_v2`.
2. **Guard.** All wgmma asm sits under `#if defined(__CUDA_ARCH_FEAT_SM90_ALL)`: `wgmma_m64n256k16_rs` and `wg_fence`/`wg_commit`/`wg_wait`, which become empty stubs otherwise. The kernel body itself is wrapped in `#if !defined(__CUDA_ARCH_FEAT_SM90_ALL)`, whose branch does `atomicOr(err, 4)` and returns (no `__trap`), `#else <body> #endif`. There is no setmaxnreg in this kernel.
3. **Distinct profile name.** The kernel is now `extern "C"`, so the profile row reads the bare name `_nvfp4_wgmma_grouped_kernel`.
4. **Image check.** `_setup_cuda_experts` logs the device capability and requires (9, 0). It logs the `cudaFuncGetAttributes` regs, local bytes, ptx and sass.
   - It asserts sass (binaryVersion) == 90, plus occupancy ≥ 1 and max threads ≥ 288.
   - **Deviation.** ptxVersion is only logged. Following the review's open question, the attributes cannot tell 90 from 90a, and a cubin-only build might report a ptx value I can't predict. The canary's error word is the decisive check.
5. **In-process canary** (`_check_cuda_canary`, new). It runs before the one-hot check: one 256-token tile on a contiguous slice of expert 0's CUDA bank (128 rows × 2 chunks), w13 (top_k 8) then w2 (top_k 1). It checks the error word (bit 4 means a non-sm_90a image, 1/2 mean barrier timeouts) and that the output is finite.
   - Every self-check launch (canary, one-hot w13/w2, e2e w13/w2, e2e full) goes through `_cuda_checked_launch`. That helper prints a flushed line before the launch, then `torch.cuda.synchronize()`, then a flushed line with the error word, so a fault names its call. There are no child processes anywhere.
6. **Master's w2 zero-store flag.** `_experts_triton`'s CUDA branch passes `state["zero_nonlocal_w2"]` for w2, exactly as master's Triton branch does. `_check_cuda_onehot` and `_check_cuda_e2e` use the same state flag instead of the `ZERO_NONLOCAL_W2` constant.
   - The kernel already skipped both MMA and stores for -1 items when zero_nonlocal is 0. Its `if (e < 0) { if (zero_nonlocal) {...} continue; }` was unchanged.
   - The one-hot check asserts the -1 block stays NaN when the flag is off.
7. **Master's 32-expert align.** The CUDA engine sits inside master's `_experts_triton`. It consumes that function's `moe_align_block_size(local_ids, 256, LOCAL_EXPERTS)` outputs, with expert index 0-31 into `_cuda_bank`, which holds the same 32 local experts, in the same order, as `state["bank"]`. `_check_cuda_e2e` now aligns with `LOCAL_EXPERTS` (it was `N_EXPERTS`), so the check exercises the same tile layout as kernel_fn.
8. **Routing.** `_experts` logs `[cuda expert gemm] PATH CHOSEN T=<T>: marlin|cuda|triton` once per token count. `_use_cuda_experts` is a shape plus a prepare() flag. The Marlin path, the router, the silu_and_mul between the GEMMs and the weighted sum are untouched.
   - Any build, arch-flag, image, canary, one-hot or e2e failure prints exactly one `[cuda expert gemm] OFF: <reason>`, and master's Triton path runs.

Review fixes applied: (3) and (4) as specified. The open question on attributes is handled as above (diagnostic only; the error word decides). The regs/local-bytes log covers the spill question.

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc diff`:
- [x] CUDA-graph capturable. kernel_fn's new code (`_experts` → `_use_cuda_experts`, `_experts_triton` CUDA branch → `_cuda_moe_gemm`) has no `.item()`/`.cpu()`/sync/tensor print. It branches only on `x.shape[0]` and prepare()'s `state["cuda_experts"]`. Buffers are the same `torch.empty` allocations sized by shape. The PATH CHOSEN print is host text, not a tensor. The pybind launch uses the current stream, with grid = SM count and no host reads.
- [x] Nothing derived from `data` is cached across calls. `_PATH_LOGGED` holds token counts (shapes) only. `state["cuda_err"]` is an error word that only the prepare() checks read.
- [x] Weights are repacked losslessly in prepare() only. `_cuda_bank` permutes the E2M1 codes and e4m3 scales into new uint8 blocks, and each expert is asserted to invert exactly. Global scales stay fp32 (the Triton bank's). NVFP4 stays NVFP4.
- [x] Activations stay BF16 in and out of both GEMMs. Dequant is exact in BF16 (no ftz, no fast-math), with fp32 accumulation. The KV cache and queries are untouched.
- [x] Attention sees every key: attention code is untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocess/multiprocessing/fork/spawn, or network. `load_inline` is torch's own JIT build, as in the parent. `os` is used only for `os.environ` (TORCH_CUDA_ARCH_LIST, restored) and `os.path.join` to read our own build.ninja.
- [x] All work happens inside the timed call. The self-checks, canary and bank are prepare()-only and untimed. There is no output caching or replay.
- [x] First measurement must be `okc check`, not a scored submit (per hypothesis/LESSONS). Noted for the dispatcher.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`):
```
24 x warn host-sync ... (outside kernel_fn's call graph; fine in prepare())   [all in prepare()-only checks]
OK with warnings: muse_prefill_layer/submission.py
```
`python3 -I -m py_compile`: ok; __pycache__ removed. The CUDA source's `#if`/`#else`/`#endif` nesting and brace count were checked as balanced by extracting the string. nvcc is not available on this node, so the source is not compiled here.

## Pre-flight audit

Audited 2026-10-24 by an opus subagent, by reading the code and running a numpy emulation. **Verdict: dispatch** (no fixes).

- **Dequant and A fragments.** A bitwise numpy emulation took random codes and e4m3 scales through `_cuda_fragment_maps` and `_cuda_bank_one`. It then decoded them as each consumer thread does: the LDS.128/LDS.64 offsets, `e2m1_pair`, `scale_pair` and the bf16 multiply. The values were placed by the m64k16 RS A-fragment layout. The result equals E2M1 x e4m3 x 2^-7 on every element of a 256x128 expert (max abs diff 0).
- **Descriptor, swizzle and accumulator layout.**
  - The B descriptor is start>>4, LBO 1, SBO 1024 (8-row groups) and layout 1 (SW128). The k16 step advances it by +32 B.
  - The producer writes 16-B chunk `j ^ (n & 7)` at `n * 128`. This matches the hardware's XOR of address bits 4-6 with bits 7-9, because the stages are 1024-aligned (37888 = 37 x 1024).
  - The RS wgmma operand order is d, a, desc, p, 1, 1, tnspB=0.
  - The epilogue's `d[4b+2h+u]` maps to (row + 8h, token 8b+2t+u), which is correct.
- **Pipeline.**
  - The full barrier counts 33 (32 noinc cp.async arrivals plus 1 expect_tx), and the empty barrier counts 8 (one per consumer warp).
  - The producer waits on the empty barrier starting at parity 1.
  - Every chunk's stage is released exactly once.
  - kc is even for K=4096 and for the canary's 128.
  - The producer and the consumers skip -1 items in the same way.
- **Gates.**
  - With `TORCH_CUDA_ARCH_LIST=9.0a`, torch's JIT `_get_cuda_arch_flags()` yields only `-gencode=arch=compute_90a,code=sm_90a`, so the build.ninja arch gate can pass.
  - The sass == 90, max_threads >= 288 and occupancy >= 1 gates fit 190.6 KB of smem and at most 224 regs at `__launch_bounds__(288, 1)`.
  - The canary, one-hot and e2e checks use the same 32-expert align and `state["zero_nonlocal_w2"]` as kernel_fn.
- **Fallback.** The build, arch gate, attrs, bank, canary, one-hot and e2e steps are all inside try blocks. Any failure prints exactly one OFF line, and master's Triton path runs.
- **Rules.**
  - The kernel_fn path branches only on the shape and the prepare() flag. It adds no allocations and no syncs, and it launches on the current stream.
  - Nothing derived from `data` is cached.
  - The bank repack is lossless, and both permutations are asserted to invert exactly.
  - No child process is started.
  - `open()` and `os.environ` were already used by the bf16-verify-streamk check run (job f5b1361a), and the judge's checks did not flag it.

Residual risk: the kernel has never been compiled or run. A hard fault in the canary, such as an illegal instruction or address from the wgmma or bulk-copy PTX, would poison the CUDA context and fail the whole run instead of falling back. The first measurement must therefore be an `okc check`.

## Result

- job `6d047cdeefd94357` (check), status done, passed True, tests 5/5
- speedup 0.9401x, 95% CI [0.9260, 0.9542], score 412102.80225236027 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: regression** (regression)
- files: `muse_prefill_layer/hypotheses/done/regression/experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc.result.json`, `muse_prefill_layer/hypotheses/done/regression/experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc.patch`; branch `hyp/muse_prefill_layer/experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1101.29 | 1116.84 | 1.0006 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1529.55 | 1507.16 | 0.9983 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3318.71 | 3302.86 | 0.999 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4849.27 | 4931.99 | 0.9982 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14705.32 | 22028.66 | 0.6746 | lost |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 14264.64 | 21471.35 | 0.6604 | lost |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 17746.69 | 25236.53 | 0.7256 | lost |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 21733.75 | 29244.66 | 0.7534 | lost |

lost on full first_turn; sliding first_turn. Expected hit: 0/3. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `_moe_nvfp4_gemm_kernel x2 (w13, w2) at T > 4096 (first_turn)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_prefill_layer "<what to fix or tune>" --kind rewrite --depends-on experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.

