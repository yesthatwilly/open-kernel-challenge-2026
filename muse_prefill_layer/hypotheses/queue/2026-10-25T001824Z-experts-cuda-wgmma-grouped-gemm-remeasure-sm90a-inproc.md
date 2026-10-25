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
status: queued
review_at: 2026-10-25T002131Z
review_companions: null
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

