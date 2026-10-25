---
problem: muse_prefill_layer
slug: experts-cuda-wgmma-grouped-gemm-remeasure-sm90a
created: 2026-10-25T001301Z
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
  nvcc received and the compile seconds. (2) FIRST LAUNCH ISOLATED (LESSONS 2026-10-25): run the
  first launch plus the one-hot check of the new kernel in a child process (multiprocessing spawn,
  synthetic weights, bounded timeout); only if the child exits clean does prepare() run the
  in-process self-checks; a child crash or illegal instruction prints OFF: <reason> and master's
  Triton path runs. (3) MASTER'S w2 ZERO-STORE FLAG: the w2 call's zero_nonlocal argument must be
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
  build, child, one-hot or e2e failure prints exactly one '[cuda expert gemm] OFF: <reason>' and
  master runs unchanged. Harvest: grep the log for the build line, '[cuda expert gemm] ON' and
  'PATH CHOSEN T=16384: cuda', and the profile for an _nvfp4_wgmma_grouped_kernel row (2 calls)
  before reading anything else.
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
review: hold
status: hold
review_at: 2026-10-25T001517Z
review_companions: null
hold_reason: "adversarial review: The rewrite design and the build fix are sound, but change item (2) spawns a multiprocessing child in prepare(), and TASK.md and the technical rules list 'start"
---

# experts-cuda-wgmma-grouped-gemm-remeasure-sm90a

## Change

BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored), not on the old branch: start from master's submission.py and hand-port the CUDA expert engine from done/neutral/experts-cuda-wgmma-mixed-input-grouped-gemm.patch (branch hyp/muse_prefill_layer/experts-cuda-wgmma-mixed-input-grouped-gemm: _CUDA_EXPERTS_CUDA/_CUDA_EXPERTS_CPP, _nvfp4_wgmma_grouped_kernel, nvfp4_wgmma_gemm, _cuda_bank, _setup_cuda_experts, _check_cuda_onehot, _check_cuda_e2e, _use_cuda_experts, the PATH CHOSEN log). The kernel design is UNCHANGED (persistent 288-thread CTAs, 2 cooperative consumer WGs on m64n256k16 RS wgmma, 1 producer warp, 5-stage ring, 1-D cp.async.bulk weight blocks in A-fragment order, BLOCK_T 256). That run (job 1821bb0e) never ran its kernel: ptxas rejected wgmma on .target sm_90 because torch's cpp_extension added its own compute_90 gencode beside -arch=sm_90a, so it measured master (REWRITE_READ_2026-10-25 #1). Changes vs that patch, and only these: (1) BUILD: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' immediately before load_inline (restore the old value after) and pass only -gencode arch=compute_90a,code=sm_90a; wrap every wgmma/fence/commit_group/wait_group (and any setmaxnreg) asm in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) with an #else that sets the error word and returns, so a stray plain-sm_90 image still compiles and the self-check turns the path OFF; log the arch flags nvcc received and the compile seconds. (2) FIRST LAUNCH ISOLATED (LESSONS 2026-10-25): run the first launch plus the one-hot check of the new kernel in a child process (multiprocessing spawn, synthetic weights, bounded timeout); only if the child exits clean does prepare() run the in-process self-checks; a child crash or illegal instruction prints OFF: <reason> and master's Triton path runs. (3) MASTER'S w2 ZERO-STORE FLAG: the w2 call's zero_nonlocal argument must be state['zero_nonlocal_w2'] exactly as master's _experts_triton passes it (OFF at first_turn now, since master's probe found the weighted sum skips remote slots); the CUDA kernel's -1-block items must skip both MMA and stores when it is False, instead of always zero-storing (that would add back ~940 MB of stores master removed). The NaN-prefilled non-local w2 comparison in _check_cuda_e2e already tolerates either setting; run it with master's flag value. (4) MASTER'S 32-EXPERT ALIGN: route the CUDA kernel inside master's _experts_triton, which already calls moe_align_block_size(local_ids, 256, LOCAL_EXPERTS) on ids that are 0-31 or -1; the kernel's expert index is therefore 0-31 into the 32-expert bank, and n_padded/block_experts come from that align (fewer padded tiles than the old 257-expert align); _cuda_bank must index the same 32 local experts as state['bank']. Do NOT touch the Marlin path at T <= 4096 (master's _align_parallel local-routes sorted_ids and _silu_mul_sorted_kernel stay as they are), the router, the silu_and_mul between the GEMMs, or the post-expert weighted sum. Fallback stays: any build, child, one-hot or e2e failure prints exactly one '[cuda expert gemm] OFF: <reason>' and master runs unchanged. Harvest: grep the log for the build line, '[cuda expert gemm] ON' and 'PATH CHOSEN T=16384: cuda', and the profile for an _nvfp4_wgmma_grouped_kernel row (2 calls) before reading anything else.

## Expected

- `*load=first_turn*` faster: Replaced: _moe_nvfp4_gemm_kernel x2, 8083-9174 us on the new master (c4d14d62) across the four first_turn shapes, ~5x the 1.67 ms math floor (~2.0-2.3 ms once BLOCK_T-256 padding is counted). Milestones as stage 1: M0 log ON + PATH CHOSEN cuda + 5/5 tests; M1 w13+w2 under 6.0 ms; M2 under 3.5 ms (~470 TFLOP/s). At M2 the first_turn load gains ~1.5x and the headline ~1.15x. Kernel-row noise is about +-6%, so judge the row, not the headline.
- `*load=turn_1*` neutral: Marlin path untouched (768.6/735.9 us on master); T <= 4096 never routes to CUDA.
- `*load=turn_4*` neutral: Marlin path untouched (2148.2/2115.3 us on master).

## Risk

The design was never measured: the only risks new to this run are the arch fix itself (verify the build line) and a hard fault in the first launch, which the child-process isolation now contains. If the log says ON and the kernel row is slower than the Triton incumbent, that is a measured stage-1 result for the next stage to tune, not a do-not-retry.

## Milestones

Rewrite stage 1 of `_moe_nvfp4_gemm_kernel x2 (w13, w2) at T > 4096 (first_turn)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T001517Z by an adversarial subagent (model opus).

**Verdict: hold.** The rewrite design and the build fix are sound, but change item (2) spawns a multiprocessing child in prepare(), and TASK.md and the technical rules list 'starts processes' as a static reject; drop (2), keep the __CUDA_ARCH_FEAT_SM90_ALL guard plus the in-process one-hot/e2e checks as the safety net, then proceed unchanged.

### Justification
A: The target is real and the largest gap in the layer: _moe_nvfp4_gemm_kernel x2 costs 8083-9174 us at every first_turn shape on master c4d14d62 (37-57% of those shapes), against a ~1.67 ms math floor (~2.0-2.3 ms with BLOCK_T-256 padding), so about 4-5x off. first_turn carries the heaviest load weight (138 on sliding ctx32768). A persistent sm_90a kernel with RS wgmma m64n256k16 on two consumer WGs, a TMA-bulk producer warp and a 5-stage ring is the right tensor-core path for this shape (T=16384, about 4K rows per local expert, K=8192-class), and M1 (<6 ms) is plausible. The expected directions agree with TASK.md: turn_1 and turn_4 stay on Marlin (T<=4096 gate), so they should be flat. The build diagnosis in REWRITE_READ_2026-10-25 #1 is specific and correct: torch's cpp_extension adds a compute_90 gencode from TORCH_CUDA_ARCH_LIST, and ptxas rejects wgmma there. Setting TORCH_CUDA_ARCH_LIST=9.0a plus the feature-macro guard fixes it. The earlier patch already honours zero_nonlocal for -1 items (producer 'if (e < 0) continue', consumer store guarded by zero_nonlocal), so item (3) only needs master's state['zero_nonlocal_w2'] to be passed through in place of the old ZERO_NONLOCAL_W2 constant. The weakest link is M2 (<3.5 ms, ~470 TFLOP/s). The RS path dequantizes NVFP4 to BF16 in the consumer warpgroups' registers, so dequant ALU work competes with wgmma issue, and with only 2 consumer WGs per SM there is little overlap headroom. This stage measures whether dequant or the producer bound first, which is fine for a rewrite stage. A secondary risk is a hard fault (illegal instruction or hang) in the first in-process launch. That would kill the CUDA context and fail the whole check run instead of falling back. The child-process isolation was meant to contain that risk, but it is not allowed (see rules_flags), so the run must accept the risk.

### Confounders on the current kernel
- stack-tonight-five-prefill (moe-align-32-experts-first-turn + w2-nonlocal-zero-store-off-first-turn): this changes the CUDA kernel's input contract (32-expert align, so fewer padded tiles and different n_padded/block_experts, and w2 zero-store OFF). The port inherits both deliberately, and the incumbent row in c4d14d62 already includes them, so the kernel-row comparison is like-for-like. If the port wrongly hard-codes zero_nonlocal=True, it adds back ~940 MB of stores and confounds the w2 row; check the '[w2 zero-store] OFF' log line.
- experts-hybrid-marlin-turn-triton-first-turn: this supplies the T>4096 gate the CUDA path hangs off, so the change is shape-disjoint from Marlin. Not a confounder if _experts keeps the gate byte-identical.
- experts-triton-w4a16-first-turn-blockt256: this fixes BLOCK_T=256 in EXPERT_CONFIGS, and the align pads to it. The CUDA kernel's TILE_T must equal cfg['BLOCK_T'] or the align output is misread. This is a correctness coupling, not a timing confounder.
- sglang-pipeline-parity: no interaction beyond providing the surrounding pipeline.

### Interference and A/B needs
No companion is warranted. This is kind: rewrite and is judged on the _nvfp4_wgmma_grouped_kernel row (2 calls) against the incumbent _moe_nvfp4_gemm_kernel row per first_turn shape, with noise of about +-6% (REWRITE_READ calibration). An ab-revert of align32/zero-store would only change the incumbent baseline that the port deliberately shares. A control would not disambiguate anything the profile row does not already show. The next stage is the follow-up.

### Open questions
- Does the ported _cuda_bank index exactly the same 32 local experts (local_start offset) as state['bank']? The old patch built it against the 257-expert align.
- Does the producer's 1-D cp.async.bulk weight-block size times 5 stages fit shared memory alongside the BF16 activation tiles for BLOCK_T=256 on both w13 (N=2*size_n) and w2 (K=size_n)? Log the smem bytes and occupancy at build.
- If the kernel row lands between the M1 and M2 milestones, is the bottleneck the RS dequant (consumer ALU) or the producer bandwidth? Ask the next stage's profile for an ncu-style split (or a no-dequant timing variant in prepare) to decide stage 2.

