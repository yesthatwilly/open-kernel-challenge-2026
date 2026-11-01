---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild
created: 2026-10-25T001223Z
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
  every wgmma and setmaxnreg body in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) ... #else __trap();
  #endif so any stray compute_90 pass still assembles; (3) add -lineinfo. Also per LESSONS
  2026-10-25, run the setup self-check's first launches in a spawned child process (same build
  dir), enabling the path only if the child exits 0 and prints PASSED, so a device fault falls
  back cleanly instead of poisoning the context. No tiling, pipeline, dequant, gate (num_tokens >=
  128, own bound constant) or layout change. Log the build line and PATH CHOSEN; a run whose log
  does not say PATH CHOSEN = wgmma is unmeasured, not neutral.
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
review: hold
status: hold
review_at: 2026-10-25T001517Z
review_companions: null
hold_reason: "adversarial review: The sm_90a build fix is correct and the rewrite is worth re-running as stage 1, but the spec also requires the self-check to run in a spawned child process, and"
---

# cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild

## Change

RE-MEASURE (stage 1b): the previous run never executed the new kernel; load_inline failed because cpp_extension appended its own -gencode arch=compute_90,code=sm_90 beside -arch=sm_90a and ptxas cannot assemble wgmma.mma_async/fence/commit_group/wait_group or setmaxnreg.inc/dec for plain sm_90, so it fell back to Marlin and measured master. Apply the archived kernel verbatim (muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-verify.patch, spec in done/neutral/2026-10-24T172038Z-cuda-nvfp4-wgmma-rega-grouped-verify.md), rebased onto current master (align-parallel-per-expert-verify), and make ONLY the build fix: (1) set os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0a' immediately before load_inline (restore the previous value after), and pass only -gencode arch=compute_90a,code=sm_90a in extra_cuda_cflags; (2) wrap every wgmma and setmaxnreg body in #if defined(__CUDA_ARCH_FEAT_SM90_ALL) ... #else __trap(); #endif so any stray compute_90 pass still assembles; (3) add -lineinfo. Also per LESSONS 2026-10-25, run the setup self-check's first launches in a spawned child process (same build dir), enabling the path only if the child exits 0 and prints PASSED, so a device fault falls back cleanly instead of poisoning the context. No tiling, pipeline, dequant, gate (num_tokens >= 128, own bound constant) or layout change. Log the build line and PATH CHOSEN; a run whose log does not say PATH CHOSEN = wgmma is unmeasured, not neutral.

## Expected

- `bs=128 *` faster: verify_16: incumbent Marlin w13+w2 316.1 us + act_and_mul 3.9 us vs floor 101.3 us; M2 = replaced kernels below incumbent (~320 us), M3 <= 170 us (~2.9 TB/s). Marlin rows move +/-20% run to run, so judge on the replaced-kernel row, not the headline
- `bs=256 *` faster: verify_32: incumbent 490.2 + 6.4 us (403.1 in the fallback rerun) vs floor 108.9 us; M3 <= 220 us via wgmma register-A, one weight stream per expert chunk
- `bs=32 *` flat: control: num_tokens < 128 runs master
- `bs=16 *` flat: control
- `bs=8 *` flat: control
- `bs=1 *` flat: control: untouched

## Risk

Stage 1b: integer --stage only, so recorded as stage 1 with depends_on the unmeasured run; REWRITE_READ_2026-10-25 section 3 says re-run as stage 1 since nothing was learned. Confound: stack checks 7e5ab95a413f4ebe and 0fde9013de97404d include the dispatch-combine rewrite on 32 < M <= 512; if one lands first, rebase on it and keep the wgmma gate on its own constant (LESSONS 2026-10-25 from17). Risk: wgmma path may still fault at runtime; the child-process check turns that into a clean fallback.

## Milestones

Rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T001517Z by an adversarial subagent (model opus).

**Verdict: hold.** The sm_90a build fix is correct and the rewrite is worth re-running as stage 1, but the spec also requires the self-check to run in a spawned child process, and TASK.md rejects any submission that 'starts processes' (a spawn context also re-imports the harness __main__), so hold until that sub-change is removed or replaced with an in-process canary.

### Justification
Design (a kind: rewrite, so judged on its replaced-kernel rows, not the headline): sound for H200 at these shapes. Master spends 292.5 us at bs=128 and 391.4 us at bs=256 in Marlin, which is 95% of each shape and 2.9x / 3.6x the 101.3 / 108.9 us floors, about 1.5 and 1.1 TB/s effective. At bs=256, roughly 350-450 local routes x 2 GEMMs is tens of us of BF16 MMA even at wgmma peak, so the argument that an mma.sync engine plus dequant cannot get near the floor holds, and wgmma with a register-A dequantized weight (swap-AB, weight rows on M) is the right tensor-core path. The bytes-in-flight plan (>= 48 KB of weight per SM across 5-6 stages, bulk copy plus TMA, 1 producer and 2 consumer warpgroups) is the right order of magnitude for ~3 TB/s. The register budget is legal: 40x128 + 232x256 = 64512 <= 65536. The control shapes (bs<128) are gated off and correctly expected flat. The build diagnosis is also correct: per REWRITE_READ section 3 and LESSONS 2026-10-25, cpp_extension added a compute_90 pass, and pinning TORCH_CUDA_ARCH_LIST=9.0a with only -gencode compute_90a/sm_90a plus the __CUDA_ARCH_FEAT_SM90_ALL guard is the documented fix. Weakest link: no instruction in this kernel has ever executed on the judge. The archived patch keeps createpolicy.fractional.L2::evict_first/evict_normal and an L2::cache_hint variant of cp.async.bulk (patch lines ~160 and ~447). The sibling swap-AB kernel trapped with illegal instruction 715, and LESSONS names that exact helper family as the prime suspect. Applying the kernel 'verbatim' therefore carries the most likely fault forward, along with setmaxnreg, mbarrier expect_tx, TMA tensor-map encoding through cudaGetDriverEntryPoint, and stmatrix.trans, all first executions. Second risk: compile time. Two families x four N buckets of a wgmma template has never compiled on the judge (the first attempt died in ptxas), so it is unknown whether it fits the 15-minute run budget alongside prepare's 300 s.

### Confounders on the current kernel
- align-parallel-per-expert-verify (master tip): the archived wgmma path calls moe_align_block_size(local_ids, 16, 32) at num_tokens >= 128, so on the target shapes it bypasses master's _align_parallel_kernel (4.0 / 6.9 us). Compare the replaced-kernel rows only (gather + w13 + w2 against Marlin x2 + act_and_mul) and report align separately, or the align delta of a few us leaks into the shape row. The rebase onto align-parallel must state which align the wgmma branch uses.
- marlin-block-m-two-smaller-on-postnorm / triton-map-align-on-block-m-restack: they set block_size_m for Marlin at bs>=128, which only shapes the incumbent row (292.5 / 391.4 us). The wgmma path uses its own block 16, so this is no confound on the kernel row, but the M2 incumbent must be read from the same run's base profile, not the 316.1 / 490.2 us quoted from an older base.
- post-expert-norm-triton-small-m: no interaction. bs>=32 still calls SGLang's fused_post_expert_norm_weighted_sum, and the wgmma w2 epilogue scatters into cache3 in the layout that combine expects.
- sglang-marlin-path-parity: this is the fallback path itself, so a run without PATH CHOSEN = WGMMA measures parity/master again, which the spec already labels unmeasured.

### Interference and A/B needs
No companion is warranted. As a rewrite it is judged on its own kernel rows (gather + w13 + w2 at bs=128/256) against the same run's Marlin + act_and_mul rows and the floor. The bs<128 shapes are built-in controls. Marlin's ±20% profile noise affects only the incumbent reference, and the ~2x gap to M3 is far larger than that. An ab-revert of any landed hypothesis would not change attribution, because the replaced kernels are disjoint from everything landed except the align call noted above, which should be reported, not A/B'd. The next stage is the follow-up.

### Open questions
- Rebase target: if stack-onewarp-hoisted-dispatchcombine(-from17), now running, lands first, it changes align and combine at bs>=128. Keep the wgmma gate on its own WG_MIN_TOKENS constant as the spec says, and confirm the w2 scatter still matches the new combine's input layout.
- Should the 'verbatim' kernel keep createpolicy + L2::cache_hint on cp.async.bulk, given that LESSONS names cache-hint helpers as the prime 715 suspect? Dropping the hint costs little (weights are streamed once) and removes the most likely trap.
- Will the build fit the 15-minute run limit? Log compile seconds, and consider trimming to the N buckets that bs=128/256/257 actually hit.
- The log string is 'PATH CHOSEN = WGMMA' (uppercase) in the patch, and the harvest grep should match it case-insensitively.

