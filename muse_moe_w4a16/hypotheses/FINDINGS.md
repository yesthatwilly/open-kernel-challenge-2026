# muse_moe_w4a16 findings

Regenerated 2026-10-28T145238Z by `churn.py index`. Read this before writing a hypothesis for muse_moe_w4a16.

## Current best

1.8941x [1.8657,1.9225] from job `0b49b1aae6a04acb` (submit), master commit `2ec8090fab`, hypothesis campaign-eb-x1v6

Today's clean-context diagnosis: `muse_moe_w4a16/hypotheses/DIAGNOSIS.md` (ranked actions and its own Do-not-try list).

## By area

### MoE align (bs<=8 one warp; 17..32 parallel; bs>=128 CUDA) + post-expert combine
- **stack-onewarp-hoisted-dispatchcombine-from17** → neutral (within-ci), 1.3237x — no shape moved outside noise. Built on master. The variant of stack-onewarp-hoisted-dispatchcombine (same three done/ patches, unchanged) plus align-parallel-from-17-tokens (ALIGN_FUSED_MAX_

### MoE align (bs<=8 one warp; bs>=128 CUDA) + post-expert combine (bs<32 hoisted; bs>=128 CUDA)
- **stack-onewarp-hoisted-dispatchcombine** → neutral (within-ci), 1.3176x — no shape moved outside noise. Built on master. Stack the three moe results of 2026-10-24 that each matched the 1.3278 best within noise, applied unchanged from their done/ patches: align-sma

### MoE align/dispatch
- **sglang-marlin-path-parity** → gain (gain), 0.9888x — gained on decode_1 decode_1; decode_16 decode_16; decode_32 decode_32; verify_1 verify_1; verify_16 verify_16; verify_32 verify_32. Replace the starter's 32-expert dense loop (dequantize every local expert to BF16 on every call, x @ w13.T over all tokens, masked weighted sum) with SGLang's M
- **align-single-token-ep** → neutral (unproven-vs-best), 0.9951x — gained on decode_1 decode_1. At num_tokens == 1 only, call SGLang's moe_align_single_token(local_ids, block_size_m) (from sglang.kernels.ops.moe.moe_align_single_token) with the expert_map-
- **marlin-block-size-m-next-larger** → regression (regression), 0.9178x — lost on decode_32 decode_32; verify_16 verify_16. Change only the block_size_m pick in kernel_fn. SGLang's loop breaks on the first candidate in (8, 16, 32, 48, 64) where num_tokens*8/32/block_size_m < 0.9, whi
- **align-256-experts-below-128** → neutral (unproven-vs-best), 0.9998x — gained on decode_1 decode_1. In kernel_fn, for num_tokens < 128 only (bs=1, 8, 16, 32 and the bs=4/33 tests), call moe_align_block_size(local_ids, block_size_m, N_EXPERTS) exactly as baseli
- **triton-map-align-small-m-port** → neutral (beats-base-not-best), 1.1850x — gained on decode_1 decode_1; decode_16 decode_16; decode_32 decode_32; verify_1 verify_1. Port muse_decode_layer moe-align-single-kernel-small-t (patch: muse_decode_layer/hypotheses/done/neutral/moe-align-single-kernel-small-t.patch; its _align_small
- **triton-map-align-verify-chunked** → failed (test:failed). Extend master's single-program map+align (_align_small_kernel, triton-map-align-small-m-port) to num_tokens > 32 (bs=128/256, and the bs=33/257/skewed-128 tests
- **triton-map-align-verify-chunked-pow2-fix** → neutral (within-ci), 1.3107x — no shape moved outside noise. Re-run triton-map-align-verify-chunked (failed at compile: tl.arange(0, BSM) with BSM=48 in prepare's warm-up raised 'arange's range must be a power of 2') with
- **align-parallel-per-expert-verify** → neutral (unproven-vs-best), 1.3249x — gained on verify_16 verify_16; verify_32 verify_32. For num_tokens > ALIGN_FUSED_MAX_TOKENS (bs=128, 256 and the bs=33/257/skewed-128 tests) when expert_map is not None, replace master's else-branch (topk_ids.to(
- **align-parallel-from-17-tokens** → neutral (unproven-vs-best), 1.3250x — no shape moved outside noise. One constant plus its warm-up: lower ALIGN_FUSED_MAX_TOKENS from 32 to 16, so 17 <= num_tokens <= 32 (decode_32 bs=32) runs master's existing 33-program _align_
- **align-small-one-warp-tiny-m** → neutral (unproven-vs-best), 1.3287x — no shape moved outside noise. num_tokens <= 8 only (decode_1 bs=1, verify_1 bs=8, the bs=1/bs=4 tests); bs=16 and above keep master's launches exactly. In _align_small (host wrapper), replac
- **align-two-phase-port-from17** → neutral (beats-base-not-best), 1.3700x — gained on verify_32 verify_32. Built on master (9080a71, submission_blob 60dfe35d); MoE align only, at 17 <= num_tokens <= 512 (decode_32 bs=32, verify_16 bs=128, verify_32 bs=256, and the bs

### MoE align/dispatch + block_size_m pick
- **triton-map-align-on-block-m-restack** → gain (gain), 1.3117x — gained on decode_1 decode_1; decode_16 decode_16; decode_32 decode_32; verify_1 verify_1. Stack the two measured, shape-disjoint moe wins: start from master (marlin-block-m-two-smaller-on-postnorm, 1.254x: block_size_m two candidates down at num_toke
- **marlin-block-m-8-verify16** → regression (regression), 1.2992x — lost on verify_16 verify_16. In _forward on master 9080a71, for 128 <= num_tokens <= 255 only (verify_16 bs=128 and the skewed bs=128 test), set block_size_m = 8 instead of master's two-sma

### MoE align/dispatch + post-expert norm+combine
- **cuda-dispatch-combine-pdl-verify** → neutral (unproven-vs-best), 1.3283x — no shape moved outside noise. Dispatch + combine only, at num_tokens > 32 (verify_16 bs=128, verify_32 bs=256, and the bs=33, bs=257 and skewed bs=128 tests). num_tokens <= 32 runs master by

### combination: naive top-row stack (decode_1 w13 atomic, decode_32 align+post-norm, verify_32 wgmma align2 front end)
- **combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2** → neutral (unproven-vs-best), 1.3866x — gained on verify_32 verify_32. COMBINATION (naive top-row stack): every per-shape winner that beat master on its own row and never reached master, each behind its own prepare-time gate, built

### combination: small-M Marlin two-stream halves x w13 split-K atomic + shared floor
- **combo-smallm-two-stream-halves-x-w13-splitk** → neutral (unproven-vs-best), 1.3887x — no shape moved outside noise. COMBINATION (conjectural pairing, small-M Marlin side): two-stream expert halves paired with w13 split-K atomic add (w2 atomic is already on master) at 8 <= T <

### combination: wgmma two-CTA/SM (prefill port) x hot-expert tile interleave + shared floor
- **combo-wg-occ2-v64-hot-interleave-align2** → neutral (unproven-vs-best), 1.4021x — gained on verify_32 verify_32. COMBINATION (conjectural pairing, wgmma side): prefill's two-CTAs-per-SM rega variant ported to moe's wgmma kernels at B256 (and as a B128 candidate), paired wi

### expert GEMM path selection (wgmma wg2 at B256 + Marlin two-stream halves at T 8-32) restack
- **restack-occ2-wg2-b256-two-stream-halves-on-master** → neutral (within-ci), 1.3996x — no shape moved outside noise. COMBINATION RESTACK of two measured per-cell wins onto current master (bdbbc97b, combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2, 1.3866x), each behind its o

### launch overhead
- **marlin-two-stream-expert-halves-small-m** → neutral (within-ci), 1.3024x — no shape moved outside noise. At 8 <= num_tokens <= 32 only (verify_1 bs=8, decode_16 bs=16, decode_32 bs=32; bs=1, the bs=4 test and num_tokens > 32 run master unchanged), split the expert 
- **post-norm-pdl-launch-small-m** → neutral (within-ci), 1.3222x — no shape moved outside noise. num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16, bs=4 test) only; master's _post_norm_combine_kernel body is otherwise unchanged and every other launch is master's.
- **wg-pdl-gather-w13-w2-chain** → neutral (unproven-vs-best), 1.3752x — no shape moved outside noise. Programmatic dependent launch across master's wgmma chain at 128 <= T <= 512 (verify_32 bs=256 runs it; verify_16 only if its bucket flag is on). Kernels, tilin

### post-expert norm+combine
- **post-expert-norm-triton-small-m** → gain (gain), 1.1297x — gained on decode_1 decode_1; decode_16 decode_16; verify_1 verify_1. For num_tokens < 32 only (bs=1, 8, 16; the bs=4 test too), replace the call to SGLang's fused_post_expert_norm_weighted_sum at the end of kernel_fn with one lau
- **post-norm-triton-all-m** → neutral (within-ci), 1.2483x — no shape moved outside noise. On top of marlin-block-m-two-smaller-on-postnorm, remove the SMALL_M_POST_NORM = 32 split at the end of kernel_fn so every num_tokens (bs=32/33/128/256/257 too)
- **post-norm-combine-hoisted-slot-loads** → neutral (unproven-vs-best), 1.3282x — no shape moved outside noise. num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16 and the bs=4 test) only; bs>=32 and every other kernel stay master byte for byte. Rewrite master's _post_norm_combin
- **post-norm-triton-hoisted-decode32** → neutral (within-ci), 1.3248x — no shape moved outside noise. num_tokens == 32 only (decode_32); bs<32 keeps master's _post_norm_combine_kernel unchanged and bs>=33 (incl. the bs=33 test) keeps SGLang's fused_post_expert_n

### w13 marlin
- **triton-w13-grouped-gemv-small-m** → regression (regression), 0.7506x — lost on decode_1 decode_1; verify_1 verify_1. For num_tokens <= 8 only (decode_1 and verify_1; all other shapes keep the parity path untouched), replace the first moe_wna16_marlin_gemm (w13, top_k=8) with t
- **marlin-workspace-blocks-per-sm-8** → neutral (unproven-vs-best), 0.9921x — no shape moved outside noise. In prepare(), build a second lock workspace marlin_make_workspace(device, max_blocks_per_sm=8) and pass it to the w13 moe_wna16_marlin_gemm call only; the w2 ca
- **w13-marlin-atomic-add-small-m** → neutral (within-ci), 0.9869x — no shape moved outside noise. For num_tokens <= 32, pass use_atomic_add=True (and use_fp32_reduce=False, which Marlin requires alongside atomic add) to the w13 moe_wna16_marlin_gemm call onl
- **w13-marlin-bf16-reduce-verify** → neutral (within-ci), 0.9840x — no shape moved outside noise. For num_tokens >= 128 (verify_16 and verify_32), pass use_fp32_reduce=False to the w13 moe_wna16_marlin_gemm call only (use_atomic_add stays False; w2 and all d
- **marlin-block-size-m-next-smaller** → gain (gain), 1.0394x — gained on verify_16 verify_16; verify_32 verify_32. Change only the block_size_m pick in kernel_fn: keep SGLang's loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) (first candidate with num_tokens*8/32/bloc
- **marlin-block-size-m-verify-two-smaller** → gain (gain), 1.0916x — gained on verify_16 verify_16; verify_32 verify_32. Change only the block_size_m pick in kernel_fn, for num_tokens >= 128 only: take SGLang's pick (the first of BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) with 
- **small-m-grouped-gemv-splitk-rewrite** → regression (regression), 0.6635x — lost on decode_1 decode_1; decode_16 decode_16; verify_1 verify_1. Stage 1 of a small-batch grouped W4A16 GEMV that replaces both moe_wna16_marlin_gemm calls when num_tokens <= 16 (decode_1 bs=1, verify_1 bs=8, decode_16 bs=16)
- **marlin-block-m-cap-16-port** → neutral (beats-base-not-best), 1.0950x — gained on verify_16 verify_16; verify_32 verify_32. Port muse_decode_layer marlin-block-m-cap-16 (landed gain 1.062x, job 0730135cdcac43b7; patch in muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.pa
- **marlin-block-m-two-smaller-on-postnorm** → gain (gain), 1.2536x — gained on verify_16 verify_16; verify_32 verify_32. Restack the landed-but-unmerged marlin-block-size-m-verify-two-smaller (1.092x on the pre-post-norm base, job ed837a4a46d44a95; patch at muse_moe_w4a16/hypothes
- **w13-marlin-atomic-add-on-master-small-m** → neutral (unproven-vs-best), 1.3558x — no shape moved outside noise. On master (9080a71, w2 atomic add already landed for T <= 32), pass use_atomic_add=True to the FIRST moe_wna16_marlin_gemm (w13, top_k=8) for num_tokens <= 32 o
- **w13-marlin-atomic-add-t1-only** → neutral (unproven-vs-best), 1.3461x — no shape moved outside noise. Restack the one measured row of w13-marlin-atomic-add-on-master-small-m (job 171684977ec1470f, 1.3558x check, on master 9080a71) with its gate narrowed to num_t

### w13 marlin + silu-mul + w2 marlin
- **cuda-nvfp4-swapab-mma-persistent-small-m** → failed (test:failed). Small-M only: num_tokens <= 32 (decode_1 bs=1, verify_1 bs=8, decode_16, decode_32 and the bs=1/bs=4 tests); num_tokens > 32 (bs=33, 128, 256, 257 tests and ver
- **cuda-nvfp4-wgmma-rega-grouped-verify** → neutral (within-ci), 1.3123x — no shape moved outside noise. Verify shapes only: num_tokens >= 128 (verify_16 bs=128, verify_32 bs=256, and the bs=128-skewed and bs=257 tests). num_tokens < 128 (incl. the bs=33 test) keep
- **cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc** → neutral (unproven-vs-best), 1.3303x — gained on verify_16 verify_16; verify_32 verify_32. RE-MEASURE (stage 1b): the previous run never executed the new kernel; load_inline failed because cpp_extension appended its own -gencode arch=compute_90,code=s
- **cuda-nvfp4-swapab-mma-persistent-small-m-inproc** → regression (regression), 0.9250x — gained on decode_1 decode_1; verify_16 verify_16; verify_32 verify_32; lost on decode_16 decode_16; decode_32 decode_32; verify_1 verify_1. FIX (stage 1b, make it run): the previous run built (51.2 s, 80-128 regs, 0 local) but its first CUDA launch in _cuda_self_check hit illegal instruction 715 (st
- **cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix** → neutral (unproven-vs-best), 1.3307x — no shape moved outside noise. Stage 2 of the wgmma register-A grouped NVFP4 rewrite (make it numerically correct). Stage 1b (job b56d17fda2d54be2) finally BUILT and RAN on the judge: TORCH_C
- **cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix** → failed (test:failed) — lost on verify_16 verify_16. Stage 3 of the wgmma register-A grouped NVFP4 rewrite, built on the stage-2 commit (cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix, claim bases a rewrite-on-rewrit
- **cuda-nvfp4-wgmma-rega-tiny-m-bucket16** → neutral (unproven-vs-best), 1.3453x — gained on decode_1 decode_1. Tiny-M branch of the wgmma register-A grouped rewrite: bs=1 (decode_1) and the bs=4 test, which master runs at 44.9 us (Marlin pair ~42 us) vs a floor of 11.4 u
- **cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate** → neutral (unproven-vs-best), 1.3744x — gained on verify_16 verify_16; verify_32 verify_32. The shippable twin of cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced: identical code (same s4 base rebased onto master 9080a71, same per-call cuTensorMapEnco
- **cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced** → neutral (unproven-vs-best), 1.3490x — gained on verify_32 verify_32; lost on verify_16 verify_16. Stage 5 of the wgmma register-A grouped NVFP4 rewrite, built on the s4 commit (cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe, job 4e7f8be895a145ba) rebased onto 
- **cuda-nvfp4-wgmma-rega-b128-n32-deep-ring** → neutral (unproven-vs-best), 1.3690x — gained on verify_16 verify_16. Built on cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced's commit (branch hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced, job d4ad4f31d2
- **cuda-fused-split-i-moe-midt-cluster-reduce** → neutral (within-ci), 1.3885x — lost on verify_32 verify_32. Stage 1 of a fused split-I MoE for mid T: 9 <= num_tokens <= 32 only (decode_16 bs=16, decode_32 bs=32). num_tokens <= 8 and >= 33 (incl. the bs=33 test) run ma
- **nvfp4-rowowner-bulk-stream-tiny-t-rewrite** → neutral (within-ci), 1.3887x — lost on verify_32 verify_32. Stage 1 of a reduce-free bulk-stream NVFP4 MoE for tiny T: 1 <= num_tokens <= 4 only (decode_1 bs=1, the bs=1 and bs=4 tests). Every other T runs master byte fo

### w2 marlin
- **w2-l2-prefetch-decode1** → regression (regression), 1.2615x — lost on decode_1 decode_1. At num_tokens == 1 only (decode_1; every other M runs master unchanged), warm L2 with the w2 weights of this token's local experts before the w13 Marlin call, s
- **w2-marlin-atomic-add-small-m** → neutral (unproven-vs-best), 1.3445x — gained on decode_1 decode_1. For num_tokens <= 32 only (bs=1, 8, 16, 32 and the bs=4 test), pass use_atomic_add=True and use_fp32_reduce=False (Marlin requires the pair) to the SECOND moe_w
- **w2-marlin-atomic-add-verify** → neutral (within-ci), 1.3407x — no shape moved outside noise. On master (9080a71), widen the landed w2 use_atomic_add path from num_tokens <= 32 to all T by giving the verify range its own bound constant W2_ATOMIC_VERIFY =

### wgmma NVFP4 grouped w13+silu/w2 at 128<=T<=512: per-state tensor maps, graph-replay safety
- **cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe** → neutral (within-ci), 1.3439x — gained on decode_1 decode_1; decode_16 decode_16; decode_32 decode_32; verify_1 verify_1; verify_16 verify_16; verify_32 verify_32. Fix the s3 judge failure (job 3bdec0e58d684c17: verify 16 bs=128 "recheck failed on repetition 2: relative L2 error 1.211e+00"). CAUSE (s3 branch submission.py)

### whole-layer fusion
- **cuda-fused-moe-bs1-persistent-w13-silu-w2** → neutral (within-ci), 1.3074x — no shape moved outside noise. bs=1 only (num_tokens == 1: decode_1 and the bs=1 test); every other M runs master unchanged. Replace the whole bs=1 MoE (_align_small, Marlin w13, silu_and_mul
- **cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc** → neutral (within-ci), 1.3257x — gained on verify_16 verify_16. Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.pa
- **cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc** → neutral (unproven-vs-best), 1.3295x — no shape moved outside noise. Stage 3 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.pa

## Open follow-ups

- cuda-fused-moe-bs1-persistent-w13-silu-w2, cuda-nvfp4-swapab-mma-persistent-small-m: rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at bs<=32`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The ne
- cuda-nvfp4-wgmma-rega-grouped-verify, cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc: rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- cuda-dispatch-combine-pdl-verify: rewrite stage 1 of `_align_parallel_kernel + fused_post_expert_norm_weighted_sum at num_tokens > 32`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which
- cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc: rewrite stage 2 of `_align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check wh
- cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc: rewrite stage 3 of `_align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check wh
- cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix: rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- cuda-nvfp4-wgmma-rega-tiny-m-bucket16: rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 1 <= num_tokens <= 7`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which
- cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe: rewrite stage 4 of `Marlin w13+silu+w2 at 128<=T<=512`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hyp
- cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate, cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced: rewrite stage 5 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=512`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mile
- cuda-nvfp4-wgmma-rega-b128-n32-deep-ring: rewrite stage 6 of `Marlin w13+silu+w2 (moe_wna16_marlin_gemm x2 + act_and_mul) at 128<=T<=255`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mile
- cuda-fused-split-i-moe-midt-cluster-reduce: rewrite stage 1 of `Marlin w13 + act_and_mul + Marlin w2 (moe_wna16_marlin_gemm x2 + silu_and_mul) and the post-norm tail at 9 <= num_tokens <= 32`: read the replaced kernels' own us per shape (profile top_kernels) again
- nvfp4-rowowner-bulk-stream-tiny-t-rewrite: rewrite stage 1 of `Marlin w13 + silu_and_mul + Marlin w2 (moe_wna16_marlin_gemm x2) at 1 <= num_tokens <= 4`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and ch
- triton-map-align-verify-chunked: none
- cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix: rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi

## Do not try (regressions, flat results, rejected)

Rewrite stages are exempt: a slow early stage is not a dead end.

- **w13-marlin-atomic-add-small-m** [w13 marlin] neutral, within-ci: For num_tokens <= 32, pass use_atomic_add=True (and use_fp32_reduce=False, which Marlin re
- **w13-marlin-bf16-reduce-verify** [w13 marlin] neutral, within-ci: For num_tokens >= 128 (verify_16 and verify_32), pass use_fp32_reduce=False to the w13 moe
- **post-norm-triton-all-m** [post-expert norm+combine] neutral, within-ci: On top of marlin-block-m-two-smaller-on-postnorm, remove the SMALL_M_POST_NORM = 32 split 
- **triton-map-align-verify-chunked-pow2-fix** [MoE align/dispatch] neutral, within-ci: Re-run triton-map-align-verify-chunked (failed at compile: tl.arange(0, BSM) with BSM=48 i
- **marlin-two-stream-expert-halves-small-m** [launch overhead] neutral, within-ci: At 8 <= num_tokens <= 32 only (verify_1 bs=8, decode_16 bs=16, decode_32 bs=32; bs=1, the 
- **post-norm-triton-hoisted-decode32** [post-expert norm+combine] neutral, within-ci: num_tokens == 32 only (decode_32); bs<32 keeps master's _post_norm_combine_kernel unchange
- **post-norm-pdl-launch-small-m** [launch overhead] neutral, within-ci: num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16, bs=4 test) only; master's _post_norm_combine_
- **stack-onewarp-hoisted-dispatchcombine** [MoE align (bs<=8 one warp; bs>=128 CUDA) + post-expert combine (bs<32 hoisted; bs>=128 CUDA)] neutral, within-ci: Stack the three moe results of 2026-10-24 that each matched the 1.3278 best within noise, 
- **stack-onewarp-hoisted-dispatchcombine-from17** [MoE align (bs<=8 one warp; 17..32 parallel; bs>=128 CUDA) + post-expert combine] neutral, within-ci: The variant of stack-onewarp-hoisted-dispatchcombine (same three done/ patches, unchanged)
- **w2-marlin-atomic-add-verify** [w2 marlin] neutral, within-ci: On master (9080a71), widen the landed w2 use_atomic_add path from num_tokens <= 32 to all 
- **restack-occ2-wg2-b256-two-stream-halves-on-master** [expert GEMM path selection (wgmma wg2 at B256 + Marlin two-stream halves at T 8-32) restack] neutral, within-ci: COMBINATION RESTACK of two measured per-cell wins onto current master (bdbbc97b, combo-top
- **triton-w13-grouped-gemv-small-m** [w13 marlin] regression, regression: For num_tokens <= 8 only (decode_1 and verify_1; all other shapes keep the parity path unt
- **marlin-block-size-m-next-larger** [MoE align/dispatch] regression, regression: Change only the block_size_m pick in kernel_fn. SGLang's loop breaks on the first candidat
- **w2-l2-prefetch-decode1** [w2 marlin] regression, regression: At num_tokens == 1 only (decode_1; every other M runs master unchanged), warm L2 with the 
- **marlin-block-m-8-verify16** [MoE align/dispatch + block_size_m pick] regression, regression: In _forward on master 9080a71, for 128 <= num_tokens <= 255 only (verify_16 bs=128 and the

## Queue and in flight

- queue: cuda-fused-moe-midm-single-launch-routed [whole-layer fusion] p2 hold
- queue: cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild [w13 marlin + silu-mul + w2 marlin] p1 hold
- queue: marlin-exec-config-autotune-mid-m [Marlin launch config (w13 + w2)] p3 hold
- queue: cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix [w13 marlin + silu-mul + w2 marlin] p3 hold
- queue: cuda-nvfp4-wgmma-rega-producer-scale-expand [w13 marlin + silu-mul + w2 marlin] p2 hold
- queue: cuda-fused-split-i-moe-tiny-t [w13 marlin + silu-mul + w2 marlin] p1 hold
- queue: wg-producer-direct-x-rows-no-gather [MoE dispatch (wgmma x gather)] p3 hold
- queue: wg2-pingpong-consumers-occ2-b256 [w13 marlin + silu-mul + w2 marlin] p2 hold
- queue: wg2-w13-occ2-nmax128-stg2-hot [w13 marlin + silu-mul] p3 hold
- queue: wg2-stream-k-tail-floor-g-over-r [w13 marlin + silu-mul + w2 marlin] p3 queued
- 21 stale holds expired to `done/expired/` (not run)
