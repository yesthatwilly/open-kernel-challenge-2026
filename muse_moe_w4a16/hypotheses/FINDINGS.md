# muse_moe_w4a16 findings

Regenerated 2026-10-25T013623Z by `churn.py index`. Read this before writing a hypothesis for muse_moe_w4a16.

## Current best

1.3278x [1.3079,1.3477] from job `1947e3ec2abb4bdf` (submit), master commit `1a5da3d22c`, hypothesis align-parallel-per-expert-verify

Where the time goes (per scored shape, top kernels):

- bs=1 load=decode_1 sharding=ep8: yours 48.8µs vs SGLang 77.1µs (SoL 11.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 44.7µs 87.9%, _post_norm_combine_kernel 3.1µs 6.2%, void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A... 1.6µs 3.1%
- bs=16 load=decode_16 sharding=ep8: yours 148.4µs vs SGLang 189.0µs (SoL 52.5µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 176.1µs 95.0%, _post_norm_combine_kernel 4.3µs 2.3%, _align_small_kernel 2.8µs 1.5%
- bs=32 load=decode_32 sharding=ep8: yours 199.9µs vs SGLang 206.4µs (SoL 70.8µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 207.5µs 94.9%, _post_expert_norm_weighted_sum_kernel 4.5µs 2.1%, _align_small_kernel 4.2µs 1.9%
- bs=8 load=verify_1 sharding=ep8: yours 107.7µs vs SGLang 144.4µs (SoL 37.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 114.0µs 93.0%, _post_norm_combine_kernel 3.8µs 3.1%, _align_small_kernel 2.7µs 2.2%
- bs=128 load=verify_16 sharding=ep8: yours 318.3µs vs SGLang 456.2µs (SoL 101.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 292.5µs 95.8%, _post_expert_norm_weighted_sum_kernel 4.9µs 1.6%, _align_parallel_kernel 4.0µs 1.3%
- bs=256 load=verify_32 sharding=ep8: yours 454.3µs vs SGLang 616.0µs (SoL 108.9µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 391.4µs 95.5%, _align_parallel_kernel 6.9µs 1.7%, _post_expert_norm_weighted_sum_kernel 6.0µs 1.5%

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

### MoE align/dispatch + block_size_m pick
- **triton-map-align-on-block-m-restack** → gain (gain), 1.3117x — gained on decode_1 decode_1; decode_16 decode_16; decode_32 decode_32; verify_1 verify_1. Stack the two measured, shape-disjoint moe wins: start from master (marlin-block-m-two-smaller-on-postnorm, 1.254x: block_size_m two candidates down at num_toke

### MoE align/dispatch + post-expert norm+combine
- **cuda-dispatch-combine-pdl-verify** → neutral (unproven-vs-best), 1.3283x — no shape moved outside noise. Dispatch + combine only, at num_tokens > 32 (verify_16 bs=128, verify_32 bs=256, and the bs=33, bs=257 and skewed bs=128 tests). num_tokens <= 32 runs master by

### launch overhead
- **marlin-two-stream-expert-halves-small-m** → neutral (within-ci), 1.3024x — no shape moved outside noise. At 8 <= num_tokens <= 32 only (verify_1 bs=8, decode_16 bs=16, decode_32 bs=32; bs=1, the bs=4 test and num_tokens > 32 run master unchanged), split the expert 
- **post-norm-pdl-launch-small-m** → neutral (within-ci), 1.3222x — no shape moved outside noise. num_tokens < SMALL_M_POST_NORM (bs=1, 8, 16, bs=4 test) only; master's _post_norm_combine_kernel body is otherwise unchanged and every other launch is master's.

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

### w13 marlin + silu-mul + w2 marlin
- **cuda-nvfp4-swapab-mma-persistent-small-m** → failed (test:failed). Small-M only: num_tokens <= 32 (decode_1 bs=1, verify_1 bs=8, decode_16, decode_32 and the bs=1/bs=4 tests); num_tokens > 32 (bs=33, 128, 256, 257 tests and ver
- **cuda-nvfp4-wgmma-rega-grouped-verify** → neutral (within-ci), 1.3123x — no shape moved outside noise. Verify shapes only: num_tokens >= 128 (verify_16 bs=128, verify_32 bs=256, and the bs=128-skewed and bs=257 tests). num_tokens < 128 (incl. the bs=33 test) keep
- **cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc** → neutral (unproven-vs-best), 1.3303x — gained on verify_16 verify_16; verify_32 verify_32. RE-MEASURE (stage 1b): the previous run never executed the new kernel; load_inline failed because cpp_extension appended its own -gencode arch=compute_90,code=s
- **cuda-nvfp4-swapab-mma-persistent-small-m-inproc** → regression (regression), 0.9250x — gained on decode_1 decode_1; verify_16 verify_16; verify_32 verify_32; lost on decode_16 decode_16; decode_32 decode_32; verify_1 verify_1. FIX (stage 1b, make it run): the previous run built (51.2 s, 80-128 regs, 0 local) but its first CUDA launch in _cuda_self_check hit illegal instruction 715 (st

### w2 marlin
- **w2-l2-prefetch-decode1** → regression (regression), 1.2615x — lost on decode_1 decode_1. At num_tokens == 1 only (decode_1; every other M runs master unchanged), warm L2 with the w2 weights of this token's local experts before the w13 Marlin call, s

### whole-layer fusion
- **cuda-fused-moe-bs1-persistent-w13-silu-w2** → neutral (within-ci), 1.3074x — no shape moved outside noise. bs=1 only (num_tokens == 1: decode_1 and the bs=1 test); every other M runs master unchanged. Replace the whole bs=1 MoE (_align_small, Marlin w13, silu_and_mul
- **cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc** → neutral (within-ci), 1.3257x — gained on verify_16 verify_16. Stage 2 of the bs=1 persistent fused MoE kernel (moe_bs1_kernel; archived at muse_moe_w4a16/hypotheses/done/neutral/cuda-fused-moe-bs1-persistent-w13-silu-w2.pa

## Open follow-ups

- align-single-token-ep: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-single-token-ep`) that adds a further change in the same area is the only way to learn more (identical code is never re-run)
- marlin-workspace-blocks-per-sm-8: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: marlin-workspace-blocks-per-sm-8`) that adds a further change in the same area is the only way to learn more (identical code is ne
- w13-marlin-atomic-add-small-m: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: w13-marlin-atomic-add-small-m`) that adds a further change in the same area is the only way to learn more (identical code is never re-run
- w13-marlin-bf16-reduce-verify: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: w13-marlin-bf16-reduce-verify`) that adds a further change in the same area is the only way to learn more (identical code is never re-run
- align-256-experts-below-128: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-256-experts-below-128`) that adds a further change in the same area is the only way to learn more (identical code is never r
- marlin-block-m-cap-16-port: beats-base-not-best: the change is not proven either way; a stacked hypothesis (`depends_on: marlin-block-m-cap-16-port`) that adds a further change in the same area is the only way to learn more (identical code is never
- triton-map-align-small-m-port: beats-base-not-best: the change is not proven either way; a stacked hypothesis (`depends_on: triton-map-align-small-m-port`) that adds a further change in the same area is the only way to learn more (identical code is ne
- post-norm-triton-all-m: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: post-norm-triton-all-m`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- triton-map-align-verify-chunked-pow2-fix: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: triton-map-align-verify-chunked-pow2-fix`) that adds a further change in the same area is the only way to learn more (identical code is n
- align-parallel-per-expert-verify: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-parallel-per-expert-verify`) that adds a further change in the same area is the only way to learn more (identical code is ne
- marlin-two-stream-expert-halves-small-m: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: marlin-two-stream-expert-halves-small-m`) that adds a further change in the same area is the only way to learn more (identical code is ne
- cuda-fused-moe-bs1-persistent-w13-silu-w2: rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at bs<=32`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The ne
- cuda-nvfp4-wgmma-rega-grouped-verify: rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- cuda-dispatch-combine-pdl-verify: rewrite stage 1 of `_align_parallel_kernel + fused_post_expert_norm_weighted_sum at num_tokens > 32`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which
- post-norm-combine-hoisted-slot-loads: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: post-norm-combine-hoisted-slot-loads`) that adds a further change in the same area is the only way to learn more (identical code i
- post-norm-triton-hoisted-decode32: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: post-norm-triton-hoisted-decode32`) that adds a further change in the same area is the only way to learn more (identical code is never re
- post-norm-pdl-launch-small-m: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: post-norm-pdl-launch-small-m`) that adds a further change in the same area is the only way to learn more (identical code is never re-run)
- align-parallel-from-17-tokens: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-parallel-from-17-tokens`) that adds a further change in the same area is the only way to learn more (identical code is never
- align-small-one-warp-tiny-m: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-small-one-warp-tiny-m`) that adds a further change in the same area is the only way to learn more (identical code is never r
- stack-onewarp-hoisted-dispatchcombine: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: stack-onewarp-hoisted-dispatchcombine`) that adds a further change in the same area is the only way to learn more (identical code is neve
- stack-onewarp-hoisted-dispatchcombine-from17: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: stack-onewarp-hoisted-dispatchcombine-from17`) that adds a further change in the same area is the only way to learn more (identical code 
- cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc: rewrite stage 2 of `_align_small + Marlin w13+w2 + act_and_mul + _post_norm_combine at num_tokens == 1`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check wh
- cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc: rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- triton-map-align-verify-chunked: none
- cuda-nvfp4-swapab-mma-persistent-small-m: rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at bs<=32`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The ne

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: triton-w13-silu-epilogue [silu-mul] p3 hold
- queue: triton-w2-grouped-gemv-small-m [w2 marlin] p3 hold
- queue: triton-small-m-threshold-32 [MoE align/dispatch] p4 hold
- queue: align-remote-sentinel-ids [MoE align/dispatch] p3 hold
- queue: marlin-two-class-block-m-decode32 [MoE align/dispatch + block_size_m pick] p2 hold
- queue: swapab-nvfp4-gemm-verify-rewrite [w13 marlin + w2 marlin] p1 hold
- queue: swapab-w13-silu-epilogue-verify [silu-mul] p2 queued
- queue: cuda-fused-moe-midm-single-launch-routed [whole-layer fusion] p2 hold
- queue: cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild [w13 marlin + silu-mul + w2 marlin] p1 hold
- queue: cuda-nvfp4-swapab-mma-persistent-small-m-childcheck [w13 marlin + silu-mul + w2 marlin] p2 hold
- queue: cuda-fused-moe-bs1-persistent-nospin-bandwidth [whole-layer fusion] p3 hold
- queue: cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc [whole-layer fusion] p3 queued
- queue: cuda-nvfp4-wgmma-rega-s2-fix-gate-from-8 [w13 marlin + silu-mul + w2 marlin] p2 hold
- queue: cuda-nvfp4-wgmma-rega-s2-per-kernel-enable [w13 marlin + silu-mul + w2 marlin] p1 hold
- queue: cuda-align-kernel-a-from17 [MoE align/dispatch] p2 hold
- queue: cuda-combine-kernel-b-bs128-only [post-expert norm+combine] p3 hold
- running: cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix [w13 marlin + silu-mul + w2 marlin] p1 running
- running: w2-marlin-atomic-add-small-m [w2 marlin] p2 running
