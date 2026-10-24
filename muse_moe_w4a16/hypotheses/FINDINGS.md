# muse_moe_w4a16 findings

Regenerated 2026-10-24T132543Z by `churn.py index`. Read this before writing a hypothesis for muse_moe_w4a16.

## Current best

1.2536x [1.2348,1.2724] from job `c00ef7295c154867` (check), master commit `cefc282ebd`, hypothesis marlin-block-m-two-smaller-on-postnorm

Where the time goes (per scored shape, top kernels):

- bs=1 load=decode_1 sharding=ep8: yours 55.5µs vs SGLang 77.8µs (SoL 11.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 45.3µs 78.5%, void moe_align_block_size_small_batch_expert_kernel<int, ... 4.4µs 7.7%, _post_norm_combine_kernel 3.5µs 6.0%
- bs=16 load=decode_16 sharding=ep8: yours 155.9µs vs SGLang 188.4µs (SoL 52.5µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 117.0µs 89.0%, void moe_align_block_size_small_batch_expert_kernel<int, ... 5.5µs 4.2%, _post_norm_combine_kernel 3.5µs 2.7%
- bs=32 load=decode_32 sharding=ep8: yours 209.7µs vs SGLang 207.5µs (SoL 70.8µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 179.7µs 91.2%, void moe_align_block_size_small_batch_expert_kernel<int, ... 6.6µs 3.4%, _post_expert_norm_weighted_sum_kernel 4.2µs 2.1%
- bs=8 load=verify_1 sharding=ep8: yours 120.4µs vs SGLang 150.9µs (SoL 37.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 65.6µs 83.4%, void moe_align_block_size_small_batch_expert_kernel<int, ... 4.9µs 6.3%, _post_norm_combine_kernel 3.3µs 4.2%
- bs=128 load=verify_16 sharding=ep8: yours 339.7µs vs SGLang 475.1µs (SoL 101.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 346.4µs 94.3%, _post_expert_norm_weighted_sum_kernel 6.4µs 1.7%, void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A... 3.9µs 1.1%
- bs=256 load=verify_32 sharding=ep8: yours 466.9µs vs SGLang 615.9µs (SoL 108.9µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 444.5µs 95.2%, void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A... 6.1µs 1.3%, _post_expert_norm_weighted_sum_kernel 5.5µs 1.2%

## By area

### MoE align/dispatch
- **sglang-marlin-path-parity** → gain (gain), 0.9888x — gained on decode_1 decode_1; decode_16 decode_16; decode_32 decode_32; verify_1 verify_1; verify_16 verify_16; verify_32 verify_32. Replace the starter's 32-expert dense loop (dequantize every local expert to BF16 on every call, x @ w13.T over all tokens, masked weighted sum) with SGLang's M
- **align-single-token-ep** → neutral (unproven-vs-best), 0.9951x — gained on decode_1 decode_1. At num_tokens == 1 only, call SGLang's moe_align_single_token(local_ids, block_size_m) (from sglang.kernels.ops.moe.moe_align_single_token) with the expert_map-
- **marlin-block-size-m-next-larger** → regression (regression), 0.9178x — lost on decode_32 decode_32; verify_16 verify_16. Change only the block_size_m pick in kernel_fn. SGLang's loop breaks on the first candidate in (8, 16, 32, 48, 64) where num_tokens*8/32/block_size_m < 0.9, whi
- **align-256-experts-below-128** → neutral (unproven-vs-best), 0.9998x — gained on decode_1 decode_1. In kernel_fn, for num_tokens < 128 only (bs=1, 8, 16, 32 and the bs=4/33 tests), call moe_align_block_size(local_ids, block_size_m, N_EXPERTS) exactly as baseli
- **triton-map-align-small-m-port** → neutral (beats-base-not-best), 1.1850x — gained on decode_1 decode_1; decode_16 decode_16; decode_32 decode_32; verify_1 verify_1. Port muse_decode_layer moe-align-single-kernel-small-t (patch: muse_decode_layer/hypotheses/done/neutral/moe-align-single-kernel-small-t.patch; its _align_small

### post-expert norm+combine
- **post-expert-norm-triton-small-m** → gain (gain), 1.1297x — gained on decode_1 decode_1; decode_16 decode_16; verify_1 verify_1. For num_tokens < 32 only (bs=1, 8, 16; the bs=4 test too), replace the call to SGLang's fused_post_expert_norm_weighted_sum at the end of kernel_fn with one lau
- **post-norm-triton-all-m** → neutral (within-ci), 1.2483x — no shape moved outside noise. On top of marlin-block-m-two-smaller-on-postnorm, remove the SMALL_M_POST_NORM = 32 split at the end of kernel_fn so every num_tokens (bs=32/33/128/256/257 too)

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

## Open follow-ups

- align-single-token-ep: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-single-token-ep`) that adds a further change in the same area is the only way to learn more (identical code is never re-run)
- marlin-workspace-blocks-per-sm-8: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: marlin-workspace-blocks-per-sm-8`) that adds a further change in the same area is the only way to learn more (identical code is ne
- w13-marlin-atomic-add-small-m: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: w13-marlin-atomic-add-small-m`) that adds a further change in the same area is the only way to learn more (identical code is never re-run
- w13-marlin-bf16-reduce-verify: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: w13-marlin-bf16-reduce-verify`) that adds a further change in the same area is the only way to learn more (identical code is never re-run
- align-256-experts-below-128: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-256-experts-below-128`) that adds a further change in the same area is the only way to learn more (identical code is never r
- marlin-block-m-cap-16-port: beats-base-not-best: the change is not proven either way; a stacked hypothesis (`depends_on: marlin-block-m-cap-16-port`) that adds a further change in the same area is the only way to learn more (identical code is never
- triton-map-align-small-m-port: beats-base-not-best: the change is not proven either way; a stacked hypothesis (`depends_on: triton-map-align-small-m-port`) that adds a further change in the same area is the only way to learn more (identical code is ne
- post-norm-triton-all-m: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: post-norm-triton-all-m`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: triton-w13-silu-epilogue [silu-mul] p3 queued
- queue: triton-w2-grouped-gemv-small-m [w2 marlin] p3 queued
- queue: triton-small-m-threshold-32 [MoE align/dispatch] p4 queued
- queue: align-remote-sentinel-ids [MoE align/dispatch] p3 hold
- running: triton-map-align-on-block-m-restack [MoE align/dispatch + block_size_m pick] p1 running
