# muse_moe_w4a16 findings

Regenerated 2026-10-24T060411Z by `churn.py index`. Read this before writing a hypothesis for muse_moe_w4a16.

## Current best

0.9888x [0.9739,1.0036] from job `fb2cc736613747df` (check), master commit `a646cb4151`, hypothesis sglang-marlin-path-parity

Where the time goes (per scored shape, top kernels):

- bs=1 load=decode_1 sharding=ep8: yours 80.2µs vs SGLang 77.0µs (SoL 11.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 41.1µs 52.6%, void at::native::reduce_kernel<512, 1, at::native::Reduce... 5.6µs 7.1%, void at::native::unrolled_elementwise_kernel<at::native::... 5.4µs 6.9%
- bs=16 load=decode_16 sharding=ep8: yours 191.4µs vs SGLang 189.8µs (SoL 52.5µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 150.9µs 75.3%, void at::native::reduce_kernel<512, 1, at::native::Reduce... 8.9µs 4.5%, void at::native::elementwise_kernel<128, 2, at::native::g... 8.6µs 4.3%
- bs=32 load=decode_32 sharding=ep8: yours 208.2µs vs SGLang 205.6µs (SoL 70.8µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 212.7µs 92.3%, void moe_align_block_size_small_batch_expert_kernel<int, ... 6.7µs 2.9%, _post_expert_norm_weighted_sum_kernel 4.1µs 1.8%
- bs=8 load=verify_1 sharding=ep8: yours 152.0µs vs SGLang 151.1µs (SoL 37.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 107.6µs 70.5%, void at::native::reduce_kernel<512, 1, at::native::Reduce... 8.9µs 5.8%, void at::native::elementwise_kernel<128, 2, at::native::g... 6.5µs 4.3%
- bs=128 load=verify_16 sharding=ep8: yours 462.5µs vs SGLang 462.4µs (SoL 101.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 495.4µs 95.8%, _post_expert_norm_weighted_sum_kernel 6.5µs 1.3%, void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A... 4.0µs 0.8%
- bs=256 load=verify_32 sharding=ep8: yours 621.3µs vs SGLang 621.4µs (SoL 108.9µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 611.8µs 96.4%, _post_expert_norm_weighted_sum_kernel 6.1µs 1.0%, void sglang::act_and_mul_kernel<__nv_bfloat16, (sglang::A... 5.5µs 0.9%

## By area

### MoE align/dispatch
- **sglang-marlin-path-parity** → gain (gain), 0.9888x — gained on decode_1 decode_1; decode_16 decode_16; decode_32 decode_32; verify_1 verify_1; verify_16 verify_16; verify_32 verify_32. Replace the starter's 32-expert dense loop (dequantize every local expert to BF16 on every call, x @ w13.T over all tokens, masked weighted sum) with SGLang's M
- **align-single-token-ep** → neutral (unproven-vs-best), 0.9951x — gained on decode_1 decode_1. At num_tokens == 1 only, call SGLang's moe_align_single_token(local_ids, block_size_m) (from sglang.kernels.ops.moe.moe_align_single_token) with the expert_map-
- **marlin-block-size-m-next-larger** → regression (regression), 0.9178x — lost on decode_32 decode_32; verify_16 verify_16. Change only the block_size_m pick in kernel_fn. SGLang's loop breaks on the first candidate in (8, 16, 32, 48, 64) where num_tokens*8/32/block_size_m < 0.9, whi

### w13 marlin
- **triton-w13-grouped-gemv-small-m** → regression (regression), 0.7506x — lost on decode_1 decode_1; verify_1 verify_1. For num_tokens <= 8 only (decode_1 and verify_1; all other shapes keep the parity path untouched), replace the first moe_wna16_marlin_gemm (w13, top_k=8) with t
- **marlin-workspace-blocks-per-sm-8** → neutral (unproven-vs-best), 0.9921x — no shape moved outside noise. In prepare(), build a second lock workspace marlin_make_workspace(device, max_blocks_per_sm=8) and pass it to the w13 moe_wna16_marlin_gemm call only; the w2 ca
- **w13-marlin-bf16-reduce-verify** → neutral (within-ci), 0.9840x — no shape moved outside noise. For num_tokens >= 128 (verify_16 and verify_32), pass use_fp32_reduce=False to the w13 moe_wna16_marlin_gemm call only (use_atomic_add stays False; w2 and all d

## Open follow-ups

- align-single-token-ep: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: align-single-token-ep`) that adds a further change in the same area is the only way to learn more (identical code is never re-run)
- marlin-workspace-blocks-per-sm-8: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: marlin-workspace-blocks-per-sm-8`) that adds a further change in the same area is the only way to learn more (identical code is ne
- w13-marlin-bf16-reduce-verify: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: w13-marlin-bf16-reduce-verify`) that adds a further change in the same area is the only way to learn more (identical code is never re-run

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: triton-w13-silu-epilogue [silu-mul] p3 queued
- queue: triton-w2-grouped-gemv-small-m [w2 marlin] p3 queued
- queue: triton-small-m-threshold-32 [MoE align/dispatch] p4 queued
- running: w13-marlin-atomic-add-small-m [w13 marlin] p4 running
