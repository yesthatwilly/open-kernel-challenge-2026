# muse_moe_w4a16 findings

Regenerated 2026-10-24T021809Z by `churn.py index`. Read this before writing a hypothesis for muse_moe_w4a16.

## Current best

0.0061x [0.0060,0.0062] from job `5e7d55323d75425e` (check), master commit `ab21d70942`

Where the time goes (per scored shape, top kernels):

- bs=1 load=decode_1 sharding=ep8: yours 35747.5µs vs SGLang 77.1µs (SoL 11.4µs) · void at::native::index_elementwise_kernel<128, 4, at::nat... 7486.2µs 20.2%, void at::native::elementwise_kernel<128, 2, at::native::g... 5473.8µs 14.7%, void at::native::vectorized_elementwise_kernel<4, at::nat... 4989.5µs 13.4%
- bs=16 load=decode_16 sharding=ep8: yours 36335.9µs vs SGLang 189.0µs (SoL 52.5µs) · void at::native::index_elementwise_kernel<128, 4, at::nat... 7488.6µs 20.0%, void at::native::elementwise_kernel<128, 2, at::native::g... 5537.3µs 14.8%, void at::native::vectorized_elementwise_kernel<4, at::nat... 4994.5µs 13.3%
- bs=32 load=decode_32 sharding=ep8: yours 37342.4µs vs SGLang 206.8µs (SoL 70.8µs) · void at::native::index_elementwise_kernel<128, 4, at::nat... 7485.9µs 19.9%, void at::native::elementwise_kernel<128, 2, at::native::g... 5551.2µs 14.8%, void at::native::vectorized_elementwise_kernel<4, at::nat... 4990.5µs 13.3%
- bs=8 load=verify_1 sharding=ep8: yours 37560.3µs vs SGLang 150.7µs (SoL 37.3µs) · void at::native::index_elementwise_kernel<128, 4, at::nat... 7487.3µs 20.0%, void at::native::elementwise_kernel<128, 2, at::native::g... 5537.9µs 14.8%, void at::native::vectorized_elementwise_kernel<4, at::nat... 4989.8µs 13.4%
- bs=128 load=verify_16 sharding=ep8: yours 37162.5µs vs SGLang 466.7µs (SoL 101.3µs) · void at::native::index_elementwise_kernel<128, 4, at::nat... 7485.0µs 19.7%, void at::native::elementwise_kernel<128, 2, at::native::g... 5654.6µs 14.9%, void at::native::vectorized_elementwise_kernel<4, at::nat... 4990.4µs 13.2%
- bs=256 load=verify_32 sharding=ep8: yours 37495.4µs vs SGLang 620.3µs (SoL 108.9µs) · void at::native::index_elementwise_kernel<128, 4, at::nat... 7493.9µs 19.5%, void at::native::elementwise_kernel<128, 2, at::native::g... 5782.1µs 15.0%, void at::native::vectorized_elementwise_kernel<4, at::nat... 4986.6µs 13.0%

## By area

### MoE align/dispatch
- **sglang-marlin-path-parity** → gain (gain), 0.9888x — gained on decode_1 decode_1; decode_16 decode_16; decode_32 decode_32; verify_1 verify_1; verify_16 verify_16; verify_32 verify_32. Replace the starter's 32-expert dense loop (dequantize every local expert to BF16 on every call, x @ w13.T over all tokens, masked weighted sum) with SGLang's M

## Open follow-ups

- none

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: triton-w13-grouped-gemv-small-m [w13 marlin] p2 queued
- queue: triton-w13-silu-epilogue [silu-mul] p3 queued
- queue: triton-w2-grouped-gemv-small-m [w2 marlin] p3 queued
- queue: align-single-token-ep [MoE align/dispatch] p4 queued
- queue: marlin-block-size-m-next-larger [MoE align/dispatch] p3 queued
- queue: marlin-workspace-blocks-per-sm-8 [w13 marlin] p4 queued
- queue: triton-small-m-threshold-32 [MoE align/dispatch] p4 queued
- queue: w13-marlin-atomic-add-small-m [w13 marlin] p4 queued
- queue: w13-marlin-bf16-reduce-verify [w13 marlin] p5 queued
