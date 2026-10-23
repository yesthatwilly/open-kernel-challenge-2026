# muse_moe_w4a16 findings

Regenerated 2026-10-23T221828Z by `churn.py index`. Read this before writing a hypothesis for muse_moe_w4a16.

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

## Open follow-ups

- none

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- empty
