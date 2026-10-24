# muse_decode_layer findings

Regenerated 2026-10-24T021809Z by `churn.py index`. Read this before writing a hypothesis for muse_decode_layer.

## Current best

0.0045x [0.0044,0.0046] from job `3c3181b1fce24b9a` (check), master commit `ab21d70942`

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60: yours 39572.9µs vs SGLang 172.3µs (SoL 40.0µs) · void at::native::index_elementwise_kernel<128, 4, at::nat... 7473.0µs 18.6%, void at::native::elementwise_kernel<128, 2, at::native::g... 5549.2µs 13.8%, void at::native::vectorized_elementwise_kernel<4, at::nat... 4994.0µs 12.4%
- batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60: yours 39359.9µs vs SGLang 179.8µs (SoL 44.2µs) · void at::native::index_elementwise_kernel<128, 4, at::nat... 7478.7µs 18.6%, void at::native::elementwise_kernel<128, 2, at::native::g... 5546.8µs 13.8%, void at::native::vectorized_elementwise_kernel<4, at::nat... 4982.8µs 12.4%
- batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic: yours 86418.4µs vs SGLang 265.8µs (SoL 76.7µs) · void gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, c... 42618.6µs 49.4%, void at::native::index_elementwise_kernel<128, 4, at::nat... 7484.7µs 8.7%, void at::native::elementwise_kernel<128, 2, at::native::g... 6659.3µs 7.7%
- batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic: yours 84451.2µs vs SGLang 430.7µs (SoL 137.5µs) · void gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, c... 42803.2µs 49.7%, void at::native::index_elementwise_kernel<128, 4, at::nat... 7476.1µs 8.7%, void at::native::elementwise_kernel<128, 2, at::native::g... 6645.5µs 7.7%
- batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic: yours 133430.0µs vs SGLang 261.2µs (SoL 84.5µs) · void gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, c... 85519.8µs 63.3%, void at::native::elementwise_kernel<128, 2, at::native::g... 7749.5µs 5.7%, void at::native::index_elementwise_kernel<128, 4, at::nat... 7483.8µs 5.5%
- batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic: yours 133660.7µs vs SGLang 623.5µs (SoL 208.1µs) · void gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, c... 85512.4µs 63.6%, void at::native::elementwise_kernel<128, 2, at::native::g... 7728.4µs 5.7%, void at::native::index_elementwise_kernel<128, 4, at::nat... 7476.1µs 5.6%
- batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8: yours 40279.2µs vs SGLang 267.9µs (SoL 62.3µs) · void at::native::index_elementwise_kernel<128, 4, at::nat... 7477.9µs 18.2%, void at::native::elementwise_kernel<128, 2, at::native::g... 5634.1µs 13.7%, void at::native::vectorized_elementwise_kernel<4, at::nat... 4992.8µs 12.2%
- batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8: yours 40510.9µs vs SGLang 270.7µs (SoL 66.4µs) · void at::native::index_elementwise_kernel<128, 4, at::nat... 7479.3µs 18.3%, void at::native::elementwise_kernel<128, 2, at::native::g... 5615.4µs 13.7%, void at::native::vectorized_elementwise_kernel<4, at::nat... 4984.1µs 12.2%
- batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 93905.3µs vs SGLang 515.1µs (SoL 126.6µs) · sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_sta... 50832.8µs 52.3%, void at::native::index_elementwise_kernel<128, 4, at::nat... 7475.6µs 7.7%, void at::native::elementwise_kernel<128, 2, at::native::g... 6799.9µs 7.0%
- batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 95959.0µs vs SGLang 702.7µs (SoL 188.7µs) · sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_sta... 50807.4µs 52.4%, void at::native::index_elementwise_kernel<128, 4, at::nat... 7472.5µs 7.7%, void at::native::elementwise_kernel<128, 2, at::native::g... 6775.4µs 7.0%
- batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 153118.9µs vs SGLang 682.8µs (SoL 143.8µs) · sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_sta... 101615.1µs 64.8%, void at::native::elementwise_kernel<128, 2, at::native::g... 8017.2µs 5.1%, void at::native::index_elementwise_kernel<128, 4, at::nat... 7474.0µs 4.8%
- batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 156958.8µs vs SGLang 1051.9µs (SoL 267.2µs) · sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_sta... 101608.9µs 65.0%, void at::native::elementwise_kernel<128, 2, at::native::g... 8021.2µs 5.1%, void at::native::index_elementwise_kernel<128, 4, at::nat... 7489.0µs 4.8%

## By area

### router GEMM
- **sglang-pipeline-parity-router-hilo-m8** → gain (gain), 1.0216x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_

## Open follow-ups

- none

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: cache-store-fused-fp8-cast [cache store] p2 queued
- queue: expert-id-map-single-kernel [MoE align/dispatch] p2 queued
- queue: fa3-num-splits-full-no-metadata [FA3 call] p3 queued
- queue: marlin-block-m-cap-16 [w13 marlin] p4 queued
- queue: moe-align-local-expert-count [MoE align/dispatch] p4 queued
- queue: pooler-gemm-splitk-autotune [pooler GEMM] p3 queued
- queue: trtllm-decode-full-attention [FA3 call] p5 queued
- queue: fa3-num-splits-sliding [FA3 call] p2 queued
- queue: moe-align-single-kernel-small-t [MoE align/dispatch] p3 queued
