# muse_attention_decode findings

Regenerated 2026-10-23T224055Z by `churn.py index`. Read this before writing a hypothesis for muse_attention_decode.

## Current best

0.0023x [0.0023,0.0024] from job `a8d647772fa945e5` (check), master commit `ab21d70942`

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1: yours 2914.1µs vs SGLang 21.0µs (SoL 0.1µs) · void gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, c... 2488.0µs 85.3%, void at::native::vectorized_gather_kernel<16, long>(char*... 99.4µs 3.4%, void at::native::elementwise_kernel<128, 2, at::native::g... 77.0µs 2.6%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1: yours 2859.7µs vs SGLang 30.8µs (SoL 4.3µs) · void gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, c... 2487.1µs 87.0%, void at::native::vectorized_gather_kernel<16, long>(char*... 99.3µs 3.5%, void at::native::unrolled_elementwise_kernel<at::native::... 72.7µs 2.5%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic: yours 49166.8µs vs SGLang 27.2µs (SoL 1.6µs) · void gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, c... 43326.0µs 87.8%, void at::native::vectorized_gather_kernel<16, long>(char*... 1706.1µs 3.5%, void at::native::elementwise_kernel<128, 2, at::native::g... 1094.4µs 2.2%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic: yours 48570.8µs vs SGLang 201.4µs (SoL 63.2µs) · void gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, c... 43323.9µs 88.4%, void at::native::vectorized_gather_kernel<16, long>(char*... 1705.8µs 3.5%, void at::native::elementwise_kernel<128, 2, at::native::g... 1079.1µs 2.2%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic: yours 97451.8µs vs SGLang 31.5µs (SoL 3.3µs) · void gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, c... 86583.1µs 87.9%, void at::native::vectorized_gather_kernel<16, long>(char*... 3411.9µs 3.5%, void at::native::elementwise_kernel<128, 2, at::native::g... 2171.2µs 2.2%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic: yours 97067.3µs vs SGLang 387.8µs (SoL 126.4µs) · void gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, c... 86809.3µs 88.4%, void at::native::vectorized_gather_kernel<16, long>(char*... 3411.1µs 3.5%, void at::native::elementwise_kernel<128, 2, at::native::g... 2161.1µs 2.2%
- batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8: yours 3508.3µs vs SGLang 23.1µs (SoL 0.1µs) · sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_sta... 2922.2µs 83.7%, void at::native::vectorized_gather_kernel<16, long>(char*... 99.3µs 2.8%, void at::native::elementwise_kernel<128, 2, at::native::g... 79.1µs 2.3%
- batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8: yours 3459.1µs vs SGLang 35.6µs (SoL 4.3µs) · sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_sta... 2920.8µs 85.1%, void at::native::vectorized_gather_kernel<16, long>(char*... 99.4µs 2.9%, void at::native::elementwise_kernel<128, 2, at::native::g... 66.8µs 1.9%
- batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 58952.4µs vs SGLang 29.2µs (SoL 1.9µs) · sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_sta... 50888.4µs 86.0%, void at::native::vectorized_gather_kernel<16, long>(char*... 1706.5µs 2.9%, void at::native::elementwise_kernel<128, 2, at::native::g... 1149.3µs 1.9%
- batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 58718.9µs vs SGLang 205.5µs (SoL 63.4µs) · sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_sta... 50869.5µs 86.3%, void at::native::vectorized_gather_kernel<16, long>(char*... 1706.3µs 2.9%, void at::native::elementwise_kernel<128, 2, at::native::g... 1148.0µs 1.9%
- batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 117643.9µs vs SGLang 34.5µs (SoL 3.7µs) · sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_sta... 101594.0µs 86.0%, void at::native::vectorized_gather_kernel<16, long>(char*... 3411.5µs 2.9%, void at::native::elementwise_kernel<128, 2, at::native::g... 2296.1µs 1.9%
- batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 117612.9µs vs SGLang 390.2µs (SoL 126.8µs) · sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_sta... 101954.3µs 86.4%, void at::native::vectorized_gather_kernel<16, long>(char*... 3411.2µs 2.9%, void at::native::elementwise_kernel<128, 2, at::native::g... 2289.5µs 1.9%

## By area

## Open follow-ups

- none

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: sglang-path-parity [SGLang path parity] p1 queued
- queue: fa3-num-splits-full-decode [FA3 call] p4 queued
- queue: fa3-num-splits-sliding [FA3 call] p3 queued
- queue: fused-preattention-triton [pre-attention norm+rope] p2 queued
- queue: triton-window-attention-sliding [FA3 call] p5 queued
- queue: fa3-drop-cu-seqlens-decode [FA3 call] p5 queued
- queue: fa3-scheduler-metadata-full-verify [FA3 call] p4 queued
- queue: triton-flash-decode-full-both-heads [FA3 call] p5 queued
- queue: fused-sliding-preattn-into-window-kernel [attention boundary] p4 queued
- queue: fused-sliding-normgate-epilogue [output norm-gate] p5 queued
