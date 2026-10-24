# muse_decode_layer findings

Regenerated 2026-10-24T070716Z by `churn.py index`. Read this before writing a hypothesis for muse_decode_layer.

## Current best

1.0200x [1.0047,1.0353] from job `5755a412dda244a1` (submit), master commit `164c1d2764`, hypothesis sglang-pipeline-parity-router-hilo-m8

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60: yours 168.7µs vs SGLang 171.2µs (SoL 40.0µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 38.7µs 22.5%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 32.5µs 18.9%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 10.0µs 5.8%
- batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60: yours 180.7µs vs SGLang 182.2µs (SoL 44.2µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 55.6µs 28.1%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 32.0µs 16.2%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 18.1µs 9.2%
- batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic: yours 269.7µs vs SGLang 270.2µs (SoL 76.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 127.0µs 42.5%, nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 27.1µs 9.1%, nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT 20.4µs 6.8%
- batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic: yours 440.6µs vs SGLang 440.5µs (SoL 137.5µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 189.4µs 41.9%, void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 107.8µs 23.8%, nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 27.1µs 6.0%
- batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic: yours 265.7µs vs SGLang 266.3µs (SoL 84.5µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 103.4µs 41.6%, nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 26.8µs 10.8%, nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT 20.6µs 8.3%
- batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic: yours 614.9µs vs SGLang 615.4µs (SoL 208.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 500.2µs 65.3%, void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 125.7µs 16.4%, nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 26.6µs 3.5%
- batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8: yours 234.0µs vs SGLang 263.2µs (SoL 62.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 106.5µs 41.1%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 33.5µs 12.9%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 9.9µs 3.8%
- batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8: yours 250.9µs vs SGLang 278.9µs (SoL 66.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 74.8µs 31.0%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 33.2µs 13.8%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 18.6µs 7.7%
- batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 521.2µs vs SGLang 520.2µs (SoL 126.6µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 388.0µs 70.9%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 28.0µs 5.1%, nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT 25.5µs 4.7%
- batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 714.3µs vs SGLang 714.3µs (SoL 188.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 369.5µs 52.4%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 191.6µs 27.2%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 28.5µs 4.0%
- batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 713.0µs vs SGLang 712.5µs (SoL 143.8µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 509.8µs 72.4%, nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 34.5µs 4.9%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 31.6µs 4.5%
- batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 1049.8µs vs SGLang 1050.5µs (SoL 267.2µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 512.9µs 46.7%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 374.0µs 34.0%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 39.1µs 3.6%

## By area

### MoE align/dispatch
- **expert-id-map-single-kernel** → neutral (unproven-vs-best), 1.0374x — no shape moved outside noise. Replace the three small kernels that turn the routing output into Marlin's local expert ids with one Triton kernel. Today kernel_fn computes local_ids = (topk_i

### cache store
- **cache-store-fused-fp8-cast** → neutral (unproven-vs-best), 1.0303x — no shape moved outside noise. In kernel_fn, replace the four separate launches between FusedQKNorm and flash_attn_with_kvcache, namely k.to(FP8), v.to(FP8), store_cache(...) and q.view(token

### pooler GEMM
- **pooler-gemm-splitk-autotune** → neutral (within-ci), 1.0189x — no shape moved outside noise. For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB weight read whose floor is 14 us), pick the GEMM form per scored T in pre

### router GEMM
- **sglang-pipeline-parity-router-hilo-m8** → gain (gain), 1.0216x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_

## Open follow-ups

- cache-store-fused-fp8-cast: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: cache-store-fused-fp8-cast`) that adds a further change in the same area is the only way to learn more (identical code is never re
- expert-id-map-single-kernel: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: expert-id-map-single-kernel`) that adds a further change in the same area is the only way to learn more (identical code is never r
- pooler-gemm-splitk-autotune: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: pooler-gemm-splitk-autotune`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: fa3-num-splits-full-no-metadata [FA3 call] p3 hold
- queue: moe-align-local-expert-count [MoE align/dispatch] p4 queued
- queue: trtllm-decode-full-attention [FA3 call] p5 queued
- queue: fa3-num-splits-sliding [FA3 call] p2 hold
- queue: moe-align-single-kernel-small-t [MoE align/dispatch] p3 queued
- running: marlin-block-m-cap-16 [w13 marlin] p4 running
