# muse_decode_layer findings

Regenerated 2026-10-24T120954Z by `churn.py index`. Read this before writing a hypothesis for muse_decode_layer.

## Current best

1.0874x [1.0710,1.1037] from job `fb07a3bd3aa54baf` (check), master commit `09da1e0d71`, hypothesis fa3-hdim128-head-pairing-batched

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60: yours 172.8µs vs SGLang 173.1µs (SoL 40.0µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 41.1µs 23.4%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 31.7µs 18.1%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 10.0µs 5.7%
- batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60: yours 182.1µs vs SGLang 183.5µs (SoL 44.2µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 41.2µs 22.3%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 32.3µs 17.5%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 18.8µs 10.2%
- batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic: yours 259.6µs vs SGLang 259.7µs (SoL 76.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 99.0µs 36.5%, nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 27.7µs 10.2%, nvjet_sm90_tst_128x16_64x11_4x1_v_bz_splitK_TNT 20.4µs 7.5%
- batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic: yours 406.8µs vs SGLang 436.1µs (SoL 137.5µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 152.3µs 35.4%, void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 108.4µs 25.2%, nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 27.1µs 6.3%
- batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic: yours 263.8µs vs SGLang 264.3µs (SoL 84.5µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 105.9µs 42.1%, nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 27.4µs 10.9%, nvjet_sm90_tst_128x32_64x10_4x1_v_bz_splitK_TNT 21.4µs 8.5%
- batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic: yours 543.5µs vs SGLang 615.3µs (SoL 208.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 290.2µs 50.3%, void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 148.1µs 25.7%, nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 26.5µs 4.6%
- batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8: yours 238.2µs vs SGLang 268.4µs (SoL 62.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 87.6µs 36.2%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 33.1µs 13.7%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 10.0µs 4.2%
- batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8: yours 247.7µs vs SGLang 278.2µs (SoL 66.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 86.7µs 34.4%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 33.6µs 13.4%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 18.6µs 7.4%
- batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 453.2µs vs SGLang 529.9µs (SoL 126.6µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 273.7µs 62.9%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 28.5µs 6.6%, nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT 26.6µs 6.1%
- batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 587.4µs vs SGLang 696.5µs (SoL 188.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 284.7µs 47.9%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 155.0µs 26.1%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 28.3µs 4.8%
- batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 637.7µs vs SGLang 723.2µs (SoL 143.8µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 429.3µs 69.0%, nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 34.1µs 5.5%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 31.3µs 5.0%
- batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 906.7µs vs SGLang 1071.8µs (SoL 267.2µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 454.1µs 47.1%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 292.1µs 30.3%, nvjet_sm90_tss_64x32_64x16_1x2_h_bz_splitK_NNT 43.7µs 4.5%

## By area

### FA3 call
- **triton-flash-decode-full-paired-kv** → neutral (unproven-vs-best), 1.0702x — gained on sliding verify_16; full verify_16. On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1) replace flash_attn_with_kvcache with a two-launch Triton flash-decoding pair buil
- **fa3-hdim128-head-pairing-batched** → neutral (unproven-vs-best), 1.0874x — gained on full decode_16, decode_32, verify_16, verify_32. Port muse_attention_decode's landed fa3-hdim128-head-pairing (1.171x) to the flash_attn_with_kvcache call in muse_decode_layer kernel_fn, gated to full-attentio

### MoE align/dispatch
- **expert-id-map-single-kernel** → neutral (unproven-vs-best), 1.0374x — no shape moved outside noise. Replace the three small kernels that turn the routing output into Marlin's local expert ids with one Triton kernel. Today kernel_fn computes local_ids = (topk_i
- **moe-align-single-kernel-small-t** → neutral (within-ci), 1.0431x — no shape moved outside noise. At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call AND the expert-id map kernel from expert-id-map-single-kernel with one 

### cache store
- **cache-store-fused-fp8-cast** → neutral (unproven-vs-best), 1.0303x — no shape moved outside noise. In kernel_fn, replace the four separate launches between FusedQKNorm and flash_attn_with_kvcache, namely k.to(FP8), v.to(FP8), store_cache(...) and q.view(token

### pooler GEMM
- **pooler-gemm-splitk-autotune** → neutral (within-ci), 1.0189x — no shape moved outside noise. For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB weight read whose floor is 14 us), pick the GEMM form per scored T in pre

### router GEMM
- **sglang-pipeline-parity-router-hilo-m8** → gain (gain), 1.0216x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_

### w13 marlin
- **marlin-block-m-cap-16** → gain (gain), 1.0621x — gained on sliding verify_16, verify_32; full verify_16, verify_32. Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first siz

## Open follow-ups

- cache-store-fused-fp8-cast: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: cache-store-fused-fp8-cast`) that adds a further change in the same area is the only way to learn more (identical code is never re
- expert-id-map-single-kernel: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: expert-id-map-single-kernel`) that adds a further change in the same area is the only way to learn more (identical code is never r
- pooler-gemm-splitk-autotune: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: pooler-gemm-splitk-autotune`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- moe-align-single-kernel-small-t: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: moe-align-single-kernel-small-t`) that adds a further change in the same area is the only way to learn more (identical code is never re-r
- triton-flash-decode-full-paired-kv: rewrite stage 1 of `FA3 on full-attention decode (q_len=1): flash_fwd main + FlashAttnFwdCombine`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- fa3-hdim128-head-pairing-batched: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: fa3-hdim128-head-pairing-batched`) that adds a further change in the same area is the only way to learn more (identical code is ne

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: fa3-num-splits-full-no-metadata [FA3 call] p3 hold
- queue: moe-align-local-expert-count [MoE align/dispatch] p4 hold
- queue: trtllm-decode-full-attention [FA3 call] p5 queued
- queue: fa3-num-splits-sliding [FA3 call] p2 hold
- queue: moe-align-single-kernel-small-t-control-capacity-256 [MoE align/dispatch] p3 hold
- queue: fa3-full-single-split-probe [FA3 call] p2 hold
- queue: fused-preattention-triton [pre-attention norm+rope] p4 queued
- queue: router-triton-splitk-side-stream [router GEMM] p1 hold
- queue: triton-flash-decode-full-verify-paired-kv [FA3 call] p5 hold
- running: triton-flash-decode-full-oversub-kv-prefetch [FA3 call] p1 running
