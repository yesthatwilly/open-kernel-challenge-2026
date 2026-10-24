# muse_decode_layer findings

Regenerated 2026-10-24T184810Z by `churn.py index`. Read this before writing a hypothesis for muse_decode_layer.

## Current best

1.2283x [1.2099,1.2468] from job `81ee6714b7614a99` (submit), master commit `f1f2f35cf7`, hypothesis small-t-launch-minimal-dense-chain-pdl

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60: yours 132.7µs vs SGLang 172.4µs (SoL 40.0µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 40.2µs 29.9%, void (anonymous namespace)::dl_pooler_router_kernel<1>(__... 21.9µs 16.2%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 13.1µs 9.7%
- batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60: yours 142.6µs vs SGLang 181.6µs (SoL 44.2µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 40.2µs 28.5%, void (anonymous namespace)::dl_pooler_router_kernel<1>(__... 21.2µs 15.1%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 18.5µs 13.1%
- batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic: yours 217.1µs vs SGLang 269.8µs (SoL 76.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 126.1µs 52.2%, nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 26.5µs 11.0%, void (anonymous namespace)::dl_pooler_router_kernel<2>(__... 23.9µs 9.9%
- batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic: yours 347.5µs vs SGLang 431.0µs (SoL 137.5µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 145.7µs 39.1%, void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 117.3µs 31.5%, nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 26.0µs 7.0%
- batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic: yours 247.0µs vs SGLang 265.6µs (SoL 84.5µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 146.1µs 53.5%, nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 28.6µs 10.5%, void (anonymous namespace)::dl_pooler_router_kernel<4>(__... 28.3µs 10.4%
- batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic: yours 535.0µs vs SGLang 624.9µs (SoL 208.1µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 291.7µs 54.0%, void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 133.4µs 24.7%, void (anonymous namespace)::dl_pooler_router_kernel<4>(__... 27.6µs 5.1%
- batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8: yours 186.5µs vs SGLang 263.3µs (SoL 62.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 96.0µs 49.2%, void (anonymous namespace)::dl_pooler_router_kernel<1>(__... 22.5µs 11.5%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 12.8µs 6.5%
- batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8: yours 195.9µs vs SGLang 272.7µs (SoL 66.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 106.7µs 48.9%, void (anonymous namespace)::dl_pooler_router_kernel<1>(__... 22.9µs 10.5%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 19.1µs 8.8%
- batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 447.6µs vs SGLang 521.4µs (SoL 126.6µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 271.6µs 63.2%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 28.2µs 6.6%, nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT 24.9µs 5.8%
- batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 599.4µs vs SGLang 704.8µs (SoL 188.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 326.1µs 51.4%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 154.0µs 24.3%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 28.4µs 4.5%
- batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 626.8µs vs SGLang 721.6µs (SoL 143.8µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 443.2µs 69.6%, nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 34.4µs 5.4%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 30.8µs 4.8%
- batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 910.9µs vs SGLang 1093.8µs (SoL 267.2µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 449.7µs 47.2%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 287.0µs 30.2%, nvjet_sm90_tss_64x32_64x16_1x2_h_bz_splitK_NNT 39.8µs 4.2%

## By area

### FA3 call
- **triton-flash-decode-full-paired-kv** → neutral (unproven-vs-best), 1.0702x — gained on sliding verify_16; full verify_16. On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1) replace flash_attn_with_kvcache with a two-launch Triton flash-decoding pair buil
- **triton-flash-decode-full-oversub-kv-prefetch** → regression (regression), 1.0413x — lost on full decode_16, decode_32; sliding verify_16. Build on stage 1 (branch hyp/muse_decode_layer/triton-flash-decode-full-paired-kv, muse_decode_layer/submission.py) and change only _full_decode_splits and the 
- **fa3-hdim128-head-pairing-batched** → neutral (unproven-vs-best), 1.0874x — gained on full decode_16, decode_32, verify_16, verify_32. Port muse_attention_decode's landed fa3-hdim128-head-pairing (1.171x) to the flash_attn_with_kvcache call in muse_decode_layer kernel_fn, gated to full-attentio
- **sliding-fused-triton-decode1-port** → neutral (unproven-vs-best), 1.1690x — gained on sliding decode_1. Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, sliding decode_1 16.56 -> 13.12 us) to muse_decode_layer kernel_fn. Copy ve
- **full-decode-cuda-splitkv-port** → neutral (unproven-vs-best), 1.1862x — gained on full decode_16, decode_32. Stage 1 of a rewrite of decode_layer full-attention decode at batch > 1, ported from muse_attention_decode master (full-decode-cuda-gather-splitkv, job 82085a4e

### MoE align/dispatch
- **expert-id-map-single-kernel** → neutral (unproven-vs-best), 1.0374x — no shape moved outside noise. Replace the three small kernels that turn the routing output into Marlin's local expert ids with one Triton kernel. Today kernel_fn computes local_ids = (topk_i
- **moe-align-single-kernel-small-t** → neutral (within-ci), 1.0431x — no shape moved outside noise. At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call AND the expert-id map kernel from expert-id-map-single-kernel with one 

### cache store
- **cache-store-fused-fp8-cast** → neutral (unproven-vs-best), 1.0303x — no shape moved outside noise. In kernel_fn, replace the four separate launches between FusedQKNorm and flash_attn_with_kvcache, namely k.to(FP8), v.to(FP8), store_cache(...) and q.view(token

### output norm-gate
- **paired-heads-pad-unpad-triton** → neutral (within-ci), 1.1654x — gained on full verify_16. Port the launch cleanup of muse_attention_decode's paired-heads-launch-cleanup (full layers there: decode_16 +6.4%, decode_32 +3.8%, verify_16 +5.9%) to master'

### pooler GEMM
- **pooler-gemm-splitk-autotune** → neutral (within-ci), 1.0189x — no shape moved outside noise. For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB weight read whose floor is 14 us), pick the GEMM form per scored T in pre
- **bf16-shape-gated-triton-gemms-small-t** → neutral (unproven-vs-best), 1.1688x — gained on sliding decode_1; full decode_1, decode_32, verify_16; lost on sliding decode_16, verify_1; full verify_1. Port muse_bf16_proj's landed bf16-shape-gated-triton-winners-combo (1.176x vs 1.151x there; code is muse_bf16_proj/submission.py on master: _bf16_splitk_kernel,

### post-expert norm+combine
- **post-expert-norm-triton-small-t** → gain (gain), 1.1635x — gained on sliding decode_1, decode_16, verify_1; full decode_1, decode_16, verify_1. Port muse_moe_w4a16's landed post-expert-norm-triton-small-m (1.130x there, job bacd1f502f4e46fa; patch muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-tri

### pre-attention norm+rope
- **fused-preattention-triton** → neutral (unproven-vs-best), 1.0970x — gained on sliding decode_32. Port muse_attention_decode's fused pre-attention Triton kernel (main tree muse_attention_decode/submission.py, _pre_attention_kernel: passed 16/16 at 1.127x the

### restack (sliding decode_1 port + paired pad/unpad + T=1 Triton GEMMs)
- **decode-restack-sliding-port-paired-pad-gemms-t1** → neutral (beats-base-not-best), 1.1937x — gained on sliding decode_1, verify_32; full decode_1, verify_16. Restack three harvested per-shape winners onto current master (post-expert-norm-triton-small-t) by a 3-way merge of their branches, resolving conflicts by hand 

### router GEMM
- **sglang-pipeline-parity-router-hilo-m8** → gain (gain), 1.0216x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_

### w13 marlin
- **marlin-block-m-cap-16** → gain (gain), 1.0621x — gained on sliding verify_16, verify_32; full verify_16, verify_32. Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first siz

### w13 marlin + silu-mul + w2 marlin
- **moe-chain-cuda-swapab-expert-once-large-t** → failed (test:failed). WHY: at T=128/256 the two Marlin MoE GEMMs are the single largest block of the whole problem: 254/271 us (sliding/full verify_16) and 443/427 us (verify_32) in 

### whole-layer fusion
- **small-t-launch-minimal-dense-chain-pdl** → gain (gain), 1.2283x — gained on sliding decode_1, decode_16, decode_32, verify_1; full decode_1, decode_16, verify_1. WHY: at T<=32 (decode_1/16/32 and verify_1: 4 of the 6 loads) master runs ~27-29 graph nodes per layer. At decode_1 sliding (145.6 us, floor 40.0) Marlin is 40.

## Open follow-ups

- cache-store-fused-fp8-cast: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: cache-store-fused-fp8-cast`) that adds a further change in the same area is the only way to learn more (identical code is never re
- expert-id-map-single-kernel: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: expert-id-map-single-kernel`) that adds a further change in the same area is the only way to learn more (identical code is never r
- pooler-gemm-splitk-autotune: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: pooler-gemm-splitk-autotune`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- moe-align-single-kernel-small-t: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: moe-align-single-kernel-small-t`) that adds a further change in the same area is the only way to learn more (identical code is never re-r
- fused-preattention-triton: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: fused-preattention-triton`) that adds a further change in the same area is the only way to learn more (identical code is never re-
- triton-flash-decode-full-paired-kv: rewrite stage 1 of `FA3 on full-attention decode (q_len=1): flash_fwd main + FlashAttnFwdCombine`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- fa3-hdim128-head-pairing-batched: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: fa3-hdim128-head-pairing-batched`) that adds a further change in the same area is the only way to learn more (identical code is ne
- bf16-shape-gated-triton-gemms-small-t: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-shape-gated-triton-gemms-small-t`) that adds a further change in the same area is the only way to learn more (identical code 
- sliding-fused-triton-decode1-port: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-fused-triton-decode1-port`) that adds a further change in the same area is the only way to learn more (identical code is n
- paired-heads-pad-unpad-triton: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: paired-heads-pad-unpad-triton`) that adds a further change in the same area is the only way to learn more (identical code is never re-run
- full-decode-cuda-splitkv-port: rewrite stage 1 of `paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer decode, batch 16/32`: read the replaced kernels' own us per shape (profile top_kernels) against the 
- decode-restack-sliding-port-paired-pad-gemms-t1: beats-base-not-best: the change is not proven either way; a stacked hypothesis (`depends_on: decode-restack-sliding-port-paired-pad-gemms-t1`) that adds a further change in the same area is the only way to learn more (id
- moe-chain-cuda-swapab-expert-once-large-t: rewrite stage 1 of `moe_align_block_size + Marlin w13 + silu_and_mul (act_and_mul_kernel) + Marlin w2 at T > 32 (verify_16 T=128, verify_32 T=256)`: read the replaced kernels' own us per shape (profile top_kernels) again

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: fa3-num-splits-full-no-metadata [FA3 call] p3 hold
- queue: moe-align-local-expert-count [MoE align/dispatch] p4 hold
- queue: trtllm-decode-full-attention [FA3 call] p5 hold
- queue: fa3-num-splits-sliding [FA3 call] p2 hold
- queue: moe-align-single-kernel-small-t-control-capacity-256 [MoE align/dispatch] p3 hold
- queue: fa3-full-single-split-probe [FA3 call] p2 hold
- queue: router-triton-splitk-side-stream [router GEMM] p1 hold
- queue: triton-flash-decode-full-verify-paired-kv [FA3 call] p5 hold
- queue: full-verify-cuda-splitkv-stage2 [FA3 call] p2 queued
- queue: bf16-m32-swapab-wgmma-port [pooler GEMM] p3 hold
- queue: moe-align-small-t-restack-on-master [MoE align/dispatch] p4 hold
- queue: fused-preattention-paired-pad-restack [pre-attention norm+rope] p5 queued
- queue: sliding-fused-batched-decode-port [FA3 call (sliding decode, batch > 1)] p1 queued
- queue: moe-dispatch-prep-on-side-stream [MoE align/dispatch (stream placement)] p2 queued
- queue: pre-expert-rmsnorm-triton [pre-expert norm] p3 queued
- queue: moe-small-t-cuda-swapab-port [w13 marlin + silu-mul + w2 marlin (T<=32)] p4 queued
- queue: moe-bs1-fused-persistent-port [w13 marlin + silu-mul + w2 marlin + post-expert combine (T=1)] p5 queued
- queue: l2-prefetch-o-pooler-during-sliding-attention [o GEMM + pooler GEMM (L2 warm-up overlapped with attention)] p4 queued
- queue: prefork-qkv-o-swapab-t16-t32 [qkv GEMM + o GEMM (before the router fork)] p4 queued
- queue: moe-align-parallel-large-t-port [MoE align/dispatch (T>32)] p5 queued
- queue: marlin-two-stream-halves-port [w13 marlin + silu-mul + w2 marlin (8<=T<=32, stream overlap)] p5 queued
