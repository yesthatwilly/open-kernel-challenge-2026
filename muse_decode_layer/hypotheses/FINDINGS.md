# muse_decode_layer findings

Regenerated 2026-10-25T202706Z by `churn.py index`. Read this before writing a hypothesis for muse_decode_layer.

## Current best

1.3142x [1.2945,1.3339] from job `a903b9d004d2452d` (submit), master commit `ec89382beb`, hypothesis combo-measured-rows-slide-batched-preattn-t8-tail

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=decode_1 page=1 repeats=60: yours 118.5µs vs SGLang 171.0µs (SoL 40.0µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 42.5µs 34.4%, void (anonymous namespace)::dl_pooler_router_kernel<1>(__... 21.5µs 17.4%, _sliding_fused_kernel 13.5µs 10.9%
- batch=1 ctx=80000 dist=soft layer=full load=decode_1 page=1 repeats=60: yours 124.3µs vs SGLang 182.1µs (SoL 44.2µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 42.6µs 32.7%, void (anonymous namespace)::dl_pooler_router_kernel<1>(__... 21.6µs 16.6%, void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 20.5µs 15.7%
- batch=16 ctx=80000 dist=soft layer=sliding load=decode_16 page=1 spread=agentic: yours 210.5µs vs SGLang 263.2µs (SoL 76.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 94.9µs 46.0%, nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 26.3µs 12.8%, void (anonymous namespace)::dl_pooler_router_kernel<2>(__... 24.6µs 11.9%
- batch=16 ctx=80000 dist=soft layer=full load=decode_16 page=1 spread=agentic: yours 290.5µs vs SGLang 430.7µs (SoL 137.5µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 110.5µs 40.8%, void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 73.6µs 27.2%, nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT 26.5µs 9.8%
- batch=32 ctx=80000 dist=soft layer=sliding load=decode_32 page=1 spread=agentic: yours 239.5µs vs SGLang 260.2µs (SoL 84.5µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 122.2µs 50.1%, nvjet_sm90_tst_64x32_64x16_2x1_v_bz_TNT 27.1µs 11.1%, void (anonymous namespace)::dl_pooler_router_kernel<4>(__... 26.6µs 10.9%
- batch=32 ctx=80000 dist=soft layer=full load=decode_32 page=1 spread=agentic: yours 446.4µs vs SGLang 620.6µs (SoL 208.1µs) · void (anonymous namespace)::splitkv_attn_sk_kernel<4, tru... 224.0µs 51.2%, void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 113.4µs 25.9%, void (anonymous namespace)::dl_pooler_router_kernel<4>(__... 26.8µs 6.1%
- batch=1 ctx=80000 dist=soft layer=sliding load=verify_1 page=1 q_len=8: yours 181.1µs vs SGLang 263.1µs (SoL 62.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 85.1µs 47.2%, void (anonymous namespace)::dl_pooler_router_kernel<1>(__... 22.5µs 12.5%, nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT 12.8µs 7.1%
- batch=1 ctx=80000 dist=soft layer=full load=verify_1 page=1 q_len=8: yours 189.5µs vs SGLang 271.3µs (SoL 66.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 85.4µs 43.9%, void (anonymous namespace)::dl_pooler_router_kernel<1>(__... 23.8µs 12.3%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 18.6µs 9.6%
- batch=16 ctx=80000 dist=soft layer=sliding load=verify_16 page=1 q_len=8 spread=agentic: yours 414.7µs vs SGLang 524.5µs (SoL 126.6µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 314.8µs 67.4%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 28.5µs 6.1%, nvjet_sm90_tst_64x64_64x13_2x1_v_bz_TNT 26.2µs 5.6%
- batch=16 ctx=80000 dist=soft layer=full load=verify_16 page=1 q_len=8 spread=agentic: yours 593.4µs vs SGLang 751.9µs (SoL 188.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 272.9µs 47.2%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 155.6µs 26.9%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 29.8µs 5.2%
- batch=32 ctx=80000 dist=soft layer=sliding load=verify_32 page=1 q_len=8 spread=agentic: yours 599.8µs vs SGLang 720.9µs (SoL 143.8µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 397.2µs 67.9%, nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 34.4µs 5.9%, nvjet_sm90_tst_128x64_64x8_1x2_h_bz_TNT 30.8µs 5.3%
- batch=32 ctx=80000 dist=soft layer=full load=verify_32 page=1 q_len=8 spread=agentic: yours 866.6µs vs SGLang 1073.7µs (SoL 267.2µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 375.5µs 44.5%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 288.4µs 34.2%, nvjet_sm90_tst_128x128_64x6_2x1_v_bz_TNT 34.0µs 4.0%

## By area

### FA3 call
- **triton-flash-decode-full-paired-kv** → neutral (unproven-vs-best), 1.0702x — gained on sliding verify_16; full verify_16. On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1) replace flash_attn_with_kvcache with a two-launch Triton flash-decoding pair buil
- **triton-flash-decode-full-oversub-kv-prefetch** → regression (regression), 1.0413x — lost on full decode_16, decode_32; sliding verify_16. Build on stage 1 (branch hyp/muse_decode_layer/triton-flash-decode-full-paired-kv, muse_decode_layer/submission.py) and change only _full_decode_splits and the 
- **fa3-hdim128-head-pairing-batched** → neutral (unproven-vs-best), 1.0874x — gained on full decode_16, decode_32, verify_16, verify_32. Port muse_attention_decode's landed fa3-hdim128-head-pairing (1.171x) to the flash_attn_with_kvcache call in muse_decode_layer kernel_fn, gated to full-attentio
- **sliding-fused-triton-decode1-port** → neutral (unproven-vs-best), 1.1690x — gained on sliding decode_1. Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, sliding decode_1 16.56 -> 13.12 us) to muse_decode_layer kernel_fn. Copy ve
- **full-decode-cuda-splitkv-port** → neutral (unproven-vs-best), 1.1862x — gained on full decode_16, decode_32. Stage 1 of a rewrite of decode_layer full-attention decode at batch > 1, ported from muse_attention_decode master (full-decode-cuda-gather-splitkv, job 82085a4e
- **full-verify-cuda-splitkv-stage2** → regression (regression), 1.1559x — lost on full verify_16, verify_32. Stage 2 of full-decode-cuda-splitkv-port: extend the CUDA split-KV path to full-layer verify blocks (q_len 8) at batch 16/32, where master still runs paired FA3

### FA3 call (full decode, all batches): split-KV A/B at 1<B<=64 and FA3 chain at B=1
- **full-decode-sk-ticket-restack-b1-b64** → neutral (within-ci), 1.2847x — gained on full decode_16, decode_32; lost on sliding verify_32. Restack onto current master (w2-marlin-atomic-add-t1-port, 1.2856 check) the two attention-side stream-K ports that each won their rows on the previous master, 

### FA3 call (full decode, batch 1)
- **full-decode1-two-level-ticket-port** → neutral (within-ci), 1.2746x — gained on sliding verify_32. Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_

### FA3 call (full decode, batch > 1)
- **full-decode-splitkv-restack-on-pdl** → neutral (unproven-vs-best), 1.2680x — gained on full decode_16, decode_32. Restack decode own neutral full-decode-cuda-splitkv-port (1.186x vs 1.166x base; its gated shapes moved full decode_16 364.9 -> 329.1 us (+12.8%) and full decod

### FA3 call (full decode, batch > 1): splitkv_attn_kernel + splitkv_combine_kernel
- **full-decode-streamk-ticket-port** → neutral (within-ci), 1.2775x — gained on full decode_16, decode_32. Stage 3 of the decode split-KV line (stage 1 full-decode-cuda-splitkv-port, stage 2 the landed restack). Built on master, which now carries the landed full-deco

### FA3 call (full decode, batch > 1): splitkv_attn_kernel bytes in flight
- **full-decode-splitkv-kv-bw-3p5** → neutral (within-ci), 1.2663x — lost on full decode_16. Stage 2 of full-decode-cuda-splitkv-port (REWRITE_READ_2026-10-25 section 2, item 2), built on the restacked code from full-decode-splitkv-restack-on-pdl. Kerne

### FA3 call (sliding decode, batch 1)
- **sliding-decode1-fused-restack-on-pdl** → neutral (unproven-vs-best), 1.2374x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16; full decode_1, decode_16, decode_32, verify_1. Restack decode own neutral sliding-fused-triton-decode1-port (sliding decode_1 145.8 -> 139.4 us, +4.4%, every other shape flat; it also carried the +8.75% slid

### FA3 call (sliding decode, batch 17..32)
- **sliding-cluster-c4-decode32-port** → neutral (within-ci), 1.2693x — no shape moved outside noise. Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_attention_
- **sliding-cluster-c4-decode32-s2-k-encode-match** → neutral (unproven-vs-best), 1.2803x — gained on full decode_1; sliding verify_32. Stage 1 (run 0fb2c945) was NEVER MEASURED: sliding_cluster_kernel C=4 built (124 regs, 32 B spill, 103 KB smem, 62 active clusters), ran without fault and its a

### FA3 call (sliding decode, batch 2..16)
- **sliding-batched-triton-b16-band** → neutral (unproven-vs-best), 1.2743x — gained on full decode_32. Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Re-apply decode's ha

### FA3 call (sliding decode, batch > 1)
- **sliding-fused-batched-decode-port** → neutral (within-ci), 1.2248x — gained on sliding decode_16, decode_32; lost on sliding verify_32; full verify_32. Port muse_attention_decode master batched sliding-decode path (landed sliding-deep-splits-on-cuda-splitkv, job 698d4cd3838c4a63: sliding decode_16 20.7 -> 15.3 

### MoE align (T>32) + sliding batch-1 attention
- **stack-align-par-sliding1** → neutral (unproven-vs-best), 1.2444x — gained on sliding decode_1, verify_32. Built on master (dense-chain PDL). Stack the two decode checks of 2026-10-24 night that each beat the 1.228 best, applied unchanged from their done/ patches: mo

### MoE align/dispatch
- **expert-id-map-single-kernel** → neutral (unproven-vs-best), 1.0374x — no shape moved outside noise. Replace the three small kernels that turn the routing output into Marlin's local expert ids with one Triton kernel. Today kernel_fn computes local_ids = (topk_i
- **moe-align-single-kernel-small-t** → neutral (within-ci), 1.0431x — no shape moved outside noise. At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call AND the expert-id map kernel from expert-id-map-single-kernel with one 

### MoE align/dispatch (T>32)
- **moe-align-parallel-large-t-port** → neutral (unproven-vs-best), 1.2433x — gained on sliding verify_16; full verify_16. Port muse_moe_w4a16 align-parallel-per-expert-verify (patch muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.patch, moe check 45eacf4ca2e

### MoE align/dispatch (stream placement)
- **moe-dispatch-prep-on-side-stream** → neutral (unproven-vs-best), 1.2294x — no shape moved outside noise. Pure stream placement, no new kernels: move the routing-only half of the expert dispatch onto state["side_stream"], after state["routing"](...), so it overlaps 

### T>32 MoE dispatch (parallel align on the side stream) + sliding batch-1 attention
- **stack-align-par-sidestream-sliding1** → neutral (unproven-vs-best), 1.2495x — gained on sliding decode_1, verify_16; full verify_16. Built on master (dense-chain PDL). moe-align-parallel-large-t-port + moe-dispatch-prep-on-side-stream + sliding-decode1-fused-restack-on-pdl. The side-stream bu

### T>32 MoE dispatch on the side stream + sliding batch-1 fused attention
- **stack-sidestream-sliding1** → neutral (unproven-vs-best), 1.2352x — gained on sliding decode_1. Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 r

### T>32 MoE dispatch on the side stream + sliding batch>1 fused attention
- **stack-sidestream-sliding-batched** → neutral (unproven-vs-best), 1.2443x — gained on sliding decode_16, decode_32. Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 r

### T>32 parallel align + sliding batch>1 fused attention
- **stack-align-par-sliding-batched** → failed (test:failed) — gained on sliding decode_16, decode_32; lost on full verify_1. Built on master (dense-chain PDL). Stack two shape-disjoint decode checks of 2026-10-24 night, applied unchanged from their done/ patches (the sliding batch-1 r

### cache store
- **cache-store-fused-fp8-cast** → neutral (unproven-vs-best), 1.0303x — no shape moved outside noise. In kernel_fn, replace the four separate launches between FusedQKNorm and flash_attn_with_kvcache, namely k.to(FP8), v.to(FP8), store_cache(...) and q.view(token

### combination: sliding attention B=2..32 + pre-attention feeds (T>1) + routing R / unpooler (2<=T<=32 / 2<=T<=8)
- **combo-measured-rows-slide-batched-preattn-t8-tail** → neutral (unproven-vs-best), 1.3142x — gained on full verify_1, verify_16. COMBINATION (naive top-row stack): the decode-measured per-shape winners that never reached master, each routed only to the shape class where it won, on master 

### full decode_1 attention side (dl3 piece A enable)
- **dl3-piece-a-gate-stride-check-fix** → neutral (unproven-vs-best), 1.3080x — gained on full decode_1. Built on master d5595769 (dense-chain-s3-t1-q-sk2-o-route-unpool landed). Piece A of s3 (kernel Q dl2_qkv + splitkv_attn_sk two-level with norm-gate epilogue + 

### o GEMM + pooler GEMM (L2 warm-up overlapped with attention)
- **l2-prefetch-o-pooler-during-sliding-attention** → regression (regression), 1.1670x — lost on sliding decode_16, decode_32, verify_1. On sliding layers at 2 <= tokens <= 32 (config gate: window != 0 and 2 <= tokens <= 32, i.e. sliding decode_16, decode_32 and verify_1), overlap an L2 warm-up o

### output norm-gate
- **paired-heads-pad-unpad-triton** → neutral (within-ci), 1.1654x — gained on full verify_16. Port the launch cleanup of muse_attention_decode's paired-heads-launch-cleanup (full layers there: decode_16 +6.4%, decode_32 +3.8%, verify_16 +5.9%) to master'

### pooler + router GEMM (kernel P, T=9..32)
- **dl-p-wide-tile-t16-t32** → neutral (within-ci), 1.2457x — no shape moved outside noise. Kernel P (dl_pooler_router_kernel<NF>) streams only 2.85 TB/s at T=32 and 3.41 at T=8 (prepare timing in 81ee6714b7614a99.log: 26.4 us vs 20.7 at T=1) because e

### pooler GEMM
- **pooler-gemm-splitk-autotune** → neutral (within-ci), 1.0189x — no shape moved outside noise. For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB weight read whose floor is 14 us), pick the GEMM form per scored T in pre
- **bf16-shape-gated-triton-gemms-small-t** → neutral (unproven-vs-best), 1.1688x — gained on sliding decode_1; full decode_1, decode_32, verify_16; lost on sliding decode_16, verify_1; full verify_1. Port muse_bf16_proj's landed bf16-shape-gated-triton-winners-combo (1.176x vs 1.151x there; code is muse_bf16_proj/submission.py on master: _bf16_splitk_kernel,

### post-MoE unpooler + residual boundaries (T<=32)
- **boundaries-unpooler-pdl-small-t** → neutral (unproven-vs-best), 1.2559x — no shape moved outside noise. Extend PDL to the stages around the landed dense chain at tokens <= DL_MAX_TOKENS (config gate plus state["dl"] and a new state["dlb_ok"] flag; T>32 master byte

### post-expert norm+combine
- **post-expert-norm-triton-small-t** → gain (gain), 1.1635x — gained on sliding decode_1, decode_16, verify_1; full decode_1, decode_16, verify_1. Port muse_moe_w4a16's landed post-expert-norm-triton-small-m (1.130x there, job bacd1f502f4e46fa; patch muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-tri

### pre-attention norm+rope
- **fused-preattention-triton** → neutral (unproven-vs-best), 1.0970x — gained on sliding decode_32. Port muse_attention_decode's fused pre-attention Triton kernel (main tree muse_attention_decode/submission.py, _pre_attention_kernel: passed 16/16 at 1.127x the
- **fused-preattention-paired-pad-restack** → neutral (unproven-vs-best), 1.2745x — gained on full decode_1, decode_16, decode_32; sliding verify_1. Restack decode's own neutral fused-preattention-triton (1.097x vs 1.087x master then; patch muse_decode_layer/hypotheses/done/neutral/fused-preattention-triton.

### qkv GEMM + o GEMM (2 <= T <= 16)
- **qkv-o-swapab16-t2-t16-no-fork** → neutral (unproven-vs-best), 1.3056x — gained on sliding decode_1; full decode_1, decode_16, decode_32; lost on sliding verify_32. Port muse_bf16_proj master's m<=16 swap-AB wgmma path (hypothesis bf16-m16-swapab-wgmma-extend: qkv 18% faster, o 10-19% at m<=16 there) to exactly two call sit

### qkv GEMM + o GEMM + attention-side small kernels (T<=32)
- **dense-chain-stage2-attn-side-pdl** → regression (regression), 1.2253x — gained on full decode_1; lost on sliding decode_16, decode_32; full decode_32. Stage 2 of the landed small-t-launch-minimal-dense-chain-pdl rewrite (master f1f2f35, its stage plan item 2). Gate: tokens <= DL_MAX_TOKENS (config) and state["

### qkv GEMM + o GEMM + attention-side small kernels (T=1 only)
- **dense-chain-stage2-attn-side-pdl-t1-gate** → neutral (unproven-vs-best), 1.2717x — gained on full decode_1, decode_32, verify_32; sliding verify_32. Restack dense-chain-stage2-attn-side-pdl onto master (claim applies its patch, depends_on) and narrow its gate from tokens <= DL_MAX_TOKENS to tokens == 1. Its 

### restack (sliding decode_1 port + paired pad/unpad + T=1 Triton GEMMs)
- **decode-restack-sliding-port-paired-pad-gemms-t1** → neutral (beats-base-not-best), 1.1937x — gained on sliding decode_1, verify_32; full decode_1, verify_16. Restack three harvested per-shape winners onto current master (post-expert-norm-triton-small-t) by a 3-way merge of their branches, resolving conflicts by hand 

### router GEMM
- **sglang-pipeline-parity-router-hilo-m8** → gain (gain), 1.0216x — gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Replace the starter's plain-PyTorch layer with SGLang's own pipeline exactly as baseline.py composes it: prepare() builds the Marlin bank via prepare_moe_nvfp4_

### routing top-k (kernel R) + post-MoE unpooler (T<=8)
- **stack-route-topk-pdl-unpool-nf1** → neutral (within-ci), 1.2671x — gained on sliding decode_1, verify_1; full decode_1. Stack two measured T=1 gains on master (BEST stack-align-par-sidestream-sliding1). Claim restacks dl-route-warp-topk-pdl onto master from its patch (depends_on)

### routing top-k + MoE align (kernel R, T<=32)
- **dl-route-warp-topk-pdl** → neutral (unproven-vs-best), 1.2599x — no shape moved outside noise. Kernel R (dl_route_norm_align_kernel in master _DL_CUDA) costs 7.4-8.0 us at every T<=32 (profile 81ee6714b7614a99; its own milestone was <= 4 us). Two changes,

### w13 marlin
- **marlin-block-m-cap-16** → gain (gain), 1.0621x — gained on sliding verify_16, verify_32; full verify_16, verify_32. Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first siz

### w13 marlin + silu-mul + w2 marlin
- **moe-chain-cuda-swapab-expert-once-large-t** → failed (test:failed). WHY: at T=128/256 the two Marlin MoE GEMMs are the single largest block of the whole problem: 254/271 us (sliding/full verify_16) and 443/427 us (verify_32) in 

### w13 marlin + silu-mul + w2 marlin (T>32)
- **moe-chain-cuda-swapab-expert-once-fix-inproc** → regression (regression), 1.1651x — gained on sliding decode_1, decode_16, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Stage 1b FIX (numbered stage 2 only because stage is an int; the original stage plan's stage 2 still comes after this) of failed moe-chain-cuda-swapab-expert-on
- **moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound** → regression (regression), 1.0637x — gained on sliding decode_1, decode_16, decode_32, verify_1; full decode_1, decode_16, decode_32, verify_1; lost on sliding verify_16, verify_32; full verify_16, verify_32. Stage 3 of the expert-once swap-AB MoE chain (moe_route_list + moe_w13_silu_swapab + moe_w2_swapab replacing moe_align + Marlin w13 + act_and_mul + Marlin w2 at

### w2 marlin (T=1)
- **w2-marlin-atomic-add-t1-port** → neutral (unproven-vs-best), 1.2856x — gained on sliding verify_32. Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_moe_w4a16'

### whole-layer fusion
- **small-t-launch-minimal-dense-chain-pdl** → gain (gain), 1.2283x — gained on sliding decode_1, decode_16, decode_32, verify_1; full decode_1, decode_16, verify_1. WHY: at T<=32 (decode_1/16/32 and verify_1: 4 of the 6 loads) master runs ~27-29 graph nodes per layer. At decode_1 sliding (145.6 us, floor 40.0) Marlin is 40.

### whole-layer fusion (T=1 dense chain: qkv/o GEMM side + routing R + unpooler)
- **dense-chain-s3-t1-q-sk2-o-route-unpool** → neutral (unproven-vs-best), 1.3018x — gained on sliding decode_1; full decode_1, decode_16, decode_32. Stage 3 of the dense-chain line (s1 small-t-launch-minimal-dense-chain-pdl landed; s2 dense-chain-stage2-attn-side-pdl-t1-gate won full decode_1 -3.3% z-5.8 on 

## Open follow-ups

- triton-flash-decode-full-paired-kv: rewrite stage 1 of `FA3 on full-attention decode (q_len=1): flash_fwd main + FlashAttnFwdCombine`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- full-decode-cuda-splitkv-port: rewrite stage 1 of `paired FA3 flash_fwd main + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full-layer decode, batch 16/32`: read the replaced kernels' own us per shape (profile top_kernels) against the 
- full-decode-splitkv-kv-bw-3p5: rewrite stage 2 of `paired FA3 flash_fwd + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full decode, 1 < batch <= 64 (now splitkv_attn_kernel + splitkv_combine_kernel)`: read the replaced kernels' own us 
- full-decode-streamk-ticket-port: rewrite stage 3 of `splitkv_attn_kernel + splitkv_combine_kernel on full decode, 1 < batch <= 64`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Mi
- full-decode1-two-level-ticket-port: rewrite stage 1 of `FA3 main + FlashAttnFwdCombine + pad/unpad elementwise + FusedRMSNormGate on full decode batch 1`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor
- sliding-cluster-c4-decode32-port: rewrite stage 1 of `FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32`: read the replaced kernels' own us per 
- sliding-cluster-c4-decode32-s2-k-encode-match: rewrite stage 2 of `FusedQKNorm/rope + FP8 casts + store_cache + FA3 main + FlashAttnFwdCombine + prepare_varlen_num_blocks + FusedRMSNormGate on sliding decode, 17 <= batch <= 32`: read the replaced kernels' own us per 
- dense-chain-s3-t1-q-sk2-o-route-unpool: rewrite stage 3 of `T=1 dense chain: nvjet qkv + FusedQKNorm/rope + FP8 casts + store_cache + nvjet o (full decode_1), routing kernel R and unpooler GEMM+pad at NF=1 (both decode_1 rows)`: read the replaced kernels' own 
- moe-chain-cuda-swapab-expert-once-large-t: rewrite stage 1 of `moe_align_block_size + Marlin w13 + silu_and_mul (act_and_mul_kernel) + Marlin w2 at T > 32 (verify_16 T=128, verify_32 T=256)`: read the replaced kernels' own us per shape (profile top_kernels) again
- stack-align-par-sliding-batched: none

## Do not try (regressions, flat results, rejected)

Rewrite stages are exempt: a slow early stage is not a dead end.

- **pooler-gemm-splitk-autotune** [pooler GEMM] neutral, within-ci: For the pooler GEMM only (ffn_input [T, 8192] bf16 x pooler [4096, 8192] bf16, a 67 MB wei
- **moe-align-single-kernel-small-t** [MoE align/dispatch] neutral, within-ci: At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call A
- **paired-heads-pad-unpad-triton** [output norm-gate] neutral, within-ci: Port the launch cleanup of muse_attention_decode's paired-heads-launch-cleanup (full layer
- **sliding-fused-batched-decode-port** [FA3 call (sliding decode, batch > 1)] neutral, within-ci: Port muse_attention_decode master batched sliding-decode path (landed sliding-deep-splits-
- **dl-p-wide-tile-t16-t32** [pooler + router GEMM (kernel P, T=9..32)] neutral, within-ci: Kernel P (dl_pooler_router_kernel<NF>) streams only 2.85 TB/s at T=32 and 3.41 at T=8 (pre
- **stack-route-topk-pdl-unpool-nf1** [routing top-k (kernel R) + post-MoE unpooler (T<=8)] neutral, within-ci: Stack two measured T=1 gains on master (BEST stack-align-par-sidestream-sliding1). Claim r
- **full-decode-sk-ticket-restack-b1-b64** [FA3 call (full decode, all batches): split-KV A/B at 1<B<=64 and FA3 chain at B=1] neutral, within-ci: Restack onto current master (w2-marlin-atomic-add-t1-port, 1.2856 check) the two attention
- **l2-prefetch-o-pooler-during-sliding-attention** [o GEMM + pooler GEMM (L2 warm-up overlapped with attention)] regression, regression: On sliding layers at 2 <= tokens <= 32 (config gate: window != 0 and 2 <= tokens <= 32, i.

## Queue and in flight

- queue: fa3-num-splits-full-no-metadata [FA3 call] p3 hold
- queue: moe-align-local-expert-count [MoE align/dispatch] p4 hold
- queue: trtllm-decode-full-attention [FA3 call] p5 hold
- queue: fa3-num-splits-sliding [FA3 call] p2 hold
- queue: moe-align-single-kernel-small-t-control-capacity-256 [MoE align/dispatch] p3 hold
- queue: fa3-full-single-split-probe [FA3 call] p2 hold
- queue: router-triton-splitk-side-stream [router GEMM] p1 hold
- queue: triton-flash-decode-full-verify-paired-kv [FA3 call] p5 hold
- queue: bf16-m32-swapab-wgmma-port [pooler GEMM] p3 hold
- queue: moe-align-small-t-restack-on-master [MoE align/dispatch] p4 hold
- queue: pre-expert-rmsnorm-triton [pre-expert norm] p3 hold
- queue: moe-small-t-cuda-swapab-port [w13 marlin + silu-mul + w2 marlin (T<=32)] p4 hold
- queue: moe-bs1-fused-persistent-port [w13 marlin + silu-mul + w2 marlin + post-expert combine (T=1)] p5 hold
- queue: prefork-qkv-o-swapab-t16-t32 [qkv GEMM + o GEMM (before the router fork)] p4 hold
- queue: marlin-two-stream-halves-port [w13 marlin + silu-mul + w2 marlin (8<=T<=32, stream overlap)] p5 hold
- queue: moe-chain-cuda-swapab-expert-once-fix [w13 marlin + silu-mul + w2 marlin (T>32)] p2 hold
- queue: sliding-cluster-c4-s3-satfinite-on-sk-ticket [FA3 call (sliding decode, batch 17..32)] p1 queued
- queue: dl3-unpool-nf1-t2-t8 [unpooler GEMM (T=2..8, dl3 piece C widened)] p2 queued
- queue: dense-qkv-o-unpool-swapab-bf16-master-port-t1-t32 [qkv GEMM + o GEMM + unpooler GEMM (T <= 32)] p1 hold
- queue: ffn-front-t128-t256-concat-route [pooler GEMM + router GEMM + routing top-k + MoE align + pre-expert norm (T > 32)] p2 queued
- queue: moe-rega-bucket-gate-port-t128-t512 [w13 marlin + silu-mul + w2 marlin (128 <= T <= 512)] p1 queued
- queue: preattn-fused-t-gt1-sk-and-fa3-feeds [pre-attention norm+rope + cache store (T > 1, every shape not routed to _sliding_forward or dl3 piece A)] p3 queued
- queue: sliding-cluster-c16-decode1-port [FA3 call (sliding decode, batch 1)] p4 queued
- queue: sliding-sk-window-port-decode32-verify1 [FA3 call (sliding decode B=17..32 and sliding verify B=1: windowed stream-K)] p2 queued
- queue: swapab16-t2-t16-port-on-master [qkv GEMM + o GEMM (2 <= T <= 16)] p1 hold
- running: combo-attn-master-sliding-race-skw-c16 [combination: sliding attention per-band race (B=1, 2..16, 17..32, verify B=1) + routing R / unpooler tail] p1 running
- running: combo-measured-rows-plus-rega-align2-large-t [combination: C1 measured stack + w13/w2 rega and align2 at 128<=T<=512] p1 running
