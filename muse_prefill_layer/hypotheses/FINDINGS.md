# muse_prefill_layer findings

Regenerated 2026-10-27T133340Z by `churn.py index`. Read this before writing a hypothesis for muse_prefill_layer.

## Current best

1.4889x [1.4666,1.5112] from job `ec3bd01859574bf1` (submit), master commit `9fbde72cc9`, hypothesis experts-rega-w13-fast-silu-epilogue

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6: yours 732.6µs vs SGLang 1101.8µs (SoL 257.0µs) · void (anonymous namespace)::rega_kernel<true, false>((ano... 302.0µs 37.7%, void (anonymous namespace)::rega_kernel<false, false>((an... 159.3µs 19.9%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 154.9µs 19.3%
- batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6: yours 1120.8µs vs SGLang 1486.6µs (SoL 496.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 420.4µs 35.5%, void (anonymous namespace)::rega_kernel<true, false>((ano... 256.4µs 21.6%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 153.1µs 12.9%
- batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6: yours 2140.1µs vs SGLang 3414.0µs (SoL 1036.3µs) · void (anonymous namespace)::rega_kernel<true, false>((ano... 682.9µs 30.9%, void (anonymous namespace)::rega_kernel<false, false>((an... 356.0µs 16.1%, nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT 327.0µs 14.8%
- batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6: yours 3698.9µs vs SGLang 4995.7µs (SoL 1992.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 1646.3µs 42.7%, void (anonymous namespace)::rega_kernel<true, false>((ano... 716.6µs 18.6%, void (anonymous namespace)::rega_kernel<false, false>((an... 381.6µs 9.9%
- batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 11731.0µs vs SGLang 18419.4µs (SoL 5518.4µs) · void (anonymous namespace)::rega_kernel<true, true>((anon... 3638.9µs 30.7%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2570.0µs 21.7%, void (anonymous namespace)::rega_kernel<false, true>((ano... 1884.8µs 15.9%
- batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6: yours 10892.9µs vs SGLang 17248.8µs (SoL 5663.4µs) · void (anonymous namespace)::rega_kernel<true, true>((anon... 3474.3µs 31.4%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2568.8µs 23.2%, void (anonymous namespace)::rega_kernel<false, true>((ano... 1798.4µs 16.3%
- batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 15080.4µs vs SGLang 21946.5µs (SoL 7710.0µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 4268.3µs 28.1%, void (anonymous namespace)::rega_kernel<true, true>((anon... 3610.2µs 23.7%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2569.8µs 16.9%
- batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 18110.3µs vs SGLang 24821.2µs (SoL 9847.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 7610.9µs 41.2%, void (anonymous namespace)::rega_kernel<true, true>((anon... 3495.5µs 18.9%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2568.5µs 13.9%

Today's clean-context diagnosis: `muse_prefill_layer/hypotheses/DIAGNOSIS.md` (ranked actions and its own Do-not-try list).

## By area

### FA3 call
- **fa3-contiguous-kv-gather** → regression (regression), 0.9879x — lost on full turn_1, turn_4. Replace the paged FA3 call with a contiguous one; everything before and after stays the parity path. After store_cache (so the new tokens' rows are in the cache

### FA3 call (CUDA sm_90a FP8 prefill flash attention with software-exp offload, stage 1: full layers, page size 1)
- **attn-full-prefill-cuda-fp8-exp-offload** → regression (regression), 0.8813x — lost on full first_turn, turn_1, turn_4. Replace FA3 (flash_attn_with_kvcache and its FlashAttnFwdCombine) on full-attention prefill layers with a hand-written sm_90a FP8 flash-attention forward kernel

### FA3 call on full-attention layers (KV source: contiguous TMA copy of the paged page-size-1 rows, per-class timing gate)
- **fa3-full-contiguous-kv-triton-gather-gated** → neutral (unproven-vs-best), 1.4587x — gained on full first_turn. Built on master (commit 1932a9c4, submission blob a2fe2a75). Only the FA3 call on full-attention layers (cfg['window'] == 0) at page_size 1 and kv_scale == 1 ch

### FA3 call on full-attention layers at page_size 1 (CUDA sm_90a FP8 wgmma flash attention, stage 1b: fix of attn-full-prefill-cuda-fp8-exp-offload with build fix, hard timing gate, no exp offload)
- **attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc** → neutral (within-ci), 1.0679x — no shape moved outside noise. BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored; its attention code is unchanged from the old base): start from master's 

### FFN boundary + unpooler GEMM (boundary chunks overlapped under unpooler chunks via a slice-writing boundary kernel)
- **boundary-overlap-s1-slice-kernel-ffn-tail** → regression (regression), 1.4438x — lost on sliding turn_1. Rewrite stage 1, built on current master (blob a2fe2a75). Goal of the line: take the two FusedPostNormResidualPreNorm boundaries (muse_spark_norm_residual_kerne

### MoE align on the Marlin path (T <= 4096) + pre-expert norm: restack of two measured row wins
- **turn-align-sglang32-prenorm-triton-restack** → neutral (within-ci), 1.0694x — no shape moved outside noise. Built on master (a81d19c). Two row-level wins already measured in the archive, applied together, shapes untouched otherwise. (1) ALIGN at 200 < T <= 4096: maste

### MoE align on the Triton path (T > 4096, first_turn)
- **align-local-parallel-first-turn** → neutral (within-ci), 1.0626x — no shape moved outside noise. Built on moe-align-local-routes-parallel-port (running, its implementation commit). In _experts_triton, when state['align_par_ok'], replace moe_align_block_size

### MoE align/dispatch
- **moe-align-side-stream** → neutral (unproven-vs-best), 1.0076x — no shape moved outside noise. Keep the parity base and move three pieces of work that depend only on the routing outputs and local_start onto the existing side stream, inside the same torch.

### MoE align/dispatch (align over the 32 local experts at T > 4096)
- **moe-align-32-experts-first-turn** → neutral (unproven-vs-best), 1.0604x — no shape moved outside noise. Built on master (635cfee). One argument in _experts_triton (T > 4096 only): moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) becomes moe_align_block_s

### MoE align/dispatch (local-routes-only align, port of muse_moe_w4a16 align-parallel-per-expert-verify)
- **moe-align-local-routes-parallel-port** → neutral (unproven-vs-best), 1.0658x — no shape moved outside noise. Built on master (635cfee). Port muse_moe_w4a16's measured align (master there: _align_parallel_kernel and its host wrapper _align_parallel in muse_moe_w4a16/sub

### MoE align/dispatch on T <= 4096 (Marlin turn_1 + ssdq turn_4): _align_parallel replaced by a two-launch chunk-count + prefix-place align
- **turn-align-two-phase-chunk-prefix** → neutral (unproven-vs-best), 1.3369x — gained on sliding turn_4. Built on master (ssdq s1b landed). Only _align_parallel's implementation changes, at its two call sites on T <= 4096: _experts_marlin (turn_1 T=753, block_size_

### MoE align/dispatch on the rega path (200 < T <= 16384): _align_parallel replaced by the two-phase chunk-count + prefix-place align, including the first_turn class
- **align-two-phase-all-t-rega** → neutral (unproven-vs-best), 1.4458x — gained on sliding turn_1, turn_4; full first_turn. Restack of turn-align-two-phase-chunk-prefix (job d792e5c9d6d94f02, neutral 1.3369 on the pre-ft master; patch muse_prefill_layer/hypotheses/done/neutral/turn-a

### MoE align/dispatch placement at T <= 4096: local-id remap + _align_parallel moved onto the router side stream, under the pooler GEMM
- **turn-align-remap-on-side-stream-under-pooler** → neutral (within-ci), 1.0683x — no shape moved outside noise. Built on master (a81d19c). In the best profile master's _align_parallel_kernel runs serially on the main stream after the router join: 16.9 us at turn_1 (1.7%) 

### attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2, persistent multi-row variant)
- **norm-boundaries-persistent-multirow** → neutral (within-ci), 1.0716x — no shape moved outside noise. Built on master (635cfee). The multi-row/persistent re-mint the review of the held norm-residual-boundaries-triton-row-kernel invited ('same one-row-per-program

### launch overhead: PDL chain across the expert block (_norm_gather_kernel -> rega w13 -> rega w2)
- **experts-rega-pdl-chain-gather-w13-w2** → neutral (within-ci), 1.4622x — no shape moved outside noise. DIAGNOSIS action 4. Built on master (blob f4751fc0); independent of the mc2 line. Prefill has no programmatic dependent launch (griddepcontrol appears 0 times i

### pre-attention norm+rope + cache store
- **qk-norm-fp8-cache-store-fusion** → neutral (unproven-vs-best), 1.0093x — no shape moved outside noise. One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8), v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per t

### pre-expert norm (MuseSpark1xRMSNorm) fused into the rega row gather
- **pre-expert-norm-fused-rega-gather** → neutral (unproven-vs-best), 1.4600x — gained on sliding first_turn. Built on current master (submission blob a2fe2a75, align-two-phase-all-t-rega landed; no depends_on). On the rega expert path only (_experts dispatches to _expe

### pre-expert norm + post-expert norm/combine (the MoE elementwise ends)
- **moe-pre-post-norm-triton-multitoken** → neutral (within-ci), 1.0714x — no shape moved outside noise. Built on master (635cfee). Two kernels at the ends of the MoE block, at every T. (1) PRE-EXPERT NORM: replace state['pre_expert_norm'](expert_input) (MuseSpark1

### qkv GEMM (output-gate columns split onto the side stream, overlapping qk_norm + cache store + FA3)
- **qkv-gate-columns-side-stream-under-attention** → neutral (within-ci), 1.4512x — no shape moved outside noise. Built on current master (submission blob a2fe2a75; no depends_on). The qkv_gate projection is one GEMM today (qkv = _gemm(attn_input, dense['qkv_gate']), N = 23

### restack: pre-expert norm + rega gather fused (all rega classes) on master 1.4587; naive per-shape winner stack
- **combo-a-master-normgather-naive-stack** → neutral (unproven-vs-best), 1.4605x — no shape moved outside noise. COMBINATION A (naive top-row stack, restack of measured ON paths only). Start from the resolved 3-way merge /home/user/git/musespark/muse_prefill_layer/hypothes

### router GEMM (hi and lo halves as one N=512 cuBLAS GEMM)
- **router-hilo-one-gemm-n512** → neutral (unproven-vs-best), 1.0607x — no shape moved outside noise. Built on master (635cfee). prepare(): add state['router_hilo_t'] = torch.cat((hi.t(), lo.t()), dim=1).contiguous(), bf16 [8192, 512] (the same BF16 hi/lo split 

### router GEMM + MoE align (T>4096) + w2 zero-store (T>4096)
- **restack-router-hilo-align32-zerostore** → neutral (unproven-vs-best), 1.0608x — lost on full first_turn. Built on master. Stack the three shape-disjoint prefill checks of 2026-10-24 23:15Z, each applied unchanged from its done/ patch: router-hilo-one-gemm-n512 (one

### router GEMM + MoE align (both paths) + w2 zero-store
- **stack-tonight-four-prefill** → neutral (unproven-vs-best), 1.0675x — no shape moved outside noise. Built on master. Stack all four prefill changes built on 2026-10-24 night, each applied unchanged: router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w

### router GEMM + MoE align (both paths) + w2 zero-store + Marlin silu-mul
- **stack-tonight-five-prefill** → neutral (unproven-vs-best), 1.0705x — no shape moved outside noise. Built on master. stack-tonight-four-prefill (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn, moe-align-local-

### router GEMM + MoE align (both paths, parallel) + w2 zero-store + Marlin silu-mul
- **stack-tonight-six-prefill** → neutral (unproven-vs-best), 1.0642x — no shape moved outside noise. Built on master. stack-tonight-five-prefill plus align-local-parallel-first-turn (the parallel local-routes align on the Triton path too, self-checked at T=1638

### silu-mul (Marlin path, T <= 4096)
- **silu-mul-triton-local-rows-marlin-turn** → neutral (within-ci), 1.0512x — no shape moved outside noise. Built on master (635cfee). In _experts_marlin only (the Marlin path, T <= MARLIN_MAX_TOKENS = 4096: turn_1 T=753, turn_4 T=3012 and every test), replace silu_an

### silu-mul (fused into the w13 Triton NVFP4 GEMM epilogue, T > 4096)
- **experts-triton-w13-fused-silu-epilogue** → neutral (within-ci), 1.0525x — lost on full first_turn. Fuse SGLang's silu_and_mul into the epilogue of the first (w13) Triton NVFP4 GEMM on the T > 4096 path (_experts_triton, first_turn only), built on master. (1) 

### silu-mul (rega w13 epilogue)
- **experts-rega-w13-fast-silu-epilogue** → gain (gain), 1.4949x — gained on sliding turn_1. Built on master (blob f4751fc0). rega_kernel<true, *> (w13) applies silu(gate) * up to its fp32 accumulators in the epilogue through silu_mul(hg, hu) = hg / (1.

### silu-mul on the Triton path (T > 4096, first_turn)
- **silu-mul-sorted-first-turn** → neutral (within-ci), 1.0636x — no shape moved outside noise. Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_triton, when state['silu_sorted_ok'], replace silu_and_mul(cache1, cache2, 

### sliding-window attention block (q FP8 cast + FA3 + FlashAttnFwdCombine + FusedRMSNormGate) on sliding layers
- **sliding-attn-block-triton-gqa-fused-normgate** → neutral (within-ci), 1.0507x — no shape moved outside noise. Built on master alone (b92a0f4, experts-hybrid-marlin-turn-triton-first-turn); touches nothing the two in-flight prefill runs touch (full-layer FA3 stays FA3, e

### w13 + w2 at turn_4 (1024 < T <= 4096) routed to the Triton NVFP4 engine with local-routes align, original non-persistent grid
- **experts-triton-turn4-persistent-local-align-control-no-persi** → neutral (within-ci), 1.0557x — no shape moved outside noise. Built on master (a81d19c). Implement only parts (1) and (3) of the parent. (1) In _experts, send 1024 < T <= 4096 to _experts_triton (set MARLIN_MAX_TOKENS = 10

### w13 + w2 expert GEMMs (rega persistent item->CTA schedule)
- **experts-rega-round-rotated-schedule** → neutral (within-ci), 1.4664x — no shape moved outside noise. Built on master (blob f4751fc0). Only the order in which rega_kernel's persistent CTAs walk items changes; every item is computed by master's code, so outputs a

### w13 + w2 expert GEMMs (rega wgmma N buckets at 16-row granularity)
- **experts-rega-fine-n-buckets** → neutral (within-ci), 1.4573x — no shape moved outside noise. Built on master (blob f4751fc0). rega sizes each item's wgmma N (tokens) and its slab TMA box from the chunk's 16-route block count nbk (1..8) with four buckets

### w13 + w2 expert GEMMs (rega_kernel consumer schedule: two-warpgroup ping-pong), every rega class
- **experts-rega-pingpong-consumers-all-t** → neutral (within-ci), 1.4478x — gained on full first_turn; lost on sliding turn_1. Built on master (1932a9c4, align-two-phase-all-t-rega). Consumer schedule of rega_kernel only. ARITHMETIC FIRST (the judge's router seed fixes the busiest GPU's

### w13 + w2 expert GEMMs (rega_kernel dequant: E4M3 scale expansion moved to idle producer warps), every rega class
- **experts-rega-producer-scale-expand-all-t** → neutral (within-ci), 1.4538x — gained on sliding first_turn; full first_turn; lost on sliding turn_1. Built on master (1932a9c4). Consumer instruction count of rega_kernel only. ARITHMETIC: turn_1 (~1.36K local routes, ~42 per expert, buckets 32/64) streams w13'

### w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft classes): 256-weight-row CTA tile with in-warpgroup dequant/MMA overlap
- **experts-rega-s4-bm256-wide-tile-high-ft** → neutral (unproven-vs-best), 1.4758x — gained on sliding first_turn, turn_1, turn_4; full first_turn, turn_4. Rewrite stage 4 of the rega line, built on master (1932a9c4; stage 2 experts-turn-rega-s4-first-turn-n128 landed, stage 3 experts-rega-low-class-two-ctas-per-sm
- **experts-rega-s4-bm256-wide-tile-high-ft-cf** → neutral (within-ci), 1.4711x — no shape moved outside noise. Confirmation rerun of experts-rega-s4-bm256-wide-tile-high-ft: its implementation commit 97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2 unchanged except one trailing 

### w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): CUDA sm_90a dequant-producer warpgroup + SS wgmma grouped GEMM, stage 1b: rebuild with the undefined-symbol fix
- **experts-turn4-ssdq-s1b-dataptr-symbol-fix** → neutral (unproven-vs-best), 1.1004x — gained on sliding turn_4; full turn_4. Re-run of experts-turn4-cuda-dequant-producer-ss-wgmma (job 40541db65f1b4ec6, neutral 1.0714), which NEVER RAN: the log reads '[ssdq] load_inline failed after 3

### w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): CUDA sm_90a warp-specialized grouped GEMM with an in-kernel dequant-producer warpgroup and SS wgmma
- **experts-turn4-cuda-dequant-producer-ss-wgmma** → neutral (within-ci), 1.0714x — no shape moved outside noise. Built on master (a81d19c). At turn_4 the Marlin pair is 2148/2115 us (64%/43% of the sliding/full turn_4 shapes) against a 306 us math floor (~3012 local routes

### w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): ssdq producer weight feed, TMA raw-code smem ring instead of register __ldg prefetch
- **experts-turn4-ssdq-s2-tma-raw-weight-ring** → neutral (unproven-vs-best), 1.1293x — gained on sliding turn_4; full turn_4. Stage 2 of the ssdq line, built on master (experts-turn4-ssdq-s1b-dataptr-symbol-fix landed: ssdq_kernel serves 1024 < T <= 4096). Only the PRODUCER's weight fe

### w13 + w2 expert GEMMs at 200 < T <= 1024 (turn_1, rega low class): two resident CTAs per SM with a 64- or 32-route slab, to hide consumer dequant latency
- **experts-rega-low-class-two-ctas-per-sm** → neutral (unproven-vs-best), 1.4173x — gained on sliding turn_1. Rewrite stage 3 of the rega line, built on master a1a5497d (experts-turn-rega-s4-first-turn-n128). It touches ONLY the 'low' class (200 < T <= REGA_LOW_MAX = 10

### w13 + w2 expert GEMMs at 200 < T <= 1024 (two CTAs/SM V64/V32 port, re-gated on master's random routing with a skew guard) stacked on combo A
- **combo-b-normgather-two-cta-low-regate** → neutral (unproven-vs-best), 1.4677x — no shape moved outside noise. COMBINATION B (conjectural: re-gate the two-CTA/SM low class on master's own gate routing, stacked on A). Start from the same merge as combination A (/home/user

### w13 + w2 expert GEMMs at T <= 4096 (turn_1 T=753, turn_4 T=3012): port of muse_moe_w4a16's replay-safe register-A swap-AB wgmma NVFP4 grouped GEMM (s4), all four N buckets
- **experts-turn-rega-s4-port-t4096** → gain (gain), 1.3275x — gained on sliding turn_1, turn_4; full turn_1, turn_4. Built on master (a81d19c, stack-tonight-five-prefill). The Marlin pair is 768.6/735.9 us at turn_1 (68%/48% of the sliding/full shape) and 2148/2115 us at turn_

### w13 + w2 expert GEMMs at T > 4096 (CUDA sm_90a wgmma W4A16 grouped GEMM, stage 1b: re-measure of experts-cuda-wgmma-mixed-input-grouped-gemm with the build fix, on the new master)
- **experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc** → regression (regression), 0.9401x — lost on full first_turn; sliding first_turn. BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored), not on the old branch: start from master's submission.py and hand-port 

### w13 + w2 expert GEMMs at T > 4096 (first_turn T=16384): extend the ported s4 register-A wgmma engine past Marlin's range, replacing the Triton NVFP4 GEMM
- **experts-turn-rega-s4-first-turn-n128** → gain (gain), 1.4060x — gained on full first_turn; sliding first_turn; lost on sliding turn_1. Stage 2, built on the branch of experts-turn-rega-s4-port-t4096 whatever its score (it carries the ported _wg_* module and its per-state scratch). First_turn is

### w13 + w2 expert GEMMs, every rega class: activation slab loaded once per 2-CTA cluster and shared by TMA multicast
- **experts-rega-mc2-slab-multicast-all-t** → neutral (within-ci), 1.4630x — no shape moved outside noise. DIAGNOSIS action 1. Built on master (combo-b-normgather-two-cta-low-regate code, blob f4751fc0). New kernel rega_mc_kernel<G, BIG> appended to the rega module s

### w13 marlin + w2 marlin (CUDA sm_90a warp-specialized wgmma W4A16 grouped GEMM, stage 1: first_turn)
- **experts-cuda-wgmma-mixed-input-grouped-gemm** → neutral (unproven-vs-best), 1.0637x — no shape moved outside noise. Replace the expert GEMM engine with a hand-written Hopper (sm_90a) mixed-input grouped GEMM in CUDA, compiled with torch.utils.cpp_extension.load_inline at impo

### w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: K-loop load predication and L2 eviction hints)
- **experts-triton-first-turn-unmasked-loads-l2-hints** → neutral (within-ci), 1.0677x — no shape moved outside noise. Built on master (635cfee). Only the K loop's loads in _moe_nvfp4_gemm_kernel change; tiles, configs, grid, dequant, dots and epilogue are unchanged. (1) UNMASKE

### w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: persistent grid, no dead CTAs)
- **experts-triton-persistent-grid-first-turn** → neutral (within-ci), 1.0726x — no shape moved outside noise. Built on master (635cfee). Only _moe_nvfp4_gemm_kernel and its launcher _nvfp4_moe_gemm change; the per-tile math (pointer setup, K loop, in-register dequant, t

### w13 marlin + w2 marlin (Triton NVFP4 engine, stage 5: packed inline-PTX dequant)
- **experts-triton-nvfp4-packed-asm-dequant** → neutral (within-ci), 1.0556x — no shape moved outside noise. Stage 5 of the Triton NVFP4 expert rewrite, built on master (stage 4, experts-hybrid-marlin-turn-triton-first-turn). Only the dequant inside _moe_nvfp4_gemm_ker

### w13 marlin + w2 marlin (Triton NVFP4 engine, stage 6: take over turn_4)
- **experts-triton-packed-dequant-turn4** → neutral (within-ci), 1.0522x — no shape moved outside noise. Stage 6 of the Triton NVFP4 expert rewrite, built on stage 5 (branch hyp/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant, whatever its score). Only t

### w13 marlin + w2 marlin (Triton NVFP4 engine, stage 7: take over turn_1 with a weight-stream config)
- **experts-triton-turn1-blockt32-stream** → regression (regression), 0.9425x — lost on sliding turn_1; full turn_1. Stage 7 of the Triton NVFP4 expert rewrite, built on stage 6 (branch hyp/muse_prefill_layer/experts-triton-packed-dequant-turn4, whatever its score). Two edits 

### w13 marlin + w2 marlin (W4A16 Triton engine, stage 3: first_turn token tile 256)
- **experts-triton-w4a16-first-turn-blockt256** → neutral (unproven-vs-best), 1.0229x — gained on full first_turn; sliding first_turn; lost on sliding turn_4. Stage 3 of the Triton NVFP4 grouped GEMM, built on stage 2 (branch hyp/muse_prefill_layer/experts-triton-w4a16-swapab-reglhs, _moe_nvfp4_gemm_kernel and EXPERT_

### w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, for the turn shapes)
- **experts-triton-w4a16-inkernel-fp4** → regression (regression), 0.6891x — lost on sliding first_turn, turn_1, turn_4; full first_turn, turn_1, turn_4. A Triton grouped-GEMM kernel with in-register NVFP4 dequant replaces both moe_wna16_marlin_gemm calls wherever master still runs Marlin (if a dequant-to-BF16 pa

### w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, stage 2: swap-AB register-LHS dot)
- **experts-triton-w4a16-swapab-reglhs** → neutral (unproven-vs-best), 1.0068x — gained on sliding first_turn, turn_1, turn_4; full first_turn, turn_1, turn_4. Stage 2 of the Triton NVFP4 grouped GEMM, built on stage 1 (branch hyp/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4, _moe_nvfp4_gemm_kernel in muse_pref

### w13 marlin + w2 marlin (block_size_m pick)
- **experts-marlin-block-m-local-rows** → regression (regression), 0.9836x — lost on sliding turn_1; full turn_1. Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py). The parity loop walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and breaks o
- **experts-marlin-block-m-48-turn1** → neutral (within-ci), 1.0030x — no shape moved outside noise. Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 512 < num_tokens <= 1024 use block_size_m = 48; every other token count k

### w13 marlin + w2 marlin (stage 4: hybrid shape gate, Marlin at T<=4096, Triton NVFP4 above)
- **experts-hybrid-marlin-turn-triton-first-turn** → gain (gain), 1.0599x — gained on sliding turn_1, turn_4; full turn_1. Stage 4 of the Triton NVFP4 expert rewrite, built on master (a1f93c2, = stage 3 experts-triton-w4a16-first-turn-blockt256). Only prepare() and _experts in muse_

### w2 (Triton NVFP4 engine at T > 4096: remote-block zero store)
- **w2-nonlocal-zero-store-off-first-turn** → neutral (unproven-vs-best), 1.0609x — no shape moved outside noise. Built on master (635cfee). One flag and one prepare() guard. Today ZERO_NONLOCAL_W2 = True makes the second _nvfp4_moe_gemm call in _experts_triton (T > 4096 on

### whole-layer parity (SGLang pipeline) + routing remap
- **sglang-pipeline-parity** → gain (gain), 1.0039x — no shape moved outside noise. Rewrite kernel_fn and prepare in muse_prefill_layer/submission.py to run SGLang's own pipeline module for module as baseline.py does: _bf16_gemm_dispatch_impl f

## Open follow-ups

- experts-triton-w4a16-swapab-reglhs: rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-triton-w4a16-first-turn-blockt256: rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-triton-nvfp4-packed-asm-dequant: rewrite stage 5 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-triton-packed-dequant-turn4: rewrite stage 6 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-cuda-wgmma-mixed-input-grouped-gemm: rewrite stage 1 of `_moe_nvfp4_gemm_kernel x2 (w13, w2) at T > 4096 in stage 1; moe_wna16_marlin_gemm x2 at T <= 4096 in stages 2-3`: read the replaced kernels' own us per shape (profile top_kernels) against the incumben
- sliding-attn-block-triton-gqa-fused-normgate: rewrite stage 1 of `q.view(T,16,64).to(FP8) + flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) + state['o_norm_gate'] (FusedRMSNormGate) on sliding layers (cfg['window'] == 2048)`: r
- attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc: rewrite stage 1 of `flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) on full-attention layers at page_size 1`: read the replaced kernels' own us per shape (profile top_kernels) again
- experts-turn4-cuda-dequant-producer-ss-wgmma, experts-turn4-ssdq-s1b-dataptr-symbol-fix: rewrite stage 1 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) at 1024 < T <= 4096 (turn_4, T=3012)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which
- experts-turn4-ssdq-s2-tma-raw-weight-ring: rewrite stage 2 of `ssdq_kernel x2 (w13, w2) at 1024 < T <= 4096 (turn_4, T=3012)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reac
- experts-rega-low-class-two-ctas-per-sm: rewrite stage 3 of `rega_kernel<true,false> + rega_kernel<false,false> (NMAX 128, 4 stages, 1 CTA/SM) in the low class 200 < T <= 1024`: read the replaced kernels' own us per shape (profile top_kernels) against the incum
- experts-rega-s4-bm256-wide-tile-high-ft: rewrite stage 4 of `rega_kernel<true,*> + rega_kernel<false,*> (128-row tile) in the high (T=3012) and ft (T=16384) classes`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and th
- experts-rega-mc2-slab-multicast-all-t: rewrite stage 1 of `rega_kernel<true,BIG> + rega_kernel<false,BIG> (w13, w2) in all three rega classes: low T<=1024, high 1024<T<=4096, ft 4096<T<=16384`: read the replaced kernels' own us per shape (profile top_kernels)

## Do not try (regressions, flat results, rejected)

Rewrite stages are exempt: a slow early stage is not a dead end.

- **experts-marlin-block-m-48-turn1** [w13 marlin + w2 marlin (block_size_m pick)] neutral, within-ci: Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 512
- **experts-triton-w13-fused-silu-epilogue** [silu-mul (fused into the w13 Triton NVFP4 GEMM epilogue, T > 4096)] neutral, within-ci: Fuse SGLang's silu_and_mul into the epilogue of the first (w13) Triton NVFP4 GEMM on the T
- **silu-mul-triton-local-rows-marlin-turn** [silu-mul (Marlin path, T <= 4096)] neutral, within-ci: In _experts_marlin only (the Marlin path, T <= MARLIN_MAX_TOKENS = 4096: turn_1 T=753, tur
- **experts-triton-persistent-grid-first-turn** [w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: persistent grid, no dead CTAs)] neutral, within-ci: Only _moe_nvfp4_gemm_kernel and its launcher _nvfp4_moe_gemm change; the per-tile math (po
- **norm-boundaries-persistent-multirow** [attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2, persistent multi-row variant)] neutral, within-ci: The multi-row/persistent re-mint the review of the held norm-residual-boundaries-triton-ro
- **moe-pre-post-norm-triton-multitoken** [pre-expert norm + post-expert norm/combine (the MoE elementwise ends)] neutral, within-ci: Two kernels at the ends of the MoE block, at every T. (1) PRE-EXPERT NORM: replace state['
- **experts-triton-first-turn-unmasked-loads-l2-hints** [w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: K-loop load predication and L2 eviction hints)] neutral, within-ci: Only the K loop's loads in _moe_nvfp4_gemm_kernel change; tiles, configs, grid, dequant, d
- **align-local-parallel-first-turn** [MoE align on the Triton path (T > 4096, first_turn)] neutral, within-ci: Built on moe-align-local-routes-parallel-port (running, its implementation commit). In _ex
- **silu-mul-sorted-first-turn** [silu-mul on the Triton path (T > 4096, first_turn)] neutral, within-ci: Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_trito
- **experts-triton-turn4-persistent-local-align-control-no-persi** [w13 + w2 at turn_4 (1024 < T <= 4096) routed to the Triton NVFP4 engine with local-routes align, original non-persistent grid] neutral, within-ci: Implement only parts (1) and (3) of the parent. (1) In _experts, send 1024 < T <= 4096 to 
- **turn-align-remap-on-side-stream-under-pooler** [MoE align/dispatch placement at T <= 4096: local-id remap + _align_parallel moved onto the router side stream, under the pooler GEMM] neutral, within-ci: In the best profile master's _align_parallel_kernel runs serially on the main stream after
- **turn-align-sglang32-prenorm-triton-restack** [MoE align on the Marlin path (T <= 4096) + pre-expert norm: restack of two measured row wins] neutral, within-ci: Two row-level wins already measured in the archive, applied together, shapes untouched oth
- **experts-rega-pingpong-consumers-all-t** [w13 + w2 expert GEMMs (rega_kernel consumer schedule: two-warpgroup ping-pong), every rega class] neutral, within-ci: Consumer schedule of rega_kernel only. ARITHMETIC FIRST (the judge's router seed fixes the
- **experts-rega-producer-scale-expand-all-t** [w13 + w2 expert GEMMs (rega_kernel dequant: E4M3 scale expansion moved to idle producer warps), every rega class] neutral, within-ci: Consumer instruction count of rega_kernel only. ARITHMETIC: turn_1 (~1.36K local routes, ~
- **qkv-gate-columns-side-stream-under-attention** [qkv GEMM (output-gate columns split onto the side stream, overlapping qk_norm + cache store + FA3)] neutral, within-ci: Built on current master (submission blob a2fe2a75; no depends_on). The qkv_gate projection
- **experts-rega-pdl-chain-gather-w13-w2** [launch overhead: PDL chain across the expert block (_norm_gather_kernel -> rega w13 -> rega w2)] neutral, within-ci: DIAGNOSIS action 4. Built on master (blob f4751fc0); independent of the mc2 line. Prefill 
- **experts-rega-round-rotated-schedule** [w13 + w2 expert GEMMs (rega persistent item->CTA schedule)] neutral, within-ci: Only the order in which rega_kernel's persistent CTAs walk items changes; every item is co
- **experts-rega-fine-n-buckets** [w13 + w2 expert GEMMs (rega wgmma N buckets at 16-row granularity)] neutral, within-ci: rega sizes each item's wgmma N (tokens) and its slab TMA box from the chunk's 16-route blo
- **fa3-contiguous-kv-gather** [FA3 call] regression, regression: Replace the paged FA3 call with a contiguous one; everything before and after stays the pa
- **experts-marlin-block-m-local-rows** [w13 marlin + w2 marlin (block_size_m pick)] regression, regression: Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py). The pari

## Queue and in flight

- queue: fa3-num-splits-cap-gated-turn [FA3 call (num_splits cap on the paged path at q_len <= 4096, per-class timing gate)] p3 hold
- queue: experts-rega-low-2cta-fullslab-mc2 [w13 + w2 expert GEMMs at 200 < T <= 1024 (rega low class, turn_1): 2 CTAs/SM with the full 128-route slab, slab multicast across a 2-CTA cluster] p4 queued
- queue: experts-rega-mc2-bm256-stack-high-ft [w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft): 256-row wide tile x 2-CTA slab multicast, stacked] p3 queued
- queue: experts-rega-mc4-cluster-high-ft [w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft): activation slab shared across a 4-CTA cluster by TMA multicast] p5 hold
- queue: experts-rega-rotate-finebucket-fastsilu-combo [w13 + w2 expert GEMMs (rega: rotated schedule + 16-row N buckets + fast silu, one module)] p2 hold
- queue: experts-rega-rotation-fastsilu-combo [w13 + w2 expert GEMMs (rega: rotated schedule + fast silu, one module)] p2 hold
- queue: experts-rega-pdl-chain-regate-iqr-slack [launch overhead (PDL chain _norm_gather_kernel -> rega w13 -> rega w2: gate rule only)] p3 queued
- queue: front-qknorm-rope-fp8-store-fusion-restack [pre-attention norm+rope + cache store (five launches -> one Triton kernel)] p2 queued
- queue: experts-rega-lpt-group-order [w13 + w2 expert GEMMs (rega persistent schedule: expert groups heaviest first)] p4 queued
- queue: experts-rega-dynamic-item-scheduler [w13 + w2 expert GEMMs (rega persistent schedule: atomic work counter)] p3 queued
- queue: experts-rega-stage-anatomy-diag [w13 + w2 expert GEMMs (prepare-only diagnostic: where a rega stage goes)] p5 queued
- queue: front-routing-pdl-launch-diet-stack [launch overhead (front fusion + routing fusion + PDL chain regate, stacked)] p4 queued
- queue: routing-triton-hilo-sum-topk-localids [routing top-k (side-stream logits sum + FusedMuseSparkRouting + local-id remap in one Triton kernel)] p3 queued
- queue: experts-rega-fine-n-high-plus-fast-silu [rega experts kernel (w13/w2 N buckets at the high class + w13 silu epilogue)] p1 hold
- queue: experts-rega-w2-epilogue-prefetch-bulk-store [w13 + w2 expert GEMMs (rega w2 consumer epilogue: loads and stores)] p1 queued
- queue: experts-rega-epilogue-wg-local-bulk-store [w13 + w2 expert GEMMs (rega consumer epilogue: barriers and store path, both GEMMs)] p2 queued
- queue: o-proj-triton-gemm-turn-gated [o GEMM (attention output projection at 200 < T <= 4096)] p2 queued
- queue: experts-rega-epilogue-overlap-next-stage [w13 + w2 expert GEMMs (rega consumer schedule across item boundaries: epilogue under the next item's first wgmma)] p3 queued
- running: attn-full-prefill-wgmma-s2-warpspec-pingpong-expoffload [FA3 call on full-attention layers at page_size 1 (CUDA sm_90a FP8 wgmma flash attention, stage 2: warp-specialized producer, 128-key tiles, intra-WG overlap + ping-pong, gated exp offload)] p2 running
- 22 stale holds expired to `done/expired/` (not run)
