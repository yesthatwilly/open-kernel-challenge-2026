# muse_prefill_layer findings

Regenerated 2026-10-25T131916Z by `churn.py index`. Read this before writing a hypothesis for muse_prefill_layer.

## Current best

1.4060x [1.3849,1.4271] from job `07a069bc69ba42a8` (check), master commit `a1a5497d7a`, hypothesis experts-turn-rega-s4-first-turn-n128

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6: yours 777.0µs vs SGLang 1074.7µs (SoL 257.0µs) · void (anonymous namespace)::rega_kernel<true, false>((ano... 282.4µs 35.5%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 157.4µs 19.8%, void (anonymous namespace)::rega_kernel<false, false>((an... 146.0µs 18.4%
- batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6: yours 1136.1µs vs SGLang 1451.7µs (SoL 496.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 418.9µs 35.2%, void (anonymous namespace)::rega_kernel<true, false>((ano... 257.8µs 21.6%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 154.0µs 12.9%
- batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6: yours 2251.4µs vs SGLang 3405.1µs (SoL 1036.3µs) · void (anonymous namespace)::rega_kernel<true, false>((ano... 731.8µs 32.2%, void (anonymous namespace)::rega_kernel<false, false>((an... 356.8µs 15.7%, nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT 331.0µs 14.6%
- batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6: yours 3785.2µs vs SGLang 4862.3µs (SoL 1992.4µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 1638.1µs 43.0%, void (anonymous namespace)::rega_kernel<true, false>((ano... 713.0µs 18.7%, void (anonymous namespace)::rega_kernel<false, false>((an... 346.9µs 9.1%
- batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 12148.3µs vs SGLang 17549.5µs (SoL 5518.4µs) · void (anonymous namespace)::rega_kernel<true, true>((anon... 3514.4µs 29.0%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2583.6µs 21.3%, void (anonymous namespace)::rega_kernel<false, true>((ano... 1724.5µs 14.2%
- batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6: yours 11658.8µs vs SGLang 17356.5µs (SoL 5663.4µs) · void (anonymous namespace)::rega_kernel<true, true>((anon... 3743.9µs 31.4%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2581.8µs 21.6%, void (anonymous namespace)::rega_kernel<false, true>((ano... 1849.2µs 15.5%
- batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 15589.4µs vs SGLang 20949.2µs (SoL 7710.0µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 4501.4µs 29.8%, void (anonymous namespace)::rega_kernel<true, true>((anon... 3086.6µs 20.5%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2576.2µs 17.1%
- batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 19167.4µs vs SGLang 24730.3µs (SoL 9847.7µs) · void cutlass::device_kernel<flash::enable_sm90_or_later<f... 8516.6µs 42.4%, void (anonymous namespace)::rega_kernel<true, true>((anon... 3688.7µs 18.4%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2615.8µs 13.0%

## By area

### FA3 call
- **fa3-contiguous-kv-gather** → regression (regression), 0.9879x — lost on full turn_1, turn_4. Replace the paged FA3 call with a contiguous one; everything before and after stays the parity path. After store_cache (so the new tokens' rows are in the cache

### FA3 call (CUDA sm_90a FP8 prefill flash attention with software-exp offload, stage 1: full layers, page size 1)
- **attn-full-prefill-cuda-fp8-exp-offload** → regression (regression), 0.8813x — lost on full first_turn, turn_1, turn_4. Replace FA3 (flash_attn_with_kvcache and its FlashAttnFwdCombine) on full-attention prefill layers with a hand-written sm_90a FP8 flash-attention forward kernel

### FA3 call on full-attention layers at page_size 1 (CUDA sm_90a FP8 wgmma flash attention, stage 1b: fix of attn-full-prefill-cuda-fp8-exp-offload with build fix, hard timing gate, no exp offload)
- **attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc** → neutral (within-ci), 1.0679x — no shape moved outside noise. BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored; its attention code is unchanged from the old base): start from master's 

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

### MoE align/dispatch placement at T <= 4096: local-id remap + _align_parallel moved onto the router side stream, under the pooler GEMM
- **turn-align-remap-on-side-stream-under-pooler** → neutral (within-ci), 1.0683x — no shape moved outside noise. Built on master (a81d19c). In the best profile master's _align_parallel_kernel runs serially on the main stream after the router join: 16.9 us at turn_1 (1.7%) 

### attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2, persistent multi-row variant)
- **norm-boundaries-persistent-multirow** → neutral (within-ci), 1.0716x — no shape moved outside noise. Built on master (635cfee). The multi-row/persistent re-mint the review of the held norm-residual-boundaries-triton-row-kernel invited ('same one-row-per-program

### pre-attention norm+rope + cache store
- **qk-norm-fp8-cache-store-fusion** → neutral (unproven-vs-best), 1.0093x — no shape moved outside noise. One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8), v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per t

### pre-expert norm + post-expert norm/combine (the MoE elementwise ends)
- **moe-pre-post-norm-triton-multitoken** → neutral (within-ci), 1.0714x — no shape moved outside noise. Built on master (635cfee). Two kernels at the ends of the MoE block, at every T. (1) PRE-EXPERT NORM: replace state['pre_expert_norm'](expert_input) (MuseSpark1

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

### silu-mul on the Triton path (T > 4096, first_turn)
- **silu-mul-sorted-first-turn** → neutral (within-ci), 1.0636x — no shape moved outside noise. Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_triton, when state['silu_sorted_ok'], replace silu_and_mul(cache1, cache2, 

### sliding-window attention block (q FP8 cast + FA3 + FlashAttnFwdCombine + FusedRMSNormGate) on sliding layers
- **sliding-attn-block-triton-gqa-fused-normgate** → neutral (within-ci), 1.0507x — no shape moved outside noise. Built on master alone (b92a0f4, experts-hybrid-marlin-turn-triton-first-turn); touches nothing the two in-flight prefill runs touch (full-layer FA3 stays FA3, e

### w13 + w2 at turn_4 (1024 < T <= 4096) routed to the Triton NVFP4 engine with local-routes align, original non-persistent grid
- **experts-triton-turn4-persistent-local-align-control-no-persi** → neutral (within-ci), 1.0557x — no shape moved outside noise. Built on master (a81d19c). Implement only parts (1) and (3) of the parent. (1) In _experts, send 1024 < T <= 4096 to _experts_triton (set MARLIN_MAX_TOKENS = 10

### w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): CUDA sm_90a dequant-producer warpgroup + SS wgmma grouped GEMM, stage 1b: rebuild with the undefined-symbol fix
- **experts-turn4-ssdq-s1b-dataptr-symbol-fix** → neutral (unproven-vs-best), 1.1004x — gained on sliding turn_4; full turn_4. Re-run of experts-turn4-cuda-dequant-producer-ss-wgmma (job 40541db65f1b4ec6, neutral 1.0714), which NEVER RAN: the log reads '[ssdq] load_inline failed after 3

### w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): CUDA sm_90a warp-specialized grouped GEMM with an in-kernel dequant-producer warpgroup and SS wgmma
- **experts-turn4-cuda-dequant-producer-ss-wgmma** → neutral (within-ci), 1.0714x — no shape moved outside noise. Built on master (a81d19c). At turn_4 the Marlin pair is 2148/2115 us (64%/43% of the sliding/full turn_4 shapes) against a 306 us math floor (~3012 local routes

### w13 + w2 expert GEMMs at 1024 < T <= 4096 (turn_4): ssdq producer weight feed, TMA raw-code smem ring instead of register __ldg prefetch
- **experts-turn4-ssdq-s2-tma-raw-weight-ring** → neutral (unproven-vs-best), 1.1293x — gained on sliding turn_4; full turn_4. Stage 2 of the ssdq line, built on master (experts-turn4-ssdq-s1b-dataptr-symbol-fix landed: ssdq_kernel serves 1024 < T <= 4096). Only the PRODUCER's weight fe

### w13 + w2 expert GEMMs at T <= 4096 (turn_1 T=753, turn_4 T=3012): port of muse_moe_w4a16's replay-safe register-A swap-AB wgmma NVFP4 grouped GEMM (s4), all four N buckets
- **experts-turn-rega-s4-port-t4096** → gain (gain), 1.3275x — gained on sliding turn_1, turn_4; full turn_1, turn_4. Built on master (a81d19c, stack-tonight-five-prefill). The Marlin pair is 768.6/735.9 us at turn_1 (68%/48% of the sliding/full shape) and 2148/2115 us at turn_

### w13 + w2 expert GEMMs at T > 4096 (CUDA sm_90a wgmma W4A16 grouped GEMM, stage 1b: re-measure of experts-cuda-wgmma-mixed-input-grouped-gemm with the build fix, on the new master)
- **experts-cuda-wgmma-grouped-gemm-remeasure-sm90a-inproc** → regression (regression), 0.9401x — lost on full first_turn; sliding first_turn. BUILDS ON THE CURRENT MASTER (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored), not on the old branch: start from master's submission.py and hand-port 

### w13 + w2 expert GEMMs at T > 4096 (first_turn T=16384): extend the ported s4 register-A wgmma engine past Marlin's range, replacing the Triton NVFP4 GEMM
- **experts-turn-rega-s4-first-turn-n128** → gain (gain), 1.4060x — gained on full first_turn; sliding first_turn; lost on sliding turn_1. Stage 2, built on the branch of experts-turn-rega-s4-port-t4096 whatever its score (it carries the ported _wg_* module and its per-state scratch). First_turn is

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

- moe-align-side-stream: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: moe-align-side-stream`) that adds a further change in the same area is the only way to learn more (identical code is never re-run)
- qk-norm-fp8-cache-store-fusion: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: qk-norm-fp8-cache-store-fusion`) that adds a further change in the same area is the only way to learn more (identical code is neve
- experts-marlin-block-m-48-turn1: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: experts-marlin-block-m-48-turn1`) that adds a further change in the same area is the only way to learn more (identical code is never re-r
- experts-triton-w4a16-swapab-reglhs: rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-triton-w4a16-first-turn-blockt256: rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-triton-nvfp4-packed-asm-dequant: rewrite stage 5 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-triton-w13-fused-silu-epilogue: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: experts-triton-w13-fused-silu-epilogue`) that adds a further change in the same area is the only way to learn more (identical code is nev
- experts-triton-packed-dequant-turn4: rewrite stage 6 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- silu-mul-triton-local-rows-marlin-turn: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: silu-mul-triton-local-rows-marlin-turn`) that adds a further change in the same area is the only way to learn more (identical code is nev
- experts-cuda-wgmma-mixed-input-grouped-gemm: rewrite stage 1 of `_moe_nvfp4_gemm_kernel x2 (w13, w2) at T > 4096 in stage 1; moe_wna16_marlin_gemm x2 at T <= 4096 in stages 2-3`: read the replaced kernels' own us per shape (profile top_kernels) against the incumben
- sliding-attn-block-triton-gqa-fused-normgate: rewrite stage 1 of `q.view(T,16,64).to(FP8) + flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) + state['o_norm_gate'] (FusedRMSNormGate) on sliding layers (cfg['window'] == 2048)`: r
- experts-triton-persistent-grid-first-turn: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: experts-triton-persistent-grid-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is 
- moe-align-local-routes-parallel-port: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: moe-align-local-routes-parallel-port`) that adds a further change in the same area is the only way to learn more (identical code i
- w2-nonlocal-zero-store-off-first-turn: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: w2-nonlocal-zero-store-off-first-turn`) that adds a further change in the same area is the only way to learn more (identical code 
- moe-align-32-experts-first-turn: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: moe-align-32-experts-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is nev
- router-hilo-one-gemm-n512: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: router-hilo-one-gemm-n512`) that adds a further change in the same area is the only way to learn more (identical code is never re-
- norm-boundaries-persistent-multirow: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: norm-boundaries-persistent-multirow`) that adds a further change in the same area is the only way to learn more (identical code is never 
- moe-pre-post-norm-triton-multitoken: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: moe-pre-post-norm-triton-multitoken`) that adds a further change in the same area is the only way to learn more (identical code is never 
- experts-triton-first-turn-unmasked-loads-l2-hints: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: experts-triton-first-turn-unmasked-loads-l2-hints`) that adds a further change in the same area is the only way to learn more (identical 
- restack-router-hilo-align32-zerostore: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: restack-router-hilo-align32-zerostore`) that adds a further change in the same area is the only way to learn more (identical code 
- stack-tonight-four-prefill: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-tonight-four-prefill`) that adds a further change in the same area is the only way to learn more (identical code is never re
- align-local-parallel-first-turn: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: align-local-parallel-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is never re-r
- stack-tonight-five-prefill: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-tonight-five-prefill`) that adds a further change in the same area is the only way to learn more (identical code is never re
- stack-tonight-six-prefill: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-tonight-six-prefill`) that adds a further change in the same area is the only way to learn more (identical code is never re-
- silu-mul-sorted-first-turn: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: silu-mul-sorted-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).
- attn-full-prefill-cuda-fp8-wgmma-hard-gate-inproc: rewrite stage 1 of `flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) on full-attention layers at page_size 1`: read the replaced kernels' own us per shape (profile top_kernels) again
- experts-triton-turn4-persistent-local-align-control-no-persi: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: experts-triton-turn4-persistent-local-align-control-no-persi`) that adds a further change in the same area is the only way to learn more 
- experts-turn4-cuda-dequant-producer-ss-wgmma: rewrite stage 1 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) at 1024 < T <= 4096 (turn_4, T=3012)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which
- turn-align-remap-on-side-stream-under-pooler: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: turn-align-remap-on-side-stream-under-pooler`) that adds a further change in the same area is the only way to learn more (identical code 
- experts-turn4-ssdq-s1b-dataptr-symbol-fix: rewrite stage 1 of `moe_wna16_marlin_gemm x2 (Marlin w13 + w2) at 1024 < T <= 4096 (turn_4, T=3012)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which
- turn-align-sglang32-prenorm-triton-restack: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: turn-align-sglang32-prenorm-triton-restack`) that adds a further change in the same area is the only way to learn more (identical code is
- experts-turn4-ssdq-s2-tma-raw-weight-ring: rewrite stage 2 of `ssdq_kernel x2 (w13, w2) at 1024 < T <= 4096 (turn_4, T=3012)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reac
- turn-align-two-phase-chunk-prefix: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: turn-align-two-phase-chunk-prefix`) that adds a further change in the same area is the only way to learn more (identical code is n

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: experts-dequant-grouped-mm-cutlass [w13 marlin + w2 marlin (GEMM engine: CUTLASS grouped GEMM instead of Triton)] p4 hold
- queue: experts-dequant-overlap-attention [w13 marlin + w2 marlin (dequant placement on a side stream)] p3 hold
- queue: experts-dequant-threshold-turn4 [w13 marlin + w2 marlin (Marlin vs dequant crossover)] p4 hold
- queue: experts-dequant-triton-moe-16k [w13 marlin + w2 marlin (expert grouped GEMM engine at 16K tokens)] p2 hold
- queue: fa3-batched-mode [FA3 call] p4 hold
- queue: experts-marlin-block-m-48-turn4 [w13 marlin + w2 marlin (block_size_m pick)] p3 hold
- queue: router-triton-hilo-splitk-turn [router GEMM] p3 hold
- queue: router-pooler-concat-gemm-turn [router GEMM + pooler GEMM (one concatenated GEMM, no side stream)] p5 hold
- queue: dense-gemm-cublaslt-algo-pick-prefill-t [dense bf16 GEMM dispatch (qkv, o, pooler, unpooler; cuBLASLt algo selection)] p2 hold
- queue: norm-residual-boundaries-triton-row-kernel [attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2)] p4 hold
- queue: first-turn-moe-dead-work-stack [MoE dispatch + silu-mul + Triton NVFP4 grid at T > 4096 (first_turn dead-work removal, combined)] p3 hold
- queue: experts-triton-turn4-persistent-local-align [w13 marlin + w2 marlin at turn_4 (route 1024 < T <= 4096 to the Triton NVFP4 engine, with persistent grid and local-routes align)] p2 hold
- queue: turn-moe-local-align-sorted-silu [MoE align/dispatch + silu-mul on the Marlin path (T <= 4096)] p3 hold
- queue: attn-full-prefill-cuda-fp8-wgmma-hard-gate [FA3 call on full-attention layers at page_size 1 (CUDA sm_90a FP8 wgmma flash attention, stage 1b: fix of attn-full-prefill-cuda-fp8-exp-offload with build fix, hard timing gate, no exp offload)] p2 hold
- queue: experts-cuda-wgmma-grouped-gemm-remeasure-sm90a [w13 + w2 expert GEMMs at T > 4096 (CUDA sm_90a wgmma W4A16 grouped GEMM, stage 1b: re-measure of experts-cuda-wgmma-mixed-input-grouped-gemm with the build fix, on the new master)] p1 hold
- queue: experts-cuda-wgmma-s2-setmaxnreg-pingpong-first-turn [w13 + w2 expert GEMMs at T > 4096 (CUDA sm_90a wgmma W4A16 grouped GEMM, stage 2: setmaxnreg register split + overlapped register-A dequant)] p2 hold
- queue: experts-triton-first-turn-per-gemm-config-pick [Triton NVFP4 engine at T > 4096 (first_turn): separate tile config for w13 and w2, picked by prepare timing] p3 hold
- queue: experts-turn1-cuda-wgmma-rega-port-bucket32 [w13 + w2 expert GEMMs at T <= 1024 (turn_1): port of muse_moe_w4a16's register-A swap-AB wgmma NVFP4 grouped GEMM, 16/32-row buckets only] p1 hold
- queue: marlin-w2-atomic-add-turn [w2 Marlin call on the Marlin path (T <= 4096): use_atomic_add, picked per shape class in prepare] p4 hold
- queue: experts-first-turn-ssdq-s2-large-m [w13 + w2 expert GEMMs at T > 4096 (first_turn T=16384): extend the ssdq dequant-producer + SS wgmma grouped GEMM to large M, replacing the Triton NVFP4 GEMM] p2 hold
- queue: experts-turn-rega-s4-w2-only-t4096 [w2 expert GEMM only at T <= 4096 (turn_1, turn_4): s4 register-A swap-AB wgmma w2 fed by master's Marlin w13 + _silu_mul_sorted output] p2 hold
- queue: experts-turn1-ssdq-s2-m64-tile [w13 + w2 expert GEMMs at T <= 1024 (turn_1 T=753): ssdq with a 64-route M tile, one consumer warpgroup per tile, dequant-once per weight stage] p3 hold
- running: align-two-phase-all-t-rega [MoE align/dispatch on the rega path (200 < T <= 16384): _align_parallel replaced by the two-phase chunk-count + prefix-place align, including the first_turn class] p1 running
- running: experts-rega-low-class-two-ctas-per-sm [w13 + w2 expert GEMMs at 200 < T <= 1024 (turn_1, rega low class): two resident CTAs per SM with a 64- or 32-route slab, to hide consumer dequant latency] p1 running
