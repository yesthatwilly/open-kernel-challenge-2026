# muse_prefill_layer findings

Regenerated 2026-10-25T000226Z by `churn.py index`. Read this before writing a hypothesis for muse_prefill_layer.

## Current best

1.0592x [1.0433,1.0751] from job `ba9c16b9cc6b4605` (submit), master commit `b92a0f4719`, hypothesis experts-hybrid-marlin-turn-triton-first-turn

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6: yours 1079.5µs vs SGLang 1088.2µs (SoL 257.0µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 752.4µs 67.4%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 154.0µs 13.8%, nvjet_sm90_tst_112x128_64x7_1x2_h_bz_TNN 39.0µs 3.5%
- batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6: yours 1419.1µs vs SGLang 1431.0µs (SoL 496.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 653.9µs 45.0%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 415.2µs 28.6%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 150.7µs 10.4%
- batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6: yours 3334.9µs vs SGLang 3340.9µs (SoL 1036.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 2217.1µs 64.3%, nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT 322.2µs 9.3%, nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_TNN 253.1µs 7.3%
- batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6: yours 4938.0µs vs SGLang 4952.8µs (SoL 1992.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 2131.0µs 43.0%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 1634.0µs 33.0%, nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT 319.2µs 6.4%
- batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 15386.3µs vs SGLang 18586.1µs (SoL 5518.4µs) · _moe_nvfp4_gemm_kernel 8873.3µs 57.4%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2499.4µs 16.2%, nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT 1366.2µs 8.8%
- batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6: yours 14275.3µs vs SGLang 17028.2µs (SoL 5663.4µs) · _moe_nvfp4_gemm_kernel 8323.7µs 58.1%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2502.5µs 17.5%, nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT 1365.9µs 9.5%
- batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 18165.9µs vs SGLang 20628.6µs (SoL 7710.0µs) · _moe_nvfp4_gemm_kernel 8338.7µs 44.8%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 4471.1µs 24.0%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2501.3µs 13.4%
- batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 22332.9µs vs SGLang 24889.1µs (SoL 9847.7µs) · _moe_nvfp4_gemm_kernel 8149.4µs 37.1%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 8006.7µs 36.5%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2497.3µs 11.4%

## By area

### FA3 call
- **fa3-contiguous-kv-gather** → regression (regression), 0.9879x — lost on full turn_1, turn_4. Replace the paged FA3 call with a contiguous one; everything before and after stays the parity path. After store_cache (so the new tokens' rows are in the cache

### FA3 call (CUDA sm_90a FP8 prefill flash attention with software-exp offload, stage 1: full layers, page size 1)
- **attn-full-prefill-cuda-fp8-exp-offload** → regression (regression), 0.8813x — lost on full first_turn, turn_1, turn_4. Replace FA3 (flash_attn_with_kvcache and its FlashAttnFwdCombine) on full-attention prefill layers with a hand-written sm_90a FP8 flash-attention forward kernel

### MoE align on the Triton path (T > 4096, first_turn)
- **align-local-parallel-first-turn** → neutral (within-ci), 1.0626x — no shape moved outside noise. Built on moe-align-local-routes-parallel-port (running, its implementation commit). In _experts_triton, when state['align_par_ok'], replace moe_align_block_size

### MoE align/dispatch
- **moe-align-side-stream** → neutral (unproven-vs-best), 1.0076x — no shape moved outside noise. Keep the parity base and move three pieces of work that depend only on the routing outputs and local_start onto the existing side stream, inside the same torch.

### MoE align/dispatch (align over the 32 local experts at T > 4096)
- **moe-align-32-experts-first-turn** → neutral (unproven-vs-best), 1.0604x — no shape moved outside noise. Built on master (635cfee). One argument in _experts_triton (T > 4096 only): moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) becomes moe_align_block_s

### MoE align/dispatch (local-routes-only align, port of muse_moe_w4a16 align-parallel-per-expert-verify)
- **moe-align-local-routes-parallel-port** → neutral (unproven-vs-best), 1.0658x — no shape moved outside noise. Built on master (635cfee). Port muse_moe_w4a16's measured align (master there: _align_parallel_kernel and its host wrapper _align_parallel in muse_moe_w4a16/sub

### pre-attention norm+rope + cache store
- **qk-norm-fp8-cache-store-fusion** → neutral (unproven-vs-best), 1.0093x — no shape moved outside noise. One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8), v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per t

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

### sliding-window attention block (q FP8 cast + FA3 + FlashAttnFwdCombine + FusedRMSNormGate) on sliding layers
- **sliding-attn-block-triton-gqa-fused-normgate** → neutral (within-ci), 1.0507x — no shape moved outside noise. Built on master alone (b92a0f4, experts-hybrid-marlin-turn-triton-first-turn); touches nothing the two in-flight prefill runs touch (full-layer FA3 stays FA3, e

### w13 marlin + w2 marlin (CUDA sm_90a warp-specialized wgmma W4A16 grouped GEMM, stage 1: first_turn)
- **experts-cuda-wgmma-mixed-input-grouped-gemm** → neutral (unproven-vs-best), 1.0637x — no shape moved outside noise. Replace the expert GEMM engine with a hand-written Hopper (sm_90a) mixed-input grouped GEMM in CUDA, compiled with torch.utils.cpp_extension.load_inline at impo

### w13 marlin + w2 marlin (Triton NVFP4 engine, stage 5: packed inline-PTX dequant)
- **experts-triton-nvfp4-packed-asm-dequant** → neutral (within-ci), 1.0556x — no shape moved outside noise. Stage 5 of the Triton NVFP4 expert rewrite, built on master (stage 4, experts-hybrid-marlin-turn-triton-first-turn). Only the dequant inside _moe_nvfp4_gemm_ker

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
- silu-mul-triton-local-rows-marlin-turn: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: silu-mul-triton-local-rows-marlin-turn`) that adds a further change in the same area is the only way to learn more (identical code is nev
- experts-cuda-wgmma-mixed-input-grouped-gemm: rewrite stage 1 of `_moe_nvfp4_gemm_kernel x2 (w13, w2) at T > 4096 in stage 1; moe_wna16_marlin_gemm x2 at T <= 4096 in stages 2-3`: read the replaced kernels' own us per shape (profile top_kernels) against the incumben
- sliding-attn-block-triton-gqa-fused-normgate: rewrite stage 1 of `q.view(T,16,64).to(FP8) + flash_attn_with_kvcache (FA3 cutlass::device_kernel<flash::...> + FlashAttnFwdCombine) + state['o_norm_gate'] (FusedRMSNormGate) on sliding layers (cfg['window'] == 2048)`: r
- moe-align-local-routes-parallel-port: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: moe-align-local-routes-parallel-port`) that adds a further change in the same area is the only way to learn more (identical code i
- w2-nonlocal-zero-store-off-first-turn: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: w2-nonlocal-zero-store-off-first-turn`) that adds a further change in the same area is the only way to learn more (identical code 
- moe-align-32-experts-first-turn: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: moe-align-32-experts-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is nev
- router-hilo-one-gemm-n512: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: router-hilo-one-gemm-n512`) that adds a further change in the same area is the only way to learn more (identical code is never re-
- restack-router-hilo-align32-zerostore: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: restack-router-hilo-align32-zerostore`) that adds a further change in the same area is the only way to learn more (identical code 
- stack-tonight-four-prefill: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-tonight-four-prefill`) that adds a further change in the same area is the only way to learn more (identical code is never re
- align-local-parallel-first-turn: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: align-local-parallel-first-turn`) that adds a further change in the same area is the only way to learn more (identical code is never re-r
- stack-tonight-five-prefill: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-tonight-five-prefill`) that adds a further change in the same area is the only way to learn more (identical code is never re
- stack-tonight-six-prefill: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: stack-tonight-six-prefill`) that adds a further change in the same area is the only way to learn more (identical code is never re-

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
- queue: experts-triton-packed-dequant-turn4 [w13 marlin + w2 marlin (Triton NVFP4 engine, stage 6: take over turn_4)] p1 queued
- queue: experts-triton-turn1-blockt32-stream [w13 marlin + w2 marlin (Triton NVFP4 engine, stage 7: take over turn_1 with a weight-stream config)] p2 queued
- queue: dense-gemm-cublaslt-algo-pick-prefill-t [dense bf16 GEMM dispatch (qkv, o, pooler, unpooler; cuBLASLt algo selection)] p2 hold
- queue: norm-residual-boundaries-triton-row-kernel [attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2)] p4 hold
- queue: experts-triton-persistent-grid-first-turn [w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: persistent grid, no dead CTAs)] p1 queued
- queue: first-turn-moe-dead-work-stack [MoE dispatch + silu-mul + Triton NVFP4 grid at T > 4096 (first_turn dead-work removal, combined)] p3 queued
- queue: experts-triton-turn4-persistent-local-align [w13 marlin + w2 marlin at turn_4 (route 1024 < T <= 4096 to the Triton NVFP4 engine, with persistent grid and local-routes align)] p2 queued
- queue: turn-moe-local-align-sorted-silu [MoE align/dispatch + silu-mul on the Marlin path (T <= 4096)] p3 queued
- queue: norm-boundaries-persistent-multirow [attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2, persistent multi-row variant)] p5 queued
- queue: moe-pre-post-norm-triton-multitoken [pre-expert norm + post-expert norm/combine (the MoE elementwise ends)] p5 queued
- queue: experts-triton-first-turn-unmasked-loads-l2-hints [w13 marlin + w2 marlin (Triton NVFP4 engine at T > 4096: K-loop load predication and L2 eviction hints)] p4 queued
- running: silu-mul-sorted-first-turn [silu-mul on the Triton path (T > 4096, first_turn)] p1 running
