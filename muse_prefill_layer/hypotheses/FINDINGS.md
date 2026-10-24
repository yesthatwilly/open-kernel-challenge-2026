# muse_prefill_layer findings

Regenerated 2026-10-24T125104Z by `churn.py index`. Read this before writing a hypothesis for muse_prefill_layer.

## Current best

1.0229x [1.0075,1.0382] from job `b33b87b5083b4d1c` (check), master commit `a1f93c2948`, hypothesis experts-triton-w4a16-first-turn-blockt256

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6: yours 1141.1µs vs SGLang 1063.9µs (SoL 257.0µs) · _moe_nvfp4_gemm_kernel 847.4µs 69.8%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 156.0µs 12.9%, nvjet_sm90_tst_112x128_64x7_1x2_h_bz_TNN 38.8µs 3.2%
- batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6: yours 1517.9µs vs SGLang 1431.8µs (SoL 496.7µs) · _moe_nvfp4_gemm_kernel 728.5µs 47.6%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 417.2µs 27.2%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 153.1µs 10.0%
- batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6: yours 3435.4µs vs SGLang 3340.9µs (SoL 1036.3µs) · _moe_nvfp4_gemm_kernel 2274.6µs 64.7%, nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT 325.1µs 9.3%, nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_TNN 253.3µs 7.2%
- batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6: yours 4975.4µs vs SGLang 4881.5µs (SoL 1992.4µs) · _moe_nvfp4_gemm_kernel 2317.7µs 44.9%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 1640.5µs 31.8%, nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT 322.6µs 6.3%
- batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 15029.5µs vs SGLang 17847.8µs (SoL 5518.4µs) · _moe_nvfp4_gemm_kernel 8573.7µs 56.5%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2514.6µs 16.6%, nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT 1375.0µs 9.1%
- batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6: yours 14426.7µs vs SGLang 17115.0µs (SoL 5663.4µs) · _moe_nvfp4_gemm_kernel 8535.2µs 58.2%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2517.6µs 17.2%, nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT 1474.5µs 10.1%
- batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 18568.9µs vs SGLang 21250.7µs (SoL 7710.0µs) · _moe_nvfp4_gemm_kernel 7959.7µs 43.9%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 4480.4µs 24.7%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2516.2µs 13.9%
- batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 22521.7µs vs SGLang 24976.4µs (SoL 9847.7µs) · _moe_nvfp4_gemm_kernel 8721.8µs 38.9%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 8017.7µs 35.7%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2509.7µs 11.2%

## By area

### FA3 call
- **fa3-contiguous-kv-gather** → regression (regression), 0.9879x — lost on full turn_1, turn_4. Replace the paged FA3 call with a contiguous one; everything before and after stays the parity path. After store_cache (so the new tokens' rows are in the cache

### MoE align/dispatch
- **moe-align-side-stream** → neutral (unproven-vs-best), 1.0076x — no shape moved outside noise. Keep the parity base and move three pieces of work that depend only on the routing outputs and local_start onto the existing side stream, inside the same torch.

### pre-attention norm+rope + cache store
- **qk-norm-fp8-cache-store-fusion** → neutral (unproven-vs-best), 1.0093x — no shape moved outside noise. One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8), v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per t

### w13 marlin + w2 marlin (W4A16 Triton engine, stage 3: first_turn token tile 256)
- **experts-triton-w4a16-first-turn-blockt256** → neutral (unproven-vs-best), 1.0229x — gained on full first_turn; sliding first_turn; lost on sliding turn_4. Stage 3 of the Triton NVFP4 grouped GEMM, built on stage 2 (branch hyp/muse_prefill_layer/experts-triton-w4a16-swapab-reglhs, _moe_nvfp4_gemm_kernel and EXPERT_

### w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, for the turn shapes)
- **experts-triton-w4a16-inkernel-fp4** → regression (regression), 0.6891x — lost on sliding first_turn, turn_1, turn_4; full first_turn, turn_1, turn_4. A Triton grouped-GEMM kernel with in-register NVFP4 dequant replaces both moe_wna16_marlin_gemm calls wherever master still runs Marlin (if a dequant-to-BF16 pa

### w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, stage 2: swap-AB register-LHS dot)
- **experts-triton-w4a16-swapab-reglhs** → neutral (unproven-vs-best), 1.0068x — gained on sliding first_turn, turn_1, turn_4; full first_turn, turn_1, turn_4. Stage 2 of the Triton NVFP4 grouped GEMM, built on stage 1 (branch hyp/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4, _moe_nvfp4_gemm_kernel in muse_pref

### w13 marlin + w2 marlin (block_size_m pick)
- **experts-marlin-block-m-local-rows** → regression (regression), 0.9836x — lost on sliding turn_1; full turn_1. Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py). The parity loop walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and breaks o
- **experts-marlin-block-m-48-turn1** → neutral (within-ci), 1.0030x — no shape moved outside noise. Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 512 < num_tokens <= 1024 use block_size_m = 48; every other token count k

### whole-layer parity (SGLang pipeline) + routing remap
- **sglang-pipeline-parity** → gain (gain), 1.0039x — no shape moved outside noise. Rewrite kernel_fn and prepare in muse_prefill_layer/submission.py to run SGLang's own pipeline module for module as baseline.py does: _bf16_gemm_dispatch_impl f

## Open follow-ups

- moe-align-side-stream: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: moe-align-side-stream`) that adds a further change in the same area is the only way to learn more (identical code is never re-run)
- qk-norm-fp8-cache-store-fusion: unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: qk-norm-fp8-cache-store-fusion`) that adds a further change in the same area is the only way to learn more (identical code is neve
- experts-marlin-block-m-48-turn1: within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: experts-marlin-block-m-48-turn1`) that adds a further change in the same area is the only way to learn more (identical code is never re-r
- experts-triton-w4a16-swapab-reglhs: rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache
- experts-triton-w4a16-first-turn-blockt256: rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reache

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: experts-dequant-grouped-mm-cutlass [w13 marlin + w2 marlin (GEMM engine: CUTLASS grouped GEMM instead of Triton)] p4 queued
- queue: experts-dequant-overlap-attention [w13 marlin + w2 marlin (dequant placement on a side stream)] p3 queued
- queue: experts-dequant-threshold-turn4 [w13 marlin + w2 marlin (Marlin vs dequant crossover)] p4 queued
- queue: experts-dequant-triton-moe-16k [w13 marlin + w2 marlin (expert grouped GEMM engine at 16K tokens)] p2 hold
- queue: fa3-batched-mode [FA3 call] p4 hold
- queue: experts-marlin-block-m-48-turn4 [w13 marlin + w2 marlin (block_size_m pick)] p3 hold
- queue: router-triton-hilo-splitk-turn [router GEMM] p3 hold
- queue: router-pooler-concat-gemm-turn [router GEMM + pooler GEMM (one concatenated GEMM, no side stream)] p5 hold
- running: experts-hybrid-marlin-turn-triton-first-turn [w13 marlin + w2 marlin (stage 4: hybrid shape gate, Marlin at T<=4096, Triton NVFP4 above)] p1 running
