# muse_prefill_layer findings

Regenerated 2026-10-24T112911Z by `churn.py index`. Read this before writing a hypothesis for muse_prefill_layer.

## Current best

1.0039x [0.9888,1.0189] from job `62025c368c3c4e8e` (check), master commit `6e616b8a05`, hypothesis sglang-pipeline-parity

Where the time goes (per scored shape, top kernels):

- batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6: yours 1078.7µs vs SGLang 1088.2µs (SoL 257.0µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 730.1µs 66.4%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 157.0µs 14.3%, nvjet_sm90_tst_112x128_64x7_1x2_h_bz_TNN 39.3µs 3.6%
- batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6: yours 1434.8µs vs SGLang 1445.1µs (SoL 496.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 680.7µs 45.4%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 419.9µs 28.0%, nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_TNN 162.1µs 10.8%
- batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6: yours 3367.0µs vs SGLang 3373.4µs (SoL 1036.3µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 2248.2µs 64.3%, nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT 327.9µs 9.4%, nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_TNN 256.8µs 7.3%
- batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6: yours 4949.8µs vs SGLang 4962.0µs (SoL 1992.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 2200.8µs 43.5%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 1656.1µs 32.7%, nvjet_sm90_tst_256x152_64x4_1x2_h_bz_coopA_TNT 325.5µs 6.4%
- batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 18279.1µs vs SGLang 18372.8µs (SoL 5518.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 11995.4µs 64.2%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2552.1µs 13.6%, nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT 1399.2µs 7.5%
- batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6: yours 16995.2µs vs SGLang 17046.7µs (SoL 5663.4µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 11106.5µs 64.1%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2550.2µs 14.7%, nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT 1500.8µs 8.7%
- batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 21535.7µs vs SGLang 21245.7µs (SoL 7710.0µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 10239.8µs 49.6%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 4545.7µs 22.0%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2544.3µs 12.3%
- batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6: yours 25286.6µs vs SGLang 25296.3µs (SoL 9847.7µs) · void sglang::device::marlin_moe::Marlin<__nv_bfloat16, 56... 10911.6µs 43.9%, void cutlass::device_kernel<flash::enable_sm90_or_later<f... 8164.1µs 32.8%, nvjet_sm90_tst_192x192_64x4_2x1_v_bz_coopB_TNN 2545.9µs 10.2%

## By area

### FA3 call
- **fa3-contiguous-kv-gather** → regression (regression), 0.9879x — lost on full turn_1, turn_4. Replace the paged FA3 call with a contiguous one; everything before and after stays the parity path. After store_cache (so the new tokens' rows are in the cache

### MoE align/dispatch
- **moe-align-side-stream** → neutral (unproven-vs-best), 1.0076x — no shape moved outside noise. Keep the parity base and move three pieces of work that depend only on the routing outputs and local_start onto the existing side stream, inside the same torch.

### pre-attention norm+rope + cache store
- **qk-norm-fp8-cache-store-fusion** → neutral (unproven-vs-best), 1.0093x — no shape moved outside noise. One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8), v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per t

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
- queue: router-pooler-concat-gemm-turn [router GEMM + pooler GEMM (one concatenated GEMM, no side stream)] p5 queued
- running: experts-triton-w4a16-first-turn-blockt256 [w13 marlin + w2 marlin (W4A16 Triton engine, stage 3: first_turn token tile 256)] p1 running
