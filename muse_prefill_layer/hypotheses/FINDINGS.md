# muse_prefill_layer findings

Regenerated 2026-10-24T045257Z by `churn.py index`. Read this before writing a hypothesis for muse_prefill_layer.

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

### whole-layer parity (SGLang pipeline) + routing remap
- **sglang-pipeline-parity** → gain (gain), 1.0039x — no shape moved outside noise. Rewrite kernel_fn and prepare in muse_prefill_layer/submission.py to run SGLang's own pipeline module for module as baseline.py does: _bf16_gemm_dispatch_impl f

## Open follow-ups

- none

## Do not retry (rejected / duplicates)

- none

## Queue and in flight

- queue: experts-dequant-grouped-mm-cutlass [w13 marlin + w2 marlin (GEMM engine: CUTLASS grouped GEMM instead of Triton)] p4 queued
- queue: experts-dequant-overlap-attention [w13 marlin + w2 marlin (dequant placement on a side stream)] p3 queued
- queue: experts-dequant-threshold-turn4 [w13 marlin + w2 marlin (Marlin vs dequant crossover)] p4 queued
- queue: experts-dequant-triton-moe-16k [w13 marlin + w2 marlin (expert grouped GEMM engine at 16K tokens)] p2 hold
- queue: experts-triton-w4a16-inkernel-fp4 [w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, for the turn shapes)] p5 queued
- queue: fa3-batched-mode [FA3 call] p4 queued
- queue: fa3-contiguous-kv-gather [FA3 call] p3 queued
- queue: qk-norm-fp8-cache-store-fusion [pre-attention norm+rope + cache store] p4 queued
- running: moe-align-side-stream [MoE align/dispatch] p2 running
