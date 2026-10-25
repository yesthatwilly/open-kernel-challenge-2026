# muse_moe_w4a16 hypotheses

Regenerated 2026-10-25T000226Z. Best: 1.3278x [1.3079,1.3477] job `1947e3ec2abb4bdf` (submit)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T225501Z | triton-w13-silu-epilogue | primary | silu-mul | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225501Z | triton-w2-grouped-gemv-small-m | primary | w2 marlin | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225501Z | sglang-marlin-path-parity | primary | MoE align/dispatch | done/gain | gain | 0.9888 | [0.974,1.004] | fb2cc736613747df | check | merged-unsubmitted | proceed | 3/3 |
| 2026-10-23T225501Z | triton-w13-grouped-gemv-small-m | primary | w13 marlin | done/regression | regression | 0.7506 | [0.739,0.762] | f31a03eb9bc6410f | check |  | proceed | 2/4 |
| 2026-10-23T225502Z | triton-small-m-threshold-32 | primary | MoE align/dispatch | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225502Z | align-single-token-ep | primary | MoE align/dispatch | done/neutral | neutral | 0.9951 | [0.980,1.010] | a7bb0cecaf8d4de1 | check |  | proceed | 3/3 |
| 2026-10-23T225502Z | marlin-workspace-blocks-per-sm-8 | primary | w13 marlin | done/neutral | neutral | 0.9921 | [0.977,1.007] | 75805dc2f18a4d06 | check |  | proceed | 2/4 |
| 2026-10-23T225502Z | w13-marlin-atomic-add-small-m | primary | w13 marlin | done/neutral | neutral | 0.9869 | [0.972,1.002] | c8ea8e21eea447f9 | check |  | proceed | 1/4 |
| 2026-10-23T225502Z | w13-marlin-bf16-reduce-verify | primary | w13 marlin | done/neutral | neutral | 0.9840 | [0.969,0.999] | 8d389f3d3e2d473f | check |  | proceed | 1/3 |
| 2026-10-23T225502Z | marlin-block-size-m-next-larger | primary | MoE align/dispatch | done/regression | regression | 0.9178 | [0.904,0.932] | 2639e3ca7f5549de | check |  | proceed | 3/5 |
| 2026-10-24T075126Z | post-expert-norm-triton-small-m | primary | post-expert norm+combine | done/gain | gain | 1.1297 | [1.113,1.147] | bacd1f502f4e46fa | check | superseded | proceed | 6/6 |
| 2026-10-24T075232Z | marlin-block-size-m-next-smaller | primary | w13 marlin | done/gain | gain | 1.0394 | [1.024,1.055] | 198f10f446a847cb | check | superseded | proceed | 5/6 |
| 2026-10-24T075232Z | align-256-experts-below-128 | primary | MoE align/dispatch | done/neutral | neutral | 0.9998 | [0.985,1.015] | e753d5b9df644ebb | check |  | proceed | 3/6 |
| 2026-10-24T075257Z | align-remote-sentinel-ids | primary | MoE align/dispatch | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T075257Z | marlin-block-size-m-verify-two-smaller | primary | w13 marlin | done/gain | gain | 1.0916 | [1.075,1.108] | ed837a4a46d44a95 | check | superseded | proceed | 6/6 |
| 2026-10-24T081339Z | small-m-grouped-gemv-splitk-rewrite | rewrite | w13 marlin | done/regression | regression | 0.6635 | [0.654,0.673] | 734a1159ff334c3d | check |  | proceed | 3/6 |
| 2026-10-24T100217Z | marlin-block-m-cap-16-port | primary | w13 marlin | done/neutral | neutral | 1.0950 | [1.079,1.111] | 1c33392914594026 | check |  | proceed | 6/6 |
| 2026-10-24T112137Z | marlin-block-m-two-smaller-on-postnorm | primary | w13 marlin | done/gain | gain | 1.2536 | [1.235,1.272] | c00ef7295c154867 | check | superseded | proceed | 6/6 |
| 2026-10-24T112155Z | triton-map-align-small-m-port | primary | MoE align/dispatch | done/neutral | neutral | 1.1850 | [1.167,1.203] | c2feb68aafb44aeb | check |  | proceed | 6/6 |
| 2026-10-24T112212Z | post-norm-triton-all-m | primary | post-expert norm+combine | done/neutral | neutral | 1.2483 | [1.230,1.267] | 2ec5387ffe8748c9 | check |  | proceed | 4/6 |
| 2026-10-24T132314Z | triton-map-align-on-block-m-restack | primary | MoE align/dispatch + block_size_m pick | done/gain | gain | 1.3117 | [1.292,1.331] | bc56182a80444d16 | check | done | proceed | 6/6 |
| 2026-10-24T140229Z | marlin-two-class-block-m-decode32 | primary | MoE align/dispatch + block_size_m pick | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T140229Z | swapab-nvfp4-gemm-verify-rewrite | rewrite | w13 marlin + w2 marlin | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T140229Z | swapab-w13-silu-epilogue-verify | rewrite | silu-mul | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T140229Z | triton-map-align-verify-chunked | primary | MoE align/dispatch | done/failed | failed |  |  | 3aa4d7e193244255 | check |  | proceed |  |
| 2026-10-24T144931Z | triton-map-align-verify-chunked-pow2-fix | primary | MoE align/dispatch | done/neutral | neutral | 1.3107 | [1.291,1.330] | c1c72d5d19ea444f | check |  | proceed | 2/4 |
| 2026-10-24T164817Z | align-parallel-per-expert-verify | primary | MoE align/dispatch | done/neutral | neutral | 1.3249 | [1.305,1.345] | 45eacf4ca2e849d2 | check | done | proceed | 4/4 |
| 2026-10-24T164836Z | w2-l2-prefetch-decode1 | primary | w2 marlin | done/regression | regression | 1.2615 | [1.243,1.280] | ac6897f284934dad | check |  | proceed | 3/4 |
| 2026-10-24T164857Z | marlin-two-stream-expert-halves-small-m | primary | launch overhead | done/neutral | neutral | 1.3024 | [1.283,1.322] | 5d630369b1634f77 | submit |  | proceed | 2/5 |
| 2026-10-24T165851Z | cuda-fused-moe-bs1-persistent-w13-silu-w2 | rewrite | whole-layer fusion | done/neutral | neutral | 1.3074 | [1.269,1.327] | 1105a02aa3fa4e56 | submit |  | proceed | 5/6 |
| 2026-10-24T165851Z | cuda-nvfp4-swapab-mma-persistent-small-m | rewrite | w13 marlin + silu-mul + w2 marlin | done/failed | failed |  |  | 8f198afc3bf14f6f | submit |  | proceed |  |
| 2026-10-24T172038Z | cuda-nvfp4-wgmma-rega-grouped-verify | rewrite | w13 marlin + silu-mul + w2 marlin | done/neutral | neutral | 1.3123 | [1.293,1.332] | 3bb822f22ebc4a5c | submit |  | proceed | 4/6 |
| 2026-10-24T172123Z | cuda-fused-moe-midm-single-launch-routed | rewrite | whole-layer fusion | queue | queued |  |  |  |  |  | proceed |  |
| 2026-10-24T183202Z | cuda-dispatch-combine-pdl-verify | rewrite | MoE align/dispatch + post-expert norm+combine | done/neutral | neutral | 1.3283 | [1.308,1.348] | c41860068fe4432f | submit |  | proceed | 4/6 |
| 2026-10-24T183618Z | post-norm-combine-hoisted-slot-loads | primary | post-expert norm+combine | done/neutral | neutral | 1.3282 | [1.308,1.348] | d7501477ce2748d5 | submit |  | proceed | 3/6 |
| 2026-10-24T183635Z | post-norm-triton-hoisted-decode32 | primary | post-expert norm+combine | queue | queued |  |  |  |  |  | proceed |  |
| 2026-10-24T183649Z | post-norm-pdl-launch-small-m | primary | launch overhead | queue | queued |  |  |  |  |  | proceed |  |
| 2026-10-24T183702Z | align-parallel-from-17-tokens | primary | MoE align/dispatch | done/neutral | neutral | 1.3250 | [1.305,1.345] | ed6d884240084d9c | submit |  | proceed | 5/6 |
| 2026-10-24T183722Z | align-small-one-warp-tiny-m | primary | MoE align/dispatch | done/neutral | neutral | 1.3287 | [1.309,1.349] | 38532ba10e394e3c | submit |  | proceed | 4/6 |

## Held

- triton-w13-silu-epilogue: stacks on triton-w13-grouped-gemv-small-m, which regressed (2-2.6x slower than Marlin at small M)
- triton-w2-grouped-gemv-small-m: stacks on triton-w13-grouped-gemv-small-m, which regressed (2-2.6x slower than Marlin at small M)
- triton-small-m-threshold-32: stacks on triton-w13-grouped-gemv-small-m, which regressed (2-2.6x slower than Marlin at small M)
- align-remote-sentinel-ids: adversarial review: Moot on master and overlapped by the running triton-map-align-small-m-port. The 3.2 us bs=1 gap it was written to explain (80.2 vs 77.0 us) no longer exists: ma
- marlin-two-class-block-m-decode32: reviewer expects flat: uniform block 8 cost only ~2.4% at bs=32, and the 239 vs 186 us premise is seed noise (unchanged bs=32 rows range 147-265 us); two extra Marlin launches (~2-4 us each) likely eat the rest
- swapab-nvfp4-gemm-verify-rewrite: reviewer: stage 1 likely 1.5-2x slower than Marlin on the replaced kernels (Marlin runs only 15-18% more blocks; prefill's Triton tile costs ~25 us vs Marlin's 11.6-16.7 us/block). Unhold after prefill's packed-asm dequant stage (experts-triton-nvfp4-packed-asm-dequant) harvests and fold that dequant in, with BLOCK_T=64 at bs=256
