# muse_bf16_proj hypotheses

Regenerated 2026-10-25T040348Z. Best: 1.2215x [1.2032,1.2398] job `60b20fd0bf924267` (submit)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T221829Z | sglang-gemm-dispatch-parity |  | bf16 GEMM dispatch | done/gain | gain | 1.0022 | [0.987,1.017] | 017804b34fed40b3 | check | merged-unsubmitted | proceed | 7/7 |
| 2026-10-23T225128Z | router-hilo-at-m8 | primary | router GEMM | done/gain | gain | 1.0219 | [1.007,1.037] | aa3d033c208e4511 | check | done | proceed | 4/4 |
| 2026-10-23T225128Z | router-hilo-addmm-epilogue | primary | router GEMM | done/neutral | neutral | 1.0256 | [1.010,1.041] | 743020b6b43d458c | check |  | proceed | 2/6 |
| 2026-10-23T225128Z | router-triton-hilo-fused | primary | router GEMM | done/regression | regression | 0.9402 | [0.926,0.954] | c2a8de5969374dff | check |  | proceed | 4/7 |
| 2026-10-23T225201Z | pooler-triton-skinny-m32 | primary | pooler GEMM | done/neutral | neutral | 0.9983 | [0.983,1.013] | 9f0ad00bbc874d31 | check |  | proceed | 4/6 |
| 2026-10-23T225201Z | unpooler-slice-triton-gemv | primary | unpooler GEMM | done/regression | regression | 0.8801 | [0.867,0.893] | 4c2b113f81434570 | check |  | proceed | 1/4 |
| 2026-10-23T225303Z | pooler-skinny-tiled-weight-layout | primary | pooler GEMM | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-23T225303Z | bf16-cublaslt-backend-toggle | primary | bf16 GEMM dispatch | done/neutral | neutral | 1.1505 | [1.133,1.168] | 5ed9ace10d9a4f68 | check |  | proceed | 3/4 |
| 2026-10-23T225303Z | o-triton-skinny-m32 | primary | o GEMM | done/neutral | neutral | 1.0057 | [0.991,1.021] | d75669d46acb42a5 | check |  | proceed | 4/5 |
| 2026-10-23T225303Z | unpooler-full-triton-skinny-m32 | primary | unpooler GEMM | done/neutral | neutral | 1.0053 | [0.990,1.020] | 635c2693bd4147a4 | check |  | proceed | 3/4 |
| 2026-10-23T225317Z | qkv-gate-triton-skinny-m32 | primary | qkv GEMM | done/regression | regression | 0.9238 | [0.910,0.938] | abcd45776f244c8c | check |  | proceed | 2/6 |
| 2026-10-23T230544Z | router-triton-hilo-fused-control-splitk-all-m | control→router-triton-hilo-fused | router GEMM | done/gain | gain | 1.1510 | [1.134,1.168] | bd9fa14151644cee | check | done | proceed | 7/7 |
| 2026-10-24T024326Z | router-triton-hilo-fused-control-hybrid-cublas-above-m32 | control→router-triton-hilo-fused-control-splitk-all-m | router GEMM | queue | hold |  |  |  |  |  | required |  |
| 2026-10-24T024434Z | router-hilo-addmm-epilogue-control-inplace-out | control→router-hilo-addmm-epilogue | router GEMM | queue | hold |  |  |  |  |  | required |  |
| 2026-10-24T032454Z | unpooler-full-triton-skinny-m32-control-blockm16-all-m | control→unpooler-full-triton-skinny-m32 | unpooler GEMM | done/neutral | neutral | 0.9990 | [0.984,1.014] | 09abdd472faa4dce | check |  | proceed | 3/4 |
| 2026-10-24T074946Z | pooler-triton-splitk2-blockn64-semaphore | primary | pooler GEMM | done/neutral | neutral | 1.1664 | [1.149,1.184] | 08ec203e5717406e | check |  | proceed | 6/7 |
| 2026-10-24T075006Z | qkv-gate-triton-splitk4-blockn64-semaphore | primary | qkv GEMM | done/neutral | neutral | 1.1431 | [1.126,1.160] | 135ef6499875416a | check |  | proceed | 4/6 |
| 2026-10-24T075025Z | router-splitk32-small-m-all-loads-in-flight | primary | router GEMM | done/neutral | neutral | 1.1427 | [1.126,1.160] | 1e339074b1844a43 | check |  | proceed | 2/6 |
| 2026-10-24T075041Z | router-splitk-semaphore-no-memset | primary | router GEMM | done/neutral | neutral | 1.1516 | [1.134,1.169] | cb76350d40ee4617 | check |  | proceed | 1/6 |
| 2026-10-24T075111Z | unpooler-slice-triton-splitk8-semaphore | primary | unpooler GEMM | done/neutral | neutral | 1.1546 | [1.137,1.172] | bb31bb36545a47bb | check |  | proceed | 5/5 |
| 2026-10-24T075347Z | pooler-triton-splitk2-blockn64-semaphore-control-blockk256 | control→pooler-triton-splitk2-blockn64-semaphore | pooler GEMM | done/neutral | neutral | 1.1620 | [1.145,1.179] | 75d595a29a4c4b57 | check |  | proceed | 5/7 |
| 2026-10-24T092553Z | pooler-unpooler-m128-wgmma-splitk-semaphore | primary | pooler GEMM + unpooler GEMM (m=128) | done/regression | regression | 1.1158 | [1.099,1.133] | d34fd76f9c444728 | check |  | proceed | 3/6 |
| 2026-10-24T092629Z | qkv-gate-o-m128-wgmma-splitk-semaphore | primary | qkv GEMM + o GEMM (m=128) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-24T092648Z | pooler-splitk2-tma-weight-stream | primary | pooler GEMM | done/neutral | neutral | 1.1693 | [1.152,1.187] | 4e89311eefdc4bc3 | check |  | proceed | 5/6 |
| 2026-10-24T103959Z | router-splitk32-small-m-all-loads-in-flight-control-stages3 | control→router-splitk32-small-m-all-loads-in-flight | router GEMM | done/neutral | neutral | 1.1518 | [1.135,1.169] | 4685bac7a662454d | check |  | proceed | 4/6 |
| 2026-10-24T112147Z | bf16-shape-gated-triton-winners-combo | primary | bf16 GEMM dispatch (shape-gated Triton routing, m<=16) | done/neutral | neutral | 1.1760 | [1.158,1.194] | 875f3e0172ee48a1 | check | merged | proceed | 14/16 |
| 2026-10-24T112213Z | bf16-m128-nn-weight-layout-probe | primary | bf16 GEMM dispatch (m>=128, weight layout) | done/neutral | neutral | 1.1580 | [1.141,1.175] | 2dc5dce078be4a24 | check |  | proceed | 7/8 |
| 2026-10-24T140147Z | bf16-m32-swapab-wgmma-triton | primary | bf16 GEMM dispatch (m=17..32, swap-AB Triton) | done/neutral | neutral | 1.1984 | [1.180,1.216] | 1602a4e947e64090 | check | done | proceed | 4/10 |
| 2026-10-24T140216Z | bf16-cublaslt-algo-pick-in-prepare-m128-m256 | primary | bf16 GEMM dispatch (m=128/256, cuBLASLt algo selection) | done/neutral | neutral | 1.1879 | [1.170,1.206] | 26a8be34a5eb4426 | submit |  | proceed | 7/10 |
| 2026-10-24T140242Z | pooler-unpooler-m128-tma-wspec-rewrite | rewrite | pooler GEMM + unpooler GEMM (m=128) | done/regression | regression | 1.1639 | [1.146,1.181] | 61da9b91dab44bf4 | submit |  | proceed | 4/7 |
| 2026-10-24T140257Z | pooler-m32-tma-splitk-extend | primary | pooler GEMM (m=17..32) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T165012Z | bf16-m16-swapab-wgmma-extend | primary | bf16 GEMM dispatch (m<=16, swap-AB Triton) | done/neutral | neutral | 1.2115 | [1.193,1.230] | 89fabef70bd745f7 | submit |  | proceed | 5/16 |
| 2026-10-24T165036Z | router-swapab-wgmma-all-m | primary | router GEMM | done/neutral | neutral | 1.1951 | [1.177,1.213] | 587e340476944b46 | submit |  | proceed | 4/10 |
| 2026-10-24T165102Z | bf16-small-m-cuda-gemv-rewrite | rewrite | bf16 GEMM dispatch (m<=8, hand-written CUDA weight stream) | done/regression | regression | 1.1667 | [1.149,1.184] | b76aca7ec1a048e7 | submit |  | proceed | 4/13 |
| 2026-10-24T172152Z | bf16-lossless-exp4-weight-stream-rewrite | rewrite | bf16 GEMM dispatch (m<=32, lossless-compressed weight stream) | done/regression | regression | 0.7071 | [0.697,0.718] | 542f2b169c934706 | submit |  | proceed | 2/17 |
| 2026-10-24T172235Z | bf16-verify-streamk-cutlass-wgmma-rewrite | rewrite | bf16 GEMM dispatch (m=129..256, persistent stream-K TMA/wgmma) | done/regression | regression | 1.1635 | [1.146,1.181] | f5b1361aaa0e48b5 | check |  | proceed | 0/8 |
| 2026-10-24T174421Z | bf16-restack-m16-swapab-router-swapab-m32 | primary | restack (m<=16 swap-AB GEMMs + router swap-AB) | done/neutral | neutral | 1.2215 | [1.203,1.240] | 1e90292c7b954d96 | submit | done | proceed | 3/4 |
| 2026-10-25T001220Z | bf16-lossless-exp4-occupancy-dequant-once-stage2 | rewrite | lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel) | done/neutral | neutral | 0.8870 | [0.874,0.900] | f1024212dd2f49f3 | check |  | proceed | 13/13 |
| 2026-10-25T003715Z | bf16-lossless-exp4-warpspec-persistent-stage3 | rewrite | lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T003758Z | router-cuda-cluster-dsmem-splitk-rewrite | rewrite | router GEMM | done/regression | regression | 1.1897 | [1.172,1.208] | dc5d03c4d9604b0e | check |  | proceed | 7/10 |
| 2026-10-25T003826Z | o-unpooler-slice-cuda-cluster-dsmem-rewrite | rewrite | o GEMM + unpooler GEMM (latency-bound small streams) | done/regression | regression | 1.1710 | [1.153,1.189] | 3f3330f70b6e4cdc | check |  | proceed | 6/10 |
| 2026-10-25T003858Z | o-swapab-m32-splitk2-two-ctas-per-sm | primary | o GEMM (m=17..32, swap-AB) | done/regression | regression | 1.1860 | [1.168,1.204] | 75f09fa68f9c4415 | check |  | proceed | 4/8 |
| 2026-10-25T003858Z | qkv-swapab-m32-splitk8-stages3 | primary | qkv GEMM (m=17..32, swap-AB) | done/regression | regression | 1.1963 | [1.178,1.214] | a61778a08ec04ea6 | check |  | proceed | 3/8 |
| 2026-10-25T003858Z | unpooler-full-m16-swapab-route | primary | unpooler GEMM | done/regression | regression | 1.1885 | [1.171,1.206] | 05782267384343eb | check |  | proceed | 5/8 |
| 2026-10-25T011407Z | pooler-unpooler-m128-nn-layout-gated | primary | pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout) | done/regression | regression | 1.1953 | [1.177,1.213] | 0daf3faa83ab4e81 | check |  | proceed | 8/9 |
| 2026-10-25T011408Z | bf16-verify-swapab-wgmma-cluster-dsmem-rewrite | rewrite | bf16 GEMM dispatch (m=65..256, swap-AB wgmma + cluster DSMEM split-K) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T011408Z | bf16-lossless-exp4-decode-only-throughput-probe | primary | lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel) | done/regression | regression | 1.1904 | [1.173,1.208] | 9148142cee4441ad | check |  | proceed | 1/1 |
| 2026-10-25T011408Z | bf16-verify-deepgemm-bf16-gemm-nt-route | primary | bf16 GEMM dispatch (m=65..256, DeepGEMM bf16) | done/regression | regression | 1.2027 | [1.185,1.221] | 2d522d47ca5c4fc9 | check |  | proceed | 5/8 |
| 2026-10-25T011408Z | pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite | rewrite | pooler GEMM + unpooler GEMM (m<=32 DRAM weight streams, bulk-TMA row-owner kernel) | done/regression | regression | 1.1828 | [1.165,1.201] | cd39d36e31814087 | check |  | proceed | 8/15 |
| 2026-10-25T035343Z | pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild | rewrite | pooler GEMM + unpooler GEMM (m<=32 DRAM weight streams, bulk-TMA row-owner kernel) | running | running |  |  |  |  |  | proceed |  |

## Held

- pooler-skinny-tiled-weight-layout: skinny-Triton family settled: pooler neutral, qkv-gate regression, unpooler-full neutral, o neutral (1.006x), and the BLOCK_M=16 control showed the m=32 penalty is not the tile (20.8 us vs 18.4 us in-run at m=16). The reviewer expected both of this hypothesis's mechanisms to be ~zero, and it builds on the unpromoted pooler branch, so a gain could not be promoted without a rebase and rerun.
- router-triton-hilo-fused-control-hybrid-cublas-above-m32: superseded: router-triton-hilo-fused-control-splitk-all-m landed (1.151x); its 3-D Triton grid beats cuBLAS hi/lo at every m (5.7-9.3 us vs 10.5-17.7 us), so a hybrid using cuBLAS above m=32 can only be slower
- router-hilo-addmm-epilogue-control-inplace-out: superseded: master's router no longer uses the cuBLAS hi/lo pair (Triton split-K landed, 1.151x), so the add node this control targets is gone; the primary read flat (1.026x, within CI)
- qkv-gate-o-m128-wgmma-splitk-semaphore: same m=128 wgmma split-K design as pooler-unpooler-m128-wgmma-splitk-semaphore (dispatching now); its reviewer asked to wait for that harvest; re-queue by hand if that kernel reaches its target
- pooler-m32-tma-splitk-extend: adversarial review: Moot under master 8b9ad43 (bf16-m32-swapab-wgmma-triton, 1.198x): pooler decode_32 m=32 already runs _bf16_swapab_kernel at 20.0 us row / 20.5 us kernel vs SGLa
- bf16-lossless-exp4-warpspec-persistent-stage3: adversarial review: Stage 2's pooler m=1 kernel took 30.8 us in the judge profile, past this hypothesis's own supersede threshold (> 28 us) and 1.62x short of the 19.0 us kill gate
- bf16-verify-swapab-wgmma-cluster-dsmem-rewrite: Reviewer: wait for the build/cluster-launch/self-check log lines of dc5d03c4d9604b0e and 3f3330f70b6e4cdc (in-flight cluster checks) before dispatch; not their perf.
