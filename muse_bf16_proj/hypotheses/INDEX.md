# muse_bf16_proj hypotheses

Regenerated 2026-10-27T103501Z. Best: 1.2735x [1.2544,1.2926] job `bc147c5b87bc465d` (submit)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T221829Z | sglang-gemm-dispatch-parity |  | bf16 GEMM dispatch | done/gain | gain | 1.0022 | [0.987,1.017] | 017804b34fed40b3 | check | merged-unsubmitted | proceed | 7/7 |
| 2026-10-23T225128Z | router-hilo-at-m8 | primary | router GEMM | done/gain | gain | 1.0219 | [1.007,1.037] | aa3d033c208e4511 | check | done | proceed | 4/4 |
| 2026-10-23T225128Z | router-hilo-addmm-epilogue | primary | router GEMM | done/neutral | neutral | 1.0256 | [1.010,1.041] | 743020b6b43d458c | check |  | proceed | 2/6 |
| 2026-10-23T225128Z | router-triton-hilo-fused | primary | router GEMM | done/regression | regression | 0.9402 | [0.926,0.954] | c2a8de5969374dff | check |  | proceed | 4/7 |
| 2026-10-23T225201Z | pooler-triton-skinny-m32 | primary | pooler GEMM | done/neutral | neutral | 0.9983 | [0.983,1.013] | 9f0ad00bbc874d31 | check |  | proceed | 4/6 |
| 2026-10-23T225201Z | unpooler-slice-triton-gemv | primary | unpooler GEMM | done/regression | regression | 0.8801 | [0.867,0.893] | 4c2b113f81434570 | check |  | proceed | 1/4 |
| 2026-10-23T225303Z | bf16-cublaslt-backend-toggle | primary | bf16 GEMM dispatch | done/neutral | neutral | 1.1505 | [1.133,1.168] | 5ed9ace10d9a4f68 | check |  | proceed | 3/4 |
| 2026-10-23T225303Z | o-triton-skinny-m32 | primary | o GEMM | done/neutral | neutral | 1.0057 | [0.991,1.021] | d75669d46acb42a5 | check |  | proceed | 4/5 |
| 2026-10-23T225303Z | unpooler-full-triton-skinny-m32 | primary | unpooler GEMM | done/neutral | neutral | 1.0053 | [0.990,1.020] | 635c2693bd4147a4 | check |  | proceed | 3/4 |
| 2026-10-23T225303Z | pooler-skinny-tiled-weight-layout | primary | pooler GEMM | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-23T225317Z | qkv-gate-triton-skinny-m32 | primary | qkv GEMM | done/regression | regression | 0.9238 | [0.910,0.938] | abcd45776f244c8c | check |  | proceed | 2/6 |
| 2026-10-23T230544Z | router-triton-hilo-fused-control-splitk-all-m | control→router-triton-hilo-fused | router GEMM | done/gain | gain | 1.1510 | [1.134,1.168] | bd9fa14151644cee | check | done | proceed | 7/7 |
| 2026-10-24T024326Z | router-triton-hilo-fused-control-hybrid-cublas-above-m32 | control→router-triton-hilo-fused-control-splitk-all-m | router GEMM | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-24T024434Z | router-hilo-addmm-epilogue-control-inplace-out | control→router-hilo-addmm-epilogue | router GEMM | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-24T032454Z | unpooler-full-triton-skinny-m32-control-blockm16-all-m | control→unpooler-full-triton-skinny-m32 | unpooler GEMM | done/neutral | neutral | 0.9990 | [0.984,1.014] | 09abdd472faa4dce | check |  | proceed | 3/4 |
| 2026-10-24T074946Z | pooler-triton-splitk2-blockn64-semaphore | primary | pooler GEMM | done/neutral | neutral | 1.1664 | [1.149,1.184] | 08ec203e5717406e | check |  | proceed | 6/7 |
| 2026-10-24T075006Z | qkv-gate-triton-splitk4-blockn64-semaphore | primary | qkv GEMM | done/neutral | neutral | 1.1431 | [1.126,1.160] | 135ef6499875416a | check |  | proceed | 4/6 |
| 2026-10-24T075025Z | router-splitk32-small-m-all-loads-in-flight | primary | router GEMM | done/neutral | neutral | 1.1427 | [1.126,1.160] | 1e339074b1844a43 | check |  | proceed | 2/6 |
| 2026-10-24T075041Z | router-splitk-semaphore-no-memset | primary | router GEMM | done/neutral | neutral | 1.1516 | [1.134,1.169] | cb76350d40ee4617 | check |  | proceed | 1/6 |
| 2026-10-24T075111Z | unpooler-slice-triton-splitk8-semaphore | primary | unpooler GEMM | done/neutral | neutral | 1.1546 | [1.137,1.172] | bb31bb36545a47bb | check |  | proceed | 5/5 |
| 2026-10-24T075347Z | pooler-triton-splitk2-blockn64-semaphore-control-blockk256 | control→pooler-triton-splitk2-blockn64-semaphore | pooler GEMM | done/neutral | neutral | 1.1620 | [1.145,1.179] | 75d595a29a4c4b57 | check |  | proceed | 5/7 |
| 2026-10-24T092553Z | pooler-unpooler-m128-wgmma-splitk-semaphore | primary | pooler GEMM + unpooler GEMM (m=128) | done/regression | regression | 1.1158 | [1.099,1.133] | d34fd76f9c444728 | check |  | proceed | 3/6 |
| 2026-10-24T092629Z | qkv-gate-o-m128-wgmma-splitk-semaphore | primary | qkv GEMM + o GEMM (m=128) | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-24T092648Z | pooler-splitk2-tma-weight-stream | primary | pooler GEMM | done/neutral | neutral | 1.1693 | [1.152,1.187] | 4e89311eefdc4bc3 | check |  | proceed | 5/6 |
| 2026-10-24T103959Z | router-splitk32-small-m-all-loads-in-flight-control-stages3 | control→router-splitk32-small-m-all-loads-in-flight | router GEMM | done/neutral | neutral | 1.1518 | [1.135,1.169] | 4685bac7a662454d | check |  | proceed | 4/6 |
| 2026-10-24T112147Z | bf16-shape-gated-triton-winners-combo | primary | bf16 GEMM dispatch (shape-gated Triton routing, m<=16) | done/neutral | neutral | 1.1760 | [1.158,1.194] | 875f3e0172ee48a1 | check | merged | proceed | 14/16 |
| 2026-10-24T112213Z | bf16-m128-nn-weight-layout-probe | primary | bf16 GEMM dispatch (m>=128, weight layout) | done/neutral | neutral | 1.1580 | [1.141,1.175] | 2dc5dce078be4a24 | check |  | proceed | 7/8 |
| 2026-10-24T140147Z | bf16-m32-swapab-wgmma-triton | primary | bf16 GEMM dispatch (m=17..32, swap-AB Triton) | done/neutral | neutral | 1.1984 | [1.180,1.216] | 1602a4e947e64090 | check | done | proceed | 4/10 |
| 2026-10-24T140216Z | bf16-cublaslt-algo-pick-in-prepare-m128-m256 | primary | bf16 GEMM dispatch (m=128/256, cuBLASLt algo selection) | done/neutral | neutral | 1.1879 | [1.170,1.206] | 26a8be34a5eb4426 | submit |  | proceed | 7/10 |
| 2026-10-24T140242Z | pooler-unpooler-m128-tma-wspec-rewrite | rewrite | pooler GEMM + unpooler GEMM (m=128) | done/regression | regression | 1.1639 | [1.146,1.181] | 61da9b91dab44bf4 | submit |  | proceed | 4/7 |
| 2026-10-24T140257Z | pooler-m32-tma-splitk-extend | primary | pooler GEMM (m=17..32) | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-24T165012Z | bf16-m16-swapab-wgmma-extend | primary | bf16 GEMM dispatch (m<=16, swap-AB Triton) | done/neutral | neutral | 1.2115 | [1.193,1.230] | 89fabef70bd745f7 | submit |  | proceed | 5/16 |
| 2026-10-24T165036Z | router-swapab-wgmma-all-m | primary | router GEMM | done/neutral | neutral | 1.1951 | [1.177,1.213] | 587e340476944b46 | submit |  | proceed | 4/10 |
| 2026-10-24T165102Z | bf16-small-m-cuda-gemv-rewrite | rewrite | bf16 GEMM dispatch (m<=8, hand-written CUDA weight stream) | done/regression | regression | 1.1667 | [1.149,1.184] | b76aca7ec1a048e7 | submit |  | proceed | 4/13 |
| 2026-10-24T172152Z | bf16-lossless-exp4-weight-stream-rewrite | rewrite | bf16 GEMM dispatch (m<=32, lossless-compressed weight stream) | done/regression | regression | 0.7071 | [0.697,0.718] | 542f2b169c934706 | submit |  | proceed | 2/17 |
| 2026-10-24T172235Z | bf16-verify-streamk-cutlass-wgmma-rewrite | rewrite | bf16 GEMM dispatch (m=129..256, persistent stream-K TMA/wgmma) | done/regression | regression | 1.1635 | [1.146,1.181] | f5b1361aaa0e48b5 | check |  | proceed | 0/8 |
| 2026-10-24T174421Z | bf16-restack-m16-swapab-router-swapab-m32 | primary | restack (m<=16 swap-AB GEMMs + router swap-AB) | done/neutral | neutral | 1.2215 | [1.203,1.240] | 1e90292c7b954d96 | submit | merged | proceed | 3/4 |
| 2026-10-25T001220Z | bf16-lossless-exp4-occupancy-dequant-once-stage2 | rewrite | lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel) | done/neutral | neutral | 0.8870 | [0.874,0.900] | f1024212dd2f49f3 | check |  | proceed | 13/13 |
| 2026-10-25T003715Z | bf16-lossless-exp4-warpspec-persistent-stage3 | rewrite | lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel) | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-25T003758Z | router-cuda-cluster-dsmem-splitk-rewrite | rewrite | router GEMM | done/regression | regression | 1.1897 | [1.172,1.208] | dc5d03c4d9604b0e | check |  | proceed | 7/10 |
| 2026-10-25T003826Z | o-unpooler-slice-cuda-cluster-dsmem-rewrite | rewrite | o GEMM + unpooler GEMM (latency-bound small streams) | done/regression | regression | 1.1710 | [1.153,1.189] | 3f3330f70b6e4cdc | check |  | proceed | 6/10 |
| 2026-10-25T003858Z | o-swapab-m32-splitk2-two-ctas-per-sm | primary | o GEMM (m=17..32, swap-AB) | done/regression | regression | 1.1860 | [1.168,1.204] | 75f09fa68f9c4415 | check |  | proceed | 4/8 |
| 2026-10-25T003858Z | qkv-swapab-m32-splitk8-stages3 | primary | qkv GEMM (m=17..32, swap-AB) | done/regression | regression | 1.1963 | [1.178,1.214] | a61778a08ec04ea6 | check |  | proceed | 3/8 |
| 2026-10-25T003858Z | unpooler-full-m16-swapab-route | primary | unpooler GEMM | done/regression | regression | 1.1885 | [1.171,1.206] | 05782267384343eb | check |  | proceed | 5/8 |
| 2026-10-25T011407Z | pooler-unpooler-m128-nn-layout-gated | primary | pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout) | done/regression | regression | 1.1953 | [1.177,1.213] | 0daf3faa83ab4e81 | check |  | proceed | 8/9 |
| 2026-10-25T011408Z | bf16-verify-swapab-wgmma-cluster-dsmem-rewrite | rewrite | bf16 GEMM dispatch (m=65..256, swap-AB wgmma + cluster DSMEM split-K) | done/neutral | neutral | 1.2155 | [1.197,1.234] | 3a7f42bc7f544886 | check |  | proceed | 3/10 |
| 2026-10-25T011408Z | bf16-lossless-exp4-decode-only-throughput-probe | primary | lossless exp4 compressed-weight stream kernel (_lossless_swapab_kernel) | done/regression | regression | 1.1904 | [1.173,1.208] | 9148142cee4441ad | check |  | proceed | 1/1 |
| 2026-10-25T011408Z | bf16-verify-deepgemm-bf16-gemm-nt-route | primary | bf16 GEMM dispatch (m=65..256, DeepGEMM bf16) | done/regression | regression | 1.2027 | [1.185,1.221] | 2d522d47ca5c4fc9 | check |  | proceed | 5/8 |
| 2026-10-25T011408Z | pooler-unpooler-full-bulk-tma-row-owner-stream-rewrite | rewrite | pooler GEMM + unpooler GEMM (m<=32 DRAM weight streams, bulk-TMA row-owner kernel) | done/regression | regression | 1.1828 | [1.165,1.201] | cd39d36e31814087 | check |  | proceed | 8/15 |
| 2026-10-25T035343Z | pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild | rewrite | pooler GEMM + unpooler GEMM (m<=32 DRAM weight streams, bulk-TMA row-owner kernel) | done/regression | regression | 1.0331 | [1.018,1.049] | 571ea532ad7a410d | check |  | proceed | 1/7 |
| 2026-10-25T042040Z | bf16-verify-deepgemm-sm90-kernel-asserted-rewrite | rewrite | bf16 GEMM dispatch (m=65..256, DeepGEMM sm90 BF16 kernel) | done/neutral | neutral | 1.2221 | [1.204,1.240] | 4c224bebdd384541 | check |  | proceed | 2/8 |
| 2026-10-25T042040Z | restack-pooler-unpooler-nn-layout-m128 | primary | pooler GEMM + unpooler GEMM (m=65..128, cuBLAS NN layout) | done/neutral | neutral | 1.2282 | [1.210,1.247] | 92cd4e758a924ca0 | check |  | proceed | 6/8 |
| 2026-10-25T042040Z | restack-router-cluster-dsmem-m8 | primary | router GEMM (m<=8) | done/neutral | neutral | 1.2199 | [1.202,1.238] | d323ce8ef6a54fde | check |  | proceed | 8/10 |
| 2026-10-25T042040Z | restack-unpooler-slice-cluster-dsmem-m8 | primary | unpooler GEMM (sliced n=1024, m<=8) | done/neutral | neutral | 1.2260 | [1.208,1.244] | e557d92969864449 | check |  | proceed | 7/9 |
| 2026-10-25T042040Z | router-cluster-dsmem-s2-flat-in-m-occupancy | rewrite | router GEMM (m<=32, cluster DSMEM kernel) | done/neutral | neutral | 1.2058 | [1.188,1.224] | ef8a8bca31ef420e | check |  | proceed | 4/10 |
| 2026-10-25T042040Z | router-m128-m256-cluster-dsmem-tiled-rewrite | rewrite | router GEMM (m=65..256) | done/neutral | neutral | 1.2189 | [1.201,1.237] | b7b2b55d04544660 | check |  | proceed | 5/8 |
| 2026-10-25T054102Z | restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 | primary | shape-gated restack: router GEMM m<=8 + unpooler-slice GEMM m<=8 + pooler/unpooler GEMM m=65..128 | done/neutral | neutral | 1.2366 | [1.218,1.255] | 84f17adc3f074a33 | check | done | proceed | 15/17 |
| 2026-10-25T110547Z | qkv-swapab-w128-splitk7-one-wave-m32 | primary | qkv GEMM (m<=32, swap-AB) | done/neutral | neutral | 1.2229 | [1.205,1.241] | c7138fe05598428e | submit |  | proceed | 1/5 |
| 2026-10-25T110625Z | bf16-verify32-cutlass-coop-256x128-splitk-stage2 | rewrite | bf16 GEMM dispatch (m=129..256, CUTLASS sm90 cooperative split-K) | done/neutral | neutral | 1.2483 | [1.230,1.267] | 92c42836caf54987 | check |  | proceed | 0/8 |
| 2026-10-25T145403Z | pooler-nn-layout-widen-m256 | primary | pooler GEMM (m=129..256, cuBLAS NN layout) | done/neutral | neutral | 1.2394 | [1.221,1.258] | 85c91578cc184287 | submit | done | proceed | 5/6 |
| 2026-10-25T145403Z | unpooler-full-m16-swapab16-blockt16 | primary | unpooler GEMM (full width n=8192, m=9..16, swap-AB BLOCK_T 16) | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-25T153141Z | bf16-verify16-cutlass-m128-onetile-splitk-rewrite | rewrite | bf16 GEMM dispatch (m=65..128, CUTLASS sm90 one-M-tile split-K) | done/neutral | neutral | 1.2423 | [1.224,1.261] | d4edd38aeed44334 | check |  | proceed | 0/6 |
| 2026-10-25T153141Z | pooler-k-halves-batched-gemm-m65-m256 | primary | pooler GEMM (m=65..256, K-split as a 2-batch cuBLAS GEMM) | done/neutral | neutral | 1.2470 | [1.228,1.266] | ce808e5a055f41b4 | submit |  | proceed | 2/3 |
| 2026-10-25T153141Z | router-hilo-tall-tile-m65-m256 | primary | router GEMM (m=65..256, hilo tile config) | done/neutral | neutral | 1.2539 | [1.235,1.273] | 864e8094b5bc4c1b | submit |  | proceed | 0/3 |
| 2026-10-25T153141Z | router-m128-m256-wgmma-swapab-tall-cluster-stage2 | rewrite | router GEMM (m=65..256, CUDA wgmma swap-AB cluster kernel) | done/neutral | neutral | 1.2612 | [1.242,1.280] | 38daa28c6a704810 | check |  | proceed | 1/3 |
| 2026-10-25T153141Z | unpooler-k-halves-batched-gemm-m65-m128 | primary | unpooler GEMM (full width, m=65..128, K-split as a 2-batch cuBLAS GEMM) | done/neutral | neutral | 1.2452 | [1.227,1.264] | e82feba3dd8d4efb | submit |  | proceed | 0/2 |
| 2026-10-25T153141Z | qkv-k-split-batched-gemm-m65-m256 | primary | qkv GEMM (m=65..256, K-split as a batched cuBLAS GEMM) | done/regression | regression | 1.2410 | [1.222,1.260] | 9d93c3d0cc184226 | submit |  | proceed | 0/3 |
| 2026-10-25T161307Z | combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt | primary | combination (pooler m=1 bulk row-owner + qkv_gate m=128 cuBLASLt single-node) | done/neutral | neutral | 1.2448 | [1.226,1.263] | 0ee2055590aa4e84 | submit |  | proceed | 2/3 |
| 2026-10-25T161307Z | combo-rowowner-reduce-free-m1-qkv-o-plus-naive | primary | combination (naive stack + bulk row-owner widened to qkv_gate/o m=1) | done/neutral | neutral | 1.2566 | [1.238,1.275] | 741072dca5fc4c9d | submit | done | proceed | 3/4 |
| 2026-10-25T161307Z | combo-cublaslt-single-node-m128-m256-plus-naive | primary | combination (naive stack + cuBLASLt single-node probe on pooler/unpooler/o m=128/256) | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-25T211856Z | pooler-k-halves-fp32-partials-restack-m128-m256 | primary | pooler GEMM (m=128/256, K-split as a 2-batch cuBLAS GEMM) | done/neutral | neutral | 1.2683 | [1.249,1.287] | de95f81782374124 | check |  | proceed | 2/3 |
| 2026-10-25T211857Z | combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16 | primary | combination (pooler K-halves m=128/256 + qkv row-owner m=8/16 + per-cell Lt plans m=128/256) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T211857Z | bf16-verify-deepgemm-sm90-explicit-instantiation-stage2 | rewrite | bf16 GEMM dispatch (m=65..256, DeepGEMM sm90 BF16 kernel) | done/neutral | neutral | 1.2606 | [1.242,1.280] | 566a79e403754c76 | check |  | proceed | 2/5 |
| 2026-10-25T211857Z | bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole | primary | bf16 GEMM dispatch (m=128/256, per-cell cuBLASLt plan selector) | done/neutral | neutral | 1.2525 | [1.234,1.271] | 37ed8722bc9b402d | check |  | proceed | 2/4 |
| 2026-10-25T211857Z | qkv-rowowner-a2-widen-m8-m16 | primary | qkv GEMM (m=8/16, bulk row-owner kernel) | done/neutral | neutral | 1.2603 | [1.241,1.279] | 0124ea8fcaf14b3c | check |  | proceed | 1/3 |
| 2026-10-26T015638Z | pooler-k-halves-fp32-partials-restack-m128-m256-cf | confirm→pooler-k-halves-fp32-partials-restack-m128-m256 | pooler GEMM (m=128/256, K-split as a 2-batch cuBLAS GEMM) | done/neutral | neutral | 1.2590 | [1.240,1.278] | caaa76cc288d47c7 | check |  | proceed |  |
| 2026-10-26T221113Z | pooler-khalf-m128-fp32-partials-triton-castadd | primary | pooler GEMM (m=128, K-split as a 2-batch cuBLAS GEMM) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T221132Z | b12-qkv-m1-r32-one-cta-per-sm-readflush-gate | primary | qkv GEMM (m=1, bulk12 12-bit row-owner CTA shape) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-26T221149Z | qkv-swapab16-five-stage-ring-m8-m16 | primary | qkv GEMM (m<=16, swap-AB Triton) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-26T221215Z | b12-unpooler-slice-m1-r16-rowowner | primary | unpooler GEMM (sliced n=1024, m=1, bulk12 12-bit row-owner) | done/neutral | neutral | 1.2737 | [1.255,1.293] | a642424734f54365 | check |  | proceed | 1/2 |
| 2026-10-26T221237Z | lt-nn-layout-cells-o-m128-qkv-m128 | primary | o GEMM + qkv GEMM (m=128, cuBLASLt plan on an NN weight copy) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-26T221256Z | lt-config-sweep-m256-beyond-heuristic | primary | bf16 GEMM dispatch (m=256, cuBLASLt configs beyond the heuristic) | queue | queued |  |  |  |  |  | required |  |

## Held

- combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16: retired per DISTILL 4.4 (2026-10-27): built on the pre-bulk12 base, its qkv cells never routed, superseded by khalf (b2) landed in 8f233e88
- pooler-khalf-m128-fp32-partials-triton-castadd: superseded: khalf (b2) landed on master by the campaign 2026-10-27 (commit 8f233e88, branch hyp/bf16-khalf), board 1.2735
- b12-qkv-m1-r32-one-cta-per-sm-readflush-gate: adversarial review: Every part of this hypothesis, (a) to (e), is already on master: it came in through d7bf1c76 (hyp/bf16-readflush, which names this slug) and landed in 8f233e88 
