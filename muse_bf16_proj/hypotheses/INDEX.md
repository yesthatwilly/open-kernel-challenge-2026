# muse_bf16_proj hypotheses

Regenerated 2026-10-24T141733Z. Best: 1.1760x [1.1584,1.1937] job `875f3e0172ee48a1` (check)

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
| 2026-10-24T140147Z | bf16-m32-swapab-wgmma-triton | primary | bf16 GEMM dispatch (m=17..32, swap-AB Triton) | running | running |  |  | 1602a4e947e64090 | check |  | proceed |  |
| 2026-10-24T140216Z | bf16-cublaslt-algo-pick-in-prepare-m128-m256 | primary | bf16 GEMM dispatch (m=128/256, cuBLASLt algo selection) | queue | queued |  |  |  |  |  | proceed |  |
| 2026-10-24T140242Z | pooler-unpooler-m128-tma-wspec-rewrite | rewrite | pooler GEMM + unpooler GEMM (m=128) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T140257Z | pooler-m32-tma-splitk-extend | primary | pooler GEMM (m=17..32) | queue | queued |  |  |  |  |  | required |  |

## Held

- pooler-skinny-tiled-weight-layout: skinny-Triton family settled: pooler neutral, qkv-gate regression, unpooler-full neutral, o neutral (1.006x), and the BLOCK_M=16 control showed the m=32 penalty is not the tile (20.8 us vs 18.4 us in-run at m=16). The reviewer expected both of this hypothesis's mechanisms to be ~zero, and it builds on the unpromoted pooler branch, so a gain could not be promoted without a rebase and rerun.
- router-triton-hilo-fused-control-hybrid-cublas-above-m32: superseded: router-triton-hilo-fused-control-splitk-all-m landed (1.151x); its 3-D Triton grid beats cuBLAS hi/lo at every m (5.7-9.3 us vs 10.5-17.7 us), so a hybrid using cuBLAS above m=32 can only be slower
- router-hilo-addmm-epilogue-control-inplace-out: superseded: master's router no longer uses the cuBLAS hi/lo pair (Triton split-K landed, 1.151x), so the add node this control targets is gone; the primary read flat (1.026x, within CI)
- qkv-gate-o-m128-wgmma-splitk-semaphore: same m=128 wgmma split-K design as pooler-unpooler-m128-wgmma-splitk-semaphore (dispatching now); its reviewer asked to wait for that harvest; re-queue by hand if that kernel reaches its target
