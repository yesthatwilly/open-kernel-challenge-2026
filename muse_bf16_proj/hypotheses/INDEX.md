# muse_bf16_proj hypotheses

Regenerated 2026-10-24T075012Z. Best: 1.1510x [1.1338,1.1683] job `bd9fa14151644cee` (check)

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
| 2026-10-23T230544Z | router-triton-hilo-fused-control-splitk-all-m | control→router-triton-hilo-fused | router GEMM | done/gain | gain | 1.1510 | [1.134,1.168] | bd9fa14151644cee | check | submitted | proceed | 7/7 |
| 2026-10-24T024326Z | router-triton-hilo-fused-control-hybrid-cublas-above-m32 | control→router-triton-hilo-fused-control-splitk-all-m | router GEMM | queue | hold |  |  |  |  |  | required |  |
| 2026-10-24T024434Z | router-hilo-addmm-epilogue-control-inplace-out | control→router-hilo-addmm-epilogue | router GEMM | queue | hold |  |  |  |  |  | required |  |
| 2026-10-24T032454Z | unpooler-full-triton-skinny-m32-control-blockm16-all-m | control→unpooler-full-triton-skinny-m32 | unpooler GEMM | done/neutral | neutral | 0.9990 | [0.984,1.014] | 09abdd472faa4dce | check |  | proceed | 3/4 |
| 2026-10-24T074946Z | pooler-triton-splitk2-blockn64-semaphore | primary | pooler GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T075006Z | qkv-gate-triton-splitk4-blockn64-semaphore | primary | qkv GEMM | queue | queued |  |  |  |  |  | required |  |

## Held

- pooler-skinny-tiled-weight-layout: skinny-Triton family settled: pooler neutral, qkv-gate regression, unpooler-full neutral, o neutral (1.006x), and the BLOCK_M=16 control showed the m=32 penalty is not the tile (20.8 us vs 18.4 us in-run at m=16). The reviewer expected both of this hypothesis's mechanisms to be ~zero, and it builds on the unpromoted pooler branch, so a gain could not be promoted without a rebase and rerun.
- router-triton-hilo-fused-control-hybrid-cublas-above-m32: superseded: router-triton-hilo-fused-control-splitk-all-m landed (1.151x); its 3-D Triton grid beats cuBLAS hi/lo at every m (5.7-9.3 us vs 10.5-17.7 us), so a hybrid using cuBLAS above m=32 can only be slower
- router-hilo-addmm-epilogue-control-inplace-out: superseded: master's router no longer uses the cuBLAS hi/lo pair (Triton split-K landed, 1.151x), so the add node this control targets is gone; the primary read flat (1.026x, within CI)
