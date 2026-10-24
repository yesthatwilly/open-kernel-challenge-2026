# muse_bf16_proj hypotheses

Regenerated 2026-10-24T035911Z. Best: 1.0307x [1.0152,1.0462] job `9aa98e889be140eb` (submit)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T221829Z | sglang-gemm-dispatch-parity |  | bf16 GEMM dispatch | done/gain | gain | 1.0022 | [0.987,1.017] | 017804b34fed40b3 | check | merged-unsubmitted | proceed | 7/7 |
| 2026-10-23T225128Z | router-hilo-at-m8 | primary | router GEMM | done/gain | gain | 1.0219 | [1.007,1.037] | aa3d033c208e4511 | check | done | proceed | 4/4 |
| 2026-10-23T225128Z | router-hilo-addmm-epilogue | primary | router GEMM | done/neutral | neutral | 1.0256 | [1.010,1.041] | 743020b6b43d458c | check |  | proceed | 2/6 |
| 2026-10-23T225128Z | router-triton-hilo-fused | primary | router GEMM | done/regression | regression | 0.9402 | [0.926,0.954] | c2a8de5969374dff | check |  | proceed | 4/7 |
| 2026-10-23T225201Z | pooler-triton-skinny-m32 | primary | pooler GEMM | done/neutral | neutral | 0.9983 | [0.983,1.013] | 9f0ad00bbc874d31 | check |  | proceed | 4/6 |
| 2026-10-23T225201Z | unpooler-slice-triton-gemv | primary | unpooler GEMM | done/regression | regression | 0.8801 | [0.867,0.893] | 4c2b113f81434570 | check |  | proceed | 1/4 |
| 2026-10-23T225303Z | bf16-cublaslt-backend-toggle | primary | bf16 GEMM dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225303Z | o-triton-skinny-m32 | primary | o GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225303Z | pooler-skinny-tiled-weight-layout | primary | pooler GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225303Z | unpooler-full-triton-skinny-m32 | primary | unpooler GEMM | running | running |  |  | 635c2693bd4147a4 | check |  | proceed |  |
| 2026-10-23T225317Z | qkv-gate-triton-skinny-m32 | primary | qkv GEMM | running | running |  |  | abcd45776f244c8c | check |  | proceed |  |
| 2026-10-23T230544Z | router-triton-hilo-fused-control-splitk-all-m | control→router-triton-hilo-fused | router GEMM | done/gain | gain | 1.1510 | [1.134,1.168] | bd9fa14151644cee | check | unmerged | proceed | 7/7 |
| 2026-10-24T024326Z | router-triton-hilo-fused-control-hybrid-cublas-above-m32 | control→router-triton-hilo-fused-control-splitk-all-m | router GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T024434Z | router-hilo-addmm-epilogue-control-inplace-out | control→router-hilo-addmm-epilogue | router GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T032454Z | unpooler-full-triton-skinny-m32-control-blockm16-all-m | control→unpooler-full-triton-skinny-m32 | unpooler GEMM | queue | queued |  |  |  |  |  | required |  |
