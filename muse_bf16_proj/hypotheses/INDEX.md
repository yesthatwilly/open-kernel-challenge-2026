# muse_bf16_proj hypotheses

Regenerated 2026-10-23T225814Z. Best: 0.9021x [0.8885,0.9156] job `366189f0f1114da1` (check)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T221829Z | sglang-gemm-dispatch-parity |  | bf16 GEMM dispatch | done/gain | gain | 1.0022 |  | 017804b34fed40b3 | check | pending | proceed | 7/7 |
| 2026-10-23T225128Z | router-hilo-addmm-epilogue | primary | router GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225128Z | router-hilo-at-m8 | primary | router GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225128Z | router-triton-hilo-fused | primary | router GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225201Z | pooler-triton-skinny-m32 | primary | pooler GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225201Z | unpooler-slice-triton-gemv | primary | unpooler GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225303Z | bf16-cublaslt-backend-toggle | primary | bf16 GEMM dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225303Z | o-triton-skinny-m32 | primary | o GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225303Z | pooler-skinny-tiled-weight-layout | primary | pooler GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225303Z | unpooler-full-triton-skinny-m32 | primary | unpooler GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225317Z | qkv-gate-triton-skinny-m32 | primary | qkv GEMM | queue | queued |  |  |  |  |  | required |  |
