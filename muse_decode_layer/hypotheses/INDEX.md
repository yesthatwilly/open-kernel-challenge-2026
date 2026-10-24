# muse_decode_layer hypotheses

Regenerated 2026-10-24T015224Z. Best: 0.0045x [0.0044,0.0046] job `3c3181b1fce24b9a` (check)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T225318Z | sglang-pipeline-parity-router-hilo-m8 | primary | router GEMM | running | running |  |  | e260dfe413e448c7 | check |  | proceed |  |
| 2026-10-23T225338Z | cache-store-fused-fp8-cast | primary | cache store | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225353Z | expert-id-map-single-kernel | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225545Z | fa3-num-splits-full-no-metadata | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225545Z | marlin-block-m-cap-16 | primary | w13 marlin | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225545Z | moe-align-local-expert-count | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225545Z | pooler-gemm-splitk-autotune | primary | pooler GEMM | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225545Z | trtllm-decode-full-attention | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225609Z | fa3-num-splits-sliding | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225609Z | moe-align-single-kernel-small-t | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
