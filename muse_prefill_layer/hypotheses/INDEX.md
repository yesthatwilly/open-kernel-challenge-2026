# muse_prefill_layer hypotheses

Regenerated 2026-10-24T024944Z. Best: none yet

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T230250Z | experts-dequant-grouped-mm-cutlass | primary | w13 marlin + w2 marlin (GEMM engine: CUTLASS grouped GEMM instead of Triton) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-overlap-attention | primary | w13 marlin + w2 marlin (dequant placement on a side stream) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-threshold-turn4 | primary | w13 marlin + w2 marlin (Marlin vs dequant crossover) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-triton-moe-16k | primary | w13 marlin + w2 marlin (expert grouped GEMM engine at 16K tokens) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-triton-w4a16-inkernel-fp4 | primary | w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, for the turn shapes) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | fa3-batched-mode | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | fa3-contiguous-kv-gather | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | moe-align-side-stream | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | qk-norm-fp8-cache-store-fusion | primary | pre-attention norm+rope + cache store | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | sglang-pipeline-parity | primary | whole-layer parity (SGLang pipeline) + routing remap | running | running |  |  | 62025c368c3c4e8e | check |  | proceed |  |
