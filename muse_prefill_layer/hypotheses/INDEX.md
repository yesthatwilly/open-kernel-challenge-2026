# muse_prefill_layer hypotheses

Regenerated 2026-10-24T075012Z. Best: 1.0039x [0.9888,1.0189] job `62025c368c3c4e8e` (check)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T230250Z | experts-dequant-grouped-mm-cutlass | primary | w13 marlin + w2 marlin (GEMM engine: CUTLASS grouped GEMM instead of Triton) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-overlap-attention | primary | w13 marlin + w2 marlin (dequant placement on a side stream) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-threshold-turn4 | primary | w13 marlin + w2 marlin (Marlin vs dequant crossover) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-triton-moe-16k | primary | w13 marlin + w2 marlin (expert grouped GEMM engine at 16K tokens) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-23T230250Z | experts-triton-w4a16-inkernel-fp4 | primary | w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, for the turn shapes) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | fa3-batched-mode | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T230250Z | qk-norm-fp8-cache-store-fusion | primary | pre-attention norm+rope + cache store | running | running |  |  | fe8883a2fc264338 | check |  | proceed |  |
| 2026-10-23T230250Z | sglang-pipeline-parity | primary | whole-layer parity (SGLang pipeline) + routing remap | done/gain | gain | 1.0039 | [0.989,1.019] | 62025c368c3c4e8e | check | merged-unsubmitted | proceed | 3/3 |
| 2026-10-23T230250Z | moe-align-side-stream | primary | MoE align/dispatch | done/neutral | neutral | 1.0076 | [0.993,1.023] | f70d1f2eaef842d2 | check |  | proceed | 1/3 |
| 2026-10-23T230250Z | fa3-contiguous-kv-gather | primary | FA3 call | done/regression | regression | 0.9879 | [0.973,1.003] | 75bbf278dd9e430e | check |  | proceed | 2/6 |

## Held

- experts-dequant-triton-moe-16k: needs human: rules check. The change materializes a full 3.2 GB BF16 copy of the NVFP4 expert weights in global memory on every call, then runs BF16 grouped GEMMs on it. TASK.md allows 'dequantizing inside your kernels' but forbids storing weights at a different precision; a per-call BF16 weight tensor is the strongest reading of that clause (reviewer's words) and may not survive the organizers' manual review. Proceed only if the user confirms it is within the rules; the adversarial review otherwise says proceed (~1.08x geomean upside).
- fa3-batched-mode: adversarial review: The occupancy mechanism is contradicted by the landed profile: the FA3 kernel already runs at the same ~590-606 TFLOP/s on turn_1 full (96 nominal CTAs) as on t
