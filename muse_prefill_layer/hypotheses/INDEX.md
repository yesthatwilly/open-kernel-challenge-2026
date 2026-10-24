# muse_prefill_layer hypotheses

Regenerated 2026-10-24T111615Z. Best: 1.0039x [0.9888,1.0189] job `62025c368c3c4e8e` (check)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T230250Z | experts-dequant-grouped-mm-cutlass | primary | w13 marlin + w2 marlin (GEMM engine: CUTLASS grouped GEMM instead of Triton) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-overlap-attention | primary | w13 marlin + w2 marlin (dequant placement on a side stream) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-threshold-turn4 | primary | w13 marlin + w2 marlin (Marlin vs dequant crossover) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-triton-moe-16k | primary | w13 marlin + w2 marlin (expert grouped GEMM engine at 16K tokens) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-23T230250Z | fa3-batched-mode | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T230250Z | sglang-pipeline-parity | primary | whole-layer parity (SGLang pipeline) + routing remap | done/gain | gain | 1.0039 | [0.989,1.019] | 62025c368c3c4e8e | check | merged-unsubmitted | proceed | 3/3 |
| 2026-10-23T230250Z | moe-align-side-stream | primary | MoE align/dispatch | done/neutral | neutral | 1.0076 | [0.993,1.023] | f70d1f2eaef842d2 | check |  | proceed | 1/3 |
| 2026-10-23T230250Z | qk-norm-fp8-cache-store-fusion | primary | pre-attention norm+rope + cache store | done/neutral | neutral | 1.0093 | [0.994,1.024] | fe8883a2fc264338 | check |  | proceed | 2/3 |
| 2026-10-23T230250Z | experts-triton-w4a16-inkernel-fp4 | rewrite | w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, for the turn shapes) | done/regression | regression | 0.6891 | [0.679,0.699] | 218515e2d0b14ce7 | check |  | proceed | 0/3 |
| 2026-10-23T230250Z | fa3-contiguous-kv-gather | primary | FA3 call | done/regression | regression | 0.9879 | [0.973,1.003] | 75bbf278dd9e430e | check |  | proceed | 2/6 |
| 2026-10-24T075409Z | experts-marlin-block-m-local-rows | primary | w13 marlin + w2 marlin (block_size_m pick) | done/regression | regression | 0.9836 | [0.969,0.998] | 8e9943ec1b004b0f | check |  | proceed | 2/3 |
| 2026-10-24T075422Z | experts-marlin-block-m-48-turn1 | primary | w13 marlin + w2 marlin (block_size_m pick) | done/neutral | neutral | 1.0030 | [0.988,1.018] | 0606f849d2f441ad | check |  | proceed | 2/3 |
| 2026-10-24T075435Z | experts-marlin-block-m-48-turn4 | primary | w13 marlin + w2 marlin (block_size_m pick) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-24T075454Z | router-triton-hilo-splitk-turn | primary | router GEMM | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T075513Z | router-pooler-concat-gemm-turn | primary | router GEMM + pooler GEMM (one concatenated GEMM, no side stream) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T092506Z | experts-triton-w4a16-swapab-reglhs | rewrite | w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, stage 2: swap-AB register-LHS dot) | done/neutral | neutral | 1.0068 | [0.992,1.022] | 00148fbedfb6414f | check |  | proceed | 1/3 |

## Held

- experts-dequant-triton-moe-16k: needs human: rules check. The change materializes a full 3.2 GB BF16 copy of the NVFP4 expert weights in global memory on every call, then runs BF16 grouped GEMMs on it. TASK.md allows 'dequantizing inside your kernels' but forbids storing weights at a different precision; a per-call BF16 weight tensor is the strongest reading of that clause (reviewer's words) and may not survive the organizers' manual review. Proceed only if the user confirms it is within the rules; the adversarial review otherwise says proceed (~1.08x geomean upside).
- fa3-batched-mode: adversarial review: The occupancy mechanism is contradicted by the landed profile: the FA3 kernel already runs at the same ~590-606 TFLOP/s on turn_1 full (96 nominal CTAs) as on t
- experts-marlin-block-m-48-turn4: block 48 at turn_1 was neutral (1.003x, 0606f849d2f441ad) and local-rows regressed (0.984x); block-m tuning on prefill is settled, and prefill has 3 checks for the rewrite
- router-triton-hilo-splitk-turn: reviewer: the side-stream router (~40-45 us) already fits under the main branch (the 39.3 us nvjet is qkv_gate, not the pooler), expected 0.3-1% inside noise; prefill has 2 checks and the rewrite's next stage needs them
