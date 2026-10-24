# muse_prefill_layer hypotheses

Regenerated 2026-10-24T184810Z. Best: 1.0592x [1.0433,1.0751] job `ba9c16b9cc6b4605` (submit)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T230250Z | experts-dequant-grouped-mm-cutlass | primary | w13 marlin + w2 marlin (GEMM engine: CUTLASS grouped GEMM instead of Triton) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-overlap-attention | primary | w13 marlin + w2 marlin (dequant placement on a side stream) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T230250Z | experts-dequant-threshold-turn4 | primary | w13 marlin + w2 marlin (Marlin vs dequant crossover) | queue | hold |  |  |  |  |  | required |  |
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
| 2026-10-24T075513Z | router-pooler-concat-gemm-turn | primary | router GEMM + pooler GEMM (one concatenated GEMM, no side stream) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T092506Z | experts-triton-w4a16-swapab-reglhs | rewrite | w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, stage 2: swap-AB register-LHS dot) | done/neutral | neutral | 1.0068 | [0.992,1.022] | 00148fbedfb6414f | check |  | proceed | 1/3 |
| 2026-10-24T111900Z | experts-triton-w4a16-first-turn-blockt256 | rewrite | w13 marlin + w2 marlin (W4A16 Triton engine, stage 3: first_turn token tile 256) | done/neutral | neutral | 1.0229 | [1.008,1.038] | b33b87b5083b4d1c | check | merged | proceed | 2/2 |
| 2026-10-24T124504Z | experts-hybrid-marlin-turn-triton-first-turn | rewrite | w13 marlin + w2 marlin (stage 4: hybrid shape gate, Marlin at T<=4096, Triton NVFP4 above) | done/gain | gain | 1.0599 | [1.044,1.076] | fb1e52d2119d4175 | check | done | proceed | 2/3 |
| 2026-10-24T140058Z | experts-triton-nvfp4-packed-asm-dequant | rewrite | w13 marlin + w2 marlin (Triton NVFP4 engine, stage 5: packed inline-PTX dequant) | done/neutral | neutral | 1.0556 | [1.040,1.071] | 49ff8248b96547de | submit |  | proceed | 0/2 |
| 2026-10-24T140119Z | experts-triton-w13-fused-silu-epilogue | primary | silu-mul (fused into the w13 Triton NVFP4 GEMM epilogue, T > 4096) | done/neutral | neutral | 1.0525 | [1.037,1.068] | 108231906a30407e | submit |  | proceed | 0/2 |
| 2026-10-24T140133Z | experts-triton-packed-dequant-turn4 | rewrite | w13 marlin + w2 marlin (Triton NVFP4 engine, stage 6: take over turn_4) | queue | queued |  |  |  |  |  | proceed |  |
| 2026-10-24T140154Z | experts-triton-turn1-blockt32-stream | rewrite | w13 marlin + w2 marlin (Triton NVFP4 engine, stage 7: take over turn_1 with a weight-stream config) | queue | queued |  |  |  |  |  | proceed |  |
| 2026-10-24T165318Z | dense-gemm-cublaslt-algo-pick-prefill-t | primary | dense bf16 GEMM dispatch (qkv, o, pooler, unpooler; cuBLASLt algo selection) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T165335Z | silu-mul-triton-local-rows-marlin-turn | primary | silu-mul (Marlin path, T <= 4096) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T165356Z | norm-residual-boundaries-triton-row-kernel | primary | attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T172035Z | experts-cuda-wgmma-mixed-input-grouped-gemm | rewrite | w13 marlin + w2 marlin (CUDA sm_90a warp-specialized wgmma W4A16 grouped GEMM, stage 1: first_turn) | done/neutral | neutral | 1.0637 | [1.048,1.080] | 1821bb0e1f7246e2 | submit |  | proceed | 0/3 |
| 2026-10-24T172146Z | attn-full-prefill-cuda-fp8-exp-offload | rewrite | FA3 call (CUDA sm_90a FP8 prefill flash attention with software-exp offload, stage 1: full layers, page size 1) | done/regression | regression | 0.8813 | [0.868,0.895] | f5c30b15fd284c15 | submit |  | proceed | 0/4 |
| 2026-10-24T183246Z | sliding-attn-block-triton-gqa-fused-normgate | rewrite | sliding-window attention block (q FP8 cast + FA3 + FlashAttnFwdCombine + FusedRMSNormGate) on sliding layers | running | running |  |  | 78d8d7bdec55491c | submit |  | proceed |  |

## Held

- experts-dequant-grouped-mm-cutlass: standing answer: the per-call BF16 dequant family stays held; pursue in-kernel dequant only (experts-cuda-wgmma-mixed-input-grouped-gemm)
- experts-dequant-overlap-attention: standing answer: the per-call BF16 dequant family stays held; pursue in-kernel dequant only (experts-cuda-wgmma-mixed-input-grouped-gemm)
- experts-dequant-threshold-turn4: standing answer: the per-call BF16 dequant family stays held; pursue in-kernel dequant only (experts-cuda-wgmma-mixed-input-grouped-gemm)
- experts-dequant-triton-moe-16k: needs human: rules check. The change materializes a full 3.2 GB BF16 copy of the NVFP4 expert weights in global memory on every call, then runs BF16 grouped GEMMs on it. TASK.md allows 'dequantizing inside your kernels' but forbids storing weights at a different precision; a per-call BF16 weight tensor is the strongest reading of that clause (reviewer's words) and may not survive the organizers' manual review. Proceed only if the user confirms it is within the rules; the adversarial review otherwise says proceed (~1.08x geomean upside).
- fa3-batched-mode: adversarial review: The occupancy mechanism is contradicted by the landed profile: the FA3 kernel already runs at the same ~590-606 TFLOP/s on turn_1 full (96 nominal CTAs) as on t
- experts-marlin-block-m-48-turn4: block 48 at turn_1 was neutral (1.003x, 0606f849d2f441ad) and local-rows regressed (0.984x); block-m tuning on prefill is settled, and prefill has 3 checks for the rewrite
- router-triton-hilo-splitk-turn: reviewer: the side-stream router (~40-45 us) already fits under the main branch (the 39.3 us nvjet is qkv_gate, not the pooler), expected 0.3-1% inside noise; prefill has 2 checks and the rewrite's next stage needs them
- router-pooler-concat-gemm-turn: adversarial review: Rule-clean, but the size of the win rests on a router side branch of ~80-90 us that the current master profile does not show. At turn_1 sliding the router pair 
- dense-gemm-cublaslt-algo-pick-prefill-t: reviewer: claim only after the bf16 precedent (bf16-cublaslt-algo-pick-in-prepare-m128-m256, 26a8be34a5eb4426) harvests; ~0.5-1.1% headline; time pooler keys with the router running or require a >=5% margin
- silu-mul-triton-local-rows-marlin-turn: projected ~0.7% is below run-to-run noise (shape medians move 2-6% between batches; see LESSONS); not worth an agent plus a run. Fold into a larger prefill change if one lands.
- norm-residual-boundaries-triton-row-kernel: reviewer: same one-row-per-program structure as the incumbent; expect 0.2-0.4% headline, below run noise; re-mint as a multi-row/persistent variant if wanted
