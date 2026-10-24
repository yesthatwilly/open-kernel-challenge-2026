# muse_decode_layer hypotheses

Regenerated 2026-10-24T075012Z. Best: 1.0621x [1.0462,1.0780] job `0730135cdcac43b7` (check)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T225318Z | sglang-pipeline-parity-router-hilo-m8 | primary | router GEMM | done/gain | gain | 1.0216 | [1.006,1.037] | e260dfe413e448c7 | check | done | proceed | 1/6 |
| 2026-10-23T225338Z | cache-store-fused-fp8-cast | primary | cache store | done/neutral | neutral | 1.0303 | [1.015,1.046] | 7264526cf9f94c37 | check |  | proceed | 2/4 |
| 2026-10-23T225353Z | expert-id-map-single-kernel | primary | MoE align/dispatch | done/neutral | neutral | 1.0374 | [1.022,1.053] | 11b8df414a2245ae | check |  | proceed | 2/4 |
| 2026-10-23T225545Z | fa3-num-splits-full-no-metadata | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T225545Z | moe-align-local-expert-count | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225545Z | trtllm-decode-full-attention | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225545Z | marlin-block-m-cap-16 | primary | w13 marlin | done/gain | gain | 1.0621 | [1.046,1.078] | 0730135cdcac43b7 | check | merged | proceed | 4/4 |
| 2026-10-23T225545Z | pooler-gemm-splitk-autotune | primary | pooler GEMM | done/neutral | neutral | 1.0189 | [1.004,1.034] | 2e1a43ab85324776 | check |  | proceed | 2/4 |
| 2026-10-23T225609Z | fa3-num-splits-sliding | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T225609Z | moe-align-single-kernel-small-t | primary | MoE align/dispatch | running | running |  |  | 0a0af8669ffe4cb0 | check |  | proceed |  |
| 2026-10-24T071442Z | moe-align-single-kernel-small-t-control-capacity-256 | control→moe-align-single-kernel-small-t | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |

## Held

- fa3-num-splits-full-no-metadata: adversarial review: An explicit num_splits cannot raise FA3's split count on this call: every flash_attn_with_kvcache call with cache_seqlens is varlen, so FA3 runs its device-side
- fa3-num-splits-sliding: adversarial review: The premise (FA3 sizes the sliding split-KV for the 80K page-table extent) is contradicted by both FA3's own get_num_splits, which clamps seqlen_k by the window
