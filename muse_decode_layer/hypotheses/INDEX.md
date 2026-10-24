# muse_decode_layer hypotheses

Regenerated 2026-10-24T053312Z. Best: 1.0200x [1.0047,1.0353] job `5755a412dda244a1` (submit)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T225318Z | sglang-pipeline-parity-router-hilo-m8 | primary | router GEMM | done/gain | gain | 1.0216 | [1.006,1.037] | e260dfe413e448c7 | check | done | proceed | 1/6 |
| 2026-10-23T225338Z | cache-store-fused-fp8-cast | primary | cache store | done/neutral | neutral | 1.0303 | [1.015,1.046] | 7264526cf9f94c37 | check |  | proceed | 2/4 |
| 2026-10-23T225353Z | expert-id-map-single-kernel | primary | MoE align/dispatch | running | running |  |  | 11b8df414a2245ae | check |  | proceed |  |
| 2026-10-23T225545Z | fa3-num-splits-full-no-metadata | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T225545Z | moe-align-local-expert-count | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225545Z | trtllm-decode-full-attention | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225545Z | marlin-block-m-cap-16 | primary | w13 marlin | running | running |  |  | 0730135cdcac43b7 | check |  | proceed |  |
| 2026-10-23T225545Z | pooler-gemm-splitk-autotune | primary | pooler GEMM | running | running |  |  | 2e1a43ab85324776 | check |  | proceed |  |
| 2026-10-23T225609Z | fa3-num-splits-sliding | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T225609Z | moe-align-single-kernel-small-t | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |

## Held

- fa3-num-splits-full-no-metadata: adversarial review: An explicit num_splits cannot raise FA3's split count on this call: every flash_attn_with_kvcache call with cache_seqlens is varlen, so FA3 runs its device-side
- fa3-num-splits-sliding: adversarial review: The premise (FA3 sizes the sliding split-KV for the 80K page-table extent) is contradicted by both FA3's own get_num_splits, which clamps seqlen_k by the window
