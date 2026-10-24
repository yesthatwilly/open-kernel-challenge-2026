# muse_decode_layer hypotheses

Regenerated 2026-10-24T135615Z. Best: 1.0874x [1.0710,1.1037] job `fb07a3bd3aa54baf` (check)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T225318Z | sglang-pipeline-parity-router-hilo-m8 | primary | router GEMM | done/gain | gain | 1.0216 | [1.006,1.037] | e260dfe413e448c7 | check | done | proceed | 1/6 |
| 2026-10-23T225338Z | cache-store-fused-fp8-cast | primary | cache store | done/neutral | neutral | 1.0303 | [1.015,1.046] | 7264526cf9f94c37 | check |  | proceed | 2/4 |
| 2026-10-23T225353Z | expert-id-map-single-kernel | primary | MoE align/dispatch | done/neutral | neutral | 1.0374 | [1.022,1.053] | 11b8df414a2245ae | check |  | proceed | 2/4 |
| 2026-10-23T225545Z | fa3-num-splits-full-no-metadata | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T225545Z | moe-align-local-expert-count | primary | MoE align/dispatch | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-23T225545Z | trtllm-decode-full-attention | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T225545Z | marlin-block-m-cap-16 | primary | w13 marlin | done/gain | gain | 1.0621 | [1.046,1.078] | 0730135cdcac43b7 | check | done | proceed | 4/4 |
| 2026-10-23T225545Z | pooler-gemm-splitk-autotune | primary | pooler GEMM | done/neutral | neutral | 1.0189 | [1.004,1.034] | 2e1a43ab85324776 | check |  | proceed | 2/4 |
| 2026-10-23T225609Z | fa3-num-splits-sliding | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T225609Z | moe-align-single-kernel-small-t | primary | MoE align/dispatch | done/neutral | neutral | 1.0431 | [1.027,1.059] | 0a0af8669ffe4cb0 | check |  | proceed | 2/6 |
| 2026-10-24T071442Z | moe-align-single-kernel-small-t-control-capacity-256 | control→moe-align-single-kernel-small-t | MoE align/dispatch | queue | hold |  |  |  |  |  | required |  |
| 2026-10-24T075400Z | fa3-full-single-split-probe | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T075400Z | router-triton-splitk-side-stream | primary | router GEMM | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T075400Z | triton-flash-decode-full-verify-paired-kv | rewrite | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T075400Z | fused-preattention-triton | primary | pre-attention norm+rope | done/neutral | neutral | 1.0970 | [1.081,1.113] | c7c26e2fd8b142a7 | check |  | proceed | 3/6 |
| 2026-10-24T075400Z | triton-flash-decode-full-paired-kv | rewrite | FA3 call | done/neutral | neutral | 1.0702 | [1.054,1.086] | e82b11f9d9f64b23 | check |  | proceed | 1/5 |
| 2026-10-24T100416Z | triton-flash-decode-full-oversub-kv-prefetch | rewrite | FA3 call | done/regression | regression | 1.0413 | [1.026,1.057] | e55908a05eec4000 | check |  | proceed | 3/5 |
| 2026-10-24T103909Z | fa3-hdim128-head-pairing-batched | primary | FA3 call | done/neutral | neutral | 1.0874 | [1.071,1.104] | fb07a3bd3aa54baf | check | merged | proceed | 7/7 |

## Held

- fa3-num-splits-full-no-metadata: adversarial review: An explicit num_splits cannot raise FA3's split count on this call: every flash_attn_with_kvcache call with cache_seqlens is varlen, so FA3 runs its device-side
- moe-align-local-expert-count: weak: the identical one-argument change (align over 32 experts instead of 256) was already measured in muse_moe_w4a16 (sglang-marlin-path-parity): verify_16/32 flat, decode_1 about 4% slower. Under the landed block cap the verify effect is a ~2 us fill launch, below resolution; decode has few checks left today. The reviewer suggests at most a T>=128-only variant later.
- trtllm-decode-full-attention: adversarial review: Hold. The spec was written on 2026-10-23 against the parity base. Since then master has landed head-paired FA3 for full layers at batch>1, so as written this ch
- fa3-num-splits-sliding: adversarial review: The premise (FA3 sizes the sliding split-KV for the 80K page-table extent) is contradicted by both FA3's own get_num_splits, which clamps seqlen_k by the window
- moe-align-single-kernel-small-t-control-capacity-256: control for moe-align-single-kernel-small-t, which was neutral; a capacity control of a non-gain cannot change a decision, and decode has 2 checks left for the attention rewrite and pairing follow-ups
- fa3-full-single-split-probe: adversarial review: The probe is sound and would give a clear reading, but neither reading can change a decision that is open right now. The rewrite it is meant to inform (triton-f
- router-triton-splitk-side-stream: reviewer: the 9.2-9.5 us splitK row is 2 calls (~4.6 us each), so the side-stream router branch (~14-18 us) already finishes before the ~22-25 us pooler and is off the critical path; most likely flat, and decode has 4 checks left
- triton-flash-decode-full-verify-paired-kv: adversarial review: Stage 1 measured the paired-KV kernel at about 1.63-1.66 TB/s, which is slower than FA3: 200.8 vs 188.8 us at decode_16 and 394.8 vs 374.4 us at decode_32. That
