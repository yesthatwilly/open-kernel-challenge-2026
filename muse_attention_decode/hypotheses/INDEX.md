# muse_attention_decode hypotheses

Regenerated 2026-10-24T100619Z. Best: 1.1284x [1.1115,1.1454] job `c297440e53334f34` (submit)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T223444Z | sglang-path-parity | primary | SGLang path parity | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T223550Z | fa3-num-splits-full-decode | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | fa3-num-splits-sliding | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | triton-window-attention-sliding | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | fused-preattention-triton | primary | pre-attention norm+rope | done/gain | gain | 1.1274 | [1.111,1.144] | 9e8fcb3fe4b04458 | check | done | proceed | 6/8 |
| 2026-10-23T223722Z | fa3-drop-cu-seqlens-decode | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T223722Z | fa3-scheduler-metadata-full-verify | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T223722Z | triton-flash-decode-full-both-heads | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T223802Z | fused-sliding-preattn-into-window-kernel | primary | attention boundary | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T223803Z | fused-sliding-normgate-epilogue | primary | output norm-gate | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225840Z | paired-heads-launch-cleanup | primary | attention boundary | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225840Z | fa3-hdim128-head-pairing | primary | FA3 call (head packing) | running | running |  |  | facc479992e24a35 | check |  | proceed |  |
| 2026-10-23T225858Z | fa3-batched-q-verify | primary | FA3 call (q layout) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fa3-num-splits-full-verify | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fa3-single-split-sliding | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fa3-window-trimmed-page-table | primary | page table (sliding) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fused-full-combine-normgate | primary | output norm-gate | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | stream-fork-preattention | primary | launch overhead | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | triton-flash-decode-deep-pipeline | primary | FA3 call (Triton pipeline) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | triton-flash-decode-full-verify | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T081233Z | sliding-single-launch-fused-attention | rewrite | sliding-layer attention (whole-path rewrite) | running | running |  |  | f389af196ddf47b1 | check |  | proceed |  |

## Held

- sglang-path-parity: adversarial review: As specified the change is baseline.py line for line (same five calls, same _NORMS/_ROPE caches, same descale and tuple handling), so churn.py commit-impl (.cla
- fa3-num-splits-sliding: num_splits is only a cap under FA3's on-device split sizing (LESSONS; decode held the same idea), and sliding-single-launch-fused-attention, now running, replaces the sliding FA3 call outright
