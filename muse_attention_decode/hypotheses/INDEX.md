# muse_attention_decode hypotheses

Regenerated 2026-10-24T120954Z. Best: 1.2686x [1.2496,1.2877] job `bd64bf9f9e904f02` (check)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T223444Z | sglang-path-parity | primary | SGLang path parity | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T223550Z | fa3-num-splits-full-decode | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | fa3-num-splits-sliding | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | triton-window-attention-sliding | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | fused-preattention-triton | primary | pre-attention norm+rope | done/gain | gain | 1.1274 | [1.111,1.144] | 9e8fcb3fe4b04458 | check | done | proceed | 6/8 |
| 2026-10-23T223722Z | fa3-drop-cu-seqlens-decode | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T223722Z | fa3-scheduler-metadata-full-verify | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T223722Z | triton-flash-decode-full-both-heads | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223802Z | fused-sliding-preattn-into-window-kernel | primary | attention boundary | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223803Z | fused-sliding-normgate-epilogue | primary | output norm-gate | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225840Z | fa3-hdim128-head-pairing | primary | FA3 call (head packing) | done/gain | gain | 1.1706 | [1.153,1.192] | facc479992e24a35 | check | merged | proceed | 7/7 |
| 2026-10-23T225840Z | paired-heads-launch-cleanup | primary | attention boundary | done/gain | gain | 1.2436 | [1.225,1.262] | 2846122f764043da | check | merged | proceed | 5/6 |
| 2026-10-23T225858Z | fa3-batched-q-verify | primary | FA3 call (q layout) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T225858Z | fa3-num-splits-full-verify | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fa3-single-split-sliding | primary | FA3 call | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fused-full-combine-normgate | primary | output norm-gate | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | stream-fork-preattention | primary | launch overhead | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | triton-flash-decode-deep-pipeline | primary | FA3 call (Triton pipeline) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | triton-flash-decode-full-verify | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fa3-window-trimmed-page-table | primary | page table (sliding) | running | running |  |  | 1f63fcbbc0394bed | check |  | proceed |  |
| 2026-10-24T081233Z | sliding-single-launch-fused-attention | rewrite | sliding-layer attention (whole-path rewrite) | done/regression | regression | 0.9452 | [0.931,0.959] | f389af196ddf47b1 | check |  | proceed | 2/7 |
| 2026-10-24T104027Z | paired-heads-launch-cleanup-ab-tune-batch-gate | ab-tune→paired-heads-launch-cleanup | FA3 call (head packing) gate | done/neutral | neutral | 1.2686 | [1.250,1.288] | bd64bf9f9e904f02 | check | merged | proceed | 5/5 |
| 2026-10-24T120440Z | sliding-fused-triton-decode1-gate | primary | sliding-layer attention (batch-1 decode gate) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T120457Z | fa3-scheduler-metadata-side-stream | primary | FA3 call (schedule, off critical path) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T120513Z | preattention-group-tiled-programs | primary | pre-attention norm+rope | running | running |  |  | 877731a3b2214ddb | check |  | proceed |  |

## Held

- sglang-path-parity: adversarial review: As specified the change is baseline.py line for line (same five calls, same _NORMS/_ROPE caches, same descale and tuple handling), so churn.py commit-impl (.cla
- fa3-num-splits-full-decode: num_splits is only a cap under FA3's on-device split sizing (LESSONS; the decode_layer equivalents were held for the same reason), and full layers are now paired, which changes FA3's split decision anyway
- fa3-num-splits-sliding: num_splits is only a cap under FA3's on-device split sizing (LESSONS; decode held the same idea), and sliding-single-launch-fused-attention, now running, replaces the sliding FA3 call outright
- triton-window-attention-sliding: moot under master 1.269x (sweep by the attention minter): its chain ends in the single fused sliding launch, measured as sliding-single-launch-fused-attention (0.945x); its one winning shape is covered by sliding-fused-triton-decode1-gate
- fa3-scheduler-metadata-full-verify: adversarial review: Master has made this moot. Full verify at batch 16/32 now runs on the head-paired path with scheduler_metadata=None, and so does full decode at batch > 1. The p
- triton-flash-decode-full-both-heads: moot under master 1.269x (sweep by the attention minter): master's paired FA3 already gathers one 128-byte row per token (156 us at decode_16); decode_layer's Triton paired-row kernel ran ~29% behind FA3
- fused-sliding-preattn-into-window-kernel: moot under master 1.269x (sweep by the attention minter): stacks on triton-window-attention-sliding (held); same measured outcome
- fused-sliding-normgate-epilogue: moot under master 1.269x (sweep by the attention minter): stacks on the held sliding Triton chain; same measured outcome
- fa3-batched-q-verify: adversarial review: Master has not made this moot, since verify still passes cu_seqlens_q. The mechanism doesn't hold up, though: cache_seqlens (seqused_k) already puts FA3 on its 
- fa3-num-splits-full-verify: moot under master 1.269x (sweep by the attention minter): num_splits is only a cap under FA3's on-device split sizing, and full verify at batch>1 is now paired with scheduler_metadata=None
- fused-full-combine-normgate: moot under master 1.269x (sweep by the attention minter): stacks on triton-flash-decode-full-both-heads (held); the paired epilogue already fuses the full-layer norm-gate
- stream-fork-preattention: moot under master 1.269x (sweep by the attention minter): targets the parity pipeline's five pre-attention launches, which master fused into one kernel
- triton-flash-decode-deep-pipeline: moot under master 1.269x (sweep by the attention minter): stacks on triton-flash-decode-full-both-heads (held)
- triton-flash-decode-full-verify: moot under master 1.269x (sweep by the attention minter): stacks on triton-flash-decode-full-both-heads (held)
- sliding-fused-triton-decode1-gate: reviewer: dispatch after fa3-window-trimmed-page-table (1f63fcbbc0394bed) harvests, which may shrink the decode_1 margin; then unhold
- fa3-scheduler-metadata-side-stream: reviewer: claim after fa3-window-trimmed-page-table (1f63fcbbc0394bed) harvests; if it lands, the sliding metadata must use its trimmed page table; then unhold
