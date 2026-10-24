# muse_attention_decode hypotheses

Regenerated 2026-10-24T184810Z. Best: 1.3949x [1.3740,1.4158] job `698d4cd3838c4a63` (submit)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T223444Z | sglang-path-parity | primary | SGLang path parity | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T223550Z | fa3-num-splits-full-decode | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | fa3-num-splits-sliding | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | triton-window-attention-sliding | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | fused-preattention-triton | primary | pre-attention norm+rope | done/gain | gain | 1.1274 | [1.111,1.144] | 9e8fcb3fe4b04458 | check | done | proceed | 6/8 |
| 2026-10-23T223722Z | fa3-drop-cu-seqlens-decode | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T223722Z | fa3-scheduler-metadata-full-verify | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T223722Z | triton-flash-decode-full-both-heads | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223802Z | fused-sliding-preattn-into-window-kernel | primary | attention boundary | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223803Z | fused-sliding-normgate-epilogue | primary | output norm-gate | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225840Z | fa3-hdim128-head-pairing | primary | FA3 call (head packing) | done/gain | gain | 1.1706 | [1.153,1.192] | facc479992e24a35 | check | superseded | proceed | 7/7 |
| 2026-10-23T225840Z | paired-heads-launch-cleanup | primary | attention boundary | done/gain | gain | 1.2436 | [1.225,1.262] | 2846122f764043da | check | superseded | proceed | 5/6 |
| 2026-10-23T225858Z | fa3-batched-q-verify | primary | FA3 call (q layout) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T225858Z | fa3-num-splits-full-verify | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fa3-single-split-sliding | primary | FA3 call | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-23T225858Z | fused-full-combine-normgate | primary | output norm-gate | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | stream-fork-preattention | primary | launch overhead | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | triton-flash-decode-deep-pipeline | primary | FA3 call (Triton pipeline) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | triton-flash-decode-full-verify | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fa3-window-trimmed-page-table | primary | page table (sliding) | done/regression | regression | 1.1411 | [1.124,1.158] | 1f63fcbbc0394bed | check |  | proceed | 1/7 |
| 2026-10-24T081233Z | sliding-single-launch-fused-attention | rewrite | sliding-layer attention (whole-path rewrite) | done/regression | regression | 0.9452 | [0.931,0.959] | f389af196ddf47b1 | check |  | proceed | 2/7 |
| 2026-10-24T104027Z | paired-heads-launch-cleanup-ab-tune-batch-gate | ab-tune→paired-heads-launch-cleanup | FA3 call (head packing) gate | done/neutral | neutral | 1.2686 | [1.250,1.288] | bd64bf9f9e904f02 | check | merged | proceed | 5/5 |
| 2026-10-24T120440Z | sliding-fused-triton-decode1-gate | primary | sliding-layer attention (batch-1 decode gate) | done/neutral | neutral | 1.2931 | [1.274,1.321] | 295727fc68e14093 | check | merged | proceed | 1/5 |
| 2026-10-24T120457Z | fa3-scheduler-metadata-side-stream | primary | FA3 call (schedule, off critical path) | done/regression | regression | 1.2245 | [1.206,1.247] | 42964d938c5842f4 | check |  | proceed | 0/5 |
| 2026-10-24T120513Z | preattention-group-tiled-programs | primary | pre-attention norm+rope | done/neutral | neutral | 1.2614 | [1.242,1.280] | 877731a3b2214ddb | check |  | proceed | 1/5 |
| 2026-10-24T140133Z | full-decode-cuda-gather-splitkv | rewrite | FA3 call (full-layer decode gather) | done/gain | gain | 1.3468 | [1.327,1.367] | 82085a4e82154393 | check | done | proceed | 4/5 |
| 2026-10-24T140200Z | sliding-fused-triton-decode-batched-deep-splits | primary | sliding-layer attention (batched decode, split depth) | done/neutral | neutral | 1.3087 | [1.289,1.328] | 179dd013d3a347e4 | check |  | proceed | 3/5 |
| 2026-10-24T140225Z | full-fused-triton-decode1-gate | primary | full-layer attention (batch-1 decode gate) | done/neutral | neutral | 1.3417 | [1.322,1.362] | af3c6a46badf4e50 | submit |  | proceed | 2/4 |
| 2026-10-24T140241Z | sliding-fused-triton-decode1-splits-32 | primary | sliding-layer attention (batch-1 decode, split depth) | done/neutral | neutral | 1.3520 | [1.332,1.376] | f9ac830932cb4eb6 | submit |  | proceed | 1/3 |
| 2026-10-24T145005Z | sliding-deep-splits-on-cuda-splitkv | primary | sliding attention, batched decode, split depth | done/gain | gain | 1.3949 | [1.374,1.416] | 698d4cd3838c4a63 | submit | done | proceed | 2/4 |
| 2026-10-24T165242Z | full-verify-cuda-streamk-splitkv | rewrite | FA3 call (full-layer verify gather) | done/regression | regression | 1.2314 | [1.210,1.250] | a43acd1ed9b24b4e | submit |  | proceed | 2/5 |
| 2026-10-24T165302Z | full-decode-cuda-fp8-qk-mma | primary | FA3 call (full-layer decode gather) | done/neutral | neutral | 1.3439 | [1.324,1.364] | b939681d3ab549d7 | submit |  | proceed | 0/3 |
| 2026-10-24T165325Z | sliding-verify-cuda-streamk-window | primary | sliding-layer attention (verify) | queue | queued |  |  |  |  |  | proceed |  |
| 2026-10-24T165342Z | full-verify1-cuda-streamk-batch1 | primary | full-layer attention (batch-1 verify gate) | queue | queued |  |  |  |  |  | proceed |  |
| 2026-10-24T172034Z | sliding-cuda-cluster-dsmem-onelaunch | rewrite | sliding-layer attention (whole-path rewrite, CUDA cluster) | done/regression | regression | 1.2206 | [1.202,1.239] | 6821e67ff3e749b7 | submit |  | proceed | 3/7 |
| 2026-10-24T172110Z | full-cuda-onelaunch-warpspec-bulk-gather | rewrite | full-layer attention (whole-path rewrite, warp-specialized one launch) | done/regression | regression | 0.8238 | [0.811,0.836] | 4994759f5b4b4278 | submit |  | proceed | 3/5 |
| 2026-10-24T183847Z | full-decode1-cuda-splitkv-chunk320 | primary | full-layer attention (batch-1 decode gate) | running | running |  |  | 8d4771feca9b4e18 | submit |  | proceed |  |
| 2026-10-24T183847Z | full-decode1-cuda-splitkv-chunk640 | primary | full-layer attention (batch-1 decode gate) | running | running |  |  | 91d541778b8f45f3 | check |  | proceed |  |
| 2026-10-24T183847Z | splitkv-full-line-gather-mapping | primary | FA3 call (full-layer decode gather) | running | running |  |  | cfa8c89e4f2b488d | check |  | proceed |  |
| 2026-10-24T183848Z | splitkv-l2-128b-prefetch-hint | primary | FA3 call (full-layer decode gather) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T183848Z | splitkv-combine-wide-mlp | primary | split-KV combine | running | running |  |  | 47967b6c4d98454b | check |  | proceed |  |
| 2026-10-24T183855Z | paired-norm-gate-token-tiled | primary | output norm-gate | running | running |  |  | 5e01430a8dfe4977 | check |  | proceed |  |
| 2026-10-24T183855Z | preattention-tiled-gate-large-tokens | primary | pre-attention norm+rope | running | running |  |  | c2ebf6ed05824692 | check |  | proceed |  |
| 2026-10-24T183855Z | sliding-decode-blockn128-single-tile | primary | sliding-layer attention (batched decode, tile size) | running | running |  |  | 88c62334775e48e4 | check |  | proceed |  |
| 2026-10-24T183855Z | sliding-decode32-splits-8 | primary | sliding-layer attention (batched decode, split depth) | running | running |  |  | 16d3fb5cae3f4141 | check |  | proceed |  |
| 2026-10-24T183855Z | sliding-fused-hoist-first-tile | primary | sliding-layer attention (batched decode, gather prologue) | running | running |  |  | ead13d58c6c843e9 | check |  | proceed |  |
| 2026-10-24T183855Z | splitkv-chunk2048-batch32 | primary | FA3 call (full-layer decode gather) | running | running |  |  | 7cfc8c5be3a04369 | check |  | proceed |  |
| 2026-10-24T183855Z | splitkv-pdl-attn-to-combine | primary | launch overhead | running | running |  |  | 113cbf3093dd40b9 | check |  | proceed |  |

## Held

- sglang-path-parity: adversarial review: As specified the change is baseline.py line for line (same five calls, same _NORMS/_ROPE caches, same descale and tuple handling), so churn.py commit-impl (.cla
- fa3-num-splits-full-decode: num_splits is only a cap under FA3's on-device split sizing (LESSONS; the decode_layer equivalents were held for the same reason), and full layers are now paired, which changes FA3's split decision anyway
- fa3-num-splits-sliding: num_splits is only a cap under FA3's on-device split sizing (LESSONS; decode held the same idea), and sliding-single-launch-fused-attention, now running, replaces the sliding FA3 call outright
- triton-window-attention-sliding: moot under master 1.269x (sweep by the attention minter): its chain ends in the single fused sliding launch, measured as sliding-single-launch-fused-attention (0.945x); its one winning shape is covered by sliding-fused-triton-decode1-gate
- fa3-drop-cu-seqlens-decode: adversarial review: Moot and unresolvable under master 1.293x. Its headline target (sliding decode_1) no longer calls FA3 at all since sliding-fused-triton-decode1-gate landed. On 
- fa3-scheduler-metadata-full-verify: adversarial review: Master has made this moot. Full verify at batch 16/32 now runs on the head-paired path with scheduler_metadata=None, and so does full decode at batch > 1. The p
- triton-flash-decode-full-both-heads: moot under master 1.269x (sweep by the attention minter): master's paired FA3 already gathers one 128-byte row per token (156 us at decode_16); decode_layer's Triton paired-row kernel ran ~29% behind FA3
- fused-sliding-preattn-into-window-kernel: moot under master 1.269x (sweep by the attention minter): stacks on triton-window-attention-sliding (held); same measured outcome
- fused-sliding-normgate-epilogue: moot under master 1.269x (sweep by the attention minter): stacks on the held sliding Triton chain; same measured outcome
- fa3-batched-q-verify: adversarial review: Master has not made this moot, since verify still passes cu_seqlens_q. The mechanism doesn't hold up, though: cache_seqlens (seqused_k) already puts FA3 on its 
- fa3-num-splits-full-verify: moot under master 1.269x (sweep by the attention minter): num_splits is only a cap under FA3's on-device split sizing, and full verify at batch>1 is now paired with scheduler_metadata=None
- fa3-single-split-sliding: reviewer expects an overall loss: one CTA per (request, KV head) streams 16-17 serial blocks, likely losing 8-12 us on sliding verify_1 and slowing decode_16/verify; only decode_32/verify_32 might gain; attention has 2 checks
- fused-full-combine-normgate: moot under master 1.269x (sweep by the attention minter): stacks on triton-flash-decode-full-both-heads (held); the paired epilogue already fuses the full-layer norm-gate
- stream-fork-preattention: moot under master 1.269x (sweep by the attention minter): targets the parity pipeline's five pre-attention launches, which master fused into one kernel
- triton-flash-decode-deep-pipeline: moot under master 1.269x (sweep by the attention minter): stacks on triton-flash-decode-full-both-heads (held)
- triton-flash-decode-full-verify: moot under master 1.269x (sweep by the attention minter): stacks on triton-flash-decode-full-both-heads (held)
- splitkv-l2-128b-prefetch-hint: adversarial review: The mechanism rests on a misreading of issue_tile: each warp-wide cp.async already touches all four 32-byte sectors of every 128-byte K/V line (chunks 0,2,4,6 a
