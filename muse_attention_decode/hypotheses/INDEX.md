# muse_attention_decode hypotheses

Regenerated 2026-10-25T154445Z. Best: 1.5017x [1.4791,1.5304] job `252abb7f11ec48c3` (check)

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
| 2026-10-24T183847Z | full-decode1-cuda-splitkv-chunk320 | primary | full-layer attention (batch-1 decode gate) | done/neutral | neutral | 1.3954 | [1.374,1.416] | 8d4771feca9b4e18 | submit |  | proceed | 3/4 |
| 2026-10-24T183847Z | full-decode1-cuda-splitkv-chunk640 | primary | full-layer attention (batch-1 decode gate) | done/neutral | neutral | 1.3805 | [1.360,1.405] | 91d541778b8f45f3 | check |  | proceed | 3/4 |
| 2026-10-24T183847Z | splitkv-full-line-gather-mapping | primary | FA3 call (full-layer decode gather) | done/neutral | neutral | 1.4160 | [1.395,1.442] | cfa8c89e4f2b488d | check |  | proceed | 0/2 |
| 2026-10-24T183848Z | splitkv-l2-128b-prefetch-hint | primary | FA3 call (full-layer decode gather) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T183848Z | splitkv-combine-wide-mlp | primary | split-KV combine | done/neutral | neutral | 1.3836 | [1.363,1.404] | 47967b6c4d98454b | check |  | proceed | 0/2 |
| 2026-10-24T183855Z | paired-norm-gate-token-tiled | primary | output norm-gate | done/neutral | neutral | 1.3911 | [1.370,1.415] | 5e01430a8dfe4977 | check |  | proceed | 0/3 |
| 2026-10-24T183855Z | preattention-tiled-gate-large-tokens | primary | pre-attention norm+rope | done/neutral | neutral | 1.3863 | [1.365,1.407] | c2ebf6ed05824692 | check |  | proceed | 1/4 |
| 2026-10-24T183855Z | sliding-decode-blockn128-single-tile | primary | sliding-layer attention (batched decode, tile size) | done/neutral | neutral | 1.3834 | [1.363,1.408] | 88c62334775e48e4 | check |  | proceed | 2/3 |
| 2026-10-24T183855Z | sliding-decode32-splits-8 | primary | sliding-layer attention (batched decode, split depth) | done/neutral | neutral | 1.3973 | [1.376,1.418] | 16d3fb5cae3f4141 | check |  | proceed | 2/3 |
| 2026-10-24T183855Z | splitkv-chunk2048-batch32 | primary | FA3 call (full-layer decode gather) | done/neutral | neutral | 1.3838 | [1.363,1.405] | 7cfc8c5be3a04369 | check |  | proceed | 1/3 |
| 2026-10-24T183855Z | splitkv-pdl-attn-to-combine | primary | launch overhead | done/neutral | neutral | 1.3929 | [1.372,1.414] | 113cbf3093dd40b9 | check |  | proceed | 0/2 |
| 2026-10-24T183855Z | sliding-fused-hoist-first-tile | primary | sliding-layer attention (batched decode, gather prologue) | done/regression | regression | 1.3449 | [1.325,1.365] | ead13d58c6c843e9 | check |  | proceed | 1/3 |
| 2026-10-25T001348Z | full-decode-splitkv-s2-streamk-ticket | rewrite | full-layer attention (decode, splitkv kernel A/B) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T001415Z | full-verify-wgmma-s3-warpgroup-per-kvhead | rewrite | full-layer attention (verify, splitkv A8/B8) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T001451Z | sliding-cluster-s2-decode-c4-pdl | rewrite | sliding-layer attention (decode, cluster kernel) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T001825Z | full-decode-splitkv-s2-streamk-ticket-inproc | rewrite | full-layer attention (decode, splitkv kernel A/B) | done/neutral | neutral | 1.4171 | [1.396,1.443] | e592b1f5d3db4f83 | check |  | proceed | 2/3 |
| 2026-10-25T001825Z | full-verify-wgmma-s3-warpgroup-per-kvhead-inproc | rewrite | full-layer attention (verify, splitkv A8/B8) | done/neutral | neutral | 1.3801 | [1.359,1.401] | 2c3f7fff72bb4e1d | check |  | proceed | 2/3 |
| 2026-10-25T001825Z | sliding-cluster-s2-decode-c4-pdl-inproc | rewrite | sliding-layer attention (decode, cluster kernel) | done/neutral | neutral | 1.4001 | [1.379,1.426] | 1990ce53e9d34073 | check |  | proceed | 2/5 |
| 2026-10-25T003621Z | full-decode-splitkv-s3-fused-preattn | rewrite | full-layer attention (decode, splitkv stream-K kernel prologue) | done/neutral | neutral | 1.4283 | [1.407,1.450] | 825baa03798f431a | check |  | proceed | 0/3 |
| 2026-10-25T003637Z | full-decode1-streamk-s3-two-level-ticket | rewrite | full-layer attention (batch-1 decode, splitkv stream-K kernel) | done/neutral | neutral | 1.4322 | [1.411,1.455] | 2a3241780c7b4c46 | check | done | proceed | 2/3 |
| 2026-10-25T003705Z | full-verify-wgmma-s4-sliding-window | rewrite | sliding-layer attention (verify, wgmma stream-K kernel) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T003705Z | full-verify1-wgmma-s4-batch1-grid-sm | rewrite | full-layer attention (batch-1 verify, wgmma stream-K kernel) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T003735Z | sliding-decode-paired-kvhead-rows | primary | sliding-layer attention (batched decode, KV row gather) | done/regression | regression | 1.2865 | [1.267,1.306] | 48ffbe5690cc4ab4 | check |  | proceed | 0/3 |
| 2026-10-25T003735Z | sliding-verify-fa3-head-pairing | primary | FA3 call (head packing, sliding verify) | done/regression | regression | 1.3475 | [1.327,1.368] | 031731b5db294a92 | check |  | proceed | 0/3 |
| 2026-10-25T004021Z | sliding-decode-paired-kvhead-rows-control-unpaired-s8 | control→sliding-decode-paired-kvhead-rows | sliding-layer attention (batched decode, split depth) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-25T014545Z | stack-streamk-ticket-cluster-c16-c4 | primary | full + sliding decode attention (stack of two pending kernels) | done/neutral | neutral | 1.4266 | [1.405,1.454] | 195dcc1209b6457b | check |  | proceed | 6/7 |
| 2026-10-25T035310Z | full-verify-wgmma-s4-multitile-fix | rewrite | full-layer attention (verify, wgmma stream-K kernel, multi-tile correctness) | done/neutral | neutral | 1.3758 | [1.355,1.396] | 51b0d5c110f3459c | check |  | proceed | 0/3 |
| 2026-10-25T042334Z | sliding-cluster-s3-verify-attn-only-phase-timed | rewrite | sliding-layer attention (verify, cluster kernel) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T042334Z | sliding-verify1-triton-pseudo-decode | primary | sliding-layer attention (batch-1 verify, Triton fused decode kernel) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T042334Z | full-decode1-sk4-cluster-dsmem-merge | rewrite | full-layer attention (batch-1 decode, stream-K kernel merge) | done/neutral | neutral | 1.4388 | [1.417,1.460] | f519261b7c2b435f | check |  | proceed | 0/3 |
| 2026-10-25T042334Z | sliding-verify-sk-pseudo-decode-s1 | rewrite | sliding-layer attention (verify, stream-K pseudo-decode kernel) | done/neutral | neutral | 1.4705 | [1.448,1.493] | 94523c3313444fd8 | check |  | proceed | 1/4 |
| 2026-10-25T042335Z | full-verify1-sk-pseudo-decode | primary | full-layer attention (batch-1 verify, stream-K pseudo-decode) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T042335Z | full-decode1-sk-pdl-from-preattn | primary | launch overhead (full decode_1, pre-attention -> stream-K kernel) | done/neutral | neutral | 1.4571 | [1.432,1.479] | c0f1945a108c4aa2 | check |  | proceed | 0/2 |
| 2026-10-25T054448Z | restack-sk-pseudo-cluster-plus-v3-fused-preattn | primary | full-layer attention (decode 1<B<=64 prologue) on top of the sliding restack | done/gain | gain | 1.4920 | [1.470,1.514] | cbd49dc94be548e7 | check | done | proceed | 3/5 |
| 2026-10-25T054448Z | full-decode1-sk-two-level-fused-preattn | rewrite | full-layer attention (batch-1 decode, stream-K prologue fusion) | done/neutral | neutral | 1.4411 | [1.419,1.463] | 2038a530edd04309 | check |  | proceed | 3/3 |
| 2026-10-25T054448Z | full-verify-wgmma-s5-qfrag-reload | rewrite | full-layer attention (verify, wgmma stream-K kernel, multi-tile Q-fragment fix) | done/neutral | neutral | 1.4345 | [1.413,1.456] | 71e40388a1904937 | check |  | proceed | 1/4 |
| 2026-10-25T054448Z | restack-sk-pseudo-b1-cluster-c16-c4 | primary | sliding-layer attention (restack of two measured row wins: verify B=1 pseudo-decode + decode cluster C=16/C=4) | done/neutral | neutral | 1.4870 | [1.465,1.509] | eb6ff0cb701d4def | check |  | proceed | 5/5 |
| 2026-10-25T054448Z | sliding-verify-sk-pseudo-s2-token-group | rewrite | sliding-layer attention (verify, stream-K pseudo-decode kernel, token grouping) | done/regression | regression | 1.4251 | [1.404,1.452] | bd0ddd413f7f4a94 | check |  | proceed | 1/4 |
| 2026-10-25T054448Z | full-decode-sk-s4-deferred-ticket-arrive | rewrite | full-layer attention (decode 1<B<=64, stream-K ticket epilogue) | done/failed | failed |  |  | 7f10d2d820cd49ee | check |  | proceed |  |
| 2026-10-25T054449Z | full-verify1-sk-two-level-token-group | rewrite | full-layer attention (batch-1 verify, stream-K two-level kernel, token grouping) | done/neutral | neutral | 1.4714 | [1.449,1.494] | 7d23e8776c194344 | check |  | proceed | 1/3 |
| 2026-10-25T054449Z | sliding-decode-sk-window-route | rewrite | sliding-layer attention (decode, stream-K windowed kernel) | done/neutral | neutral | 1.4973 | [1.475,1.520] | 118c6de9704146f3 | check |  | proceed | 2/4 |
| 2026-10-25T110636Z | restack-sliding-decode-sk-window-on-cluster-master | primary | sliding-layer attention (decode, stream-K windowed kernel) restack | done/neutral | neutral | 1.5017 | [1.479,1.530] | 252abb7f11ec48c3 | submit | merged | proceed | 5/5 |
| 2026-10-25T110716Z | sliding-decode-sk-s3-fused-preattn-rope | rewrite | sliding-layer attention (decode, stream-K windowed kernel, pre-attention fold) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-25T150036Z | preattention-tiled-every-verify-block | primary | pre-attention norm+rope | queue | queued |  |  |  |  |  | required |  |
| 2026-10-25T150036Z | epilogue-gate-l2-prefetch-cuda | primary | output norm-gate (gate operand prefetch in the CUDA epilogues) | done/neutral | neutral | 1.5020 | [1.479,1.525] | 917a9689ba4645bd | submit |  | proceed | 0/6 |
| 2026-10-25T153027Z | full-decode1-sk-two-level-deep-ring | primary | full-layer attention (batch-1 decode, stream-K ring depth) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-25T153027Z | full-verify-fa3-num-splits-band-gate | primary | FA3 call (full verify split count) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-25T153027Z | full-verify-wgmma-ws-s1-producer-split | rewrite | full-layer attention (verify, wgmma stream-K kernel, warp-specialized pipeline) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-25T153027Z | sliding-decode1-cluster-per-kvhead | primary | sliding-layer attention (batch-1 decode, cluster kernel work split) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-25T153027Z | sliding-verify-fa3-num-splits-band-gate | primary | FA3 call (sliding verify split count) | queue | queued |  |  |  |  |  | required |  |
| 2026-10-25T153027Z | sliding-verify-wgmma-window-s1 | rewrite | sliding-layer attention (verify, wgmma stream-K windowed kernel) | queue | queued |  |  |  |  |  | required |  |

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
- full-decode-splitkv-s2-streamk-ticket: superseded by full-decode-splitkv-s2-streamk-ticket-inproc: the original spec runs a self-check in a child process, which the rules reject (starting processes)
- full-verify-wgmma-s3-warpgroup-per-kvhead: superseded by full-verify-wgmma-s3-warpgroup-per-kvhead-inproc: the original spec runs a self-check in a child process, which the rules reject (starting processes)
- sliding-cluster-s2-decode-c4-pdl: superseded by sliding-cluster-s2-decode-c4-pdl-inproc: the original spec runs a self-check in a child process, which the rules reject (starting processes)
- full-verify-wgmma-s4-sliding-window: adversarial review: Sequencing hold, not a rejection of the design. The stage-3 kernel this builds on never ran in the timed call. It failed its own self-check (case 2: B=6, kv_sca
- full-verify1-wgmma-s4-batch1-grid-sm: adversarial review: Hold by this hypothesis's own condition ('worth running only once stage 3 shows A8 at or near its gate'). Stage 3's wgmma kernel failed its self-check (case 2, 
- sliding-cluster-s3-verify-attn-only-phase-timed: Reviewer: proceed but wait for stack-streamk-ticket-cluster-c16-c4 (195dcc1209b6457b) and build on its branch (raw s2 carries the C=8 band, +5.7% on sliding decode_16). Port setup_one<8>/launch<8> back from stage 1 593859d; mask rows to p-(W-1)<=t<=p; phase timing in prepare only.
- sliding-verify1-triton-pseudo-decode: adversarial review: HOLD for two reasons. (1) The key range is wrong: seq = seq_lens[b]-7+j treats seq_lens as the post-block length, but seq_lens is the prefix length (positions =
- full-verify1-sk-pseudo-decode: Reviewer: low priority; dispatch after full-verify-wgmma-s4-multitile-fix (51b0d5c110f3459c) and after sliding-verify-sk-pseudo-decode-s1 harvest; build on s1 PSEUDO code with window=0; if the wgmma fix works, full-verify1-wgmma-s4-batch1-grid-sm owns this row.
