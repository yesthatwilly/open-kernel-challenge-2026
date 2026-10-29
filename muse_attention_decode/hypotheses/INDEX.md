# muse_attention_decode hypotheses

Regenerated 2026-10-29T223319Z. Best: 1.6340x [1.6094,1.6585] job `4e386647edd845f7` (submit)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T223444Z | sglang-path-parity | primary | SGLang path parity | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-23T223550Z | triton-window-attention-sliding | primary | FA3 call | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | fused-preattention-triton | primary | pre-attention norm+rope | done/gain | gain | 1.1274 | [1.111,1.144] | 9e8fcb3fe4b04458 | check | done | proceed | 6/8 |
| 2026-10-23T223550Z | fa3-num-splits-full-decode | primary | FA3 call | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-23T223550Z | fa3-num-splits-sliding | primary | FA3 call | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-23T223722Z | fa3-drop-cu-seqlens-decode | primary | FA3 call | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-23T223722Z | fa3-scheduler-metadata-full-verify | primary | FA3 call | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-23T223722Z | triton-flash-decode-full-both-heads | primary | FA3 call | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-23T223802Z | fused-sliding-preattn-into-window-kernel | primary | attention boundary | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-23T223803Z | fused-sliding-normgate-epilogue | primary | output norm-gate | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-23T225840Z | fa3-hdim128-head-pairing | primary | FA3 call (head packing) | done/gain | gain | 1.1706 | [1.153,1.192] | facc479992e24a35 | check | superseded | proceed | 7/7 |
| 2026-10-23T225840Z | paired-heads-launch-cleanup | primary | attention boundary | done/gain | gain | 1.2436 | [1.225,1.262] | 2846122f764043da | check | superseded | proceed | 5/6 |
| 2026-10-23T225858Z | fused-full-combine-normgate | primary | output norm-gate | queue | hold |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fa3-window-trimmed-page-table | primary | page table (sliding) | done/regression | regression | 1.1411 | [1.124,1.158] | 1f63fcbbc0394bed | check |  | proceed | 1/7 |
| 2026-10-23T225858Z | fa3-batched-q-verify | primary | FA3 call (q layout) | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-23T225858Z | fa3-num-splits-full-verify | primary | FA3 call | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | fa3-single-split-sliding | primary | FA3 call | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-23T225858Z | stream-fork-preattention | primary | launch overhead | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | triton-flash-decode-deep-pipeline | primary | FA3 call (Triton pipeline) | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-23T225858Z | triton-flash-decode-full-verify | primary | FA3 call | done/expired | expired |  |  |  |  |  | required |  |
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
| 2026-10-24T165325Z | sliding-verify-cuda-streamk-window | primary | sliding-layer attention (verify) | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-24T165342Z | full-verify1-cuda-streamk-batch1 | primary | full-layer attention (batch-1 verify gate) | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-24T172034Z | sliding-cuda-cluster-dsmem-onelaunch | rewrite | sliding-layer attention (whole-path rewrite, CUDA cluster) | done/regression | regression | 1.2206 | [1.202,1.239] | 6821e67ff3e749b7 | submit |  | proceed | 3/7 |
| 2026-10-24T172110Z | full-cuda-onelaunch-warpspec-bulk-gather | rewrite | full-layer attention (whole-path rewrite, warp-specialized one launch) | done/regression | regression | 0.8238 | [0.811,0.836] | 4994759f5b4b4278 | submit |  | proceed | 3/5 |
| 2026-10-24T183847Z | full-decode1-cuda-splitkv-chunk320 | primary | full-layer attention (batch-1 decode gate) | done/neutral | neutral | 1.3954 | [1.374,1.416] | 8d4771feca9b4e18 | submit |  | proceed | 3/4 |
| 2026-10-24T183847Z | full-decode1-cuda-splitkv-chunk640 | primary | full-layer attention (batch-1 decode gate) | done/neutral | neutral | 1.3805 | [1.360,1.405] | 91d541778b8f45f3 | check |  | proceed | 3/4 |
| 2026-10-24T183847Z | splitkv-full-line-gather-mapping | primary | FA3 call (full-layer decode gather) | done/neutral | neutral | 1.4160 | [1.395,1.442] | cfa8c89e4f2b488d | check |  | proceed | 0/2 |
| 2026-10-24T183848Z | splitkv-combine-wide-mlp | primary | split-KV combine | done/neutral | neutral | 1.3836 | [1.363,1.404] | 47967b6c4d98454b | check |  | proceed | 0/2 |
| 2026-10-24T183848Z | splitkv-l2-128b-prefetch-hint | primary | FA3 call (full-layer decode gather) | done/expired | expired |  |  |  |  |  | hold |  |
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
| 2026-10-25T004021Z | sliding-decode-paired-kvhead-rows-control-unpaired-s8 | control→sliding-decode-paired-kvhead-rows | sliding-layer attention (batched decode, split depth) | done/expired | expired |  |  |  |  |  | required |  |
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
| 2026-10-25T110716Z | sliding-decode-sk-s3-fused-preattn-rope | rewrite | sliding-layer attention (decode, stream-K windowed kernel, pre-attention fold) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T150036Z | preattention-tiled-every-verify-block | primary | pre-attention norm+rope | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T150036Z | epilogue-gate-l2-prefetch-cuda | primary | output norm-gate (gate operand prefetch in the CUDA epilogues) | done/neutral | neutral | 1.5020 | [1.479,1.525] | 917a9689ba4645bd | submit |  | proceed | 0/6 |
| 2026-10-25T153027Z | sliding-verify-fa3-num-splits-band-gate | primary | FA3 call (sliding verify split count) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T153027Z | full-decode1-sk-two-level-deep-ring | primary | full-layer attention (batch-1 decode, stream-K ring depth) | done/neutral | neutral | 1.5109 | [1.488,1.534] | 26c049edfdd842b3 | submit |  | proceed | 3/5 |
| 2026-10-25T153027Z | full-verify-fa3-num-splits-band-gate | primary | FA3 call (full verify split count) | done/neutral | neutral | 1.5216 | [1.499,1.551] | 849f2733f8ea4888 | check |  | proceed | 2/5 |
| 2026-10-25T153027Z | full-verify-wgmma-ws-s1-producer-split | rewrite | full-layer attention (verify, wgmma stream-K kernel, warp-specialized pipeline) | done/neutral | neutral | 1.5157 | [1.493,1.544] | e701c51d7b944a13 | check |  | proceed | 1/5 |
| 2026-10-25T153027Z | sliding-decode1-cluster-per-kvhead | primary | sliding-layer attention (batch-1 decode, cluster kernel work split) | done/neutral | neutral | 1.5120 | [1.489,1.535] | 4b608df595b046fb | submit | done | proceed | 5/5 |
| 2026-10-25T153027Z | sliding-verify-wgmma-window-s1 | rewrite | sliding-layer attention (verify, wgmma stream-K windowed kernel) | done/neutral | neutral | 1.5164 | [1.494,1.539] | cd01da3909cc4f4c | check |  | proceed | 2/5 |
| 2026-10-25T161608Z | combo-b1-chain-c320-fused-pre-pdl | primary | combination (conjectural: full decode_1 chunk320 + fused pre-attention prologue + PDL to combine) | done/neutral | neutral | 1.5295 | [1.507,1.552] | 8713665f3064462c | submit |  | proceed | 3/4 |
| 2026-10-25T161608Z | combo-naive-c320-fd1-tiled-preattn-verify | primary | combination (naive top-row stack: full decode_1 chunk320 + verify tiled pre-attention) | done/neutral | neutral | 1.5094 | [1.487,1.538] | b9d062aaa94c4cbf | submit |  | proceed | 5/6 |
| 2026-10-25T161609Z | combo-tiled-preattn-neighbour-regate-c320 | primary | combination (conjectural: tiled pre-attention as every pair's first node, sd B=16 and skF selectors re-decided, + chunk320) | done/neutral | neutral | 1.5199 | [1.497,1.543] | 52cbfaebab9646f2 | submit |  | proceed | 3/7 |
| 2026-10-25T190812Z | merge-perkvhead-c320-sv32-tiled | primary | full decode_1 split-KV + sliding verify_32 pre-attention | done/neutral | neutral | 1.5347 | [1.512,1.558] | a78f29aeab9b47cb | submit |  | proceed | 2/3 |
| 2026-10-25T212048Z | full-verify-wgmma-s6-split-ring-2cta | rewrite | full-layer attention (verify, wgmma stream-K kernel, per-warpgroup ring) | done/neutral | neutral | 1.5331 | [1.510,1.556] | 4899ab49d74f4df7 | check |  | proceed | 1/5 |
| 2026-10-25T212052Z | full-verify1-wgmma-s7-grid-sm-two-level | rewrite | full-layer attention (batch-1 verify, wgmma stream-K kernel, two-level ticket) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T212052Z | full-decode-sk-s5-deferred-ticket-fault-fix | rewrite | full-layer attention (decode 1<B<=64, stream-K ticket epilogue) | done/neutral | neutral | 1.5162 | [1.493,1.539] | f1d6d493c24244f4 | check |  | proceed | 1/5 |
| 2026-10-25T212052Z | merge-perkvhead-v3chain-sv32-tiled | primary | full decode_1 split-KV launch chain (merge onto per-kvhead master, with SV32 tiled pre-attention) | done/neutral | neutral | 1.5360 | [1.513,1.565] | b4b20425eaff4942 | check |  | proceed | 3/5 |
| 2026-10-25T212052Z | sliding-verify1-pseudo-fused-preattn-rope | primary | sliding-layer attention (batch-1 verify, PSEUDO stream-K kernel, pre-attention fold) | done/neutral | neutral | 1.6191 | [1.595,1.643] | 7284659f6c4d4035 | check |  | proceed | 1/5 |
| 2026-10-26T222037Z | sk-full-line-gather | primary | full-layer attention (decode 1<B<=64, stream-K kernel K/V gather mapping) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T222100Z | sliding-verify-wgmma-window-s2-split-combine | rewrite | sliding-layer attention (verify B=2..32, wgmma window kernel merge) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T222117Z | c320-combine-onepass | primary | full decode_1 split-KV combine (c320 chain) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T222131Z | c320-pdl-early-trigger | primary | launch overhead (full decode_1 c320 kernel A -> combine PDL) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T222147Z | combo-sk-full-line-c320-tail | primary | combination (full decode B>=2 gather mapping + full decode_1 combine tail) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T222203Z | sliding-verify-wgmma-window-s2b-fast-merge | rewrite | sliding-layer attention (verify B=2..32, wgmma window kernel merge) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-26T222216Z | c320-attn-full-line-gather | primary | full-layer attention (batch-1 decode, c320 kernel A K/V gather mapping) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T222231Z | sliding-verify-wgmma-split-combine-on-v3chain | primary | sliding-layer attention (verify B=2..32, wgmma window kernel merge) on the v3chain base | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T222249Z | sliding-decode1-perkvhead-launch-select | primary | sliding-layer attention (batch-1 decode, cluster kernel launch configuration) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T222305Z | sk-full-line-5stage-select | primary | full-layer attention (decode 1<B<=64, stream-K ring depth after the whole-line remap) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T222324Z | sliding-decode-band2-sk-gate | primary | sliding-layer attention (decode B=2..16, stream-K window route gate) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-26T222353Z | full-verify-wgmma-s7-split-combine | rewrite | full-layer attention (verify B=2..64, wgmma stream-K kernel merge) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-27T132640Z | sk-ticket-merge-onepass | primary | stream-K ticket merge (splitkv_attn_sk_kernel merge_range: full decode 1<B<=64, PSEUDO sliding verify_1, sk-window sliding decode_32) | done/neutral | neutral | 1.6177 | [1.593,1.642] | ba8c8a6103124f71 | check |  | proceed | 3/5 |
| 2026-10-27T132706Z | sliding-decode1-ph-push-merge | primary | sliding-layer attention (batch-1 decode, PERHEAD cluster DSMEM merge: push instead of pull) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-27T132752Z | attn-fast-math-epilogues | primary | output norm-gate (CUDA epilogue math: o/L, RMSNorm reciprocal sqrt, sigmoid) in the routed attention kernels | queue | hold |  |  |  |  |  | required |  |
| 2026-10-27T132814Z | combo-tail-onepass-push-fastepi | primary | combination (stream-K ticket merge onepass + PERHEAD push merge + fast-math epilogues: the serialized merge/epilogue tails) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-27T143409Z | sk-onepass-verify1-timing-fix | primary | stream-K ticket merge onepass twin at sliding verify B=1 (the gate's timing inputs) | done/neutral | neutral | 1.6219 | [1.598,1.652] | bfb0457c9ae84322 | check |  | proceed | 0/3 |
| 2026-10-29T154210Z | fa3-metadata-fold-preattn-full-verify | primary | FA3 call (scheduler metadata folded into pre-attention, full verify) | done/neutral | neutral | 1.6336 | [1.609,1.666] | 6d1082a6c0dd4f80 | check |  | proceed | 2/5 |
| 2026-10-29T215944Z | full-verify-paged-gather-gate-kernel | primary | full-layer verify attention (FA3 call at verify_16 / verify_32) | done/neutral | neutral | 1.6154 | [1.591,1.640] | 06c051723643462f | check |  | proceed | 0/4 |
| 2026-10-29T222928Z | full-verify-paged-gather-gate-kernel-r2 | primary | full-layer verify attention (FA3 call at verify_16 / verify_32) | running | running |  |  | 184991095dff4d50 | check |  | proceed |  |

## Held

- triton-window-attention-sliding: moot under master 1.269x (sweep by the attention minter): its chain ends in the single fused sliding launch, measured as sliding-single-launch-fused-attention (0.945x); its one winning shape is covered by sliding-fused-triton-decode1-gate
- fa3-drop-cu-seqlens-decode: adversarial review: Moot and unresolvable under master 1.293x. Its headline target (sliding decode_1) no longer calls FA3 at all since sliding-fused-triton-decode1-gate landed. On 
- fused-full-combine-normgate: moot under master 1.269x (sweep by the attention minter): stacks on triton-flash-decode-full-both-heads (held); the paired epilogue already fuses the full-layer norm-gate
- full-decode-splitkv-s2-streamk-ticket: superseded by full-decode-splitkv-s2-streamk-ticket-inproc: the original spec runs a self-check in a child process, which the rules reject (starting processes)
- full-verify-wgmma-s3-warpgroup-per-kvhead: superseded by full-verify-wgmma-s3-warpgroup-per-kvhead-inproc: the original spec runs a self-check in a child process, which the rules reject (starting processes)
- sliding-cluster-s2-decode-c4-pdl: superseded by sliding-cluster-s2-decode-c4-pdl-inproc: the original spec runs a self-check in a child process, which the rules reject (starting processes)
- full-verify-wgmma-s4-sliding-window: adversarial review: Sequencing hold, not a rejection of the design. The stage-3 kernel this builds on never ran in the timed call. It failed its own self-check (case 2: B=6, kv_sca
- full-verify1-wgmma-s4-batch1-grid-sm: adversarial review: Hold by this hypothesis's own condition ('worth running only once stage 3 shows A8 at or near its gate'). Stage 3's wgmma kernel failed its self-check (case 2, 
- sliding-cluster-s3-verify-attn-only-phase-timed: Reviewer: proceed but wait for stack-streamk-ticket-cluster-c16-c4 (195dcc1209b6457b) and build on its branch (raw s2 carries the C=8 band, +5.7% on sliding decode_16). Port setup_one<8>/launch<8> back from stage 1 593859d; mask rows to p-(W-1)<=t<=p; phase timing in prepare only.
- sliding-verify1-triton-pseudo-decode: adversarial review: HOLD for two reasons. (1) The key range is wrong: seq = seq_lens[b]-7+j treats seq_lens as the post-block length, but seq_lens is the prefix length (positions =
- full-verify1-sk-pseudo-decode: Reviewer: low priority; dispatch after full-verify-wgmma-s4-multitile-fix (51b0d5c110f3459c) and after sliding-verify-sk-pseudo-decode-s1 harvest; build on s1 PSEUDO code with window=0; if the wgmma fix works, full-verify1-wgmma-s4-batch1-grid-sm owns this row.
- sliding-decode-sk-s3-fused-preattn-rope: killed 2026-10-27 by the campaign (check 877a6ba5: B=2..16 failed correctness runs, B=17..32 x0.99-1.005)
- preattention-tiled-every-verify-block: adversarial review: Sixth scored run of the same _pre_attention_tiled_kernel; its parent preattention-tiled-gate-large-tokens is on Do-not-try. The only new part (gate widened to S
- sliding-verify-fa3-num-splits-band-gate: screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: sliding verify B=2..32 on s2b kernels, FA3 not called
- full-verify1-wgmma-s7-grid-sm-two-level: released 2026-10-29T151514Z: zombie claim (stage claimed since 2026-10-27T212857Z, no job in 24 h); re-review to run it
- sk-full-line-gather: stale 2026-10-27: already run and landed by the campaign (sk full-line gather, c320 onepass, PDL trigger on master since 10-27), or superseded by s2b (scored a76ee9b0 / sw17 4e386647)
- sliding-verify-wgmma-window-s2-split-combine: superseded 2026-10-27: s2 was checked and killed by the campaign; its successor s2b landed on master (scored a76ee9b0, then sw17 4e386647 1.6340)
- c320-combine-onepass: stale 2026-10-27: already run and landed by the campaign (sk full-line gather, c320 onepass, PDL trigger on master since 10-27), or superseded by s2b (scored a76ee9b0 / sw17 4e386647)
- c320-pdl-early-trigger: stale 2026-10-27: already run and landed by the campaign (sk full-line gather, c320 onepass, PDL trigger on master since 10-27), or superseded by s2b (scored a76ee9b0 / sw17 4e386647)
- combo-sk-full-line-c320-tail: screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: full-line gather + c320 onepass landed; early trigger measured, not routed
- sliding-verify-wgmma-window-s2b-fast-merge: adversarial review: This hypothesis has already run and is on master: it was implemented as 9f428796 (hyp/attn-s2b on hyp/attn-c320 0453ca70), checked at af08448e82c04fdf (1.6247),
- c320-attn-full-line-gather: screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.2%
- sliding-verify-wgmma-split-combine-on-v3chain: stale 2026-10-27: already run and landed by the campaign (sk full-line gather, c320 onepass, PDL trigger on master since 10-27), or superseded by s2b (scored a76ee9b0 / sw17 4e386647)
- sliding-decode1-perkvhead-launch-select: screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.5%
- sk-full-line-5stage-select: screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.2%
- sliding-decode-band2-sk-gate: screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.5% (p~0.4), cheap stacker at most
- full-verify-wgmma-s7-split-combine: adversarial review: Hold. Not on master (no s7 module; full verify at B>1 still runs paired FA3 at 164/304 us), but s6's own DIAG (archive full-verify-wgmma-s6-split-ring-2cta/4899
- sliding-decode1-ph-push-merge: screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.6%, high risk
- attn-fast-math-epilogues: screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.2%
- combo-tail-onepass-push-fastepi: screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.6%; onepass on Do-not-try
