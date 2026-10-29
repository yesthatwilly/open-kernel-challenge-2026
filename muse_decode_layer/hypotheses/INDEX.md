# muse_decode_layer hypotheses

Regenerated 2026-10-29T194717Z. Best: 1.6904x [1.6650,1.7158] job `7cc034f917274439` (submit)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T225318Z | sglang-pipeline-parity-router-hilo-m8 | primary | router GEMM | done/gain | gain | 1.0216 | [1.006,1.037] | e260dfe413e448c7 | check | done | proceed | 1/6 |
| 2026-10-23T225338Z | cache-store-fused-fp8-cast | primary | cache store | done/neutral | neutral | 1.0303 | [1.015,1.046] | 7264526cf9f94c37 | check |  | proceed | 2/4 |
| 2026-10-23T225353Z | expert-id-map-single-kernel | primary | MoE align/dispatch | done/neutral | neutral | 1.0374 | [1.022,1.053] | 11b8df414a2245ae | check |  | proceed | 2/4 |
| 2026-10-23T225545Z | marlin-block-m-cap-16 | primary | w13 marlin | done/gain | gain | 1.0621 | [1.046,1.078] | 0730135cdcac43b7 | check | done | proceed | 4/4 |
| 2026-10-23T225545Z | pooler-gemm-splitk-autotune | primary | pooler GEMM | done/neutral | neutral | 1.0189 | [1.004,1.034] | 2e1a43ab85324776 | check |  | proceed | 2/4 |
| 2026-10-23T225545Z | fa3-num-splits-full-no-metadata | primary | FA3 call | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-23T225545Z | moe-align-local-expert-count | primary | MoE align/dispatch | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-23T225545Z | trtllm-decode-full-attention | primary | FA3 call | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-23T225609Z | moe-align-single-kernel-small-t | primary | MoE align/dispatch | done/neutral | neutral | 1.0431 | [1.027,1.059] | 0a0af8669ffe4cb0 | check |  | proceed | 2/6 |
| 2026-10-23T225609Z | fa3-num-splits-sliding | primary | FA3 call | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-24T071442Z | moe-align-single-kernel-small-t-control-capacity-256 | control→moe-align-single-kernel-small-t | MoE align/dispatch | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-24T075400Z | fused-preattention-triton | primary | pre-attention norm+rope | done/neutral | neutral | 1.0970 | [1.081,1.113] | c7c26e2fd8b142a7 | check |  | proceed | 3/6 |
| 2026-10-24T075400Z | triton-flash-decode-full-paired-kv | rewrite | FA3 call | done/neutral | neutral | 1.0702 | [1.054,1.086] | e82b11f9d9f64b23 | check |  | proceed | 1/5 |
| 2026-10-24T075400Z | fa3-full-single-split-probe | primary | FA3 call | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-24T075400Z | router-triton-splitk-side-stream | primary | router GEMM | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-24T075400Z | triton-flash-decode-full-verify-paired-kv | rewrite | FA3 call | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-24T100416Z | triton-flash-decode-full-oversub-kv-prefetch | rewrite | FA3 call | done/regression | regression | 1.0413 | [1.026,1.057] | e55908a05eec4000 | check |  | proceed | 3/5 |
| 2026-10-24T103909Z | fa3-hdim128-head-pairing-batched | primary | FA3 call | done/neutral | neutral | 1.0874 | [1.071,1.104] | fb07a3bd3aa54baf | check | merged | proceed | 7/7 |
| 2026-10-24T140057Z | post-expert-norm-triton-small-t | primary | post-expert norm+combine | done/gain | gain | 1.1635 | [1.146,1.181] | 9dcaec316b3045a7 | check | done | proceed | 6/6 |
| 2026-10-24T140116Z | bf16-shape-gated-triton-gemms-small-t | primary | pooler GEMM | done/neutral | neutral | 1.1688 | [1.151,1.186] | 6076ae1f250143aa | submit |  | proceed | 2/6 |
| 2026-10-24T140135Z | sliding-fused-triton-decode1-port | primary | FA3 call | done/neutral | neutral | 1.1690 | [1.152,1.187] | 12198572363b49a4 | submit |  | proceed | 5/5 |
| 2026-10-24T140156Z | paired-heads-pad-unpad-triton | primary | output norm-gate | done/neutral | neutral | 1.1654 | [1.148,1.183] | a92b64be4c6547d3 | submit |  | proceed | 4/7 |
| 2026-10-24T165154Z | full-decode-cuda-splitkv-port | rewrite | FA3 call | done/neutral | neutral | 1.1862 | [1.168,1.204] | e789032128df4291 | submit |  | proceed | 5/5 |
| 2026-10-24T165224Z | full-verify-cuda-splitkv-stage2 | rewrite | FA3 call | done/regression | regression | 1.1559 | [1.139,1.173] | e9d5eeffddef4f70 | check |  | proceed | 3/5 |
| 2026-10-24T165242Z | bf16-m32-swapab-wgmma-port | primary | pooler GEMM | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T165302Z | moe-align-small-t-restack-on-master | primary | MoE align/dispatch | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-24T165619Z | fused-preattention-paired-pad-restack | primary | pre-attention norm+rope | done/neutral | neutral | 1.2745 | [1.255,1.294] | 5e9b5464d1444708 | check |  | proceed | 4/6 |
| 2026-10-24T172107Z | moe-chain-cuda-swapab-expert-once-large-t | rewrite | w13 marlin + silu-mul + w2 marlin | done/failed | failed |  |  | 1bba1bab70da4f61 | submit |  | proceed |  |
| 2026-10-24T172138Z | small-t-launch-minimal-dense-chain-pdl | rewrite | whole-layer fusion | done/gain | gain | 1.2283 | [1.210,1.247] | 81ee6714b7614a99 | submit | done | proceed | 6/6 |
| 2026-10-24T174421Z | decode-restack-sliding-port-paired-pad-gemms-t1 | primary | restack (sliding decode_1 port + paired pad/unpad + T=1 Triton GEMMs) | done/neutral | neutral | 1.1937 | [1.176,1.212] | 4739dd4590524487 | submit |  | proceed | 3/5 |
| 2026-10-24T184247Z | sliding-fused-batched-decode-port | primary | FA3 call (sliding decode, batch > 1) | done/neutral | neutral | 1.2248 | [1.206,1.243] | b01fd2df41a040b3 | check |  | proceed | 5/5 |
| 2026-10-24T184303Z | moe-dispatch-prep-on-side-stream | primary | MoE align/dispatch (stream placement) | done/neutral | neutral | 1.2294 | [1.211,1.248] | fe3422db9ea540d8 | check |  | proceed | 0/6 |
| 2026-10-24T184316Z | pre-expert-rmsnorm-triton | primary | pre-expert norm | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-24T184335Z | moe-small-t-cuda-swapab-port | rewrite | w13 marlin + silu-mul + w2 marlin (T<=32) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-24T184355Z | moe-bs1-fused-persistent-port | primary | w13 marlin + silu-mul + w2 marlin + post-expert combine (T=1) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T184430Z | l2-prefetch-o-pooler-during-sliding-attention | primary | o GEMM + pooler GEMM (L2 warm-up overlapped with attention) | done/regression | regression | 1.1670 | [1.150,1.185] | 96ee63e15edb438c | check |  | proceed | 3/6 |
| 2026-10-24T184451Z | prefork-qkv-o-swapab-t16-t32 | primary | qkv GEMM + o GEMM (before the router fork) | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-24T184511Z | moe-align-parallel-large-t-port | primary | MoE align/dispatch (T>32) | done/neutral | neutral | 1.2433 | [1.225,1.262] | b42e33406f2d4978 | check |  | proceed | 3/4 |
| 2026-10-24T184527Z | marlin-two-stream-halves-port | primary | w13 marlin + silu-mul + w2 marlin (8<=T<=32, stream overlap) | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-24T224424Z | dense-chain-stage2-attn-side-pdl | rewrite | qkv GEMM + o GEMM + attention-side small kernels (T<=32) | done/regression | regression | 1.2253 | [1.207,1.244] | b31b8453208a471c | check |  | proceed | 3/7 |
| 2026-10-24T224452Z | boundaries-unpooler-pdl-small-t | primary | post-MoE unpooler + residual boundaries (T<=32) | done/neutral | neutral | 1.2559 | [1.237,1.275] | 1dfe83fdbeea458a | check |  | proceed | 2/6 |
| 2026-10-24T224452Z | dl-route-warp-topk-pdl | primary | routing top-k + MoE align (kernel R, T<=32) | done/neutral | neutral | 1.2599 | [1.241,1.279] | b471a950805c4053 | check |  | proceed | 2/6 |
| 2026-10-24T224506Z | dl-p-wide-tile-t16-t32 | primary | pooler + router GEMM (kernel P, T=9..32) | done/neutral | neutral | 1.2457 | [1.227,1.264] | 9ad8e2cf869c4b4f | check |  | proceed | 2/4 |
| 2026-10-24T224525Z | full-decode-splitkv-restack-on-pdl | primary | FA3 call (full decode, batch > 1) | done/neutral | neutral | 1.2680 | [1.249,1.287] | c243a29f26dc4725 | check | done | required | 5/5 |
| 2026-10-24T224525Z | sliding-decode1-fused-restack-on-pdl | primary | FA3 call (sliding decode, batch 1) | done/neutral | neutral | 1.2374 | [1.219,1.256] | 30027c28212946fd | check |  | required | 1/4 |
| 2026-10-24T233529Z | stack-align-par-sliding1 | primary | MoE align (T>32) + sliding batch-1 attention | done/neutral | neutral | 1.2444 | [1.226,1.263] | d41952184b714e37 | check |  | required | 1/1 |
| 2026-10-24T233637Z | stack-sidestream-sliding1 | primary | T>32 MoE dispatch on the side stream + sliding batch-1 fused attention | done/neutral | neutral | 1.2352 | [1.217,1.254] | 158fed699fdb4d0e | check |  | required | 1/1 |
| 2026-10-24T233637Z | stack-align-par-sliding-batched | primary | T>32 parallel align + sliding batch>1 fused attention | done/failed | failed |  |  | b94e797333fb4a27 | check |  | required | 0/1 |
| 2026-10-24T233638Z | stack-sidestream-sliding-batched | primary | T>32 MoE dispatch on the side stream + sliding batch>1 fused attention | done/neutral | neutral | 1.2443 | [1.226,1.263] | 2193fc0dc9be4b3c | check |  | required | 1/1 |
| 2026-10-24T233721Z | stack-align-par-sidestream-sliding1 | primary | T>32 MoE dispatch (parallel align on the side stream) + sliding batch-1 attention | done/neutral | neutral | 1.2495 | [1.231,1.268] | cc92097b64784c0f | check | done | required | 1/1 |
| 2026-10-25T001250Z | moe-chain-cuda-swapab-expert-once-fix | rewrite | w13 marlin + silu-mul + w2 marlin (T>32) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T001251Z | full-decode-splitkv-kv-bw-3p5 | rewrite | FA3 call (full decode, batch > 1): splitkv_attn_kernel bytes in flight | done/neutral | neutral | 1.2663 | [1.247,1.285] | 6af3b0e3311943ee | check |  | proceed | 3/5 |
| 2026-10-25T001824Z | moe-chain-cuda-swapab-expert-once-fix-inproc | rewrite | w13 marlin + silu-mul + w2 marlin (T>32) | done/regression | regression | 1.1651 | [1.148,1.183] | 4ebd055adab34ecd | check |  | proceed | 2/4 |
| 2026-10-25T014617Z | moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound | rewrite | w13 marlin + silu-mul + w2 marlin (T>32) | done/regression | regression | 1.0637 | [1.048,1.080] | 51d047cbc5ff450a | check |  | proceed | 0/4 |
| 2026-10-25T035400Z | stack-route-topk-pdl-unpool-nf1 | primary | routing top-k (kernel R) + post-MoE unpooler (T<=8) | done/neutral | neutral | 1.2671 | [1.248,1.286] | 87211e1362b64c4c | check |  | proceed | 2/3 |
| 2026-10-25T035414Z | dense-chain-stage2-attn-side-pdl-t1-gate | primary | qkv GEMM + o GEMM + attention-side small kernels (T=1 only) | done/neutral | neutral | 1.2717 | [1.253,1.291] | c6cb53d8ebc2478a | check |  | proceed | 2/3 |
| 2026-10-25T042121Z | full-decode-streamk-ticket-port | rewrite | FA3 call (full decode, batch > 1): splitkv_attn_kernel + splitkv_combine_kernel | done/neutral | neutral | 1.2775 | [1.258,1.297] | 6d48932b230d4067 | check |  | proceed | 5/5 |
| 2026-10-25T042121Z | full-decode1-two-level-ticket-port | rewrite | FA3 call (full decode, batch 1) | done/neutral | neutral | 1.2746 | [1.255,1.294] | 5e44c7aedecd4cc1 | check |  | proceed | 2/5 |
| 2026-10-25T042121Z | sliding-batched-triton-b16-band | primary | FA3 call (sliding decode, batch 2..16) | done/neutral | neutral | 1.2743 | [1.255,1.293] | 02bf417257e14a3d | check |  | proceed | 3/5 |
| 2026-10-25T042121Z | sliding-cluster-c4-decode32-port | rewrite | FA3 call (sliding decode, batch 17..32) | done/neutral | neutral | 1.2693 | [1.250,1.288] | 0fb2c9451c5d4283 | check |  | proceed | 4/5 |
| 2026-10-25T042121Z | w2-marlin-atomic-add-t1-port | primary | w2 marlin (T=1) | done/neutral | neutral | 1.2856 | [1.266,1.305] | 7a9f5de5aecc4aba | check | done | proceed | 2/4 |
| 2026-10-25T054108Z | full-decode-sk-ticket-restack-b1-b64 | primary | FA3 call (full decode, all batches): split-KV A/B at 1<B<=64 and FA3 chain at B=1 | done/neutral | neutral | 1.2847 | [1.265,1.304] | 9c99379067084f6e | check |  | proceed | 4/5 |
| 2026-10-25T054127Z | sliding-cluster-c4-decode32-s2-k-encode-match | rewrite | FA3 call (sliding decode, batch 17..32) | done/neutral | neutral | 1.2803 | [1.261,1.299] | 7c04214e90354dfa | check |  | proceed | 2/5 |
| 2026-10-25T054152Z | dense-chain-s3-t1-q-sk2-o-route-unpool | rewrite | whole-layer fusion (T=1 dense chain: qkv/o GEMM side + routing R + unpooler) | done/neutral | neutral | 1.3018 | [1.282,1.321] | 34a9be06918d448b | check | superseded | proceed | 3/5 |
| 2026-10-25T110639Z | sliding-cluster-c4-s3-satfinite-on-sk-ticket | rewrite | FA3 call (sliding decode, batch 17..32) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T110701Z | qkv-o-swapab16-t2-t16-no-fork | primary | qkv GEMM + o GEMM (2 <= T <= 16) | done/neutral | neutral | 1.3056 | [1.286,1.325] | b0057d2e6cef4d5a | submit |  | proceed | 3/6 |
| 2026-10-25T145938Z | dl3-unpool-nf1-t2-t8 | primary | unpooler GEMM (T=2..8, dl3 piece C widened) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T145938Z | dl3-piece-a-gate-stride-check-fix | primary | full decode_1 attention side (dl3 piece A enable) | done/neutral | neutral | 1.3080 | [1.288,1.328] | 933d3f825a044d36 | submit | done | proceed | 7/7 |
| 2026-10-25T153205Z | dense-qkv-o-unpool-swapab-bf16-master-port-t1-t32 | primary | qkv GEMM + o GEMM + unpooler GEMM (T <= 32) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T153205Z | ffn-front-t128-t256-concat-route | rewrite | pooler GEMM + router GEMM + routing top-k + MoE align + pre-expert norm (T > 32) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T153205Z | moe-rega-bucket-gate-port-t128-t512 | rewrite | w13 marlin + silu-mul + w2 marlin (128 <= T <= 512) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T153205Z | preattn-fused-t-gt1-sk-and-fa3-feeds | primary | pre-attention norm+rope + cache store (T > 1, every shape not routed to _sliding_forward or dl3 piece A) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T153205Z | sliding-cluster-c16-decode1-port | primary | FA3 call (sliding decode, batch 1) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T153205Z | sliding-sk-window-port-decode32-verify1 | primary | FA3 call (sliding decode B=17..32 and sliding verify B=1: windowed stream-K) | done/neutral | neutral | 1.3117 | [1.292,1.331] | 3117f2e423d44cb7 | check |  | proceed | 3/6 |
| 2026-10-25T161651Z | combo-attn-master-sliding-race-skw-c16 | primary | combination: sliding attention per-band race (B=1, 2..16, 17..32, verify B=1) + routing R / unpooler tail | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T161651Z | combo-measured-rows-plus-rega-align2-large-t | primary | combination: C1 measured stack + w13/w2 rega and align2 at 128<=T<=512 | done/neutral | neutral | 1.3164 | [1.297,1.336] | eaa8cc0f18f64bfb | submit |  | proceed | 1/7 |
| 2026-10-25T161651Z | combo-measured-rows-slide-batched-preattn-t8-tail | primary | combination: sliding attention B=2..32 + pre-attention feeds (T>1) + routing R / unpooler (2<=T<=32 / 2<=T<=8) | done/neutral | neutral | 1.3142 | [1.295,1.334] | a903b9d004d2452d | submit | done | proceed | 3/8 |
| 2026-10-25T202335Z | swapab16-t2-t16-port-on-master | primary | qkv GEMM + o GEMM (2 <= T <= 16) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T212731Z | fsi-moe-midt-decode-port | rewrite | w13 marlin + silu-mul + w2 marlin + post-expert combine (9 <= T <= 32) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T212731Z | fsi-moe-tiny-t-decode-port | rewrite | w13 marlin + silu-mul + w2 marlin + post-expert combine (1 <= T <= 8) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T212731Z | marlin-bsm8-t17-t32-w13-atomic-t1 | primary | w13 marlin + w2 marlin (call knobs: block_size_m 8 at 17 <= T <= 32; w13 use_atomic_add at T = 1) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T212731Z | marlin-hot-cold-two-class-bsm-t128-t256 | primary | w13 marlin + w2 marlin (block_size_m per expert class, 128 <= T <= 256) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-25T212731Z | wg2-occ2-b256-decode-port | primary | w13 marlin + silu-mul + w2 marlin (T = 256; T = 128 candidate) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-29T190020Z | hothalf-a3-exactn-ring-wh | primary | moe hot half (wg_kernel w13/w2, dq_stage) at T=128/256 | running | running |  |  | 44f303647100481b | check |  | proceed |  |

## Held

- bf16-m32-swapab-wgmma-port: reviewer: ~0.15% headline (3-5 us at decode_32), inside drift; edits the same call sites as bf16-shape-gated-triton-gemms-small-t (in the judge). Revisit as part of a combined decode GEMM restack after that harvests (per-layer kernel names for attribution)
- moe-small-t-cuda-swapab-port: its parent, moe cuda-nvfp4-swapab-mma-persistent-small-m (8f198afc3bf14f6f), failed its tests on 2026-10-24; the minter made dispatch conditional on that parent winning
- moe-bs1-fused-persistent-port: adversarial review: Hold: the hypothesis gates its own dispatch on the moe parent (1105a02aa3fa4e56) being faster than moe master at bs=1, and it was not (bs=1 50.93 us vs 48.37 us
- moe-chain-cuda-swapab-expert-once-fix: Retired: its design runs the self-check in a child process, which the rules statically reject; superseded by the -inproc line (LESSONS 2026-10-25).
- sliding-cluster-c4-s3-satfinite-on-sk-ticket: screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: base never landed; sliding B=17..32 already on stream-K window kernel
- dl3-unpool-nf1-t2-t8: superseded: the unpooler path is already on master (decode minter, 2026-10-25)
- dense-qkv-o-unpool-swapab-bf16-master-port-t1-t32: adversarial review: Hold. The conjecture is that bf16 master's 'newer' swap-AB kernel will run in decode at its standalone 9.7/3.9 us. That premise is false, and the transfer has a
- ffn-front-t128-t256-concat-route: screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: w6b-t128diet cells landed
- moe-rega-bucket-gate-port-t128-t512: superseded: decode's own selector switched rega OFF at T=128 (384.8 vs Marlin 327.2 us) and T=256 (454.6 vs 440.0) on 2026-10-25; dead in decode (decode minter, 2026-10-25)
- preattn-fused-t-gt1-sk-and-fa3-feeds: adversarial review: Duplicate of master: the landed combo-measured-rows-slide-batched-preattn-t8-tail (ec89382b, BEST 1.3142x) carries _pre_attention_kernel as source S3 (PAIRED an
- sliding-cluster-c16-decode1-port: adversarial review: Plausible but small (~2 us on the 118.5 us sliding decode_1 row, ~0.25% headline); its own first measurement must be an okc check, and it is an unchecked cross-
- combo-attn-master-sliding-race-skw-c16: superseded: remainder equals sliding-sk-window-port-decode32-verify1 + sliding-cluster-c16-decode1-port, both queued (decode minter, 2026-10-25)
- swapab16-t2-t16-port-on-master: adversarial review: Hold. The swap-AB in-gate effect (T=8/16) was already measured kernel-by-kernel in the parent's profile b0057d2e6cef4d5a: ~1.5-2 us per in-gate shape, ~+0.3% he
- fsi-moe-midt-decode-port: screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: Engine A v4c at T=9..32
- fsi-moe-tiny-t-decode-port: screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: Engine A replaced Marlin at T<=8
- marlin-bsm8-t17-t32-w13-atomic-t1: screen 2026-10-28 1520Z (value bar, vs master): STALE: Marlin no longer on the T<=32 path
- marlin-hot-cold-two-class-bsm-t128-t256: screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: hotcold landed (H=-16)
- wg2-occ2-b256-decode-port: screen 2026-10-28 1520Z (value bar, vs master): BELOW-BAR ~0.4%; Marlin premise gone
