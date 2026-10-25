# muse_decode_layer hypotheses

Regenerated 2026-10-25T034141Z. Best: 1.2521x [1.2333,1.2709] job `39bcf05afc4e4dd3` (submit)

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
| 2026-10-24T140057Z | post-expert-norm-triton-small-t | primary | post-expert norm+combine | done/gain | gain | 1.1635 | [1.146,1.181] | 9dcaec316b3045a7 | check | done | proceed | 6/6 |
| 2026-10-24T140116Z | bf16-shape-gated-triton-gemms-small-t | primary | pooler GEMM | done/neutral | neutral | 1.1688 | [1.151,1.186] | 6076ae1f250143aa | submit |  | proceed | 2/6 |
| 2026-10-24T140135Z | sliding-fused-triton-decode1-port | primary | FA3 call | done/neutral | neutral | 1.1690 | [1.152,1.187] | 12198572363b49a4 | submit |  | proceed | 5/5 |
| 2026-10-24T140156Z | paired-heads-pad-unpad-triton | primary | output norm-gate | done/neutral | neutral | 1.1654 | [1.148,1.183] | a92b64be4c6547d3 | submit |  | proceed | 4/7 |
| 2026-10-24T165154Z | full-decode-cuda-splitkv-port | rewrite | FA3 call | done/neutral | neutral | 1.1862 | [1.168,1.204] | e789032128df4291 | submit |  | proceed | 5/5 |
| 2026-10-24T165224Z | full-verify-cuda-splitkv-stage2 | rewrite | FA3 call | done/regression | regression | 1.1559 | [1.139,1.173] | e9d5eeffddef4f70 | check |  | proceed | 3/5 |
| 2026-10-24T165242Z | bf16-m32-swapab-wgmma-port | primary | pooler GEMM | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T165302Z | moe-align-small-t-restack-on-master | primary | MoE align/dispatch | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T165619Z | fused-preattention-paired-pad-restack | primary | pre-attention norm+rope | queue | queued |  |  |  |  |  | proceed |  |
| 2026-10-24T172107Z | moe-chain-cuda-swapab-expert-once-large-t | rewrite | w13 marlin + silu-mul + w2 marlin | done/failed | failed |  |  | 1bba1bab70da4f61 | submit |  | proceed |  |
| 2026-10-24T172138Z | small-t-launch-minimal-dense-chain-pdl | rewrite | whole-layer fusion | done/gain | gain | 1.2283 | [1.210,1.247] | 81ee6714b7614a99 | submit | done | proceed | 6/6 |
| 2026-10-24T174421Z | decode-restack-sliding-port-paired-pad-gemms-t1 | primary | restack (sliding decode_1 port + paired pad/unpad + T=1 Triton GEMMs) | done/neutral | neutral | 1.1937 | [1.176,1.212] | 4739dd4590524487 | submit |  | proceed | 3/5 |
| 2026-10-24T184247Z | sliding-fused-batched-decode-port | primary | FA3 call (sliding decode, batch > 1) | done/neutral | neutral | 1.2248 | [1.206,1.243] | b01fd2df41a040b3 | check |  | proceed | 5/5 |
| 2026-10-24T184303Z | moe-dispatch-prep-on-side-stream | primary | MoE align/dispatch (stream placement) | done/neutral | neutral | 1.2294 | [1.211,1.248] | fe3422db9ea540d8 | check |  | proceed | 0/6 |
| 2026-10-24T184316Z | pre-expert-rmsnorm-triton | primary | pre-expert norm | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T184335Z | moe-small-t-cuda-swapab-port | rewrite | w13 marlin + silu-mul + w2 marlin (T<=32) | queue | hold |  |  |  |  |  | required |  |
| 2026-10-24T184355Z | moe-bs1-fused-persistent-port | primary | w13 marlin + silu-mul + w2 marlin + post-expert combine (T=1) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T184430Z | l2-prefetch-o-pooler-during-sliding-attention | primary | o GEMM + pooler GEMM (L2 warm-up overlapped with attention) | done/regression | regression | 1.1670 | [1.150,1.185] | 96ee63e15edb438c | check |  | proceed | 3/6 |
| 2026-10-24T184451Z | prefork-qkv-o-swapab-t16-t32 | primary | qkv GEMM + o GEMM (before the router fork) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T184511Z | moe-align-parallel-large-t-port | primary | MoE align/dispatch (T>32) | done/neutral | neutral | 1.2433 | [1.225,1.262] | b42e33406f2d4978 | check |  | proceed | 3/4 |
| 2026-10-24T184527Z | marlin-two-stream-halves-port | primary | w13 marlin + silu-mul + w2 marlin (8<=T<=32, stream overlap) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-24T224424Z | dense-chain-stage2-attn-side-pdl | rewrite | qkv GEMM + o GEMM + attention-side small kernels (T<=32) | done/regression | regression | 1.2253 | [1.207,1.244] | b31b8453208a471c | check |  | proceed | 3/7 |
| 2026-10-24T224452Z | boundaries-unpooler-pdl-small-t | primary | post-MoE unpooler + residual boundaries (T<=32) | done/neutral | neutral | 1.2559 | [1.237,1.275] | 1dfe83fdbeea458a | check |  | proceed | 2/6 |
| 2026-10-24T224452Z | dl-route-warp-topk-pdl | primary | routing top-k + MoE align (kernel R, T<=32) | done/neutral | neutral | 1.2599 | [1.241,1.279] | b471a950805c4053 | check |  | proceed | 2/6 |
| 2026-10-24T224506Z | dl-p-wide-tile-t16-t32 | primary | pooler + router GEMM (kernel P, T=9..32) | done/neutral | neutral | 1.2457 | [1.227,1.264] | 9ad8e2cf869c4b4f | check |  | proceed | 2/4 |
| 2026-10-24T224525Z | full-decode-splitkv-restack-on-pdl | primary | FA3 call (full decode, batch > 1) | done/neutral | neutral | 1.2680 | [1.249,1.287] | c243a29f26dc4725 | check |  | required | 5/5 |
| 2026-10-24T224525Z | sliding-decode1-fused-restack-on-pdl | primary | FA3 call (sliding decode, batch 1) | done/neutral | neutral | 1.2374 | [1.219,1.256] | 30027c28212946fd | check |  | required | 1/4 |
| 2026-10-24T233529Z | stack-align-par-sliding1 | primary | MoE align (T>32) + sliding batch-1 attention | done/neutral | neutral | 1.2444 | [1.226,1.263] | d41952184b714e37 | check |  | required | 1/1 |
| 2026-10-24T233637Z | stack-sidestream-sliding1 | primary | T>32 MoE dispatch on the side stream + sliding batch-1 fused attention | done/neutral | neutral | 1.2352 | [1.217,1.254] | 158fed699fdb4d0e | check |  | required | 1/1 |
| 2026-10-24T233637Z | stack-align-par-sliding-batched | primary | T>32 parallel align + sliding batch>1 fused attention | done/failed | failed |  |  | b94e797333fb4a27 | check |  | required | 0/1 |
| 2026-10-24T233638Z | stack-sidestream-sliding-batched | primary | T>32 MoE dispatch on the side stream + sliding batch>1 fused attention | done/neutral | neutral | 1.2443 | [1.226,1.263] | 2193fc0dc9be4b3c | check |  | required | 1/1 |
| 2026-10-24T233721Z | stack-align-par-sidestream-sliding1 | primary | T>32 MoE dispatch (parallel align on the side stream) + sliding batch-1 attention | done/neutral | neutral | 1.2495 | [1.231,1.268] | cc92097b64784c0f | check | done | required | 1/1 |
| 2026-10-25T001250Z | moe-chain-cuda-swapab-expert-once-fix | rewrite | w13 marlin + silu-mul + w2 marlin (T>32) | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T001251Z | full-decode-splitkv-kv-bw-3p5 | rewrite | FA3 call (full decode, batch > 1): splitkv_attn_kernel bytes in flight | done/neutral | neutral | 1.2663 | [1.247,1.285] | 6af3b0e3311943ee | check |  | proceed | 3/5 |
| 2026-10-25T001824Z | moe-chain-cuda-swapab-expert-once-fix-inproc | rewrite | w13 marlin + silu-mul + w2 marlin (T>32) | done/regression | regression | 1.1651 | [1.148,1.183] | 4ebd055adab34ecd | check |  | proceed | 2/4 |
| 2026-10-25T014617Z | moe-chain-cuda-swapab-expert-once-s3-rebase-fp32ref-bound | rewrite | w13 marlin + silu-mul + w2 marlin (T>32) | queue | queued |  |  |  |  |  | required |  |

## Held

- fa3-num-splits-full-no-metadata: adversarial review: An explicit num_splits cannot raise FA3's split count on this call: every flash_attn_with_kvcache call with cache_seqlens is varlen, so FA3 runs its device-side
- moe-align-local-expert-count: weak: the identical one-argument change (align over 32 experts instead of 256) was already measured in muse_moe_w4a16 (sglang-marlin-path-parity): verify_16/32 flat, decode_1 about 4% slower. Under the landed block cap the verify effect is a ~2 us fill launch, below resolution; decode has few checks left today. The reviewer suggests at most a T>=128-only variant later.
- trtllm-decode-full-attention: adversarial review: Hold. The spec was written on 2026-10-23 against the parity base. Since then master has landed head-paired FA3 for full layers at batch>1, so as written this ch
- fa3-num-splits-sliding: adversarial review: The premise (FA3 sizes the sliding split-KV for the 80K page-table extent) is contradicted by both FA3's own get_num_splits, which clamps seqlen_k by the window
- moe-align-single-kernel-small-t-control-capacity-256: control for moe-align-single-kernel-small-t, which was neutral; a capacity control of a non-gain cannot change a decision, and decode has 2 checks left for the attention rewrite and pairing follow-ups
- fa3-full-single-split-probe: adversarial review: The probe is sound and would give a clear reading, but neither reading can change a decision that is open right now. The rewrite it is meant to inform (triton-f
- router-triton-splitk-side-stream: reviewer: the 9.2-9.5 us splitK row is 2 calls (~4.6 us each), so the side-stream router branch (~14-18 us) already finishes before the ~22-25 us pooler and is off the critical path; most likely flat, and decode has 4 checks left
- triton-flash-decode-full-verify-paired-kv: adversarial review: Stage 1 measured the paired-KV kernel at about 1.63-1.66 TB/s, which is slower than FA3: 200.8 vs 188.8 us at decode_16 and 394.8 vs 374.4 us at decode_32. That
- bf16-m32-swapab-wgmma-port: reviewer: ~0.15% headline (3-5 us at decode_32), inside drift; edits the same call sites as bf16-shape-gated-triton-gemms-small-t (in the judge). Revisit as part of a combined decode GEMM restack after that harvests (per-layer kernel names for attribution)
- moe-align-small-t-restack-on-master: projected +0.5-0.9% is below noise, and small-t-launch-minimal-dense-chain-pdl (built for the reset) replaces the same align at T<=32. Revisit only if that rewrite fails.
- pre-expert-rmsnorm-triton: adversarial review: Stale premise: master (small-t-launch-minimal-dense-chain-pdl, f1f2f35) already folds the pre-expert RMSNorm into dl_route_norm_align at T <= 32, so 4 of the 6 
- moe-small-t-cuda-swapab-port: its parent, moe cuda-nvfp4-swapab-mma-persistent-small-m (8f198afc3bf14f6f), failed its tests on 2026-10-24; the minter made dispatch conditional on that parent winning
- moe-bs1-fused-persistent-port: adversarial review: Hold: the hypothesis gates its own dispatch on the moe parent (1105a02aa3fa4e56) being faster than moe master at bs=1, and it was not (bs=1 50.93 us vs 48.37 us
- prefork-qkv-o-swapab-t16-t32: adversarial review: Its main purpose no longer holds on the current master: small-t-launch-minimal-dense-chain-pdl removed the router side stream at T<=32, so there is no fork for 
- marlin-two-stream-halves-port: adversarial review: Hold: the moe parent it ports (marlin-two-stream-expert-halves-small-m, job 5d630369b1634f77) came back neutral, with every scored shape flat and verify_1 1.5% 
- moe-chain-cuda-swapab-expert-once-fix: adversarial review: Hold for a rules problem only: fix (1), the central fix, runs the self-check in a torch.multiprocessing spawn child or subprocess, and TASK.md says a submission
