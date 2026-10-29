# muse_moe_w4a16 hypotheses

Regenerated 2026-10-29T202413Z. Best: 1.9827x [1.9529,2.0124] job `54b70249868f468b` (check)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T225501Z | sglang-marlin-path-parity | primary | MoE align/dispatch | done/gain | gain | 0.9888 | [0.974,1.004] | fb2cc736613747df | check | merged-unsubmitted | proceed | 3/3 |
| 2026-10-23T225501Z | triton-w13-grouped-gemv-small-m | primary | w13 marlin | done/regression | regression | 0.7506 | [0.739,0.762] | f31a03eb9bc6410f | check |  | proceed | 2/4 |
| 2026-10-23T225501Z | triton-w13-silu-epilogue | primary | silu-mul | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-23T225501Z | triton-w2-grouped-gemv-small-m | primary | w2 marlin | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-23T225502Z | align-single-token-ep | primary | MoE align/dispatch | done/neutral | neutral | 0.9951 | [0.980,1.010] | a7bb0cecaf8d4de1 | check |  | proceed | 3/3 |
| 2026-10-23T225502Z | marlin-workspace-blocks-per-sm-8 | primary | w13 marlin | done/neutral | neutral | 0.9921 | [0.977,1.007] | 75805dc2f18a4d06 | check |  | proceed | 2/4 |
| 2026-10-23T225502Z | w13-marlin-atomic-add-small-m | primary | w13 marlin | done/neutral | neutral | 0.9869 | [0.972,1.002] | c8ea8e21eea447f9 | check |  | proceed | 1/4 |
| 2026-10-23T225502Z | w13-marlin-bf16-reduce-verify | primary | w13 marlin | done/neutral | neutral | 0.9840 | [0.969,0.999] | 8d389f3d3e2d473f | check |  | proceed | 1/3 |
| 2026-10-23T225502Z | marlin-block-size-m-next-larger | primary | MoE align/dispatch | done/regression | regression | 0.9178 | [0.904,0.932] | 2639e3ca7f5549de | check |  | proceed | 3/5 |
| 2026-10-23T225502Z | triton-small-m-threshold-32 | primary | MoE align/dispatch | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-24T075126Z | post-expert-norm-triton-small-m | primary | post-expert norm+combine | done/gain | gain | 1.1297 | [1.113,1.147] | bacd1f502f4e46fa | check | superseded | proceed | 6/6 |
| 2026-10-24T075232Z | marlin-block-size-m-next-smaller | primary | w13 marlin | done/gain | gain | 1.0394 | [1.024,1.055] | 198f10f446a847cb | check | superseded | proceed | 5/6 |
| 2026-10-24T075232Z | align-256-experts-below-128 | primary | MoE align/dispatch | done/neutral | neutral | 0.9998 | [0.985,1.015] | e753d5b9df644ebb | check |  | proceed | 3/6 |
| 2026-10-24T075257Z | marlin-block-size-m-verify-two-smaller | primary | w13 marlin | done/gain | gain | 1.0916 | [1.075,1.108] | ed837a4a46d44a95 | check | superseded | proceed | 6/6 |
| 2026-10-24T075257Z | align-remote-sentinel-ids | primary | MoE align/dispatch | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-24T081339Z | small-m-grouped-gemv-splitk-rewrite | rewrite | w13 marlin | done/regression | regression | 0.6635 | [0.654,0.673] | 734a1159ff334c3d | check |  | proceed | 3/6 |
| 2026-10-24T100217Z | marlin-block-m-cap-16-port | primary | w13 marlin | done/neutral | neutral | 1.0950 | [1.079,1.111] | 1c33392914594026 | check |  | proceed | 6/6 |
| 2026-10-24T112137Z | marlin-block-m-two-smaller-on-postnorm | primary | w13 marlin | done/gain | gain | 1.2536 | [1.235,1.272] | c00ef7295c154867 | check | superseded | proceed | 6/6 |
| 2026-10-24T112155Z | triton-map-align-small-m-port | primary | MoE align/dispatch | done/neutral | neutral | 1.1850 | [1.167,1.203] | c2feb68aafb44aeb | check |  | proceed | 6/6 |
| 2026-10-24T112212Z | post-norm-triton-all-m | primary | post-expert norm+combine | done/neutral | neutral | 1.2483 | [1.230,1.267] | 2ec5387ffe8748c9 | check |  | proceed | 4/6 |
| 2026-10-24T132314Z | triton-map-align-on-block-m-restack | primary | MoE align/dispatch + block_size_m pick | done/gain | gain | 1.3117 | [1.292,1.331] | bc56182a80444d16 | check | done | proceed | 6/6 |
| 2026-10-24T140229Z | triton-map-align-verify-chunked | primary | MoE align/dispatch | done/failed | failed |  |  | 3aa4d7e193244255 | check |  | proceed |  |
| 2026-10-24T140229Z | marlin-two-class-block-m-decode32 | primary | MoE align/dispatch + block_size_m pick | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-24T140229Z | swapab-nvfp4-gemm-verify-rewrite | rewrite | w13 marlin + w2 marlin | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-24T140229Z | swapab-w13-silu-epilogue-verify | rewrite | silu-mul | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-24T144931Z | triton-map-align-verify-chunked-pow2-fix | primary | MoE align/dispatch | done/neutral | neutral | 1.3107 | [1.291,1.330] | c1c72d5d19ea444f | check |  | proceed | 2/4 |
| 2026-10-24T164817Z | align-parallel-per-expert-verify | primary | MoE align/dispatch | done/neutral | neutral | 1.3249 | [1.305,1.345] | 45eacf4ca2e849d2 | check | done | proceed | 4/4 |
| 2026-10-24T164836Z | w2-l2-prefetch-decode1 | primary | w2 marlin | done/regression | regression | 1.2615 | [1.243,1.280] | ac6897f284934dad | check |  | proceed | 3/4 |
| 2026-10-24T164857Z | marlin-two-stream-expert-halves-small-m | primary | launch overhead | done/neutral | neutral | 1.3024 | [1.283,1.322] | 5d630369b1634f77 | submit |  | proceed | 2/5 |
| 2026-10-24T165851Z | cuda-fused-moe-bs1-persistent-w13-silu-w2 | rewrite | whole-layer fusion | done/neutral | neutral | 1.3074 | [1.269,1.327] | 1105a02aa3fa4e56 | submit |  | proceed | 5/6 |
| 2026-10-24T165851Z | cuda-nvfp4-swapab-mma-persistent-small-m | rewrite | w13 marlin + silu-mul + w2 marlin | done/failed | failed |  |  | 8f198afc3bf14f6f | submit |  | proceed |  |
| 2026-10-24T172038Z | cuda-nvfp4-wgmma-rega-grouped-verify | rewrite | w13 marlin + silu-mul + w2 marlin | done/neutral | neutral | 1.3123 | [1.293,1.332] | 3bb822f22ebc4a5c | submit |  | proceed | 4/6 |
| 2026-10-24T172123Z | cuda-fused-moe-midm-single-launch-routed | rewrite | whole-layer fusion | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-24T183202Z | cuda-dispatch-combine-pdl-verify | rewrite | MoE align/dispatch + post-expert norm+combine | done/neutral | neutral | 1.3283 | [1.308,1.348] | c41860068fe4432f | submit |  | proceed | 4/6 |
| 2026-10-24T183618Z | post-norm-combine-hoisted-slot-loads | primary | post-expert norm+combine | done/neutral | neutral | 1.3282 | [1.308,1.348] | d7501477ce2748d5 | submit |  | proceed | 3/6 |
| 2026-10-24T183635Z | post-norm-triton-hoisted-decode32 | primary | post-expert norm+combine | done/neutral | neutral | 1.3248 | [1.305,1.345] | 47d186dd91974d84 | check |  | proceed | 5/6 |
| 2026-10-24T183649Z | post-norm-pdl-launch-small-m | primary | launch overhead | done/neutral | neutral | 1.3222 | [1.302,1.342] | 5202cf15936c4492 | check |  | proceed | 3/6 |
| 2026-10-24T183702Z | align-parallel-from-17-tokens | primary | MoE align/dispatch | done/neutral | neutral | 1.3250 | [1.305,1.345] | ed6d884240084d9c | submit |  | proceed | 5/6 |
| 2026-10-24T183722Z | align-small-one-warp-tiny-m | primary | MoE align/dispatch | done/neutral | neutral | 1.3287 | [1.309,1.349] | 38532ba10e394e3c | submit |  | proceed | 4/6 |
| 2026-10-25T000436Z | stack-onewarp-hoisted-dispatchcombine | primary | MoE align (bs<=8 one warp; bs>=128 CUDA) + post-expert combine (bs<32 hoisted; bs>=128 CUDA) | done/neutral | neutral | 1.3176 | [1.298,1.337] | 7e5ab95a413f4ebe | check |  | required | 0/1 |
| 2026-10-25T000511Z | stack-onewarp-hoisted-dispatchcombine-from17 | primary | MoE align (bs<=8 one warp; 17..32 parallel; bs>=128 CUDA) + post-expert combine | done/neutral | neutral | 1.3237 | [1.304,1.344] | 0fde9013de97404d | check |  | required | 0/1 |
| 2026-10-25T001223Z | cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild | rewrite | w13 marlin + silu-mul + w2 marlin | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T001240Z | cuda-nvfp4-swapab-mma-persistent-small-m-childcheck | rewrite | w13 marlin + silu-mul + w2 marlin | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-25T001254Z | cuda-fused-moe-bs1-persistent-nospin-bandwidth | rewrite | whole-layer fusion | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-25T001824Z | cuda-fused-moe-bs1-persistent-nospin-bandwidth-inproc | rewrite | whole-layer fusion | done/neutral | neutral | 1.3257 | [1.306,1.368] | 7a6919420b724ee7 | check |  | proceed | 4/6 |
| 2026-10-25T001824Z | cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc | rewrite | w13 marlin + silu-mul + w2 marlin | done/neutral | neutral | 1.3303 | [1.310,1.350] | b56d17fda2d54be2 | check |  | proceed | 6/6 |
| 2026-10-25T001824Z | cuda-nvfp4-swapab-mma-persistent-small-m-inproc | rewrite | w13 marlin + silu-mul + w2 marlin | done/regression | regression | 0.9250 | [0.911,0.939] | fecc63ef567b4dc2 | check |  | proceed | 1/6 |
| 2026-10-25T001825Z | cuda-fused-moe-bs1-persistent-bandwidth-s3-inproc | rewrite | whole-layer fusion | done/neutral | neutral | 1.3295 | [1.310,1.349] | cd0f6b08a6dd4165 | check |  | proceed | 5/6 |
| 2026-10-25T011255Z | cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix | rewrite | w13 marlin + silu-mul + w2 marlin | done/neutral | neutral | 1.3307 | [1.311,1.351] | 583a73418fbf4d01 | check |  | proceed | 4/6 |
| 2026-10-25T011332Z | cuda-nvfp4-wgmma-rega-s2-fix-gate-from-8 | rewrite | w13 marlin + silu-mul + w2 marlin | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-25T011332Z | cuda-nvfp4-wgmma-rega-s2-per-kernel-enable | rewrite | w13 marlin + silu-mul + w2 marlin | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-25T011401Z | w2-marlin-atomic-add-small-m | primary | w2 marlin | done/neutral | neutral | 1.3445 | [1.324,1.365] | a2744bc496b84790 | check | done | proceed | 3/6 |
| 2026-10-25T011401Z | cuda-align-kernel-a-from17 | primary | MoE align/dispatch | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-25T011401Z | cuda-combine-kernel-b-bs128-only | primary | post-expert norm+combine | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-25T035329Z | cuda-nvfp4-wgmma-rega-grouped-s3-wide-bucket-fix | rewrite | w13 marlin + silu-mul + w2 marlin | done/failed | failed |  |  | 3bdec0e58d684c17 | check |  | proceed | 0/3 |
| 2026-10-25T042150Z | cuda-nvfp4-wgmma-rega-tiny-m-bucket16 | rewrite | w13 marlin + silu-mul + w2 marlin | done/neutral | neutral | 1.3453 | [1.325,1.366] | 3df5c69750884600 | check |  | proceed | 6/6 |
| 2026-10-25T042150Z | w13-marlin-atomic-add-on-master-small-m | primary | w13 marlin | done/neutral | neutral | 1.3558 | [1.335,1.376] | 171684977ec1470f | check |  | proceed | 4/6 |
| 2026-10-25T042150Z | cuda-nvfp4-wgmma-rega-midm-bucket-cap32 | rewrite | w13 marlin + silu-mul + w2 marlin | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-25T042150Z | cuda-nvfp4-wgmma-rega-w2-only-midm | rewrite | w2 marlin | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-25T042151Z | marlin-exec-config-autotune-mid-m | primary | Marlin launch config (w13 + w2) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T042151Z | w2-marlin-atomic-add-verify | primary | w2 marlin | done/neutral | neutral | 1.3407 | [1.321,1.361] | ccbcda65da584be0 | check |  | proceed | 4/6 |
| 2026-10-25T044131Z | cuda-nvfp4-wgmma-rega-grouped-s4-replay-safe | rewrite | wgmma NVFP4 grouped w13+silu/w2 at 128<=T<=512: per-state tensor maps, graph-replay safety | done/neutral | neutral | 1.3439 | [1.324,1.364] | 4e7f8be895a145ba | check |  | proceed | 0/3 |
| 2026-10-25T054545Z | cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix | rewrite | w13 marlin + silu-mul + w2 marlin | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T054545Z | cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate | rewrite | w13 marlin + silu-mul + w2 marlin | done/neutral | neutral | 1.3744 | [1.354,1.395] | 7be125e9e1ea42e3 | check | done | proceed | 4/4 |
| 2026-10-25T054545Z | cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-forced | rewrite | w13 marlin + silu-mul + w2 marlin | done/neutral | neutral | 1.3490 | [1.329,1.369] | d4ad4f31d2764050 | check |  | proceed | 5/6 |
| 2026-10-25T054545Z | w13-marlin-atomic-add-t1-only | primary | w13 marlin | done/neutral | neutral | 1.3461 | [1.326,1.366] | 1048c6edf3e54892 | check |  | proceed | 3/4 |
| 2026-10-25T054545Z | marlin-block-m-8-verify16 | primary | MoE align/dispatch + block_size_m pick | done/regression | regression | 1.2992 | [1.280,1.319] | d0811c27500c416a | check |  | proceed | 3/4 |
| 2026-10-25T110521Z | cuda-nvfp4-wgmma-rega-b128-n32-deep-ring | rewrite | w13 marlin + silu-mul + w2 marlin | done/neutral | neutral | 1.3690 | [1.348,1.389] | 6759e36bbd4648fc | check |  | proceed | 3/3 |
| 2026-10-25T110544Z | cuda-nvfp4-wgmma-rega-b256-n64-deep-ring | rewrite | w13 marlin + silu-mul + w2 marlin | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-25T125414Z | cuda-nvfp4-wgmma-rega-pingpong-consumers | rewrite | w13 marlin + silu-mul + w2 marlin | done/expired | expired |  |  |  |  |  | required |  |
| 2026-10-25T125512Z | cuda-nvfp4-wgmma-rega-producer-scale-expand | rewrite | w13 marlin + silu-mul + w2 marlin | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T145952Z | align-two-phase-port-from17 | primary | MoE align/dispatch | done/neutral | neutral | 1.3700 | [1.349,1.391] | c05a8f7326004be6 | submit | done | proceed | 4/6 |
| 2026-10-25T145952Z | silu-mul-triton-row-tiled-gated | primary | silu-mul | done/expired | expired |  |  |  |  |  | proceed |  |
| 2026-10-25T150301Z | restack-bucket-gate-on-master-align-atomic | primary | expert GEMM path selection (wgmma vs Marlin) restack | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-25T153417Z | cuda-fused-split-i-moe-tiny-t | rewrite | w13 marlin + silu-mul + w2 marlin | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-25T153417Z | wg-producer-direct-x-rows-no-gather | primary | MoE dispatch (wgmma x gather) | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T153417Z | cuda-fused-split-i-moe-midt-cluster-reduce | rewrite | w13 marlin + silu-mul + w2 marlin | done/neutral | neutral | 1.3885 | [1.368,1.409] | 6fe2a63f490f49a3 | check |  | proceed | 3/6 |
| 2026-10-25T153417Z | wg-pdl-gather-w13-w2-chain | primary | launch overhead | done/neutral | neutral | 1.3752 | [1.355,1.396] | 667b64e3fb1343ec | submit | done | proceed | 3/4 |
| 2026-10-25T153417Z | wg-stream-k-tail-split | primary | w13 marlin + silu-mul + w2 marlin | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-25T153417Z | wg-two-ctas-per-sm-port-b128-n32 | primary | w13 marlin + silu-mul + w2 marlin | done/expired | expired |  |  |  |  |  | hold |  |
| 2026-10-25T161613Z | combo-smallm-two-stream-halves-x-w13-splitk | primary | combination: small-M Marlin two-stream halves x w13 split-K atomic + shared floor | done/neutral | neutral | 1.3887 | [1.368,1.409] | 9a455a2aae7842cb | submit |  | proceed | 1/6 |
| 2026-10-25T161613Z | combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2 | primary | combination: naive top-row stack (decode_1 w13 atomic, decode_32 align+post-norm, verify_32 wgmma align2 front end) | done/neutral | neutral | 1.3866 | [1.366,1.407] | b395f5b6097348c7 | submit | done | proceed | 4/6 |
| 2026-10-25T161613Z | combo-wg-occ2-v64-hot-interleave-align2 | primary | combination: wgmma two-CTA/SM (prefill port) x hot-expert tile interleave + shared floor | done/neutral | neutral | 1.4021 | [1.381,1.423] | 31c27e4b8fc84bd2 | submit |  | proceed | 4/5 |
| 2026-10-25T211728Z | restack-occ2-wg2-b256-two-stream-halves-on-master | primary | expert GEMM path selection (wgmma wg2 at B256 + Marlin two-stream halves at T 8-32) restack | done/neutral | neutral | 1.3996 | [1.379,1.421] | 6ee7b6e302a348c8 | check |  | proceed | 3/6 |
| 2026-10-25T211822Z | nvfp4-rowowner-bulk-stream-tiny-t-rewrite | rewrite | w13 marlin + silu-mul + w2 marlin | done/neutral | neutral | 1.3887 | [1.368,1.409] | c08dbd737e29442d | check |  | proceed | 2/4 |
| 2026-10-25T211843Z | wg2-pingpong-consumers-occ2-b256 | primary | w13 marlin + silu-mul + w2 marlin | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T211909Z | wg2-w13-occ2-nmax128-stg2-hot | primary | w13 marlin + silu-mul | queue | hold |  |  |  |  |  | proceed |  |
| 2026-10-25T211926Z | wg2-stream-k-tail-floor-g-over-r | primary | w13 marlin + silu-mul + w2 marlin | queue | hold |  |  |  |  |  | hold |  |
| 2026-10-29T151134Z | cmbf-combine-fused-w2-epilogue-t8-32 | primary | post-expert norm+combine | done/neutral | neutral | 1.9675 | [1.938,2.004] | a5aee75176174ce4 | check |  | proceed | 5/6 |
| 2026-10-29T162119Z | cmbf-combine-fused-w2-epilogue-t8-32-cf | confirm→cmbf-combine-fused-w2-epilogue-t8-32 | post-expert norm+combine | queue | queued |  |  |  |  |  | proceed |  |
| 2026-10-29T195507Z | hothalf-wh-a5b-port | primary | moe hot half (wg_kernel w13/w2, dq_stage) at T=128/256 | done/gain | gain | 1.9827 | [1.953,2.012] | 54b70249868f468b | check | merged | proceed | 0/3 |

## Held

- cuda-fused-moe-midm-single-launch-routed: moe per-shape read 2026-10-25: its parent cuda-nvfp4-swapab-mma-persistent-small-m hit an illegal instruction on first launch (0/0 tests); hold until a fixed swap-AB runs on the CUDA path and reaches its M2
- cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild: adversarial review: The sm_90a build fix is correct and the rewrite is worth re-running as stage 1, but the spec also requires the self-check to run in a spawned child process, and
- marlin-exec-config-autotune-mid-m: Reviewer: no evidence SGLang moe_wna16_marlin_gemm takes thread_k/thread_n/blocks_per_sm; a print-only kwargs probe + cold-L2 config timing table rides in w13-marlin-atomic-add-on-master-small-m prepare(). Build only if a config is >=8% faster on two routings; fp32 reference; w2 in master atomic mode; T=8/16/32 only.
- cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix: superseded (phase C 2026-10-27 2215Z): implemented, but its data-derived table routes B128 -> master and B256 -> wgmma for both GEMMs, exactly the s5 bucket-gate twin (4e4a0e26 / 7be125e9, 1.3744x); paths (b)/(c) are never selected, and the base is the stale s5 commit (master 1.7187), so a check answers no new question. Implementation left unstaged in worktree .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-s6-per-kernel-bucket-mix (never built by nvcc).
- cuda-nvfp4-wgmma-rega-producer-scale-expand: adversarial review: This design has already been measured and refuted. muse_prefill_layer's experts-rega-producer-scale-expand-all-t (job 6be2a8d3, written as an explicit 'port of 
- cuda-fused-split-i-moe-tiny-t: adversarial review: Hold. The file is written against a master that no longer exists at T<=8. Today's master (86e4a4d9) runs Engine A v1/v4 there, with decode_1 at 35.0 us and veri
- wg-producer-direct-x-rows-no-gather: own condition: first measurement must be an okc check (new cp.async/mbarrier producer can fault or hang); only scored submits remained on 2026-10-25, and a gain on master cannot move the board (best is the occ2 branch). Unhold when checks reset; apply the reviewer's fence.proxy.async and nvalid-bound fixes when building
- wg2-pingpong-consumers-occ2-b256: killed line (X5 2026-10-27): wg2 re-split measured x0.405/0.882 (SINGLE-PASS s1 killed table), ping-pong consumer WGs x1.18-1.53 slower (prefill rega-pp, campaign K5 prior); the T<=8 incumbent is Engine A, not the wg consumer; adversary of cuda-fused-split-i-moe-tiny-t asked to close remaining wg-consumer items
- wg2-w13-occ2-nmax128-stg2-hot: killed line (X5 2026-10-27): wg2 hot-half re-split measured x0.405/0.882 on the judge (SINGLE-PASS s1); S1a's node build shows today's wg hot half already runs at the K4 K-step rate (S/wg x0.89-0.95), so occupancy/stage tweaks of wg2 have no headroom to find
- wg2-stream-k-tail-floor-g-over-r: adversarial review: Stale and superseded: the change edits wg2_kernel at OCC2 / G=264 on the restack-occ2-wg2-b256 base, but that base never landed (it is on FINDINGS' Do-not-try l
