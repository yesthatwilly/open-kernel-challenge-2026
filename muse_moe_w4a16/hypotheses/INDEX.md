# muse_moe_w4a16 hypotheses

Regenerated 2026-10-24T080316Z. Best: 0.9888x [0.9739,1.0036] job `fb2cc736613747df` (check)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T225501Z | triton-w13-silu-epilogue | primary | silu-mul | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225501Z | triton-w2-grouped-gemv-small-m | primary | w2 marlin | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225501Z | sglang-marlin-path-parity | primary | MoE align/dispatch | done/gain | gain | 0.9888 | [0.974,1.004] | fb2cc736613747df | check | merged-unsubmitted | proceed | 3/3 |
| 2026-10-23T225501Z | triton-w13-grouped-gemv-small-m | primary | w13 marlin | done/regression | regression | 0.7506 | [0.739,0.762] | f31a03eb9bc6410f | check |  | proceed | 2/4 |
| 2026-10-23T225502Z | triton-small-m-threshold-32 | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225502Z | align-single-token-ep | primary | MoE align/dispatch | done/neutral | neutral | 0.9951 | [0.980,1.010] | a7bb0cecaf8d4de1 | check |  | proceed | 3/3 |
| 2026-10-23T225502Z | marlin-workspace-blocks-per-sm-8 | primary | w13 marlin | done/neutral | neutral | 0.9921 | [0.977,1.007] | 75805dc2f18a4d06 | check |  | proceed | 2/4 |
| 2026-10-23T225502Z | w13-marlin-atomic-add-small-m | primary | w13 marlin | done/neutral | neutral | 0.9869 | [0.972,1.002] | c8ea8e21eea447f9 | check |  | proceed | 1/4 |
| 2026-10-23T225502Z | w13-marlin-bf16-reduce-verify | primary | w13 marlin | done/neutral | neutral | 0.9840 | [0.969,0.999] | 8d389f3d3e2d473f | check |  | proceed | 1/3 |
| 2026-10-23T225502Z | marlin-block-size-m-next-larger | primary | MoE align/dispatch | done/regression | regression | 0.9178 | [0.904,0.932] | 2639e3ca7f5549de | check |  | proceed | 3/5 |
| 2026-10-24T075126Z | post-expert-norm-triton-small-m | primary | post-expert norm+combine | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T075232Z | align-256-experts-below-128 | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T075232Z | marlin-block-size-m-next-smaller | primary | w13 marlin | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T075257Z | align-remote-sentinel-ids | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-24T075257Z | marlin-block-size-m-verify-two-smaller | primary | w13 marlin | queue | queued |  |  |  |  |  | required |  |
