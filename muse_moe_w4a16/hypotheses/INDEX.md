# muse_moe_w4a16 hypotheses

Regenerated 2026-10-24T024944Z. Best: 0.9888x [0.9739,1.0036] job `fb2cc736613747df` (check)

| created | slug | kind | area | where | outcome | speedup | CI | job | mode | promotion | review | expected hit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23T225501Z | triton-w13-silu-epilogue | primary | silu-mul | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225501Z | triton-w2-grouped-gemv-small-m | primary | w2 marlin | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225501Z | triton-w13-grouped-gemv-small-m | primary | w13 marlin | running | running |  |  | f31a03eb9bc6410f | check |  | proceed |  |
| 2026-10-23T225501Z | sglang-marlin-path-parity | primary | MoE align/dispatch | done/gain | gain | 0.9888 | [0.974,1.004] | fb2cc736613747df | check | merged-unsubmitted | proceed | 3/3 |
| 2026-10-23T225502Z | align-single-token-ep | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225502Z | marlin-workspace-blocks-per-sm-8 | primary | w13 marlin | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225502Z | triton-small-m-threshold-32 | primary | MoE align/dispatch | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225502Z | w13-marlin-atomic-add-small-m | primary | w13 marlin | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225502Z | w13-marlin-bf16-reduce-verify | primary | w13 marlin | queue | queued |  |  |  |  |  | required |  |
| 2026-10-23T225502Z | marlin-block-size-m-next-larger | primary | MoE align/dispatch | running | running |  |  | 2639e3ca7f5549de | check |  | proceed |  |
