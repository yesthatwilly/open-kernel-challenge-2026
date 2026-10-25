# Hypotheses across problems

Regenerated 2026-10-25T013723Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.3949x | 28 | 2 | 5 | 19 | 7 | 0 | 0 | 16 | 10 |
| muse_bf16_proj | 1.2215x | 7 | 9 | 3 | 22 | 8 | 0 | 0 | 9 | 10 |
| muse_decode_layer | 1.2521x | 18 | 3 | 4 | 22 | 4 | 2 | 0 | 13 | 9 |
| muse_moe_w4a16 | 1.3278x | 15 | 3 | 6 | 23 | 5 | 2 | 0 | 11 | 10 |
| muse_prefill_layer | 1.0744x | 18 | 3 | 2 | 22 | 5 | 0 | 0 | 13 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 16 | 10 | 2 | 2 | 2 | 23 |  |
| muse_bf16_proj | 9 | 10 | 9 | 0 | 0 | 18 | queue has 0 eligible hypotheses; 18 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 13 | 9 | 3 | 2 | 1 | 19 |  |
| muse_moe_w4a16 | 11 | 10 | 3 | 0 | 0 | 20 | queue has 0 eligible hypotheses; 20 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 13 | 10 | 3 | 5 | 2 | 17 |  |
