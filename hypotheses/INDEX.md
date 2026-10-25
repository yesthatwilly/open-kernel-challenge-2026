# Hypotheses across problems

Regenerated 2026-10-25T010545Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.3949x | 28 | 4 | 5 | 17 | 7 | 0 | 0 | 16 | 10 |
| muse_bf16_proj | 1.2215x | 6 | 5 | 3 | 22 | 8 | 0 | 0 | 13 | 10 |
| muse_decode_layer | 1.2521x | 20 | 3 | 4 | 22 | 2 | 2 | 0 | 15 | 9 |
| muse_moe_w4a16 | 1.3278x | 12 | 3 | 6 | 20 | 5 | 2 | 0 | 13 | 10 |
| muse_prefill_layer | 1.0744x | 18 | 2 | 2 | 22 | 5 | 0 | 0 | 15 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 16 | 10 | 4 | 0 | 0 | 25 | queue has 0 eligible hypotheses; 25 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 13 | 10 | 5 | 1 | 1 | 21 | queue has 1 eligible hypotheses; 21 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 15 | 9 | 3 | 4 | 2 | 19 |  |
| muse_moe_w4a16 | 13 | 10 | 3 | 0 | 0 | 22 | queue has 0 eligible hypotheses; 22 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 15 | 10 | 2 | 6 | 2 | 18 |  |
