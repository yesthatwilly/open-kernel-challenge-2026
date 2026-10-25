# Hypotheses across problems

Regenerated 2026-10-25T034141Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.3949x | 26 | 1 | 5 | 21 | 9 | 0 | 0 | 14 | 10 |
| muse_bf16_proj | 1.2215x | 7 | 0 | 3 | 22 | 17 | 0 | 0 | 9 | 10 |
| muse_decode_layer | 1.2521x | 18 | 0 | 4 | 25 | 5 | 2 | 0 | 12 | 9 |
| muse_moe_w4a16 | 1.3278x | 15 | 0 | 6 | 26 | 5 | 2 | 0 | 10 | 10 |
| muse_prefill_layer | 1.0744x | 18 | 0 | 2 | 24 | 6 | 0 | 0 | 13 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 14 | 10 | 1 | 3 | 2 | 21 |  |
| muse_bf16_proj | 9 | 10 | 0 | 0 | 0 | 19 | queue has 0 eligible hypotheses; 19 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 12 | 9 | 0 | 2 | 2 | 19 |  |
| muse_moe_w4a16 | 10 | 10 | 0 | 0 | 0 | 20 | queue has 0 eligible hypotheses; 20 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 13 | 10 | 0 | 3 | 2 | 20 |  |
