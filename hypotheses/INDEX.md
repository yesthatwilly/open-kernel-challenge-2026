# Hypotheses across problems

Regenerated 2026-10-25T000557Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.3949x | 19 | 0 | 5 | 17 | 7 | 0 | 0 | 20 | 10 |
| muse_bf16_proj | 1.1959x | 5 | 1 | 3 | 21 | 7 | 0 | 0 | 20 | 10 |
| muse_decode_layer | 1.2283x | 21 | 1 | 4 | 20 | 2 | 2 | 0 | 20 | 9 |
| muse_moe_w4a16 | 1.3278x | 10 | 2 | 6 | 17 | 4 | 2 | 0 | 19 | 10 |
| muse_prefill_layer | 1.0705x | 19 | 1 | 2 | 19 | 4 | 0 | 0 | 19 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 20 | 10 | 0 | 0 | 0 | 30 | queue has 0 eligible hypotheses; 30 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 20 | 10 | 1 | 0 | 0 | 30 | queue has 0 eligible hypotheses; 30 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 20 | 9 | 1 | 6 | 2 | 23 |  |
| muse_moe_w4a16 | 19 | 10 | 2 | 3 | 2 | 25 |  |
| muse_prefill_layer | 19 | 10 | 1 | 8 | 2 | 20 |  |
