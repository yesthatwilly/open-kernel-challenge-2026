# Hypotheses across problems

Regenerated 2026-10-24T173236Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.3949x | 18 | 4 | 5 | 6 | 3 | 0 | 0 | 0 | 4 |
| muse_bf16_proj | 1.1959x | 5 | 4 | 3 | 20 | 4 | 0 | 0 | 0 | 4 |
| muse_decode_layer | 1.1665x | 12 | 3 | 3 | 10 | 1 | 0 | 0 | 0 | 4 |
| muse_moe_w4a16 | 1.3076x | 8 | 4 | 6 | 10 | 4 | 1 | 0 | 0 | 9 |
| muse_prefill_layer | 1.0592x | 13 | 2 | 2 | 7 | 3 | 0 | 0 | 0 | 7 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0 | 4 | 4 | 0 | 0 | 4 | queue has 0 eligible hypotheses; 4 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 0 | 4 | 4 | 0 | 0 | 4 | queue has 0 eligible hypotheses; 4 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 0 | 4 | 3 | 2 | 1 | 2 |  |
| muse_moe_w4a16 | 0 | 9 | 4 | 1 | 0 | 6 | queue has 1 eligible hypotheses; 6 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 0 | 7 | 2 | 2 | 1 | 5 |  |
