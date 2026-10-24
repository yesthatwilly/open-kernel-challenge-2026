# Hypotheses across problems

Regenerated 2026-10-24T162228Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.3468x | 19 | 0 | 4 | 4 | 3 | 0 | 0 | 0 | 9 |
| muse_bf16_proj | 1.1984x | 7 | 0 | 3 | 17 | 4 | 0 | 0 | 0 | 8 |
| muse_decode_layer | 1.1635x | 11 | 0 | 3 | 7 | 1 | 0 | 0 | 0 | 8 |
| muse_moe_w4a16 | 1.3117x | 7 | 0 | 6 | 9 | 3 | 1 | 0 | 2 | 10 |
| muse_prefill_layer | 1.0599x | 12 | 0 | 2 | 5 | 3 | 0 | 0 | 0 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0 | 9 | 0 | 3 | 1 | 5 |  |
| muse_bf16_proj | 0 | 8 | 0 | 3 | 1 | 5 |  |
| muse_decode_layer | 0 | 8 | 0 | 3 | 2 | 4 |  |
| muse_moe_w4a16 | 2 | 10 | 0 | 0 | 0 | 7 | queue has 0 eligible hypotheses; 7 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 0 | 10 | 0 | 2 | 2 | 7 |  |
