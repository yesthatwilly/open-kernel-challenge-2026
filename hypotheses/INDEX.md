# Hypotheses across problems

Regenerated 2026-10-24T135615Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.2931x | 16 | 0 | 3 | 3 | 3 | 0 | 0 | 2 | 9 |
| muse_bf16_proj | 1.1433x | 4 | 0 | 3 | 16 | 4 | 0 | 0 | 1 | 8 |
| muse_decode_layer | 1.0874x | 8 | 0 | 2 | 7 | 1 | 0 | 0 | 1 | 8 |
| muse_moe_w4a16 | 1.3117x | 4 | 0 | 6 | 8 | 3 | 0 | 0 | 4 | 10 |
| muse_prefill_layer | 1.0599x | 8 | 0 | 2 | 5 | 3 | 0 | 0 | 0 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 2 | 9 | 0 | 0 | 0 | 11 | queue has 0 eligible hypotheses; 11 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 1 | 8 | 0 | 0 | 0 | 9 | queue has 0 eligible hypotheses; 9 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 1 | 8 | 0 | 0 | 0 | 9 | queue has 0 eligible hypotheses; 9 more could be absorbed today (run /hypothesize muse_decode_layer) |
| muse_moe_w4a16 | 4 | 10 | 0 | 0 | 0 | 9 | queue has 0 eligible hypotheses; 9 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 0 | 10 | 0 | 0 | 0 | 9 | queue has 0 eligible hypotheses; 9 more could be absorbed today (run /hypothesize muse_prefill_layer) |
