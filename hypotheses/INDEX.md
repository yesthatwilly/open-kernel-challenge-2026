# Hypotheses across problems

Regenerated 2026-10-24T111615Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.1706x | 17 | 2 | 2 | 0 | 1 | 0 | 0 | 6 | 9 |
| muse_bf16_proj | 1.1433x | 5 | 4 | 3 | 9 | 4 | 0 | 0 | 4 | 8 |
| muse_decode_layer | 1.0635x | 10 | 1 | 2 | 5 | 0 | 0 | 0 | 3 | 8 |
| muse_moe_w4a16 | 1.1297x | 4 | 1 | 4 | 5 | 3 | 0 | 0 | 8 | 10 |
| muse_prefill_layer | 1.0039x | 8 | 0 | 1 | 4 | 3 | 0 | 0 | 2 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 6 | 9 | 2 | 10 | 2 | 4 |  |
| muse_bf16_proj | 4 | 8 | 4 | 1 | 1 | 10 |  |
| muse_decode_layer | 3 | 8 | 1 | 4 | 1 | 6 |  |
| muse_moe_w4a16 | 8 | 10 | 1 | 0 | 0 | 15 | queue has 0 eligible hypotheses; 15 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 2 | 10 | 0 | 1 | 1 | 11 | queue has 1 eligible hypotheses; 11 more could be absorbed today (run /hypothesize muse_prefill_layer) |
