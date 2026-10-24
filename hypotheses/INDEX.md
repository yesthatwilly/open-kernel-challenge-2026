# Hypotheses across problems

Regenerated 2026-10-24T115933Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.2436x | 16 | 1 | 3 | 1 | 1 | 0 | 0 | 5 | 9 |
| muse_bf16_proj | 1.1433x | 6 | 3 | 3 | 11 | 4 | 0 | 0 | 3 | 8 |
| muse_decode_layer | 1.0635x | 9 | 1 | 2 | 6 | 0 | 0 | 0 | 2 | 8 |
| muse_moe_w4a16 | 1.1297x | 5 | 2 | 4 | 6 | 3 | 0 | 0 | 6 | 10 |
| muse_prefill_layer | 1.0039x | 8 | 1 | 1 | 4 | 3 | 0 | 0 | 1 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 5 | 9 | 1 | 8 | 2 | 4 |  |
| muse_bf16_proj | 3 | 8 | 3 | 0 | 0 | 10 | queue has 0 eligible hypotheses; 10 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 2 | 8 | 1 | 3 | 1 | 6 |  |
| muse_moe_w4a16 | 6 | 10 | 2 | 1 | 1 | 12 | queue has 1 eligible hypotheses; 12 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 1 | 10 | 1 | 1 | 1 | 9 |  |
