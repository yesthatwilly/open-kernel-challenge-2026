# Hypotheses across problems

Regenerated 2026-10-24T103713Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.1706x | 18 | 0 | 2 | 0 | 1 | 0 | 0 | 8 | 9 |
| muse_bf16_proj | 1.1433x | 8 | 1 | 3 | 9 | 3 | 0 | 0 | 8 | 8 |
| muse_decode_layer | 1.0635x | 10 | 0 | 2 | 5 | 0 | 0 | 0 | 4 | 8 |
| muse_moe_w4a16 | 0.9888x | 5 | 3 | 1 | 5 | 3 | 0 | 0 | 9 | 10 |
| muse_prefill_layer | 1.0039x | 8 | 1 | 1 | 3 | 3 | 0 | 0 | 2 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 8 | 9 | 0 | 11 | 1 | 5 |  |
| muse_bf16_proj | 8 | 8 | 1 | 3 | 2 | 12 |  |
| muse_decode_layer | 4 | 8 | 0 | 4 | 1 | 8 |  |
| muse_moe_w4a16 | 9 | 10 | 3 | 1 | 1 | 17 | queue has 1 eligible hypotheses; 17 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 2 | 10 | 1 | 2 | 1 | 9 |  |
