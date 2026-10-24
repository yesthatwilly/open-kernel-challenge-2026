# Hypotheses across problems

Regenerated 2026-10-24T104514Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.1706x | 17 | 2 | 2 | 0 | 1 | 0 | 0 | 6 | 9 |
| muse_bf16_proj | 1.1433x | 5 | 5 | 3 | 9 | 3 | 0 | 0 | 5 | 8 |
| muse_decode_layer | 1.0635x | 10 | 1 | 2 | 5 | 0 | 0 | 0 | 3 | 8 |
| muse_moe_w4a16 | 0.9888x | 4 | 4 | 1 | 5 | 3 | 0 | 0 | 8 | 10 |
| muse_prefill_layer | 1.0039x | 8 | 1 | 1 | 3 | 3 | 0 | 0 | 2 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 6 | 9 | 2 | 10 | 2 | 4 |  |
| muse_bf16_proj | 5 | 8 | 5 | 1 | 1 | 11 |  |
| muse_decode_layer | 3 | 8 | 1 | 4 | 1 | 6 |  |
| muse_moe_w4a16 | 8 | 10 | 4 | 0 | 0 | 17 | queue has 0 eligible hypotheses; 17 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 2 | 10 | 1 | 1 | 1 | 10 |  |
