# Hypotheses across problems

Regenerated 2026-10-25T112626Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.4920x | 29 | 1 | 6 | 32 | 10 | 1 | 0 | 0 | 7 |
| muse_bf16_proj | 1.2491x | 7 | 1 | 3 | 30 | 18 | 0 | 0 | 0 | 8 |
| muse_decode_layer | 1.2726x | 17 | 1 | 4 | 36 | 6 | 2 | 0 | 0 | 6 |
| muse_moe_w4a16 | 1.3439x | 21 | 1 | 6 | 32 | 6 | 3 | 0 | 1 | 9 |
| muse_prefill_layer | 1.1076x | 24 | 2 | 2 | 31 | 6 | 0 | 0 | 5 | 9 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0 | 7 | 1 | 0 | 0 | 7 | queue has 0 eligible hypotheses; 7 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 0 | 8 | 1 | 1 | 0 | 7 |  |
| muse_decode_layer | 0 | 6 | 1 | 1 | 0 | 5 |  |
| muse_moe_w4a16 | 1 | 9 | 1 | 3 | 1 | 6 |  |
| muse_prefill_layer | 5 | 9 | 2 | 4 | 1 | 9 |  |
