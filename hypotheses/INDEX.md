# Hypotheses across problems

Regenerated 2026-10-28T145237Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.6340x | 30 | 1 | 6 | 48 | 10 | 1 | 0 | 5 | 10 |
| muse_bf16_proj | 1.2794x | 9 | 0 | 3 | 49 | 20 | 0 | 0 | 8 | 9 |
| muse_decode_layer | 1.6551x | 18 | 0 | 4 | 41 | 6 | 2 | 0 | 5 | 4 |
| muse_moe_w4a16 | 1.8941x | 10 | 0 | 6 | 42 | 6 | 3 | 0 | 7 | 4 |
| muse_prefill_layer | 1.5008x | 17 | 1 | 5 | 52 | 7 | 0 | 0 | 15 | 9 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 5 | 10 | 1 | 8 | 1 | 7 |  |
| muse_bf16_proj | 8 | 9 | 0 | 4 | 1 | 13 |  |
| muse_decode_layer | 5 | 4 | 0 | 6 | 1 | 3 |  |
| muse_moe_w4a16 | 7 | 4 | 0 | 1 | 1 | 10 | queue has 1 eligible hypotheses; 10 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 15 | 9 | 1 | 10 | 2 | 14 |  |
