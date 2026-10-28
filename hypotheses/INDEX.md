# Hypotheses across problems

Regenerated 2026-10-28T113415Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.6340x | 30 | 1 | 6 | 48 | 10 | 1 | 0 | 20 | 10 |
| muse_bf16_proj | 1.2740x | 9 | 0 | 3 | 49 | 20 | 0 | 0 | 20 | 10 |
| muse_decode_layer | 1.6428x | 18 | 0 | 4 | 41 | 6 | 2 | 0 | 13 | 6 |
| muse_moe_w4a16 | 1.8822x | 10 | 0 | 6 | 42 | 6 | 3 | 0 | 14 | 5 |
| muse_prefill_layer | 1.4897x | 17 | 1 | 5 | 52 | 7 | 0 | 0 | 20 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 20 | 10 | 1 | 10 | 1 | 20 |  |
| muse_bf16_proj | 20 | 10 | 0 | 4 | 1 | 26 |  |
| muse_decode_layer | 13 | 6 | 0 | 6 | 1 | 13 |  |
| muse_moe_w4a16 | 14 | 5 | 0 | 1 | 1 | 18 | queue has 1 eligible hypotheses; 18 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 20 | 10 | 1 | 10 | 2 | 20 |  |
