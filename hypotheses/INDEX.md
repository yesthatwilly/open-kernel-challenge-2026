# Hypotheses across problems

Regenerated 2026-10-27T212819Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.6340x | 31 | 0 | 6 | 48 | 10 | 1 | 0 | 12 | 5 |
| muse_bf16_proj | 1.2740x | 9 | 0 | 3 | 49 | 20 | 0 | 0 | 13 | 8 |
| muse_decode_layer | 1.5920x | 18 | 0 | 4 | 41 | 6 | 2 | 0 | 0 | 0 |
| muse_moe_w4a16 | 1.7187x | 10 | 0 | 6 | 42 | 6 | 3 | 0 | 11 | 8 |
| muse_prefill_layer | 1.4897x | 17 | 1 | 5 | 52 | 7 | 0 | 0 | 13 | 8 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 12 | 5 | 0 | 11 | 1 | 6 |  |
| muse_bf16_proj | 13 | 8 | 0 | 4 | 1 | 17 | queue has 4 eligible hypotheses; 17 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 0 | 0 | 0 | 6 | 0 | 0 |  |
| muse_moe_w4a16 | 11 | 8 | 0 | 5 | 2 | 14 |  |
| muse_prefill_layer | 13 | 8 | 1 | 10 | 2 | 11 |  |
