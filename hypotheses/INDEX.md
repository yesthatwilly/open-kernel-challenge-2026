# Hypotheses across problems

Regenerated 2026-10-27T121142Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.6340x | 28 | 0 | 6 | 46 | 10 | 1 | 0 | 14 | 6 |
| muse_bf16_proj | 1.2735x | 5 | 0 | 3 | 46 | 20 | 0 | 0 | 16 | 9 |
| muse_decode_layer | 1.5697x | 18 | 0 | 4 | 41 | 6 | 2 | 0 | 5 | 2 |
| muse_moe_w4a16 | 1.7187x | 10 | 0 | 6 | 42 | 6 | 3 | 0 | 12 | 8 |
| muse_prefill_layer | 1.4889x | 14 | 1 | 5 | 48 | 7 | 0 | 0 | 17 | 9 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 14 | 6 | 0 | 12 | 1 | 8 |  |
| muse_bf16_proj | 16 | 9 | 0 | 2 | 1 | 23 | queue has 2 eligible hypotheses; 23 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 5 | 2 | 0 | 6 | 1 | 1 |  |
| muse_moe_w4a16 | 12 | 8 | 0 | 5 | 1 | 15 |  |
| muse_prefill_layer | 17 | 9 | 1 | 8 | 2 | 18 |  |
