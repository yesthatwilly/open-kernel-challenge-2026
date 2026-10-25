# Hypotheses across problems

Regenerated 2026-10-25T131915Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.4825x | 29 | 0 | 6 | 33 | 10 | 1 | 0 | 0 | 7 |
| muse_bf16_proj | 1.2491x | 7 | 0 | 3 | 31 | 18 | 0 | 0 | 0 | 8 |
| muse_decode_layer | 1.2726x | 17 | 0 | 4 | 37 | 6 | 2 | 0 | 0 | 6 |
| muse_moe_w4a16 | 1.3439x | 22 | 0 | 6 | 34 | 6 | 3 | 0 | 0 | 9 |
| muse_prefill_layer | 1.4060x | 22 | 2 | 4 | 33 | 6 | 0 | 0 | 1 | 9 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0 | 7 | 0 | 1 | 0 | 6 |  |
| muse_bf16_proj | 0 | 8 | 0 | 1 | 0 | 7 |  |
| muse_decode_layer | 0 | 6 | 0 | 1 | 0 | 5 |  |
| muse_moe_w4a16 | 0 | 9 | 0 | 4 | 0 | 5 |  |
| muse_prefill_layer | 1 | 9 | 2 | 0 | 0 | 8 | queue has 0 eligible hypotheses; 8 more could be absorbed today (run /hypothesize muse_prefill_layer) |
