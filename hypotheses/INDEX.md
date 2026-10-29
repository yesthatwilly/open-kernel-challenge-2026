# Hypotheses across problems

Regenerated 2026-10-29T154921Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.6340x | 31 | 0 | 6 | 49 | 10 | 1 | 0 | 20 | 10 |
| muse_bf16_proj | 1.2794x | 9 | 0 | 3 | 50 | 20 | 0 | 0 | 19 | 10 |
| muse_decode_layer | 1.6904x | 18 | 0 | 4 | 41 | 6 | 2 | 0 | 18 | 9 |
| muse_moe_w4a16 | 1.9521x | 10 | 1 | 6 | 42 | 6 | 3 | 0 | 12 | 7 |
| muse_prefill_layer | 1.5008x | 18 | 1 | 5 | 53 | 7 | 0 | 0 | 19 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 20 | 10 | 0 | 1 | 1 | 29 | queue has 1 eligible hypotheses |
| muse_bf16_proj | 19 | 10 | 0 | 0 | 0 | 29 | queue empty: only leads clearing the value bar |
| muse_decode_layer | 18 | 9 | 0 | 0 | 0 | 27 | queue empty: only leads clearing the value bar |
| muse_moe_w4a16 | 12 | 7 | 1 | 0 | 0 | 19 | queue empty: only leads clearing the value bar |
| muse_prefill_layer | 19 | 10 | 1 | 0 | 0 | 29 | queue empty: only leads clearing the value bar |
