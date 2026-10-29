# Hypotheses across problems

Regenerated 2026-10-29T184217Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.6340x | 30 | 0 | 6 | 50 | 10 | 1 | 0 | 19 | 10 |
| muse_bf16_proj | 1.2794x | 9 | 0 | 3 | 50 | 20 | 0 | 0 | 19 | 10 |
| muse_decode_layer | 1.6904x | 18 | 0 | 4 | 41 | 6 | 2 | 0 | 18 | 9 |
| muse_moe_w4a16 | 1.9521x | 11 | 0 | 6 | 43 | 6 | 3 | 0 | 11 | 7 |
| muse_prefill_layer | 1.5362x | 18 | 0 | 6 | 55 | 7 | 0 | 0 | 16 | 9 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 19 | 10 | 0 | 0 | 0 | 29 | queue empty: only leads clearing the value bar |
| muse_bf16_proj | 19 | 10 | 0 | 0 | 0 | 29 | queue empty: only leads clearing the value bar |
| muse_decode_layer | 18 | 9 | 0 | 0 | 0 | 27 | queue empty: only leads clearing the value bar |
| muse_moe_w4a16 | 11 | 7 | 0 | 1 | 1 | 17 | queue has 1 eligible hypotheses |
| muse_prefill_layer | 16 | 9 | 0 | 0 | 0 | 25 | queue empty: only leads clearing the value bar |
