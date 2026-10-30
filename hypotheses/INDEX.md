# Hypotheses across problems

Regenerated 2026-10-30T000047Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.6340x | 30 | 0 | 6 | 53 | 10 | 1 | 0 | 20 | 10 |
| muse_bf16_proj | 1.2839x | 9 | 0 | 5 | 50 | 20 | 0 | 0 | 20 | 10 |
| muse_decode_layer | 1.7051x | 18 | 0 | 5 | 43 | 6 | 2 | 0 | 20 | 10 |
| muse_moe_w4a16 | 2.0201x | 12 | 0 | 8 | 44 | 6 | 3 | 0 | 20 | 10 |
| muse_prefill_layer | 1.5721x | 18 | 0 | 7 | 56 | 7 | 0 | 0 | 20 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 20 | 10 | 0 | 0 | 0 | 30 | queue empty: only leads clearing the value bar |
| muse_bf16_proj | 20 | 10 | 0 | 0 | 0 | 30 | queue empty: only leads clearing the value bar |
| muse_decode_layer | 20 | 10 | 0 | 0 | 0 | 30 | queue empty: only leads clearing the value bar |
| muse_moe_w4a16 | 20 | 10 | 0 | 1 | 1 | 29 | queue has 1 eligible hypotheses |
| muse_prefill_layer | 20 | 10 | 0 | 0 | 0 | 30 | queue empty: only leads clearing the value bar |
