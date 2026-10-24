# Hypotheses across problems

Regenerated 2026-10-24T184810Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.3949x | 19 | 11 | 5 | 7 | 6 | 0 | 0 | 0 | 0 |
| muse_bf16_proj | 1.1959x | 5 | 1 | 3 | 21 | 7 | 0 | 0 | 0 | 0 |
| muse_decode_layer | 1.2283x | 21 | 0 | 4 | 12 | 1 | 1 | 0 | 10 | 0 |
| muse_moe_w4a16 | 1.3278x | 10 | 1 | 6 | 16 | 4 | 2 | 0 | 0 | 0 |
| muse_prefill_layer | 1.0592x | 13 | 1 | 2 | 8 | 4 | 0 | 0 | 10 | 4 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0 | 0 | 11 | 0 | 0 | 0 | 11 in flight (cap 8) |
| muse_bf16_proj | 0 | 0 | 1 | 0 | 0 | 0 |  |
| muse_decode_layer | 10 | 0 | 0 | 11 | 3 | 0 |  |
| muse_moe_w4a16 | 0 | 0 | 1 | 3 | 0 | 0 |  |
| muse_prefill_layer | 10 | 4 | 1 | 1 | 1 | 13 | queue has 1 eligible hypotheses; 13 more could be absorbed today (run /hypothesize muse_prefill_layer) |
