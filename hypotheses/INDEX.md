# Hypotheses across problems

Regenerated 2026-10-24T165806Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.3492x | 18 | 5 | 4 | 4 | 3 | 0 | 0 | 0 | 5 |
| muse_bf16_proj | 1.1984x | 5 | 5 | 3 | 17 | 4 | 0 | 0 | 0 | 4 |
| muse_decode_layer | 1.1665x | 12 | 4 | 3 | 7 | 1 | 0 | 0 | 0 | 4 |
| muse_moe_w4a16 | 1.3076x | 7 | 3 | 6 | 9 | 3 | 1 | 0 | 0 | 9 |
| muse_prefill_layer | 1.0599x | 13 | 2 | 2 | 5 | 3 | 0 | 0 | 0 | 7 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0 | 5 | 5 | 0 | 0 | 5 | queue has 0 eligible hypotheses; 5 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 0 | 4 | 5 | 0 | 0 | 4 | queue has 0 eligible hypotheses; 4 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 0 | 4 | 4 | 1 | 1 | 3 |  |
| muse_moe_w4a16 | 0 | 9 | 3 | 0 | 0 | 6 | queue has 0 eligible hypotheses; 6 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 0 | 7 | 2 | 1 | 1 | 6 | queue has 1 eligible hypotheses; 6 more could be absorbed today (run /hypothesize muse_prefill_layer) |
