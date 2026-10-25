# Hypotheses across problems

Regenerated 2026-10-25T051707Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.4322x | 28 | 3 | 5 | 24 | 9 | 0 | 0 | 8 | 9 |
| muse_bf16_proj | 1.2215x | 6 | 4 | 3 | 25 | 18 | 0 | 0 | 1 | 10 |
| muse_decode_layer | 1.2856x | 16 | 3 | 4 | 30 | 6 | 2 | 0 | 3 | 7 |
| muse_moe_w4a16 | 1.3439x | 18 | 3 | 6 | 27 | 5 | 3 | 0 | 6 | 9 |
| muse_prefill_layer | 1.0744x | 19 | 3 | 2 | 26 | 6 | 0 | 0 | 8 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 8 | 9 | 3 | 0 | 0 | 16 | queue has 0 eligible hypotheses; 16 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 1 | 10 | 4 | 0 | 0 | 10 | queue has 0 eligible hypotheses; 10 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 3 | 7 | 3 | 0 | 0 | 9 | queue has 0 eligible hypotheses; 9 more could be absorbed today (run /hypothesize muse_decode_layer) |
| muse_moe_w4a16 | 6 | 9 | 3 | 0 | 0 | 14 | queue has 0 eligible hypotheses; 14 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 8 | 10 | 3 | 0 | 0 | 17 | queue has 0 eligible hypotheses; 17 more could be absorbed today (run /hypothesize muse_prefill_layer) |
