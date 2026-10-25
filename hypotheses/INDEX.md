# Hypotheses across problems

Regenerated 2026-10-25T084200Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.4441x | 29 | 7 | 5 | 27 | 9 | 0 | 0 | 8 | 9 |
| muse_bf16_proj | 1.2215x | 6 | 0 | 3 | 30 | 18 | 0 | 0 | 0 | 10 |
| muse_decode_layer | 1.2726x | 16 | 3 | 4 | 33 | 6 | 2 | 0 | 3 | 7 |
| muse_moe_w4a16 | 1.3439x | 21 | 0 | 6 | 31 | 6 | 3 | 0 | 3 | 9 |
| muse_prefill_layer | 1.0744x | 26 | 0 | 2 | 29 | 6 | 0 | 0 | 8 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 8 | 9 | 7 | 0 | 0 | 17 | queue has 0 eligible hypotheses; 17 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 0 | 10 | 0 | 0 | 0 | 10 | queue has 0 eligible hypotheses; 10 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 3 | 7 | 3 | 0 | 0 | 10 | queue has 0 eligible hypotheses; 10 more could be absorbed today (run /hypothesize muse_decode_layer) |
| muse_moe_w4a16 | 3 | 9 | 0 | 2 | 1 | 10 |  |
| muse_prefill_layer | 8 | 10 | 0 | 4 | 2 | 14 |  |
