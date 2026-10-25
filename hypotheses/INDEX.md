# Hypotheses across problems

Regenerated 2026-10-25T040348Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.3949x | 25 | 3 | 5 | 21 | 9 | 0 | 0 | 12 | 10 |
| muse_bf16_proj | 1.2215x | 7 | 1 | 3 | 22 | 17 | 0 | 0 | 9 | 10 |
| muse_decode_layer | 1.2521x | 16 | 4 | 4 | 25 | 5 | 2 | 0 | 10 | 9 |
| muse_moe_w4a16 | 1.3278x | 15 | 1 | 6 | 26 | 5 | 2 | 0 | 10 | 10 |
| muse_prefill_layer | 1.0744x | 16 | 2 | 2 | 24 | 6 | 0 | 0 | 11 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 12 | 10 | 3 | 0 | 0 | 21 | queue has 0 eligible hypotheses; 21 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 9 | 10 | 1 | 0 | 0 | 19 | queue has 0 eligible hypotheses; 19 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 10 | 9 | 4 | 0 | 0 | 18 | queue has 0 eligible hypotheses; 18 more could be absorbed today (run /hypothesize muse_decode_layer) |
| muse_moe_w4a16 | 10 | 10 | 1 | 0 | 0 | 20 | queue has 0 eligible hypotheses; 20 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 11 | 10 | 2 | 1 | 1 | 19 | queue has 1 eligible hypotheses; 19 more could be absorbed today (run /hypothesize muse_prefill_layer) |
