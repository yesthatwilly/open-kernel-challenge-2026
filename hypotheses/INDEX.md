# Hypotheses across problems

Regenerated 2026-10-25T003348Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.3949x | 22 | 3 | 5 | 17 | 7 | 0 | 0 | 20 | 10 |
| muse_bf16_proj | 1.2215x | 5 | 1 | 3 | 21 | 8 | 0 | 0 | 18 | 10 |
| muse_decode_layer | 1.2521x | 21 | 4 | 4 | 20 | 2 | 2 | 0 | 18 | 9 |
| muse_moe_w4a16 | 1.3278x | 12 | 5 | 6 | 19 | 4 | 2 | 0 | 17 | 10 |
| muse_prefill_layer | 1.0744x | 20 | 3 | 2 | 20 | 4 | 0 | 0 | 18 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 20 | 10 | 3 | 0 | 0 | 30 | queue has 0 eligible hypotheses; 30 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 18 | 10 | 1 | 0 | 0 | 27 | queue has 0 eligible hypotheses; 27 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 18 | 9 | 4 | 5 | 2 | 21 |  |
| muse_moe_w4a16 | 17 | 10 | 5 | 0 | 0 | 26 | queue has 0 eligible hypotheses; 26 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 18 | 10 | 3 | 7 | 2 | 20 |  |
