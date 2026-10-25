# Hypotheses across problems

Regenerated 2026-10-25T043621Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.4322x | 28 | 3 | 5 | 24 | 9 | 0 | 0 | 11 | 10 |
| muse_bf16_proj | 1.2215x | 6 | 4 | 3 | 25 | 18 | 0 | 0 | 2 | 10 |
| muse_decode_layer | 1.2680x | 16 | 3 | 4 | 30 | 6 | 2 | 0 | 4 | 9 |
| muse_moe_w4a16 | 1.3439x | 18 | 2 | 6 | 27 | 5 | 3 | 0 | 7 | 9 |
| muse_prefill_layer | 1.0744x | 20 | 2 | 2 | 26 | 6 | 0 | 0 | 11 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 11 | 10 | 3 | 0 | 0 | 21 | queue has 0 eligible hypotheses; 21 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 2 | 10 | 4 | 0 | 0 | 11 | queue has 0 eligible hypotheses; 11 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 4 | 9 | 3 | 0 | 0 | 12 | queue has 0 eligible hypotheses; 12 more could be absorbed today (run /hypothesize muse_decode_layer) |
| muse_moe_w4a16 | 7 | 9 | 2 | 0 | 0 | 15 | queue has 0 eligible hypotheses; 15 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 11 | 10 | 2 | 1 | 1 | 20 | queue has 1 eligible hypotheses; 20 more could be absorbed today (run /hypothesize muse_prefill_layer) |
