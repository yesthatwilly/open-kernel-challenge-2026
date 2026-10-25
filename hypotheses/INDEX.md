# Hypotheses across problems

Regenerated 2026-10-25T091116Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.4920x | 28 | 1 | 6 | 31 | 10 | 1 | 0 | 0 | 9 |
| muse_bf16_proj | 1.2215x | 6 | 0 | 3 | 30 | 18 | 0 | 0 | 0 | 10 |
| muse_decode_layer | 1.2726x | 16 | 1 | 4 | 35 | 6 | 2 | 0 | 0 | 7 |
| muse_moe_w4a16 | 1.3439x | 20 | 0 | 6 | 32 | 6 | 3 | 0 | 2 | 9 |
| muse_prefill_layer | 1.0744x | 24 | 0 | 2 | 31 | 6 | 0 | 0 | 6 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0 | 9 | 1 | 0 | 0 | 8 | queue has 0 eligible hypotheses; 8 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 0 | 10 | 0 | 0 | 0 | 10 | queue has 0 eligible hypotheses; 10 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 0 | 7 | 1 | 0 | 0 | 6 | queue has 0 eligible hypotheses; 6 more could be absorbed today (run /hypothesize muse_decode_layer) |
| muse_moe_w4a16 | 2 | 9 | 0 | 2 | 1 | 9 |  |
| muse_prefill_layer | 6 | 10 | 0 | 4 | 2 | 12 |  |
