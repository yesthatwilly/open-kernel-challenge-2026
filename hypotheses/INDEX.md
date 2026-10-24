# Hypotheses across problems

Regenerated 2026-10-24T032801Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0.0023x | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 10 |
| muse_bf16_proj | 1.0307x | 6 | 4 | 2 | 1 | 2 | 0 | 0 | 15 | 9 |
| muse_decode_layer | 1.0200x | 7 | 2 | 1 | 0 | 0 | 0 | 0 | 9 | 9 |
| muse_moe_w4a16 | 0.9888x | 5 | 4 | 1 | 0 | 0 | 0 | 0 | 16 | 10 |
| muse_prefill_layer | — | 9 | 1 | 0 | 0 | 0 | 0 | 0 | 9 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 10 | 10 | 0 | 0 | 0 | 20 | queue has 0 eligible hypotheses; 20 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 15 | 9 | 4 | 3 | 2 | 20 |  |
| muse_decode_layer | 9 | 9 | 2 | 6 | 1 | 11 |  |
| muse_moe_w4a16 | 16 | 10 | 4 | 2 | 2 | 23 |  |
| muse_prefill_layer | 9 | 10 | 1 | 0 | 0 | 18 | queue has 0 eligible hypotheses; 18 more could be absorbed today (run /hypothesize muse_prefill_layer) |
