# Hypotheses across problems

Regenerated 2026-10-24T063614Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0.0023x | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 10 |
| muse_bf16_proj | 1.1510x | 4 | 2 | 3 | 3 | 3 | 0 | 0 | 13 | 9 |
| muse_decode_layer | 1.0200x | 5 | 2 | 1 | 2 | 0 | 0 | 0 | 6 | 9 |
| muse_moe_w4a16 | 0.9888x | 3 | 0 | 1 | 4 | 2 | 0 | 0 | 14 | 10 |
| muse_prefill_layer | 1.0039x | 7 | 1 | 1 | 1 | 0 | 0 | 0 | 7 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 10 | 10 | 0 | 0 | 0 | 20 | queue has 0 eligible hypotheses; 20 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 13 | 9 | 2 | 2 | 2 | 19 |  |
| muse_decode_layer | 6 | 9 | 2 | 3 | 1 | 11 |  |
| muse_moe_w4a16 | 14 | 10 | 0 | 0 | 0 | 24 | queue has 0 eligible hypotheses; 24 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 7 | 10 | 1 | 3 | 1 | 13 |  |
