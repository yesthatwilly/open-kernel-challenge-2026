# Hypotheses across problems

Regenerated 2026-10-24T024944Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0.0023x | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 10 |
| muse_bf16_proj | 0.9021x | 7 | 2 | 2 | 1 | 2 | 0 | 0 | 17 | 9 |
| muse_decode_layer | 0.0045x | 8 | 1 | 1 | 0 | 0 | 0 | 0 | 9 | 9 |
| muse_moe_w4a16 | 0.9888x | 7 | 2 | 1 | 0 | 0 | 0 | 0 | 18 | 10 |
| muse_prefill_layer | — | 9 | 1 | 0 | 0 | 0 | 0 | 0 | 10 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 10 | 10 | 0 | 0 | 0 | 20 | queue has 0 eligible hypotheses; 20 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 17 | 9 | 2 | 5 | 2 | 20 |  |
| muse_decode_layer | 9 | 9 | 1 | 7 | 1 | 10 |  |
| muse_moe_w4a16 | 18 | 10 | 2 | 4 | 2 | 23 |  |
| muse_prefill_layer | 10 | 10 | 1 | 0 | 0 | 19 | queue has 0 eligible hypotheses; 19 more could be absorbed today (run /hypothesize muse_prefill_layer) |
