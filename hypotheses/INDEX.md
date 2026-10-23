# Hypotheses across problems

Regenerated 2026-10-23T225814Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0.0023x | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 18 | 10 |
| muse_bf16_proj | 0.9021x | 10 | 0 | 1 | 0 | 0 | 0 | 0 | 17 | 10 |
| muse_decode_layer | 0.0045x | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 10 |
| muse_moe_w4a16 | 0.0061x | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 10 |
| muse_prefill_layer | — | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 18 | 10 | 0 | 0 | 0 | 28 | queue has 0 eligible hypotheses; 28 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 17 | 10 | 0 | 6 | 4 | 21 |  |
| muse_decode_layer | 19 | 10 | 0 | 1 | 1 | 28 | queue has 1 eligible hypotheses; 28 more could be absorbed today (run /hypothesize muse_decode_layer) |
| muse_moe_w4a16 | 19 | 10 | 0 | 1 | 1 | 28 | queue has 1 eligible hypotheses; 28 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 19 | 10 | 0 | 0 | 0 | 29 | queue has 0 eligible hypotheses; 29 more could be absorbed today (run /hypothesize muse_prefill_layer) |
