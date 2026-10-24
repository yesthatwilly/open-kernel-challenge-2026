# Hypotheses across problems

Regenerated 2026-10-24T021809Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0.0023x | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 10 |
| muse_bf16_proj | 0.9021x | 7 | 0 | 4 | 0 | 1 | 0 | 0 | 19 | 10 |
| muse_decode_layer | 0.0045x | 9 | 0 | 1 | 0 | 0 | 0 | 0 | 10 | 10 |
| muse_moe_w4a16 | 0.0061x | 9 | 0 | 1 | 0 | 0 | 0 | 0 | 20 | 10 |
| muse_prefill_layer | — | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 10 | 10 | 0 | 0 | 0 | 20 | queue has 0 eligible hypotheses; 20 more could be absorbed today (run /hypothesize muse_attention_decode) |
| muse_bf16_proj | 19 | 10 | 0 | 7 | 2 | 19 |  |
| muse_decode_layer | 10 | 10 | 0 | 8 | 1 | 11 |  |
| muse_moe_w4a16 | 20 | 10 | 0 | 6 | 2 | 24 |  |
| muse_prefill_layer | 10 | 10 | 0 | 1 | 1 | 19 |  |
