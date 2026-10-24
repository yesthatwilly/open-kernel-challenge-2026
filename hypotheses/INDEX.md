# Hypotheses across problems

Regenerated 2026-10-24T141733Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.2931x | 18 | 2 | 3 | 3 | 3 | 0 | 0 | 1 | 9 |
| muse_bf16_proj | 1.1760x | 7 | 1 | 3 | 16 | 4 | 0 | 0 | 0 | 8 |
| muse_decode_layer | 1.0874x | 11 | 1 | 2 | 7 | 1 | 0 | 0 | 0 | 8 |
| muse_moe_w4a16 | 1.3117x | 7 | 1 | 6 | 8 | 3 | 0 | 0 | 3 | 10 |
| muse_prefill_layer | 1.0599x | 12 | 0 | 2 | 5 | 3 | 0 | 0 | 0 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1 | 9 | 2 | 2 | 2 | 7 |  |
| muse_bf16_proj | 0 | 8 | 1 | 3 | 0 | 4 |  |
| muse_decode_layer | 0 | 8 | 1 | 3 | 1 | 4 |  |
| muse_moe_w4a16 | 3 | 10 | 1 | 0 | 0 | 8 | queue has 0 eligible hypotheses; 8 more could be absorbed today (run /hypothesize muse_moe_w4a16) |
| muse_prefill_layer | 0 | 10 | 0 | 2 | 2 | 7 |  |
