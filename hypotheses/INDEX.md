# Hypotheses across problems

Regenerated 2026-10-26T044456Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.5295x | 16 | 0 | 6 | 46 | 10 | 1 | 0 | 14 | 10 |
| muse_bf16_proj | 1.2566x | 1 | 0 | 3 | 45 | 19 | 0 | 0 | 12 | 10 |
| muse_decode_layer | 1.3142x | 18 | 0 | 4 | 41 | 6 | 2 | 0 | 19 | 10 |
| muse_moe_w4a16 | 1.4021x | 10 | 0 | 6 | 42 | 6 | 3 | 0 | 17 | 10 |
| muse_prefill_layer | 1.4677x | 5 | 1 | 4 | 45 | 7 | 0 | 0 | 16 | 10 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 14 | 10 | 0 | 4 | 1 | 20 |  |
| muse_bf16_proj | 12 | 10 | 0 | 1 | 1 | 21 | queue has 1 eligible hypotheses; 21 more could be absorbed today (run /hypothesize muse_bf16_proj) |
| muse_decode_layer | 19 | 10 | 0 | 6 | 1 | 23 |  |
| muse_moe_w4a16 | 17 | 10 | 0 | 5 | 1 | 22 |  |
| muse_prefill_layer | 16 | 10 | 1 | 4 | 2 | 22 |  |
