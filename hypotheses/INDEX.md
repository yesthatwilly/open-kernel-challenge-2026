# Hypotheses across problems

Regenerated 2026-10-25T150116Z by `churn.py index`. Per-problem detail in `<problem>/hypotheses/INDEX.md`; learnings in `FINDINGS.md`.

| problem | best | queue | running | gain | neutral | regression | failed | rejected | checks left | scored left |
|---|---|---|---|---|---|---|---|---|---|---|
| muse_attention_decode | 1.5017x | 31 | 0 | 6 | 33 | 10 | 1 | 0 | 0 | 7 |
| muse_bf16_proj | 1.2491x | 9 | 0 | 3 | 31 | 18 | 0 | 0 | 0 | 8 |
| muse_decode_layer | 1.3018x | 19 | 0 | 4 | 37 | 6 | 2 | 0 | 0 | 6 |
| muse_moe_w4a16 | 1.3727x | 24 | 0 | 6 | 34 | 6 | 3 | 0 | 0 | 8 |
| muse_prefill_layer | 1.4458x | 22 | 0 | 4 | 35 | 6 | 0 | 0 | 0 | 8 |

## Today's budget

| problem | checks left | scored left | in flight | eligible queue | dispatch now | shortfall | note |
|---|---|---|---|---|---|---|---|
| muse_attention_decode | 0 | 7 | 0 | 3 | 1 | 4 |  |
| muse_bf16_proj | 0 | 8 | 0 | 3 | 2 | 5 |  |
| muse_decode_layer | 0 | 6 | 0 | 3 | 1 | 3 |  |
| muse_moe_w4a16 | 0 | 8 | 0 | 6 | 2 | 2 |  |
| muse_prefill_layer | 0 | 8 | 0 | 0 | 0 | 7 | queue has 0 eligible hypotheses; 7 more could be absorbed today (run /hypothesize muse_prefill_layer) |
