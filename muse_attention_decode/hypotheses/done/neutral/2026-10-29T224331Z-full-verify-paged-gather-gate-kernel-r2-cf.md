---
problem: muse_attention_decode
slug: full-verify-paged-gather-gate-kernel-r2-cf
created: 2026-10-29T224331Z
base: master
area: "full-layer verify attention (FA3 call at verify_16 / verify_32)"
change: "Confirmation rerun of full-verify-paged-gather-gate-kernel-r2: its implementation commit 37649103a331b49c92b219f1c9d06caaf5d82fc9 unchanged except one trailing comment, run as a check and pooled with job 184991095dff4d50 at harvest."
expected: []
risk: low
depends_on: full-verify-paged-gather-gate-kernel-r2
priority: 1
kind: confirm
controls_for: full-verify-paged-gather-gate-kernel-r2
confirm_of: full-verify-paged-gather-gate-kernel-r2
confirm_of_job: 184991095dff4d50
review: proceed
status: done
stage: harvested
claimed_at: 2026-10-29T224342Z
base_sha: 37649103a331b49c92b219f1c9d06caaf5d82fc9
base_blob: 0f7d6195263717e5dff426814b77ed679a083408
branch: hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2-cf
worktree: .worktrees/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2-cf
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T224343Z
impl_commit: aaae17696203ec6225081ff36f595518bbcb9cef
submission_sha256: 0792aaa3f717e9b1f8f4f7a5c4061e2d66760d4e4390bce2ac8d54ad399d480f
norm_sha256: 139856ac1aefff5e19ea405807d72d6e1149eed43376726291b8bea4e188fa1b
submission_blob: 0039c56e69b9ce34892424ff02ed8cbd68224630
lint_warnings: 105
attempt_file: 2026-10-29T224344Z-check-full-verify-paged-gather-gate-kernel-r2-cf.md
prepared_at: 2026-10-29T224344Z
job: a47caf0724de40ad
dispatched_at: 2026-10-29T224359Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.6464170555542041
speedup_ci: [1.6217207997208911, 1.671113311387517]
harvested_at: 2026-10-29T225407Z
best_at_harvest: 1.6339575246416982
promotion: null
mechanism: null
---

# full-verify-paged-gather-gate-kernel-r2-cf

## Change

Confirmation rerun of full-verify-paged-gather-gate-kernel-r2: its implementation commit 37649103a331b49c92b219f1c9d06caaf5d82fc9 unchanged except one trailing comment, run as a check and pooled with job 184991095dff4d50 at harvest.

## Why

Queued by churn.py (harvest): `full-verify-paged-gather-gate-kernel-r2` ran 1.6422x against best 1.6340x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.

## Implementation notes

No code change: `full-verify-paged-gather-gate-kernel-r2` implementation commit `37649103a331b49c92b219f1c9d06caaf5d82fc9` plus one trailing comment, so okc times it again instead of returning the earlier job. Written by churn.py claim.

## Rules check

- [x] CUDA-graph capturable: `_fvg_forward` makes Triton + one ext launch on the current stream; no `.item()/.cpu()/synchronize()` in kernel_fn's call graph (lint: none); branches only on config (`_fvg_gate`) and shapes (`_fvg_shape_ok`); lens/page table read on device; capture verified in prepare by `_alt_check` and the race graphs.
- [x] Nothing derived from `data` is cached across calls: `_FVG_BUF` holds partials/counters sized from G and batch only; `_FVG_ROUTE` frozen at import from synthetic races.
- [x] Weights: n/a: attention problem has no weights; nothing repacked.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 (read as e4m3, untouched format); queries rounded to FP8 E4M3 (as FA3); P f16 and fp32 O/softmax (finer than FA3); output bf16.
- [x] Attention sees every key: stream-K covers every 64-key tile of every request; only causal / length masking on the last tiles.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import (own fp64 reference), subprocesses (a `threading.Thread` around load_inline, as master's builds) or network.
- [x] All work happens inside the timed call: pre-attention, attention, combine and epilogue all run per call; no output caching or replay.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode`):

```
warn  host-sync  ... (outside kernel_fn's call graph; fine in prepare())   [x ~105, master's prepare + [fvg] prepare lines]
warn  dtype      L7646: q = torch.full(..., dtype=torch.uint8 ...).view(FP8)   [master, pre-existing]
OK with warnings: .../muse_attention_decode/submission.py
```
`python3 -I -m py_compile`: OK (__pycache__ removed).

Copied from `full-verify-paged-gather-gate-kernel-r2`, whose code this is; the only change is a comment.

## Result

- job `a47caf0724de40ad` (check), status done, passed True, tests 16/16
- speedup 1.6464x, 95% CI [1.6217, 1.6711], score 2287.573484066112 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.6339575246416982, "ci": [1.6094481617720726, 1.6584668875113235], "job": "4e386647edd845f7"}; base result: muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel-r2.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel-r2-cf.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel-r2-cf.patch`; branch `hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2-cf`
- confirm of `full-verify-paged-gather-gate-kernel-r2`: pooled 1.6422x (job 184991095dff4d50) and 1.6464x (job a47caf0724de40ad) → 1.6443x; best 1.6340x, two-run margin 1.07% → neutral (unproven-vs-best)

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 11.26 | 10.81 | 1.0543 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 15.74 | 15.82 | 0.9996 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.44 | 15.41 | 0.995 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 103.45 | 104.59 | 0.992 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 16.94 | 17.47 | 0.9734 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 189.43 | 190.28 | 0.9978 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 13.46 | 13.11 | 1.0011 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.35 | 30.11 | 1.0018 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 19.21 | 19.4 | 0.9992 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 151.44 | 151.72 | 0.9968 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 27.83 | 27.64 | 0.9933 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 282.33 | 282.99 | 0.9981 | flat |

gained on sliding decode_1. Noise ±1.5%.

## Follow-ups

- pooled verdict for `full-verify-paged-gather-gate-kernel-r2`: neutral (unproven-vs-best); this file is the record of the rerun only.
- unproven-vs-best: faster than the best but inside the drift margin; churn queues one confirm rerun when the margin allows (CONFIRM_MIN), and pools the two runs.

