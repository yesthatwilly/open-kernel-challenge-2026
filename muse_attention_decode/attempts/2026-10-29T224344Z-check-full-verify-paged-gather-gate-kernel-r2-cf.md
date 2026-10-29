# muse_attention_decode · 2026-10-29T224344Z · check · full-verify-paged-gather-gate-kernel-r2-cf

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-29T224331Z-full-verify-paged-gather-gate-kernel-r2-cf.md` (branch `hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2-cf`).

- **Job:** `a47caf0724de40ad` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `aaae1769`

## Goal
Confirmation rerun of full-verify-paged-gather-gate-kernel-r2: its implementation commit 37649103a331b49c92b219f1c9d06caaf5d82fc9 unchanged except one trailing comment, run as a check and pooled with job 184991095dff4d50 at harvest.

## What changed, and where
No code change: `full-verify-paged-gather-gate-kernel-r2` implementation commit `37649103a331b49c92b219f1c9d06caaf5d82fc9` plus one trailing comment, so okc times it again instead of returning the earlier job. Written by churn.py claim.

## Rules check (each must be true before the run)
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

## Risks / what could fail
low

## Result (after the run)
- job `a47caf0724de40ad` (check), status done, passed True, tests 16/16
- speedup 1.6464x, 95% CI [1.6217, 1.6711], score 2287.573484066112 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.6339575246416982, "ci": [1.6094481617720726, 1.6584668875113235], "job": "4e386647edd845f7"}; base result: muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel-r2.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel-r2-cf.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-verify-paged-gather-gate-kernel-r2-cf.patch`; branch `hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2-cf`
- confirm of `full-verify-paged-gather-gate-kernel-r2`: pooled 1.6422x (job 184991095dff4d50) and 1.6464x (job a47caf0724de40ad) → 1.6443x; best 1.6340x, two-run margin 1.07% → neutral (unproven-vs-best)
