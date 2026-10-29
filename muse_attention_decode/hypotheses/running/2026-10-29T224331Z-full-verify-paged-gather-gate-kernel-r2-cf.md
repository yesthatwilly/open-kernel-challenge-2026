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
status: running
stage: claimed
claimed_at: 2026-10-29T224342Z
base_sha: 37649103a331b49c92b219f1c9d06caaf5d82fc9
base_blob: 0f7d6195263717e5dff426814b77ed679a083408
branch: hyp/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2-cf
worktree: .worktrees/muse_attention_decode/full-verify-paged-gather-gate-kernel-r2-cf
mode: null
restack_of: null
stale_base: null
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

