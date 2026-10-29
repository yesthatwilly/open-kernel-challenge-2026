# muse_bf16_proj · 2026-10-29T212738Z · check · b12-batched-escape-m8-32-race-cf

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-29T212710Z-b12-batched-escape-m8-32-race-cf.md` (branch `hyp/muse_bf16_proj/b12-batched-escape-m8-32-race-cf`).

- **Job:** `c0f8969e76cf4cb7` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `1a8b6edc`

## Goal
Confirmation rerun of b12-batched-escape-m8-32-race: its implementation commit 84995d9e32799d8c940b71c9d279bedaad12e6d1 unchanged except one trailing comment, run as a check and pooled with job bced57b42b87444d at harvest.

## What changed, and where
No code change: `b12-batched-escape-m8-32-race` implementation commit `84995d9e32799d8c940b71c9d279bedaad12e6d1` plus one trailing comment, so okc times it again instead of returning the earlier job. Written by churn.py claim.

## Rules check (each must be true before the run)
Each item was checked against `git -C .worktrees/muse_bf16_proj/b12-batched-escape-m8-32-race diff`.
- [x] CUDA-graph capturable: kernel_fn adds only `_b12m_route` (cfg, shapes, dtype, contiguity, `data_ptr() % 16` and state flags) and `_b12m_forward` (`torch.empty` plus one extension launch whose host-side tensor-map encode is not a stream op). Every `.item()` and synchronize sits in prepare-only helpers (`_b12m_rows`, `_b12m_prepare`). The graph capture and replay were validated on the H100.
- [x] Nothing derived from `data` is cached: the tensor map is a stack-local built on each call from that call's x, and `_B12M_SUM` and `_B12M_BUILD` hold only prepare text and build state.
- [x] Weights: a lossless 12-bit codec (section (10)'s `_b12_encode`, host + device round trip bit-exact), with escapes holding the true bf16 bits. Accumulation is fp32, followed by one bf16 rounding.
- [x] Activations stay BF16 (x is read as bf16, the output is bf16). The KV-cache / query item is n/a: this is not attention.
- [x] Attention full window: n/a, not attention.
- [x] No timing patches, harness frames or I/O, module edits, reference import, subprocesses or network: `load_inline` only (as in master), and the reference is imported only by the node driver outside the submission.
- [x] All work happens inside the timed call: one kernel per m=8 call, with no output caching or replay.
- [x] The first line `# okc: problem=muse_bf16_proj` is kept. py_compile is OK and __pycache__ was removed.

okc_lint: `OK with warnings: .../muse_bf16_proj/submission.py`. All 58 warnings are host-sync/dtype items "outside kernel_fn's call graph; fine in prepare()". There are no errors.

Copied from `b12-batched-escape-m8-32-race`, whose code this is; the only change is a comment.

## Risks / what could fail
low

## Result (after the run)
- job `c0f8969e76cf4cb7` (check), status done, passed True, tests 9/9
- speedup 1.3076x, 95% CI [1.2880, 1.3272], score 3694.432176104382 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.279360358306445, "ci": [1.2601699529318484, 1.2985507636810416], "job": "3e3ffc5afc57433b"}; base result: muse_bf16_proj/hypotheses/done/neutral/b12-batched-escape-m8-32-race.result.json
- **outcome: gain** (gain)
- files: `muse_bf16_proj/hypotheses/done/gain/b12-batched-escape-m8-32-race-cf.result.json`, `muse_bf16_proj/hypotheses/done/gain/b12-batched-escape-m8-32-race-cf.patch`; branch `hyp/muse_bf16_proj/b12-batched-escape-m8-32-race-cf`
- confirm of `b12-batched-escape-m8-32-race`: pooled 1.2871x (job bced57b42b87444d) and 1.3076x (job c0f8969e76cf4cb7) → 1.2974x; best 1.2794x, two-run margin 1.07% → gain (gain-pooled)
