# muse_attention_decode · 2026-10-24T184519Z · check · sliding-decode32-splits-8

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183855Z-sliding-decode32-splits-8.md` (branch `hyp/muse_attention_decode/sliding-decode32-splits-8`).

- **Job:** `16d3fb5cae3f4141` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `fda25ba`

## Goal
Config-only change in _sliding_forward: at length == 1 use splits = 16 if batch <= 16 else 8 (today 16 at every batch). At batch 32 that is 512 programs of 256 old keys (4 BLOCK_N tiles) instead of 1024 programs of 128 keys, and half the partials and ticket merges. The workspace sizing and counters follow `splits` as now; nothing else changes.

## What changed, and where
In `_sliding_forward` (worktree muse_attention_decode/submission.py, line 355) the decode split count is now `splits = (16 if batch <= 16 else 8) if length == 1 else _num_splits(batch, hkv)`, gated on cfg batch, with B<=16 kept at 16 as the review asks. The comment above it was updated. No deviation from the spec.

- `ws_acc`, `ws_ml` and the launch grid `(batch, hkv, splits)` use `splits` directly, so at B=32 they are sized for S=8.
- `S=splits` is the constexpr passed to the kernel.
- Counters are keyed on (batch, hkv), not S.
- There is no `prepare()` and no warm-up in this file that depends on S, so none needed changing.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the diff is one scalar chosen from cfg; no syncs, no value-dependent control flow.
- [x] Nothing derived from `data` is cached across calls: the diff adds no state.
- [x] Weights: n/a, no weight change in the diff.
- [x] Activations BF16, KV cache FP8, queries at most FP8: unchanged.
- [x] Attention sees every key: S=8 chunk = 256, and 8*256 >= 2047, so the whole window is covered; no skipping.
- [x] No timing patches, harness access, module edits, subprocesses or network: the diff is a split-count line only.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: `CLEAN: submission.py`. py_compile passed and __pycache__ was removed.

## Risks / what could fail
low
Touches master's _sliding_fused_kernel, which sliding-cuda-cluster-dsmem-onelaunch (in the judge now) replaces on every sliding shape at tp8/page 1; if that rewrite lands this is moot and should be held. Useful only as a tuning of the incumbent if the rewrite falls back. Judge on the _sliding_fused_kernel row, not the headline (sliding decode_1 drifted 2-3.5% between run batches).

## Result (after the run)
- job `16d3fb5cae3f4141` (check), status done, passed True, tests 16/16
- speedup 1.3973x, 95% CI [1.3763, 1.4182], score 2695.5036271878203 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-decode32-splits-8.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-decode32-splits-8.patch`; branch `hyp/muse_attention_decode/sliding-decode32-splits-8`
