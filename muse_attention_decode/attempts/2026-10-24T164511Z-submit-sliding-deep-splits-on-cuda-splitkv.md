# muse_attention_decode · 2026-10-24T164511Z · submit · sliding-deep-splits-on-cuda-splitkv

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T145005Z-sliding-deep-splits-on-cuda-splitkv.md` (branch `hyp/muse_attention_decode/sliding-deep-splits-on-cuda-splitkv`).

- **Job:** `698d4cd3838c4a63` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `54501c2`

## Goal
Stack sliding-fused-triton-decode-batched-deep-splits (1.309x on the pre-CUDA master; sliding decode_16 gained 33%, decode_32 +3%, every other shape flat) onto master full-decode-cuda-gather-splitkv (1.347x). The result is a clean 3-way merge (git merge-file, no conflicts): the fused sliding kernel's gate widens from batch == 1 to every batch at q_len 1, and _sliding_forward uses 16 splits for decode (length 1). The CUDA split-KV path (full layers, 1 < batch <= 64, q_len 1) and everything else are master's. Shapes are disjoint: the merge touches only sliding decode.

## What changed, and where
Built by churnsubs as a clean 3-way merge (`git merge-file`, no conflicts) of
hyp/muse_attention_decode/sliding-fused-triton-decode-batched-deep-splits onto master
(full-decode-cuda-gather-splitkv). The diff against master is exactly the parent's two sliding-only edits:
the fused sliding gate widens from batch == 1 to every batch at q_len 1, and `_sliding_forward` uses 16
splits for decode. The reviewer confirmed the merge. Runs as a scored submit (checks spent; the user's
standing answer allows reviewed experiments on scored runs).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/sliding-deep-splits-on-cuda-splitkv diff` (two config-only edits).

- [x] CUDA-graph capturable: both edits are on config values (window, q_len, length); no host syncs added.
- [x] Nothing derived from `data` is cached across calls.
- [x] Weights: n/a: no weights in this problem.
- [x] FP8 KV cache and FP8 E4M3 queries: unchanged (the fused kernel's existing precision path).
- [x] Attention sees every key: the fused kernel's splits cover the whole window, as in the parent run that passed 16/16.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] First line marker kept; no triton.autotune.

okc_lint: CLEAN.

## Risks / what could fail
low
Minted by churnsubs to restack a harvested per-shape win onto the new master; merged file at /home/user/.claude/jobs/7156a0da/tmp/merge/a_merged.py.

## Result (after the run)
- job `698d4cd3838c4a63` (submit), status done, passed True, tests 16/16
- speedup 1.3949x, 95% CI [1.3740, 1.4158], score 2700.100822688804 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3492076314709118, "ci": [1.328969516998848, 1.3729544014677721], "job": "de486093311d4bb9"}; base result: muse_attention_decode/hypotheses/done/gain/full-decode-cuda-gather-splitkv.result.json
- **outcome: gain** (gain)
- files: `muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json`, `muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.patch`; branch `hyp/muse_attention_decode/sliding-deep-splits-on-cuda-splitkv`
