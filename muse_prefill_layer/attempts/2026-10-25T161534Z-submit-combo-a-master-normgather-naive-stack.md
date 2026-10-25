# muse_prefill_layer · 2026-10-25T161534Z · submit · combo-a-master-normgather-naive-stack

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-25T161233Z-combo-a-master-normgather-naive-stack.md` (branch `hyp/muse_prefill_layer/combo-a-master-normgather-naive-stack`).

- **Job:** `9bfc488c36c84924` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `f2fdd734`

## Goal
COMBINATION A (naive top-row stack, restack of measured ON paths only). Start from the resolved 3-way merge /home/user/git/musespark/muse_prefill_layer/hypotheses/archive/pre-expert-norm-fused-rega-gather/master_plus_normgather.merged.py (master 020566aa / blob of muse_prefill_layer/submission.py with md5 82c368f4..., i.e. master 1.4587 contiguous-KV FA3 + rega first-turn + align2, plus pre-expert-norm-fused-rega-gather from hyp/muse_prefill_layer/pre-expert-norm-fused-rega-gather impl bd60e2db, scored edf68fa3131c4bf8 1.4600). Copy it over muse_prefill_layer/submission.py in the worktree and change nothing else. Verify the diff against master contains only the norm-gather hunks (the _norm_gather_setup block after _fa3c_setup, the fused _norm_gather_kernel Triton kernel, the prepare() call after _fa3c_setup, and the _experts_rega dispatch that uses the fused norm+gather when state flags say so); every master string (_REGA_*, _rega_setup, _fa3c_*, align2) must be byte-identical to master. SOURCES AND ROUTING: (1) master (all paths, unchanged). (2) pre-expert-norm-fused-rega-gather, bd60e2db, _norm_gather_kernel + _norm_gather_setup: replaces state['pre_expert_norm'](expert_input) followed by master's rega_gather with one fused RMSNorm*(1+w) + gather-to-sorted-slots Triton kernel, ONLY where rega is ON: T in (200,1024] low, (1024,4096] high, (4096,16384] ft; every other T and every non-rega path runs master's norm + gather. Its own prepare gate stays as built: self-check within 1 ulp with round_w=True on random/skewed routings per class, whole-expert-output rel L2 <= 1e-3, two-state replay rel L2 <= 1e-3, then cold-L2 graph timing fused vs pair at ON <= 0.9 (it measured 0.348 / 0.329 / 0.327). It must print one '[norm-gather] PATH CHOSEN' line covering the three classes. NOT included, and why (all from the prepare logs, not the row deltas): experts-rega-producer-scale-expand-all-t (6be2a8d3) printed PATH CHOSEN master in low/high/ft (sx/master 1.099 / 1.033 / 1.038), experts-rega-pingpong-consumers-all-t (5fbaec9a) printed PATH CHOSEN master in all three (1.471 / 1.179 / 1.239), experts-rega-low-class-two-ctas-per-sm (1a77da43) printed PATH CHOSEN low: master, qkv-gate-columns-side-stream (b8b8536c) was ON only at the sliding high class at 0.959 with a flat row. Their row 'wins' therefore measured master and are run-to-run noise (about +-3-4% per row; contiguous-KV also moved sliding first_turn +2.3% where it cannot act). PREPARE BUDGET: master ends its own setup at about 124.3-124.8 s (before-rega 76.9, build 38.8-39.3, checks 8 s; fa3c setup 1.1 s), norm-gather setup 0.7 s, total about 125.5 s of 300 s. REGA_FT_START_DEADLINE_S 150 s and REGA_COPY_DEADLINE_S 255 s are untouched. No nvcc module is added. RULES: unchanged from the two scored runs this file merges (both passed the judge); kernel_fn adds only a dict lookup on prepare-set bools keyed by the T class.

## What changed, and where
Built by the orchestrator (no builder agent): submission.py is the archived 3-way merge
muse_prefill_layer/hypotheses/archive/pre-expert-norm-fused-rega-gather/master_plus_normgather.merged.py
(master's contiguous-KV FA3 + pre-expert-norm-fused-rega-gather bd60e2db; both sides' added blocks kept,
prepare order _rega_setup -> _fa3c_setup -> _norm_gather_setup). The Opus adversarial review verified the
merge semantically (diff vs master = only norm-gather hunks; diff vs bd60e2db = only the four fa3c hunks;
deadlines intact). Both halves passed their own Opus audits and scored runs today (1cc92a43, edf68fa3), so
no separate audit was spent on the unchanged code.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: union of two audited diffs; kernel_fn adds only shape/flag-routed launches; no sync, item, cpu, print, value-sized allocation.
- [x] Nothing derived from `data` is cached across calls.
- [x] Weights: unchanged (bf16 checkpoint norm weight; NVFP4 experts untouched).
- [x] Activations BF16 as master; KV FP8; q E4M3 as master.
- [x] Attention sees every key (contiguous-KV gather is byte-exact and zero-fills past cache_seqlens).
- [x] No timing patches, harness access, module edits, reference import, subprocess or network.
- [x] All work inside the timed call; no output caching or replay; per-state buffers only.

okc_lint: OK with warnings (prepare-only host-sync/dtype notes); undef.py clean.

## Risks / what could fail
medium

## Result (after the run)
- job `9bfc488c36c84924` (submit), status done, passed True, tests 5/5
- speedup 1.4605x, 95% CI [1.4386, 1.4824], score 265256.0799646167 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4599652306075455, "ci": [1.4380657521484324, 1.4818647090666586], "job": "edf68fa3131c4bf8"}; base result: muse_prefill_layer/hypotheses/done/neutral/fa3-full-contiguous-kv-triton-gather-gated.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/combo-a-master-normgather-naive-stack.result.json`, `muse_prefill_layer/hypotheses/done/neutral/combo-a-master-normgather-naive-stack.patch`; branch `hyp/muse_prefill_layer/combo-a-master-normgather-naive-stack`
