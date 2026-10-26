# muse_bf16_proj · 2026-10-26T020132Z · check · pooler-k-halves-fp32-partials-restack-m128-m256-cf

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-26T015638Z-pooler-k-halves-fp32-partials-restack-m128-m256-cf.md` (branch `hyp/muse_bf16_proj/pooler-k-halves-fp32-partials-restack-m128-m256-cf`).

- **Job:** `caaa76cc288d47c7` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `549b453b`

## Goal
Confirmation rerun of pooler-k-halves-fp32-partials-restack-m128-m256: its implementation commit 659ff4783582d8103e09cf89dd5fae13e57f96cf unchanged except one trailing comment, run as a check and pooled with job de95f81782374124 at harvest.

## What changed, and where
No code change: `pooler-k-halves-fp32-partials-restack-m128-m256` implementation commit `659ff4783582d8103e09cf89dd5fae13e57f96cf` plus one trailing comment, so okc times it again instead of returning the earlier job. Written by churn.py claim.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn branch is torch.bmm/torch.add/torch.empty on shapes and flags only; no .item()/.cpu()/sync/print/try (all syncs, prints and try blocks are in prepare()).
- [x] Nothing derived from `data` is cached: only the kind choice from synthetic prepare-time x, stored per state; kernel_fn reads x and cfg only.
- [x] Weights: no new weight bytes; reuses master's lossless bf16 NN repack pooler_t (view only). fp32 only in internal partials.
- [x] Activations BF16 where SGLang keeps BF16: x and the output are bf16; the output of both kinds is bf16 (b: one rounding from an fp32 add).
- [x] Attention: n/a, no attention in this problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds only torch ops, torch.profiler in prepare and prints.
- [x] All work happens inside the timed call; no output caching or replay (fresh output tensor per call).
- [x] Review required fixes applied (see Implementation notes).

okc_lint: `OK with warnings: muse_bf16_proj/submission.py` (only pre-existing-style host-sync/dtype warnings in prepare(); no errors). `python3 -I -m py_compile` passed; __pycache__ deleted.

Copied from `pooler-k-halves-fp32-partials-restack-m128-m256`, whose code this is; the only change is a comment.

## Risks / what could fail
low

## Result (after the run)
- job `caaa76cc288d47c7` (check), status done, passed True, tests 9/9
- speedup 1.2590x, 95% CI [1.2401, 1.2779], score 3837.014330381884 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256-cf.result.json`, `muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256-cf.patch`; branch `hyp/muse_bf16_proj/pooler-k-halves-fp32-partials-restack-m128-m256-cf`
- confirm of `pooler-k-halves-fp32-partials-restack-m128-m256`: pooled 1.2683x (job de95f81782374124) and 1.2590x (job caaa76cc288d47c7) → 1.2636x; best 1.2566x, two-run margin 1.07% → neutral (unproven-vs-best)
