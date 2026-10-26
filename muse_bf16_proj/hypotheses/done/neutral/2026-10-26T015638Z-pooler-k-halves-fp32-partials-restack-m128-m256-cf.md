---
problem: muse_bf16_proj
slug: pooler-k-halves-fp32-partials-restack-m128-m256-cf
created: 2026-10-26T015638Z
base: master
area: "pooler GEMM (m=128/256, K-split as a 2-batch cuBLAS GEMM)"
change: "Confirmation rerun of pooler-k-halves-fp32-partials-restack-m128-m256: its implementation commit 659ff4783582d8103e09cf89dd5fae13e57f96cf unchanged except one trailing comment, run as a check and pooled with job de95f81782374124 at harvest."
expected: []
risk: low
depends_on: pooler-k-halves-fp32-partials-restack-m128-m256
priority: 1
kind: confirm
controls_for: pooler-k-halves-fp32-partials-restack-m128-m256
confirm_of: pooler-k-halves-fp32-partials-restack-m128-m256
confirm_of_job: de95f81782374124
review: proceed
status: done
stage: harvested
claimed_at: 2026-10-26T020127Z
base_sha: 659ff4783582d8103e09cf89dd5fae13e57f96cf
base_blob: ff7580f8edf36c612459db87204b32d412842b5b
branch: hyp/muse_bf16_proj/pooler-k-halves-fp32-partials-restack-m128-m256-cf
worktree: .worktrees/muse_bf16_proj/pooler-k-halves-fp32-partials-restack-m128-m256-cf
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T020132Z
impl_commit: 549b453b11a5381b322b4554849388d4b2b37f2f
submission_sha256: f6937fdbd9b2ccd8716ed8c2ed23aea243a19a9aff097a53b7d8b9eb0d637216
norm_sha256: 478b2209429ec85fd18bb83e264144195eb06c7769c7a6c9c4e2999fb92989da
submission_blob: 4cfa21c6157c58d30290c826b6e7dc400aaa4829
lint_warnings: 36
attempt_file: 2026-10-26T020132Z-check-pooler-k-halves-fp32-partials-restack-m128-m256-cf.md
prepared_at: 2026-10-26T020132Z
job: caaa76cc288d47c7
dispatched_at: 2026-10-26T020142Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.258999728447511
speedup_ci: [1.2401147325207982, 1.2778847243742235]
harvested_at: 2026-10-26T023858Z
best_at_harvest: 1.2566457115954903
promotion: null
---

# pooler-k-halves-fp32-partials-restack-m128-m256-cf

## Change

Confirmation rerun of pooler-k-halves-fp32-partials-restack-m128-m256: its implementation commit 659ff4783582d8103e09cf89dd5fae13e57f96cf unchanged except one trailing comment, run as a check and pooled with job de95f81782374124 at harvest.

## Why

Queued by churn.py (harvest): `pooler-k-halves-fp32-partials-restack-m128-m256` ran 1.2683x against best 1.2566x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.

## Implementation notes

No code change: `pooler-k-halves-fp32-partials-restack-m128-m256` implementation commit `659ff4783582d8103e09cf89dd5fae13e57f96cf` plus one trailing comment, so okc times it again instead of returning the earlier job. Written by churn.py claim.

## Rules check

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

## Result

- job `caaa76cc288d47c7` (check), status done, passed True, tests 9/9
- speedup 1.2590x, 95% CI [1.2401, 1.2779], score 3837.014330381884 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256-cf.result.json`, `muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-fp32-partials-restack-m128-m256-cf.patch`; branch `hyp/muse_bf16_proj/pooler-k-halves-fp32-partials-restack-m128-m256-cf`
- confirm of `pooler-k-halves-fp32-partials-restack-m128-m256`: pooled 1.2683x (job de95f81782374124) and 1.2590x (job caaa76cc288d47c7) → 1.2636x; best 1.2566x, two-run margin 1.07% → neutral (unproven-vs-best)

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 10.92 | 11.12 | 0.9724 | flat |
| layer=o load=decode_1 m=1 | 62 | 6.11 | 6.09 | 0.969 | lost |
| layer=pooler load=decode_1 m=1 | 62 | 17.81 | 17.85 | 0.9909 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 4.96 | 5.26 | 0.9499 | lost |
| layer=router load=decode_1 m=1 | 62 | 4.87 | 5.26 | 0.9465 | lost |
| layer=qkv_gate load=decode_16 m=16 | 62 | 12.99 | 12.96 | 0.9069 | lost |
| layer=o load=decode_16 m=16 | 62 | 5.9 | 6.36 | 0.9705 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.52 | 19.39 | 0.9994 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.4 | 18.69 | 0.9974 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.5 | 5.7 | 0.9918 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.63 | 13.93 | 0.9568 | lost |
| layer=o load=decode_32 m=32 | 62 | 6.35 | 6.18 | 1.0405 | gained |
| layer=pooler load=decode_32 m=32 | 62 | 19.98 | 20.09 | 1.0076 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.72 | 18.34 | 1.0076 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.31 | 5.86 | 1.063 | gained |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.74 | 12.7 | 1.0009 | flat |
| layer=o load=verify_1 m=8 | 62 | 6.12 | 6.1 | 1.0297 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 19.27 | 18.86 | 1.0117 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 5.27 | 5.55 | 1.0181 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.49 | 5.26 | 1.0421 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 15.99 | 16.07 | 0.9637 | lost |
| layer=o load=verify_16 m=128 | 62 | 7.54 | 7.54 | 1.001 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 21.36 | 21.81 | 0.9792 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 18.54 | 18.87 | 0.985 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.54 | 7.59 | 0.9833 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.7 | 19.57 | 0.9923 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.23 | 9.66 | 0.9893 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 28.59 | 26.98 | 1.0405 | gained |
| layer=unpooler load=verify_32 m=256 | 62 | 24.54 | 24.55 | 0.9841 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.24 | 9.47 | 0.9826 | flat |

gained on o decode_32; router decode_32, verify_1; pooler verify_32; lost on o decode_1; unpooler decode_1; router decode_1; qkv_gate decode_16, decode_32, verify_16. Noise ±1.5%.

## Follow-ups

- pooled verdict for `pooler-k-halves-fp32-partials-restack-m128-m256`: neutral (unproven-vs-best); this file is the record of the rerun only.
- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: pooler-k-halves-fp32-partials-restack-m128-m256-cf`) that adds a further change in the same area is the only way to learn more.

