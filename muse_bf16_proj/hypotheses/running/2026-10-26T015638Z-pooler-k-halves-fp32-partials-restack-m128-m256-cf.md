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
status: running
stage: dispatched
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

