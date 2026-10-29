---
problem: muse_bf16_proj
slug: b12-batched-escape-m8-32-race-cf
created: 2026-10-29T212710Z
base: master
area: "bf16 dense GEMM cells at m=8..32 (pooler, qkv_gate, o): bulk12 kernels"
change: "Confirmation rerun of b12-batched-escape-m8-32-race: its implementation commit 84995d9e32799d8c940b71c9d279bedaad12e6d1 unchanged except one trailing comment, run as a check and pooled with job bced57b42b87444d at harvest."
expected: []
risk: low
depends_on: b12-batched-escape-m8-32-race
priority: 1
kind: confirm
controls_for: b12-batched-escape-m8-32-race
confirm_of: b12-batched-escape-m8-32-race
confirm_of_job: bced57b42b87444d
review: proceed
status: running
stage: implemented
claimed_at: 2026-10-29T212727Z
base_sha: 84995d9e32799d8c940b71c9d279bedaad12e6d1
base_blob: 5ef62a2078f6922e02a5c5c4e9aa9c7a1b3e31ab
branch: hyp/muse_bf16_proj/b12-batched-escape-m8-32-race-cf
worktree: .worktrees/muse_bf16_proj/b12-batched-escape-m8-32-race-cf
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-29T212737Z
impl_commit: 1a8b6edc3f59c32b01680754d13eafcfb94f3eff
submission_sha256: f5fa3671f2bbe9a6872de3d2af728cdea07bad7ae6ddb50b5b865d90464d3dd3
norm_sha256: 6a001bc2811e29072474f4cb3f5160bc7340a3492911325721a6db4baeaac7af
submission_blob: ededf8220bf604b63b4e8564beadcf3a6556ef8b
lint_warnings: 58
---

# b12-batched-escape-m8-32-race-cf

## Change

Confirmation rerun of b12-batched-escape-m8-32-race: its implementation commit 84995d9e32799d8c940b71c9d279bedaad12e6d1 unchanged except one trailing comment, run as a check and pooled with job bced57b42b87444d at harvest.

## Why

Queued by churn.py (harvest): `b12-batched-escape-m8-32-race` ran 1.2871x against best 1.2794x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.

## Implementation notes

No code change: `b12-batched-escape-m8-32-race` implementation commit `84995d9e32799d8c940b71c9d279bedaad12e6d1` plus one trailing comment, so okc times it again instead of returning the earlier job. Written by churn.py claim.

## Rules check

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

