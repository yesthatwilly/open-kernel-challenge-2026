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
status: done
stage: harvested
claimed_at: 2026-10-29T212727Z
base_sha: 84995d9e32799d8c940b71c9d279bedaad12e6d1
base_blob: 5ef62a2078f6922e02a5c5c4e9aa9c7a1b3e31ab
branch: hyp/muse_bf16_proj/b12-batched-escape-m8-32-race-cf
worktree: .worktrees/muse_bf16_proj/b12-batched-escape-m8-32-race-cf
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T212737Z
impl_commit: 1a8b6edc3f59c32b01680754d13eafcfb94f3eff
submission_sha256: f5fa3671f2bbe9a6872de3d2af728cdea07bad7ae6ddb50b5b865d90464d3dd3
norm_sha256: 6a001bc2811e29072474f4cb3f5160bc7340a3492911325721a6db4baeaac7af
submission_blob: ededf8220bf604b63b4e8564beadcf3a6556ef8b
lint_warnings: 58
attempt_file: 2026-10-29T212738Z-check-b12-batched-escape-m8-32-race-cf.md
prepared_at: 2026-10-29T212738Z
job: c0f8969e76cf4cb7
dispatched_at: 2026-10-29T212752Z
outcome: gain
verdict_detail: gain
speedup: 1.3075893045880378
speedup_ci: [1.2879754650192172, 1.3272031441568581]
harvested_at: 2026-10-29T213647Z
best_at_harvest: 1.279360358306445
promotion: via-parent
mechanism: null
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

## Result

- job `c0f8969e76cf4cb7` (check), status done, passed True, tests 9/9
- speedup 1.3076x, 95% CI [1.2880, 1.3272], score 3694.432176104382 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.279360358306445, "ci": [1.2601699529318484, 1.2985507636810416], "job": "3e3ffc5afc57433b"}; base result: muse_bf16_proj/hypotheses/done/neutral/b12-batched-escape-m8-32-race.result.json
- **outcome: gain** (gain)
- files: `muse_bf16_proj/hypotheses/done/gain/b12-batched-escape-m8-32-race-cf.result.json`, `muse_bf16_proj/hypotheses/done/gain/b12-batched-escape-m8-32-race-cf.patch`; branch `hyp/muse_bf16_proj/b12-batched-escape-m8-32-race-cf`
- confirm of `b12-batched-escape-m8-32-race`: pooled 1.2871x (job bced57b42b87444d) and 1.3076x (job c0f8969e76cf4cb7) → 1.2974x; best 1.2794x, two-run margin 1.07% → gain (gain-pooled)

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 10.58 | 10.37 | 1.0094 | flat |
| layer=o load=decode_1 m=1 | 62 | 5.55 | 5.55 | 1.0152 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 14.97 | 14.77 | 0.9935 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 4.99 | 4.84 | 1.0357 | gained |
| layer=router load=decode_1 m=1 | 62 | 5.26 | 4.74 | 1.063 | gained |
| layer=qkv_gate load=decode_16 m=16 | 62 | 12.89 | 12.62 | 1.0237 | flat |
| layer=o load=decode_16 m=16 | 62 | 6.31 | 6.28 | 0.9942 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.58 | 19.25 | 1.014 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.71 | 18.32 | 1.0255 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.25 | 4.94 | 1.0535 | gained |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.38 | 13.53 | 1.0479 | gained |
| layer=o load=decode_32 m=32 | 62 | 6.7 | 6.33 | 1.0342 | gained |
| layer=pooler load=decode_32 m=32 | 62 | 20.21 | 19.79 | 1.0108 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.68 | 17.93 | 1.0314 | gained |
| layer=router load=decode_32 m=32 | 62 | 5.85 | 5.54 | 1.0471 | gained |
| layer=qkv_gate load=verify_1 m=8 | 62 | 11.68 | 11.45 | 1.0304 | gained |
| layer=o load=verify_1 m=8 | 62 | 6.23 | 5.9 | 0.9935 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 16.01 | 15.86 | 0.9981 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 5.61 | 5.32 | 1.0172 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.34 | 5.07 | 1.0422 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 15.74 | 15.61 | 0.9997 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.41 | 7.38 | 1.0211 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 22.34 | 22.1 | 0.9988 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 19.32 | 19.1 | 1.0128 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.57 | 6.79 | 1.0774 | gained |
| layer=qkv_gate load=verify_32 m=256 | 62 | 20.0 | 19.75 | 1.0141 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.66 | 9.58 | 1.0085 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 30.15 | 30.04 | 1.0203 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.22 | 24.39 | 1.0011 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.28 | 9.04 | 1.0066 | flat |

gained on unpooler decode_1, decode_32; router decode_1, decode_16, decode_32, verify_1, verify_16; qkv_gate decode_32, verify_1; o decode_32. Noise ±1.5%.

## Follow-ups

- pooled verdict for `b12-batched-escape-m8-32-race`: gain (gain-pooled); this file is the record of the rerun only.
- candidate for promotion; stack further changes with `depends_on` once it lands.

