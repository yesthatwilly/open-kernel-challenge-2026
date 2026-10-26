---
problem: muse_prefill_layer
slug: experts-rega-s4-bm256-wide-tile-high-ft-cf
created: 2026-10-26T023858Z
base: master
area: "w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft classes): 256-weight-row CTA tile with in-warpgroup dequant/MMA overlap"
change: "Confirmation rerun of experts-rega-s4-bm256-wide-tile-high-ft: its implementation commit 97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2 unchanged except one trailing comment, run as a check and pooled with job a6f6063051f14750 at harvest."
expected: []
risk: low
depends_on: experts-rega-s4-bm256-wide-tile-high-ft
priority: 1
kind: confirm
controls_for: experts-rega-s4-bm256-wide-tile-high-ft
confirm_of: experts-rega-s4-bm256-wide-tile-high-ft
confirm_of_job: a6f6063051f14750
review: proceed
status: done
stage: harvested
claimed_at: 2026-10-26T214345Z
base_sha: 97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2
base_blob: 82449fd7e7fae52ba13d26109cc48a9fd256b40c
branch: hyp/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft-cf
worktree: .worktrees/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft-cf
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-26T214402Z
impl_commit: 912b3ea2ef4cfe5a47f0904e5d9a5ac38a8c92ea
submission_sha256: 263f2e32390c52ac7c285f0b126d42ab3f13bacc3328fc182f1ab39ba25e52ba
norm_sha256: 0c499f6341769f494a929dd9b5fdb8c2a8b71d1fea622a4e6f5687d48fc94a1b
submission_blob: 5d7e2cb0f68f381154fb868581be341a63e9aca3
lint_warnings: 106
attempt_file: 2026-10-26T214409Z-check-experts-rega-s4-bm256-wide-tile-high-ft-cf.md
prepared_at: 2026-10-26T214409Z
job: 13cfaf1e25c341af
dispatched_at: 2026-10-26T214415Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.47107534271079
speedup_ci: [1.4490092125701282, 1.4931414728514518]
harvested_at: 2026-10-26T220301Z
best_at_harvest: 1.4677493909229
promotion: null
---

# experts-rega-s4-bm256-wide-tile-high-ft-cf

## Change

Confirmation rerun of experts-rega-s4-bm256-wide-tile-high-ft: its implementation commit 97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2 unchanged except one trailing comment, run as a check and pooled with job a6f6063051f14750 at harvest.

## Why

Queued by churn.py (harvest): `experts-rega-s4-bm256-wide-tile-high-ft` ran 1.4758x against best 1.4677x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.

## Implementation notes

No code change: `experts-rega-s4-bm256-wide-tile-high-ft` implementation commit `97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2` plus one trailing comment, so okc times it again instead of returning the earlier job. Written by churn.py claim.

## Rules check

- [x] CUDA-graph capturable: `_experts_rega` adds only a dict lookup of a prepare-set int by the token count and one launch; no `.item()`/`.cpu()`/`synchronize()`/prints/allocation on values in kernel_fn (all syncs/prints are in prepare-only `_rega_wide_*`).
- [x] Nothing derived from `data` is cached: state gets only `rega_wide` (ints) and `rega_wide_ext` (module); tensor maps are encoded per call; no module/static/C++ global holds pointers (source audit `_rega_wide_audit` + `_rega_audit_cpp`).
- [x] Weights: unchanged bank from `_rega_bank`; NVFP4 codes and E4M3 scales dequantized in registers inside the kernel; nothing repacked or stored at another precision.
- [x] Activations BF16 (slab and outputs bf16), fp32 accumulation; KV cache and queries untouched.
- [x] Attention untouched; every route is computed by the wide kernel (same chunk/tile coverage as master, exact-compare gate).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses (in-process `load_inline` only, as master's builds) or network.
- [x] All work inside the timed call; no output caching or replay (the copy/replay check refills inputs per state).
- [x] Builds only inside prepare; build error, attribute miss, spill, exact-compare miss, deadline or gate miss leaves master's `rega_gemm` path (flag 0); wide has its own per-state flag and never disables an unrelated landed flag.
- [x] Bounded waits: `mbar_wait` (2^26-cycle bound, error word 1) on every mbarrier; no spin elsewhere; launch markers err[1]/err[2]; non-sm_90a image sets error word 2.
- [x] Prepare budget: wide start deadline 175 s, step deadline 250 s; prints `skipped: deadline`.

okc_lint: `OK with warnings` (0 errors; 101 host-sync warnings, all in prepare-only code outside kernel_fn's call graph, and 5 pre-existing dtype warnings). `python3 -I -m py_compile`: ok; `__pycache__` removed.

Copied from `experts-rega-s4-bm256-wide-tile-high-ft`, whose code this is; the only change is a comment.

## Result

- job `13cfaf1e25c341af` (check), status done, passed True, tests 5/5
- speedup 1.4711x, 95% CI [1.4490, 1.4931], score 263347.4226317467 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4677493909229, "ci": [1.4457331500590564, 1.4897656317867434], "job": "73b40cd13fdf4bb0"}; base result: muse_prefill_layer/hypotheses/done/neutral/experts-rega-s4-bm256-wide-tile-high-ft.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-s4-bm256-wide-tile-high-ft-cf.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-s4-bm256-wide-tile-high-ft-cf.patch`; branch `hyp/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft-cf`
- confirm of `experts-rega-s4-bm256-wide-tile-high-ft`: pooled 1.4758x (job a6f6063051f14750) and 1.4711x (job 13cfaf1e25c341af) → 1.4734x; best 1.4677x, two-run margin 1.07% → neutral (unproven-vs-best)

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 769.33 | 727.71 | 1.015 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1145.38 | 1101.56 | 1.0103 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 2148.97 | 2094.58 | 0.9982 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 3725.86 | 3682.33 | 0.9941 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 10973.2 | 11119.1 | 0.9786 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 10828.31 | 11000.86 | 0.9794 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14827.97 | 14714.82 | 0.98 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18238.85 | 18343.87 | 0.9939 | flat |

no shape moved outside noise. Noise ±1.5%.

## Follow-ups

- pooled verdict for `experts-rega-s4-bm256-wide-tile-high-ft`: neutral (unproven-vs-best); this file is the record of the rerun only.
- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: experts-rega-s4-bm256-wide-tile-high-ft-cf`) that adds a further change in the same area is the only way to learn more.

