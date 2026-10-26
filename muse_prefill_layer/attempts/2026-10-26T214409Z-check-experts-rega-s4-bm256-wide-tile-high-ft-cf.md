# muse_prefill_layer · 2026-10-26T214409Z · check · experts-rega-s4-bm256-wide-tile-high-ft-cf

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-26T023858Z-experts-rega-s4-bm256-wide-tile-high-ft-cf.md` (branch `hyp/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft-cf`).

- **Job:** `13cfaf1e25c341af` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `912b3ea2`

## Goal
Confirmation rerun of experts-rega-s4-bm256-wide-tile-high-ft: its implementation commit 97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2 unchanged except one trailing comment, run as a check and pooled with job a6f6063051f14750 at harvest.

## What changed, and where
No code change: `experts-rega-s4-bm256-wide-tile-high-ft` implementation commit `97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2` plus one trailing comment, so okc times it again instead of returning the earlier job. Written by churn.py claim.

## Rules check (each must be true before the run)
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

## Risks / what could fail
low

## Result (after the run)
- job `13cfaf1e25c341af` (check), status done, passed True, tests 5/5
- speedup 1.4711x, 95% CI [1.4490, 1.4931], score 263347.4226317467 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4677493909229, "ci": [1.4457331500590564, 1.4897656317867434], "job": "73b40cd13fdf4bb0"}; base result: muse_prefill_layer/hypotheses/done/neutral/experts-rega-s4-bm256-wide-tile-high-ft.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-s4-bm256-wide-tile-high-ft-cf.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-s4-bm256-wide-tile-high-ft-cf.patch`; branch `hyp/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft-cf`
- confirm of `experts-rega-s4-bm256-wide-tile-high-ft`: pooled 1.4758x (job a6f6063051f14750) and 1.4711x (job 13cfaf1e25c341af) → 1.4734x; best 1.4677x, two-run margin 1.07% → neutral (unproven-vs-best)
