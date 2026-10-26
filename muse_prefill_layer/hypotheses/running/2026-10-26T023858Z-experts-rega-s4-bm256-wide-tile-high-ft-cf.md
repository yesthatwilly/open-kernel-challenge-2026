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
status: running
stage: implemented
claimed_at: 2026-10-26T214345Z
base_sha: 97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2
base_blob: 82449fd7e7fae52ba13d26109cc48a9fd256b40c
branch: hyp/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft-cf
worktree: .worktrees/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft-cf
mode: null
restack_of: null
stale_base: true
implemented_at: 2026-10-26T214356Z
impl_commit: 912b3ea2ef4cfe5a47f0904e5d9a5ac38a8c92ea
submission_sha256: 263f2e32390c52ac7c285f0b126d42ab3f13bacc3328fc182f1ab39ba25e52ba
norm_sha256: 0c499f6341769f494a929dd9b5fdb8c2a8b71d1fea622a4e6f5687d48fc94a1b
submission_blob: 5d7e2cb0f68f381154fb868581be341a63e9aca3
lint_warnings: 106
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

