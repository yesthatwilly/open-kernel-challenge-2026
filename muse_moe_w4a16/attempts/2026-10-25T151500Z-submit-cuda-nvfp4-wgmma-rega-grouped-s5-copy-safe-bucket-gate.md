# muse_moe_w4a16 · 2026-10-25T151500Z · submit · cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate

Hypothesis: `muse_moe_w4a16/hypotheses/done/neutral/2026-10-25T054545Z-cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate.md` (branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-s5-copy-safe-bucket-gate`).

- **Job:** `4e4a0e26c85944f2` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `fc959d9c` (identical to the check run 7be125e9e1ea42e3)

## Goal
Scored submit of the bucket-gate code exactly as checked (1.3744x [1.354,1.395], CI fully above the moe board 1.3439). The user said on 2026-10-25 ~15:00Z "submit any scored runs at your discretion". Not landed on master: its base predates master's align-parallel-per-expert-verify and w2-marlin-atomic-add-small-m promotions (stale base, 828-line diff), so a hand merge comes later if the score holds.

## What changed, and where
Nothing since the check run: same commit fc959d9c, worktree checked out detached.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn adds only a Python branch on shape + two prepare-time booleans; no sync/.item/allocation on values (checked in diff: selector code is prepare-only, called from `_wg_setup`).
- [x] Nothing derived from `data` cached across calls: flags come from synthetic prepare-time routings only.
- [x] Weights: unchanged from s5-forced (lossless retile in prepare, NVFP4 stays NVFP4, dequant in registers).
- [x] Activations BF16 where SGLang keeps BF16: unchanged.
- [x] Attention: n/a (MoE problem); every route attended (no skipping), unchanged.
- [x] No timing patches, harness/IO/secrets, module edits, reference import, subprocesses or network: no subprocess/fork/spawn added; selector only uses torch events/graphs.
- [x] All work inside the timed call; no output caching or replay. No module/static globals hold device pointers (selector tensors are locals; `_wg_audit_globals` unchanged and still run; flags are in state).
- [x] Two-state judge copies: flags are plain bools in state and survive a tree-map clone; copy check (A, deepcopy B, clone C) unchanged.

okc_lint: `OK with warnings` (0 errors; only host-sync/dtype warnings outside kernel_fn's call graph, prepare-only). undef.py: `submission.py []` (no undefined names). py_compile OK, __pycache__ removed.

## Risks / what could fail
low: identical code already passed 5/5 tests on the check run.

## Result (after the run)
