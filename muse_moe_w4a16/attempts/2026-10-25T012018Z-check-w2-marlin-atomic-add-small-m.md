# muse_moe_w4a16 · 2026-10-25T012018Z · check · w2-marlin-atomic-add-small-m

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-25T011401Z-w2-marlin-atomic-add-small-m.md` (branch `hyp/muse_moe_w4a16/w2-marlin-atomic-add-small-m`).

- **Job:** `a2744bc496b84790` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `6960c4c`

## Goal
For num_tokens <= 32 only (bs=1, 8, 16, 32 and the bs=4 test), pass use_atomic_add=True and use_fp32_reduce=False (Marlin requires the pair) to the SECOND moe_wna16_marlin_gemm call (w2) only, by building a separate kwargs dict from master's common for that call; the w13 call and every num_tokens > 32 call keep master's common exactly. Reason: w13-marlin-atomic-add-small-m tried this on w13 only and was neutral, but w2 is the other shape: size_n = H = 4096 (half of w13's 8192) at the same K = I = 4096, so Marlin has half as many N tiles to spread over 132 SMs and slices K more, i.e. more of w2's time is the K-slice fp32 global reduce that atomic add replaces. y (cache3) is then accumulated in place, so make sure cache3 is zeroed where Marlin expects it (Marlin's moe path with use_atomic_add zeroes or requires a zeroed C; read sglang's moe_wna16_marlin_gemm wrapper and zero cache3 with one .zero_() only if the wrapper does not, inside kernel_fn, graph-safe). Self-check in prepare: at T in {1, 4, 8, 16, 32} compare the final output against master's path (rel-L2 <= 5e-3, finite), else keep master and print '[w2-atomic] PATH CHOSEN = atomic|master (<reason>)'.

## What changed, and where
- Worktree submission.py only. kernel_fn is now a thin wrapper over `_forward(data, state, w2_mode)`; the body is unchanged except the w2 call, which uses `w2_common = dict(common, use_atomic_add=True)` when `w2_mode != "master"` and num_tokens <= 32 (`W2_ATOMIC_MAX_TOKENS`). w13 and every T > 32 call keep master's `common` exactly.
- Review corrections applied: use_fp32_reduce stays True; cache3 still aliases cache13 (holds w13's output, so non-zero); prepare() prints atomic enabled / memset in path.
- Deviation: sglang source is not readable on this node, so zeroing is decided empirically. `_w2_atomic_self_check` (prepare, synthetic seeded inputs only) tries "atomic" (no memset) at T in {1,4,8,16,32}, then "atomic_zero" (one `cache3.zero_()` before w2, graph-safe) if that fails, else "master". Gate: finite and rel-L2 vs master <= 5e-3. Binding exceptions are caught and logged. Chosen mode is stored in state["w2_mode"] (set in prepare, not derived from inputs).
- Prints: `[w2-atomic] PATH CHOSEN = atomic|master (<reason>)` and a line with atomic enabled, fp32_reduce=True, memset in path.
- Not run (no GPU). slice_count printing was skipped (needs Marlin internals).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds no sync/.item()/.cpu() in kernel_fn path; branch on num_tokens and state string only; `zero_()` is a graph-safe kernel. Syncs/.item() are only in prepare().
- [x] Nothing derived from `data` is cached: only w2_mode from prepare's synthetic inputs.
- [x] Weights: untouched; NVFP4 stays NVFP4.
- [x] Activations BF16 as before; unchanged.
- [x] Attention: n/a, no attention in this problem.
- [x] No timing patches, harness access, subprocesses or network in diff.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `OK with warnings: submission.py` (only host-sync warnings, all in prepare(), outside kernel_fn's call graph). py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
LESSONS lists use_atomic_add as a knob that did nothing, but that was measured on w13 only; this is the untested w2 half. If a zero-fill launch is needed it costs ~1-2 us and may cancel the gain; read the Marlin row and the memset row.

## Result (after the run)
- job `a2744bc496b84790` (check), status done, passed True, tests 5/5
- speedup 1.3445x, 95% CI [1.3244, 1.3647], score 10002.413200587025 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-small-m.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/w2-marlin-atomic-add-small-m.patch`; branch `hyp/muse_moe_w4a16/w2-marlin-atomic-add-small-m`
