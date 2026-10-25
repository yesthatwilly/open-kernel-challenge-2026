# muse_decode_layer · 2026-10-25T045638Z · submit · w2-marlin-atomic-add-t1-port

Hypothesis: `muse_decode_layer/hypotheses/done/neutral/2026-10-25T042121Z-w2-marlin-atomic-add-t1-port.md` (branch `hyp/muse_decode_layer/w2-marlin-atomic-add-t1-port`).

- **Job:** `3fd3fa1f96d045dd` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `daed376`

## Goal
Promote check-proven gain w2-marlin-atomic-add-t1-port (1.2856x [1.2663,1.3049], check job 7a9f5de5aecc4aba) to the board.

Built on master, which now carries the landed full-decode-splitkv-restack-on-pdl (CUDA split-KV kernel A/B on full decode 1 < batch <= 64). Port muse_moe_w4a16's w2-marlin-atomic-add-small-m (branch hyp/muse_moe_w4a16/w2-marlin-atomic-add-small-m; bs=1 48.8 -> 44.9 us, slow mode 44/100 -> 10/100 replays). In _experts (the two moe_wna16_marlin_gemm calls), at num_tokens == 1 ONLY (config: every decode_1 shape), build a separate kwargs dict for the SECOND call (w2, size_n=size_k=hidden) with use_atomic_add=True and use_fp32_reduce=False; the w13 call and every num_tokens > 1 call keep `common` unchanged. Because w2 then accumulates into its C buffer, give that path a cache3 that is zeroed (a dedicated slice with .zero_() inside kernel_fn, graph-safe) unless sglang's wrapper already zeroes C with atomic add (read the wrapper; do not double-zero). Prepare self-check at T=1 vs master's path: rel L2 vs an fp32-dequant reference no worse than 1.5x master's own, finite, else keep master and print PATH CHOSEN.

## What changed, and where
Edited only `muse_decode_layer/submission.py` in the worktree (+88/-3).
- `_experts(..., w2_mode=None)`: w2_mode defaults to `state["w2_mode"]`. When it is not "master" AND `num_tokens == 1` AND `align is not None` (kernel R / dense-chain path) AND `dispatch is None`, the w2 `moe_wna16_marlin_gemm` call gets `dict(common, use_atomic_add=True)`. The w13 call, every T>1 call and the align=None path keep `common` unchanged.
- Review corrections applied (review wins over the spec text): `use_fp32_reduce` stays True; `cache3` still aliases `cache13` (no dedicated zeroed slice); selector order "atomic" (no memset) -> "atomic_zero" (`cache3.zero_()` before w2) -> master, as moe.
- `_w2_atomic_self_check(state, device)` runs in prepare() after `_dl_setup`/`_align_par_setup`. It drives kernel R (`_dl_route`) at T=1 with crafted logits (1-4 forced local experts, the rest remote, 6 local_start values, 12 trials), then compares the post-combine `_experts` output with `w2_mode="atomic"/"atomic_zero"` against `w2_mode="master"` on identical inputs and the same align (rel L2 <= 5e-3, moe's tolerance, finite). Requires both local and remote slots to have been exercised, else master. If `state["dl"]` is None (dense chain off) the mode is master. Every launch is announced with a flushed print and followed by `torch.cuda.synchronize()` plus a flushed print. Loud `[w2-atomic] PATH CHOSEN = ...` line (T=1 chosen mode, T>1 MASTER).
- Deviation: the self-check compares to master's w2 numerics (as moe did), not to an fp32-dequant reference; the review sets the comparison target as master within moe's tolerance.
- No GPU here: nothing was run.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds only a kwarg choice (config-only: num_tokens, align presence, prepare-time mode) and an optional `cache3.zero_()`; no sync/.item in kernel_fn; self-check is prepare()-only.
- [x] Nothing derived from `data` is cached across calls: the mode is chosen from synthetic inputs in prepare().
- [x] Weights: unchanged (n/a: no repacking touched).
- [x] Activations BF16 / KV FP8: unchanged; w2 still bf16 output, fp32 reduce kept.
- [x] Attention sees every key: n/a, attention untouched.
- [x] No timing patches, harness, module edits, reference import, subprocess or child processes: none added.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: OK with warnings (only pre-existing and new prepare()-only host-sync warnings, no errors). py_compile passed; __pycache__ removed.

## Risks / what could fail
low

## Result (after the run)
Promotion passed: 1.2726x [1.2535,1.2917], board: {"rank": 396, "team": "yesthatwilly", "speedup": 1.2726, "first": false, "ahead": 395, "to_first": 1.7955}
