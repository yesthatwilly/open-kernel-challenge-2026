# muse_moe_w4a16 · 2026-10-30T0525Z · check · moefix-no-bf16-atomic-reduce

- **Job:** `54c956f1296c4b02` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `aae2fe76` (branch ladder/moefix, from master 94958afa / blob 88edbb9f = board run 45ef593d)

## Goal
Compliance, not speed. Remove the latent non-compliant Marlin fallback (BF16 atomic split-K reduce) flagged by
campaigns/2026-10-30-public-ladder/research/compliance-audit-1030.md. Expected score: same as 45ef593d / 922ddf4c
within noise (~2.07-2.08x), because every scored T (1, 8, 16, 32, 128, 256) is timed on Engine A / hot-half / XL
routes, not on Marlin.

## What changed, and where
- prepare (`_prepare_main`): `_w2_atomic_self_check` no longer runs; `state["w2_mode"]` stays "master". Prints
  `[w2-atomic] PATH CHOSEN = master (compliance ...)`.
- `_combo_setup`: phase F1 (w13 Marlin `use_atomic_add=True` at T == 1) removed; `w13_mode` stays "master".
- `W2_ATOMIC_MAX_TOKENS` 32 -> 0 and `W13_ATOMIC_MAX_TOKENS` 1 -> 0 (belt and braces: the atomic branches in
  `_moe_marlin` are unreachable even if a mode were set).
- No CUDA source changed: all 31 CUDA source strings are byte-identical to master (AST compare).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn's call graph; branching
      on shapes and prepare flags only (okc_lint: OK with warnings, all prepare-only).
- [x] Nothing derived from `data` is cached across calls.
- [x] Weights: repacked losslessly in `prepare()` only (asserted round trips); NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16. **Reduce precision, stated explicitly:** every Marlin call on every
      route (fallback `_moe_marlin`, wgmma-gate reference, hot-half hybrid cold Marlin) runs
      `use_atomic_add=False, use_fp32_reduce=True`, i.e. split-K partials are reduced in FP32 exactly as SGLang
      (baseline.py:137-138). Engine A / wg / wh / XL: every MMA is `f32.bf16.bf16` (FP32 accumulate), split-K and
      stream-K partials live in FP32 workspaces summed in a fixed order; the only CUDA atomics are int tickets/counters.
      Post-norm/combine accumulates in FP32 with one BF16 store. No BF16 atomic reduce anywhere.
- [x] Attention rules: n/a (MoE).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call; no output caching or replay.

## Risks / what could fail
- Engine A gates compare against kernel_fn's Marlin path; that reference is now the (slower, FP32-reduce) SGLang
  reduce, so Engine A wins its gates at least as often. A missed gate now costs speed, not compliance.
- Prepare is slightly shorter (no w2 self-check, no F1 phase); later deadlines only get more slack.

## Result (after the run)

## Result
- job `54c956f1296c4b02` (check), passed 5/5, **2.089x [2.058, 2.120]** vs board 2.0793: level (scored routes unchanged, as predicted).
