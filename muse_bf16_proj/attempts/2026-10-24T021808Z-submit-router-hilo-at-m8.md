# muse_bf16_proj · 2026-10-24T021808Z · submit · router-hilo-at-m8

Hypothesis: `muse_bf16_proj/hypotheses/done/gain/2026-10-23T225128Z-router-hilo-at-m8.md` (branch `hyp/muse_bf16_proj/router-hilo-at-m8`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `d2eb172`

## Goal
Promote check-proven gain router-hilo-at-m8 (1.0219x [1.0066,1.0372], check job aa3d033c208e4511) to the board.

In kernel_fn's router branch, lower the threshold at which the router switches from F.linear(x.float(), router) to the BF16 hi + lo pair: change ROUTER_FP32_MAX_TOKENS from 8 to 1, so m=8 (and the m=7/m=9 tests) take the two torch.mm(x, router_hi_t/router_lo_t, out_dtype=torch.float32) GEMMs that the parity base already builds in prepare(), while m=1 keeps the FP32 F.linear path exactly as before. Nothing else changes: the bf16 layers, prepare(), and the hi/lo tensors stay as in sglang-gemm-dispatch-parity. Mechanism: at m=8 the FP32 path runs two sm80_xmma f32 SIMT GEMM kernels (9.3 + 8.3 us) plus the x.float() cast (2.2 us) for 26.6 us reported; the hi/lo path at m=16 runs two nvjet bf16 tensor-core splitK GEMMs + two splitKreduce + one add for 14.8 us. m=8 should land near the m=16 number. The rules allow hi/lo with FP32 accumulation at any batch size.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/router-hilo-at-m8`, base `e64f27e` (the parity implementation commit).
One constant changed in `muse_bf16_proj/submission.py`: `ROUTER_FP32_MAX_TOKENS = 8` -> `1`, so the
router branch `if x.shape[0] <= ROUTER_FP32_MAX_TOKENS: return F.linear(x.float(), router)` keeps
only m=1 on fp32 and every m >= 2 takes the two `torch.mm(x, router_hi_t / router_lo_t,
out_dtype=torch.float32)` GEMMs the parity base builds in `prepare()`. The module docstring's "up to
8 tokens" now reads "at 1 token". Nothing else touched: bf16 layers, `prepare()`, hi/lo tensors.
Deviation from the spec text, per the adversarial review: only the m=8 router test changes path
(m=7 is a pooler test; m=9 router was already on hi/lo under the parent), not "the m=7/m=9 tests".

## Rules check (each must be true before the run)
Checked against `git diff e64f27e` in the worktree (2 lines: the constant and the docstring):

- [x] CUDA-graph capturable: the only branch is on `x.shape[0]` (a shape) and `config["layer"]`;
      no `.item()`/`.cpu()`/`synchronize()`/tensor prints; no allocation on input values.
- [x] Nothing derived from `data` is cached: `state` comes from `prepare()` (weights only).
- [x] Weights: bf16 weights unchanged; the fp32 router is kept and additionally held as a BF16
      hi + lo pair with FP32 accumulation, the rule's one named exception, allowed at any batch size.
- [x] Activations: bf16 in; the router's x stays bf16 on the hi/lo path (SGLang upcasts to fp32 at
      m<=8, which is finer, not coarser, than keeping bf16); fp32 router output as before.
- [x] Attention/every-key rule: not applicable (no attention in this problem).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses
      or network: imports are torch and SGLang's dispatch function, as in the parent.
- [x] All work happens inside the timed call; no output caching or replay.

Lint (`okc_lint.py`): CLEAN, no findings.

## Risks / what could fail
low

## Result (after the run)
(pending)
