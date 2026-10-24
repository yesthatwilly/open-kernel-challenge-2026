# muse_bf16_proj · 2026-10-23T230319Z · check · router-hilo-at-m8

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-23T225128Z-router-hilo-at-m8.md` (branch `hyp/muse_bf16_proj/router-hilo-at-m8`).

- **Job:** `aa3d033c208e4511` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `b71f98b`

## Goal
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
Correctness: the m=8 router test and the m=9 outliers test now run on hi/lo; L2 tol is 1e-4 and probe rows 1e-3, and hi+lo residual is ~2^-16 relative per weight, so margin is ~10x on both. The m=7 pooler test is unaffected. If the m=8 hi/lo path were slower than expected it would show only on verify_1. The reviewer of the parity run named this as the first real hypothesis; it is deliberately the smallest possible step (threshold 8 to 1) so the m=1 gemv path is left as a control; m=1 on hi/lo is a separate question (five launches vs three).

## Result (after the run)
- job `aa3d033c208e4511` (check), status done, passed True, tests 9/9
- speedup 1.0219x, 95% CI [1.0066, 1.0372], score 4727.328283175937 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 0.9020752041800139, "ci": [0.8885440761173137, 0.9156063322427139], "job": "366189f0f1114da1"}; base result: muse_bf16_proj/hypotheses/done/gain/sglang-gemm-dispatch-parity.result.json
- **outcome: gain** (gain)
- files: `muse_bf16_proj/hypotheses/done/gain/router-hilo-at-m8.result.json`, `muse_bf16_proj/hypotheses/done/gain/router-hilo-at-m8.patch`; branch `hyp/muse_bf16_proj/router-hilo-at-m8`
