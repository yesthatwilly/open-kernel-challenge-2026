# muse_bf16_proj · 2026-10-25T012003Z · check · pooler-unpooler-m128-nn-layout-gated

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T011407Z-pooler-unpooler-m128-nn-layout-gated.md` (branch `hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated`).

- **Job:** `0daf3faa83ab4e81` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `41146e7`

## Goal
Built on master (blob a8bb42a). Narrow bf16-m128-nn-weight-layout-probe (done/neutral, 1.158x on the older dba4ffd master) to the only rows it won. Measured per shape in that run: pooler verify_16 m=128 25.21 -> 22.86 us (1.120x) and unpooler verify_16 m=128 19.65 -> 19.15 (1.043x); it LOST qkv_gate verify_32 m=256 (0.941x), and the other m=128/256 rows were flat. Master's incumbent on those two rows is cuBLAS TN through _bf16_gemm_dispatch_impl: pooler nvjet_sm90_tst_64x64_64x13_2x1 25.7 us kernel (row 25.55, ~2.6 TB/s against a 14.6 us floor), unpooler nvjet_sm90_tst_128x64_64x8_1x2 20.4 us (row 19.79). Change: in prepare(), for pooler and unpooler ONLY, store a lossless transposed copy state['pooler_t'] = W_pooler.t().contiguous() and state['unpooler_t'] = W_unpooler.t().contiguous() ([K, N] bf16 row-major, same values and dtype; the originals stay for every other path; +134 MB once, untimed). In kernel_fn, add one branch placed just before master's cuBLAS fallback: if layer in {pooler, unpooler} and 65 <= cfg m <= 128 and cfg n equals the full weight width, return torch.mm(x, state[layer + '_t']) (a contiguous bf16 [m, n] with fp32 accumulation inside cuBLAS, the same math). Gate only on layer and cfg m/n. Everything else is byte-identical to master: m <= 32 Triton paths, the router, qkv_gate and o at every m, and pooler/unpooler at m >= 129 (so m=256 stays TN, where the probe showed nothing, and the m=130 and m=1000 unpooler tests stay on master). The attempt file records the nvjet kernel name and us for the two target rows from the profile.

## What changed, and where
Restack of bf16-m128-nn-weight-layout-probe: the staged 3-way-applied patch applied the NN path to all four bf16 layers at m >= 128. Narrowed in `muse_bf16_proj/submission.py` (worktree) only:
- Constants `NN_GATED_LAYERS = ("pooler", "unpooler")`, `NN_GATED_MIN_M = 65`, `NN_GATED_MAX_M = 128` replace the probe's `NN_LAYERS`/`NN_MIN_M` (own gate, so it stacks cleanly with bf16-verify-deepgemm-bf16-gemm-nt-route).
- `prepare()`: lossless `state["pooler_t"]` and `state["unpooler_t"]` = `W.t().contiguous()` ([K, N] bf16); qkv_gate and o copies from the probe removed (no +memory for them).
- `kernel_fn`: inside the `layer != "router"` fallback, before master's cuBLAS call: if layer in the gated set and 65 <= cfg m <= 128 and cfg n == transposed weight's N, return `torch.mm(x, state[layer + "_t"])`. Gate reads only layer and cfg m/n.
- Module docstring updated. Everything else equals master: m <= 32 Triton paths, router, qkv_gate and o at all m, pooler/unpooler at m >= 129 (TN).
- Deviation: none. Note the earlier Triton routes (pooler/unpooler shape gates, swap-AB) still take precedence where they apply, as on master.
- Not run (no GPU). For the attempt file: record NN nvjet kernel names/us for the two m=128 rows and list the test m values in 65..128.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds only a cfg-gated torch.mm; no .item()/.cpu()/sync, no input-value control flow.
- [x] Nothing derived from `data` is cached across calls: the only new state is weight transposes made in prepare().
- [x] Weights: repacked losslessly in prepare() only; BF16 stays BF16 (transpose copy, same values/dtype).
- [x] Activations BF16 where SGLang keeps BF16: x passed unchanged to torch.mm, bf16 output with fp32 accumulation inside cuBLAS.
- [x] n/a: attention not touched.
- [x] No timing patches, harness edits, reference import, subprocesses or network in the diff.
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint output:
```
CLEAN: muse_bf16_proj/submission.py
```
py_compile: OK; __pycache__ removed.

## Risks / what could fail
low
Low. Pure dispatch, no kernel. The probe's gain came from one run against an older master; LESSONS says cuBLAS rows at m>=128 swing between runs (pooler m=128 nvjet 23.9 vs 25.7 us across runs), so a per-shape gain under ~7% needs a repeat before it counts. The headline effect is small (~+0.5-0.7%: one load of six), so judge on the two rows.

## Result (after the run)
- job `0daf3faa83ab4e81` (check), status done, passed True, tests 9/9
- speedup 1.1953x, 95% CI [1.1774, 1.2133], score 4041.3571243986717 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-m128-nn-layout-gated.result.json`, `muse_bf16_proj/hypotheses/done/regression/pooler-unpooler-m128-nn-layout-gated.patch`; branch `hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated`
