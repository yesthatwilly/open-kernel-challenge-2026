# muse_bf16_proj · 2026-10-25T042357Z · check · restack-pooler-unpooler-nn-layout-m128

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T042040Z-restack-pooler-unpooler-nn-layout-m128.md` (branch `hyp/muse_bf16_proj/restack-pooler-unpooler-nn-layout-m128`).

- **Job:** `92cd4e758a924ca0` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `47465fb`

## Goal
Built on master (b99b1dd, the landed restack). Apply the exact change of branch hyp/muse_bf16_proj/pooler-unpooler-m128-nn-layout-gated (job 0daf3faa) to the restack: in prepare(), store state['pooler_t'] = W_pooler.t().contiguous() and state['unpooler_t'] = W_unpooler.t().contiguous() (lossless [K, N] bf16 copies; the originals stay). In kernel_fn add one branch just before the `if layer != 'router'` cuBLAS fallback: if layer in {pooler, unpooler} and 65 <= cfg m <= 128 and cfg n is the full width -> return torch.mm(x, state[layer + '_t']). m >= 129 keeps TN; m <= 32 keeps the restack's swap-AB paths. Nothing else changes. Record the nvjet NN kernel names for the two target rows.

## What changed, and where
- Applied done/regression/pooler-unpooler-m128-nn-layout-gated.patch with `git apply -3`. One conflict (prepare(): the restack replaced the ROUTER_WARM_MS warm-up loop with `_router_prepare(state)`); hand-merged by keeping `_router_prepare(state)` and adding the `_t` copy loop before it. The docstring and constants hunks merged cleanly. Own constants kept: NN_GATED_LAYERS, NN_GATED_MIN_M=65, NN_GATED_MAX_M=128.
- prepare(): `state[name + "_t"] = state[name].t().contiguous()` for pooler and unpooler, a real [K, N] contiguous copy (not a view). Originals stay.
- kernel_fn: branch inside `if layer != "router":`, after every Triton gate (swapab16, pooler, qkv, unpooler, o, swapab 17..32) and immediately before the `_bf16_gemm_dispatch_impl` fallback. Gate: layer in NN_GATED_LAYERS, 65 <= cfg m <= 128, cfg n == state[layer+"_t"].shape[1]; returns `torch.mm(x, state[layer+"_t"])`. m>=129 and m<=32 paths unchanged.
- Review fixes applied: branch placement, constants, real contiguous copy as above. The review's kill-criterion correction wins over the spec's absolute-us kill: judge on paired ratios of the two rows (pooler >= 1.06x with `_splitK_NNT` in the profile, unpooler >= 1.02x with `_NNT`), not on absolute us or the headline label.
- No TASK test covers m=65..128 for pooler/unpooler (tests use m=130 and m=1000 for unpooler, outside the gate); only the timed m=128 calls are checked against the reference. Record this in the attempt file.
- Stacking: the queued m=65..256 rewrites (bf16-verify-swapab-wgmma-cluster-dsmem-rewrite, qkv-gate-o-m128-wgmma-splitk-semaphore) must order their branch against this gate explicitly; whichever lands second decides precedence.
- Diff: 1 file, +17/-1. Not run (no GPU on this node).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: torch.mm on cfg-only gate, no syncs/allocation on input values (diff checked).
- [x] Nothing derived from `data` cached: `_t` copies come from weights in prepare().
- [x] Weights repacked losslessly in prepare() only: bf16 transpose copy.
- [x] Activations BF16 unchanged; KV cache n/a: not touched.
- [x] Attention n/a: projection layers only.
- [x] No timing patches, harness access, module edits, subprocesses or network (diff checked).
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `CLEAN: .../restack-pooler-unpooler-nn-layout-m128/muse_bf16_proj/submission.py`; py_compile OK; __pycache__ removed.

## Risks / what could fail
low
Kill: pooler verify_16 row not <= 24.0 us (6% under restack) with the NN kernel in the profile. +134 MB device memory, untimed. Estimated headline +0.003 over the restack.

## Result (after the run)
- job `92cd4e758a924ca0` (check), status done, passed True, tests 9/9
- speedup 1.2282x, 95% CI [1.2098, 1.2466], score 3933.285014089467 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "1e90292c7b954d96"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/restack-pooler-unpooler-nn-layout-m128.result.json`, `muse_bf16_proj/hypotheses/done/neutral/restack-pooler-unpooler-nn-layout-m128.patch`; branch `hyp/muse_bf16_proj/restack-pooler-unpooler-nn-layout-m128`
