# muse_bf16_proj · 2026-10-25T004327Z · check · unpooler-full-m16-swapab-route

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T003858Z-unpooler-full-m16-swapab-route.md` (branch `hyp/muse_bf16_proj/unpooler-full-m16-swapab-route`).

- **Job:** `05782267384343eb` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `9e2225c`

## Goal
Built on master (8b9ad43, blob a8bb42a). On master the full-weight unpooler (n=8192, k=4096, 67 MB) at 9 <= m <= 16 falls through to _bf16_gemm_dispatch_impl (cuBLAS nvjet_sm90_tst_64x16, 19.3 us kernel, row 18.5 at decode_16), while at m 17..32 it takes the swap-AB Triton kernel _bf16_swapab_kernel (SWAPAB_CFG unpooler: SPLIT_K 1, BLOCK_K 256, 4 stages, grid 128 CTAs; 18.7 us kernel at m=32, row 18.1). bf16-m16-swapab-wgmma-extend covered pooler/qkv/o/slice at m<=16 but never the full unpooler. Change exactly one gate: add a per-layer lower bound, SWAPAB_MIN_M_BY_LAYER = {"unpooler": 9} (others keep SWAPAB_MIN_M = 17), used by _swapab_route_ok, so unpooler with cfg n == 8192 at 9..16 takes the same swap-AB kernel with the same constexprs (BLOCK_T 32, token columns >= m masked; M is not specialized so no new compile). In _swapab_prepare extend the unpooler warm-up range to 9..32. The slice (cfg n == 1024, m <= 8) is still caught earlier by _unpool_shape_ok and is unaffected; m=130 and m=1000 tests unaffected. No other layer, constant or kernel changes.

## What changed, and where
Edited only muse_bf16_proj/submission.py in the worktree (+7/-3). Added SWAPAB_MIN_M_BY_LAYER = {"unpooler": 9}; _swapab_route_ok uses SWAPAB_MIN_M_BY_LAYER.get(layer, SWAPAB_MIN_M) as its lower bound; _swapab_prepare's warm loop now uses the same per-layer minimum, so the unpooler is warmed at m 9..31 (m=32 by the first warm call), others stay 17..31. Per the review, the minimum drives the warm loop rather than a hardcoded 9..32. No deviation from the spec. The slice (_unpool_shape_ok, n=1024) is checked earlier in kernel_fn and is unaffected.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff only changes a constant lookup on cfg["m"]/layer; no sync, allocation or value-dependent control flow in kernel_fn.
- [x] Nothing derived from data is cached across calls: warm-up uses synthetic randn x in prepare only.
- [x] Weights: unchanged; BF16 stays BF16, no repack.
- [x] Activations BF16: unchanged (route requires x bf16).
- [x] Attention: n/a, no attention in this problem.
- [x] No timing patches, harness access, subprocesses or network added.
- [x] All work inside the timed call; no output caching or replay.

okc_lint output:
```
CLEAN: muse_bf16_proj/submission.py
```

## Risks / what could fail
low
Small, one-shape change (~0.5-1 us on one 62-weight shape, ~+1% on decode_16); attributable from the unpooler decode_16 row alone. Risk: swap-AB at m=16 wastes half the wgmma n32 columns, but the kernel is weight-stream-bound so that is free; if the row is not below 18.5 the gate goes back.

## Result (after the run)
- job `05782267384343eb` (check), status done, passed True, tests 9/9
- speedup 1.1885x, 95% CI [1.1706, 1.2063], score 4064.7835902195084 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/unpooler-full-m16-swapab-route.result.json`, `muse_bf16_proj/hypotheses/done/regression/unpooler-full-m16-swapab-route.patch`; branch `hyp/muse_bf16_proj/unpooler-full-m16-swapab-route`
