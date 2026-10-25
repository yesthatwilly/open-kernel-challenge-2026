# muse_bf16_proj · 2026-10-25T012031Z · check · bf16-verify-deepgemm-bf16-gemm-nt-route

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T011408Z-bf16-verify-deepgemm-bf16-gemm-nt-route.md` (branch `hyp/muse_bf16_proj/bf16-verify-deepgemm-bf16-gemm-nt-route`).

- **Job:** `2d522d47ca5c4fc9` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `ba33022`

## Goal
Built on master (blob a8bb42a). At m=128/256 all four bf16 layers still run cuBLAS nvjet and sit at ~1.0x SGLang (master profile 1602a4e947e64090): qkv_gate m=128 nvjet 96x64 splitK 13.0 + cublasLt splitKreduce 3.0 us (two nodes, row 16.81, floor 8.4), m=256 nvjet 96x64 19.4 (row 19.62, 498 TFLOP/s, floor 9.8); pooler m=128 25.7 (row 25.55, floor 14.6), m=256 29.5 (583 TFLOP/s, floor 17.4); o m=128 5.8, m=256 8.3; unpooler m=128 20.4, m=256 24.6 (~700 TFLOP/s). Our own kernels have lost here three ways (Triton 128x128 split-K, Triton TMA+warp-spec, CUTLASS stream-K: qkv m=256 28.5 us, every m=256 row worse), and prepare-time algo picking is a dead knob. Untried: DeepGEMM, which the image ships (its include dir was found by the stream-K build: deep_gemm/include with CUTLASS 4.2.1) and whose recent releases have dense BF16 GEMMs (deep_gemm.bf16_gemm_nt and friends) with analytic block/multicast/SM-count heuristics tuned for exactly these skinny-M, deep-K Hopper shapes, no timing search. Change: at import, try `import deep_gemm`; print deep_gemm.__version__ (or __file__) and whether hasattr(deep_gemm, 'bf16_gemm_nt'); if the module or the function is missing print '[deepgemm] unavailable, master kept' and change nothing. In prepare(), for each of qkv_gate, o, pooler, unpooler and m in {128, 256}: allocate out = torch.empty(m, n, bf16), call deep_gemm.bf16_gemm_nt(x, W, out) on a synthetic bf16 x (this also triggers DeepGEMM's JIT compile, untimed; print the seconds), torch.cuda.synchronize(), then self-check against x.double() @ W.double().T: rel L2 <= 3e-3, a one-hot power-of-two probe row within 1e-4, finite. Wrap each layer in try/except; a failure prints '[deepgemm] <layer> DISABLED: <reason>' and that layer keeps master. Also run m=130 for unpooler and m=200 for each layer through the self-check so the 65..256 route is covered. Record state['dg_ok'][layer]. In kernel_fn add one branch just before master's cuBLAS fallback: if state['dg_ok'][layer] and 65 <= cfg m <= 256 and cfg n is the full width and x is contiguous bf16 [m, k], out = torch.empty(m, n, bf16); deep_gemm.bf16_gemm_nt(x, W, out); return out. If the signature differs in the installed version (e.g. it takes (a, b, d, c=None) or tuples), adapt to it and print the signature used. Weights are read in place (W [n, k] row-major is DeepGEMM's NT 'b'), no repack, no precision change. m <= 32 paths, the router, and the m=512/1000 tests are unchanged. Print '[deepgemm] PATH CHOSEN <layer>: deepgemm|master'. At harvest, read the PATH CHOSEN lines and confirm a deep_gemm kernel in top_kernels on the m=128/256 rows before reading any number. Check DeepGEMM's launch is CUDA-graph capturable (no host sync, no allocation inside it once JIT-compiled): run one torch.cuda.graph capture + replay per layer in prepare and disable the layer if capture raises.

## What changed, and where
Edited only `muse_bf16_proj/submission.py` in the worktree (+157 lines, nothing removed).
- Module-level guarded `import deep_gemm`; `_dg_find_fn()` prints module path, version and every bf16/gemm candidate name, then picks `bf16_gemm_nt` (fallbacks `gemm_bf16_nt`, `bf16_gemm_nt_contiguous`). A loud `[deepgemm] AVAILABLE` or `[deepgemm] unavailable, master kept` line results. Signature is printed via inspect and the call is `fn(x, W, out)`.
- `prepare()` (last step, `_dg_prepare`): for pooler, qkv_gate, unpooler, o and EVERY routed m in (128, 130, 200, 256): eager call (JIT), fp32 matmul self-check (rel L2 <= 3e-3), one-hot probe (<= 1e-4), then torch.cuda.graph capture + replay checked again. Per (layer, m) failure prints `DISABLED` and only that m is dropped. Elapsed time is tracked from prepare() start; past 200 s remaining (layer, m) are skipped (gate off). Prints `PATH CHOSEN <layer> m=<m>: deepgemm|master`.
- `kernel_fn`: one branch just before master's cuBLAS fallback, gated on the exact verified m set (not 65..256, per the review), cfg n == full weight rows, contiguous bf16 x. Router, m<=32 paths and m=512/1000 are unchanged.
- Deviation: gate is the warmed set {128,130,200,256}, not the whole 65..256, so no JIT can happen in a timed or captured call. Chosen tile config is not introspectable from here; not printed. Nothing could be run on this node (no GPU, no deep_gemm).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: branch has only torch.empty + the DeepGEMM launch, no sync/.item()/value-dependent control flow; capture tested per (layer, m) in prepare. Syncs are prepare-only.
- [x] Nothing derived from `data` is cached: state holds only weight-derived/verified-m sets built from synthetic x.
- [x] Weights: read in place, no repack; BF16 stays BF16.
- [x] Activations BF16, output BF16; no precision change.
- [x] Attention: n/a, dense projections only.
- [x] No timing patches, harness access, subprocess or network added by me (DeepGEMM's own JIT is a library compile; no timing-based selection).
- [x] All work inside the timed call; no output caching or replay.

okc_lint output:
```
warn  host-sync    L713: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L736: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L741: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
```
py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
Medium. (a) The installed DeepGEMM may predate the BF16 GEMMs (then the run measures master: read the [deepgemm] lines first). (b) Its JIT needs nvcc and a writable cache dir at prepare time; load_inline builds have run on the judge, so nvcc exists, but print the JIT seconds against the 10-minute budget. (c) DeepGEMM's heuristics favour large M; at m=128/256 with M on the token side it may choose block_m 64/128 and lose like CUTLASS did; per-shape rows decide, gate per layer at harvest. (d) If it allocates or syncs inside the call, graph capture fails in prepare and the layer is disabled. Rules: an in-image library (rules list DeepGEMM explicitly), no timing-based selection.

## Result (after the run)
- job `2d522d47ca5c4fc9` (check), status done, passed True, tests 9/9
- speedup 1.2027x, 95% CI [1.1846, 1.2207], score 4016.7905493453336 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/bf16-verify-deepgemm-bf16-gemm-nt-route.result.json`, `muse_bf16_proj/hypotheses/done/regression/bf16-verify-deepgemm-bf16-gemm-nt-route.patch`; branch `hyp/muse_bf16_proj/bf16-verify-deepgemm-bf16-gemm-nt-route`
