# muse_bf16_proj · 2026-10-24T175228Z · submit · bf16-restack-m16-swapab-router-swapab-m32

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T174421Z-bf16-restack-m16-swapab-router-swapab-m32.md` (branch `hyp/muse_bf16_proj/bf16-restack-m16-swapab-router-swapab-m32`).

- **Job:** `1e90292c7b954d96` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `8f97b72`

## Goal
Restack two harvested per-shape winners onto current master (bf16-m32-swapab-wgmma-triton) by a 3-way git merge-file of their patches. (1) bf16-m16-swapab-wgmma-extend as is: qkv m=1/8/16 kernels went 11.3-12.3 to 9.4-10.1 us, o 4.6-5.2 to 4.1-4.4, unpooler slice 4.4-4.9 to 3.9-4.3. Its patch is in done/neutral/bf16-m16-swapab-wgmma-extend.patch on branch hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend. (2) router-swapab-wgmma-all-m, but gated to m<=32 only. Above that, keep master's BLOCK_M=64 hilo router kernel, which already emits wgmma (the m=128/256 rows read -2%/flat). Patch on branch hyp/muse_bf16_proj/router-swapab-wgmma-all-m. The harvest analysis found the merge-file of the two clean. Do NOT include bf16-cublaslt-algo-pick (it lost at m=256). No other change.

## What changed, and where
- Base: worktree submission.py at master blob a8bb42a. `git merge-file` of bb41afb (bf16-m16-swapab-wgmma-extend) and 64894c9 (router-swapab-wgmma-all-m) over a8bb42a exited 0 (clean); that result is the starting point. Outside the router region the file equals bb41afb: the diff vs bb41afb touches only the module docstring's router paragraph, the ROUTER_* constants and _router_* functions, the one `_router_prepare(state)` call in prepare() and the router tail of kernel_fn.
- Gate (inclusive, `ROUTER_SWAPAB_MAX_M = 32`): new `_router_path(x, hi, n, stages)` (shapes only) returns "swapab" for m <= 32 (bucket present in state["router_stages"]), "hilo" for m > 32, "" otherwise. kernel_fn sends "swapab" to the swap-AB `_router_forward`, "hilo" to `_router_hilo_forward`, and only "" to the F.linear fp32 fallback.
- `_router_hilo_forward` is master's `_router_forward` verbatim, with only its name and its config call renamed (`_router_config` -> `_router_hilo_config`). `_router_hilo_config` is master's `_router_config` logic verbatim (docstring changed only). `_router_hilo_kernel`, ROUTER_NUM_STAGES = 3 and ROUTER_NUM_WARPS = 4 are unchanged. Verified with an AST source diff against a8bb42a: forward and kernel identical modulo rename, config identical except the docstring. So for m > 32 the launched path (config, grid, kernel, stages, warps, zeroed output) is master's.
- `_router_config` (swap-AB) trimmed to its two m <= 32 buckets: (16, 32) at m <= 16, (32, 32) at m <= 32. The patch's m > 32 rows were dropped since they are never used.
- `_router_shape_ok` takes SPLIT_K (and the N tile) from the path actually used: swap-AB SPLIT_K 32 / BLOCK_W 64 at m <= 32; `_router_hilo_config` SPLIT_K 16/8 / BLOCK_N at m > 32 (exactly master's check).
- `_router_prepare`: m in ROUTER_WARM_MS <= 32 (1, 8, 16, 32) choose stages per swap-AB bucket and warm as in the patch; m = 64/128/256 warm master's hilo forward. After warm-up, for the problem's [256, 8192] router, it asserts every ROUTER_WARM_MS m resolves to "swapab" or "hilo" (never the fallback) and prints `[router] m=<m>: <path>`.
- Dropped `_router_hilo_report` (prepare-only diagnostic) and the duplicate `_ptx_counts` helper; the router report line now uses the m16 piece's `_print_ptx_counts(tag, compiled)` (loses red/atom counts in that log line; diagnostic only).
- Deviation from spec: none in substance. Diff stat vs master: 1 file, +364 / -23.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: router routing in kernel_fn branches on x.shape / cfg["n"] / the prepare-time stage dict only; no .item()/.cpu()/synchronize()/prints in kernel_fn (prints and the assert are prepare-only). The m16 piece is unchanged from its measured patch (routes on cfg and shapes).
- [x] Nothing derived from `data` is cached across calls: state only gains swapab16 workspaces/tickets/stages and router_stages, all built in prepare() from weights and synthetic x.
- [x] Weights: router as bf16 hi + lo with fp32 accumulation (as master); bf16 weights read in place as bf16.
- [x] Activations BF16 where SGLang keeps BF16 (bf16 x in, bf16 out for the four layers, fp32 router out). KV cache / queries: n/a, no attention in this problem.
- [x] Attention sees every key: n/a, no attention in this problem (dense projections only).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds only Triton launches, torch allocations, prepare-time prints and an assert.
- [x] All work happens inside the timed call; no output caching or replay (each call zeroes or allocates its output and launches).
- [x] First line still `# okc: problem=muse_bf16_proj`; no triton.autotune.

okc_lint output:
```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/bf16-restack-m16-swapab-router-swapab-m32/muse_bf16_proj/submission.py
```
`python3 -I -m py_compile` passed; __pycache__ removed from the worktree.

## Risks / what could fail
low
Restack of measured pieces; expected headline about 1.215-1.222 vs 1.196 (harvest analysis). Run as a scored experiment; if it gains, it is the board entry.

## Result (after the run)
- job `1e90292c7b954d96` (submit), status done, passed True, tests 9/9
- speedup 1.2215x, 95% CI [1.2032, 1.2398], score 3954.807696982551 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1958517819004566, "ci": [1.1779140051719497, 1.2137895586289633], "job": "eeb9985396764ac2"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.patch`; branch `hyp/muse_bf16_proj/bf16-restack-m16-swapab-router-swapab-m32`
