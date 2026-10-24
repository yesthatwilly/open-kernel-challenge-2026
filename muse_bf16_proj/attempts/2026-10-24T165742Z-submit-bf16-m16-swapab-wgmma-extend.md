# muse_bf16_proj · 2026-10-24T165742Z · submit · bf16-m16-swapab-wgmma-extend

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T165012Z-bf16-m16-swapab-wgmma-extend.md` (branch `hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend`).

- **Job:** `89fabef70bd745f7` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `bdeadf4`

## Goal
Built on master (8b9ad43, submission blob a8bb42a). Master's m<=16 Triton paths all put the tokens on tl.dot's M side with BLOCK_M=16 (pooler _bf16_splitk_kernel, qkv_gate _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler slice _bf16_splitk_kernel at n==1024, m<=8). Triton's MMAv3 (wgmma) needs a 64-row M tile per warpgroup, so these lower to mma.sync; at m=32 the same orientation gave wgmma=0 and swapping it to weight-on-LHS gave wgmma and +3.6..9.3% (bf16-m32-swapab-wgmma-triton). Change: route those same m<=16 shapes to master's existing _bf16_swapab_kernel (weight rows [BLOCK_W=64, BLOCK_K] on tl.dot's LHS, x^T [BLOCK_K, BLOCK_T] on the RHS) with BLOCK_T=16 (wgmma m64n16k16; token columns >= m masked; m=1 and m=8 run the same BLOCK_T=16 compile, M is not specialized). Per-layer config, same as master's swap-AB table except BLOCK_T and the unpooler slice: pooler (N 4096, K 8192) SPLIT_K 2, BLOCK_K 256, 4 stages, grid (64, 2); qkv_gate (2304, 8192) SPLIT_K 4, BLOCK_K 128, 4 stages, grid (36, 4); o (8192, 1024) SPLIT_K 1, BLOCK_K 256, 4 stages, grid (128, 1); unpooler slice (n == 1024, K 4096, w = the first 1024 rows of the full weight read in place, stride_wn 4096) SPLIT_K 8, BLOCK_K 128, 3 stages, grid (16, 8). Add SWAPAB16_CFG with those rows and new prepare()-allocated workspaces/tickets under own keys (ws fp32 [SPLIT_K, 16, n], cnt int32 [n // 64] zeroed) so master's m=17..32 buffers are untouched; WS_M=16. Gate in kernel_fn BEFORE master's m<=16 branches: layer in {pooler, qkv_gate, o} and 1 <= cfg m <= 16 with master's n/k, or layer == unpooler and cfg n == 1024 (m <= 8); x bf16, stride(1) == 1. The unpooler full weight at m=16 (cuBLAS nvjet) and every m>=17 shape keep master. prepare(): warm every m in 1..16 per layer (1..8 for the unpooler slice) on synthetic x, re-zero counters, and print per layer the report=True line (n_regs, n_spills, wgmma, mma.sync, cp.async), plus the same PTX counts for master's incumbent m<=16 kernels compiled at m=16, so the log confirms the incumbents are mma.sync and the new path wgmma. If a 4-stage variant is OutOfResources, step down to 3 (master's fallback). Master's old m<=16 kernels stay in the file but are no longer routed for these shapes.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/bf16-m16-swapab-wgmma-extend`, `muse_bf16_proj/submission.py` only. The diff only adds lines (+203, 0 removed); every line of master's code is still there.

- New section `(6)` after `_swapab_prepare`:
  - `SWAPAB16_CFG` uses the keys `pooler` (4096, 8192, SK 2, BK 256, 4 st), `qkv_gate` (2304, 8192, SK 4, BK 128, 4 st), `o` (8192, 1024, SK 1, BK 256, 4 st) and `unpooler_slice` (1024, 4096, SK 8, BK 128, 3 st).
  - `SWAPAB16_BLOCK_T = SWAPAB16_WS_M = 16`.
- `_swapab16_forward` is a separate launcher for the unchanged `_bf16_swapab_kernel`. It passes `WS_M=16` and `BLOCK_T=16`, as the review REQUIRED. Master's `_swapab_forward`, which hardcodes 32/32, is not reused and not modified. The unpooler slice passes the full `[8192, 4096]` weight in place (`stride_wn = w.stride(0) = 4096`), with grid `(1024 // 64, 8)`, so only the first 1024 rows are read.
- `_swapab16_route_ok` is a separate gate. Master's `_swapab_route_ok` (m >= 17, full weight shape) is not used here.
  - It requires `1 <= cfg m <= 16`, the key's cfg n/k, `x.shape == (m, k)`, x bf16 and `stride(1) == 1`, and w bf16 with `stride(1) == 1`.
  - pooler, qkv_gate and o also need `w.shape == (n, k)`. The unpooler slice needs `w.shape[0] >= 1024` and `w.shape[1] == 4096`.
  - `_swapab16_key` maps the unpooler to `unpooler_slice` only when `cfg n == 1024`. So the unpooler's full width at m = 16 (cuBLAS) and m = 130 / 1000 keep master.
- Buffers have their own keys: `swapab16_ws_{pooler,qkv_gate,unpooler_slice}` fp32 `[SPLIT_K, 16, n]` and `swapab16_cnt_*` int32 `[n // 64]`, zeroed. Master's `swapab_*`, `pooler_*`, `qkv_*` and `unpool_*` buffers are untouched by the routed path.
- `_swapab16_prepare`:
  - It warms m = 1..16 per layer (1..8 for the slice) on synthetic randn x. The first launch is at the top m with `report=True`.
  - On OutOfResources it steps down to 3 stages (`SWAPAB_FALLBACK_STAGES`), and records `state["swapab16_stages"]`.
  - It then calls `_incumbent16_report`, which relaunches master's four incumbent m <= 16 kernels once each at m = 16 on zero x, with the same constants and their own buffers, and prints `n_regs / n_spills / wgmma / mma.sync / cp.async` as `[incumbent16] ...` lines. After that it re-zeroes every counter it touched.
  - The new path prints `[swapab16] <key> m=...: ...` lines.
- `kernel_fn` checks the new gate first, before master's m <= 16 branches. Master's m <= 16 kernels stay in the file but are no longer reached for these shapes.
- The module docstring has a paragraph describing the new path.
- Deviation 1: the slice gate accepts `1 <= m <= 16` rather than only m <= 8. `cfg n == 1024` only occurs at m <= 8. The 16-row workspace covers m <= 16, and M is not specialized (`do_not_specialize=["M"]`; `stride_xm = k` and `stride_om = n` are shape constants), so any m 9..16 would hit the same compile. The warm-up is 1..8 as specified.
- Deviation 2: the incumbent PTX report relaunches the kernels directly instead of changing master's forward helpers to return the compiled handle. This keeps master's functions byte-identical.
- Open questions from the review:
  - wgmma at n16: the `[swapab16]` lines answer it in the log.
  - Same cache entry for m = 1 and m = 16: yes by construction. M is in `do_not_specialize` and the strides are shape constants. The full 1..16 warm-up also guarantees no compile in a timed call even if Triton specialized further.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/bf16-m16-swapab-wgmma-extend diff`.

- [x] CUDA-graph capturable. There is no `.item()`, `.cpu()`, `synchronize()` or tensor print in `kernel_fn` or `_swapab16_forward`. Prints are only in prepare(), and they print compile metadata, not tensors. The only allocation in the timed path is `torch.empty` of the output, whose shape comes from `x.shape` and the static cfg, never from input values. Branches are on cfg layer, m, n, k and on shapes, dtypes and strides.
- [x] Nothing derived from `data` is cached across calls. The workspaces are overwritten every call, and the reducer reads only token rows < M written in the same launch. The counters self-reset. Warm-ups use synthetic x.
- [x] Weights: no repack and no precision change. bf16 weights are read in place, and the unpooler slice reads the full weight's first 1024 rows via its stride. The router is untouched.
- [x] Activations stay bf16. Accumulation is fp32 in the MMA. Split-K partials are fp32, summed in fixed order 0..SPLIT_K-1, and stored bf16 once, the same as the incumbent.
- [x] n/a: attention. This problem has no attention, KV cache or queries.
- [x] No timing patches, harness frames or I/O, secrets, module edits, reference import, subprocesses or network. The diff only adds Triton launches, torch allocations and prepare-time prints.
- [x] All work happens inside the timed call, with no output caching or replay. Each call launches the full GEMM on its fresh x.
- [x] First-line marker `# okc: problem=muse_bf16_proj` kept (line 1 unchanged).
- [x] No `triton.autotune`. The config is static per layer key.
- [x] Workspace sizing: the `[SPLIT_K, 16, n]` workspace is indexed with `WS_M=16` and the tokens are masked `< M <= 16`, so nothing is written out of bounds. This is the review's REQUIRED fix.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/bf16-m16-swapab-wgmma-extend/muse_bf16_proj/submission.py
lint rc=0
```

`python3 -I -m py_compile` passed, and the leftover `__pycache__` was removed from the worktree.

## Risks / what could fail
medium
At m<=16 these shapes are HBM-bound, not MMA-bound, so the m=32 gain (which removed 255-reg tokens-on-M spills) may not transfer: m=16 incumbents already run 96-ish regs. The plausible win is the async wgmma letting more loads stay in flight and fewer registers per CTA; the prepare() PTX/regs print attributes it either way. wgmma n16 wastes nothing at m=16 but computes 15 dead columns at m=1 (free under the BW bound). Per-shape judging: if only some layers gain, a combo keeps those (as with bf16-shape-gated-triton-winners-combo).

## Result (after the run)
- job `89fabef70bd745f7` (submit), status done, passed True, tests 9/9
- speedup 1.2115x, 95% CI [1.1934, 1.2297], score 3987.29254440901 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1958517819004566, "ci": [1.1779140051719497, 1.2137895586289633], "job": "eeb9985396764ac2"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-m16-swapab-wgmma-extend.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-m16-swapab-wgmma-extend.patch`; branch `hyp/muse_bf16_proj/bf16-m16-swapab-wgmma-extend`
