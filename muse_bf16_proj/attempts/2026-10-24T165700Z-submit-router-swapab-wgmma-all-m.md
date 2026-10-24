# muse_bf16_proj · 2026-10-24T165700Z · submit · router-swapab-wgmma-all-m

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T165036Z-router-swapab-wgmma-all-m.md` (branch `hyp/muse_bf16_proj/router-swapab-wgmma-all-m`).

- **Job:** `587e340476944b46` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `ced97b5`

## Goal
Built on master (8b9ad43). Master's router is _router_hilo_kernel with the tokens on tl.dot's M side: grid (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K), (BLOCK_M, BLOCK_N, SPLIT_K) = (16, 32, 16) at m<=16, (32, 32, 16) at m<=32, (64, 64, 16) at m<=128, (64, 64, 8) above, acc += x_tile @ hi_tile; acc += x_tile @ lo_tile, fp32 atomic_add into a torch.zeros output. At m<=32 the 16/32-row token tile cannot lower to wgmma (needs 64 M rows per warpgroup; the bf16 m=32 run measured wgmma=0 for tokens-on-M), and at m=128/256 the M-blocking re-reads every hi/lo weight tile 2x / 4x (grid M dim 2 / 4). Change: replace the router launch with a new _router_swapab_kernel, the router analogue of master's _bf16_swapab_kernel: program (pid_t, pid_w, pid_k) loads hi and lo tiles [BLOCK_W=64, BLOCK_K] of the [256, 8192] row-major router_hi / router_lo (as stored, K-contiguous) as tl.dot's LHS and the x^T tile [BLOCK_K, BLOCK_T] (tokens >= m masked to 0) as the RHS, acc[64, BLOCK_T] fp32 = tl.dot(hi, xt, acc); acc = tl.dot(lo, xt, acc), then fp32 tl.atomic_add of acc transposed into out[m, 256] (masked on tokens < m), out zeroed right before the launch exactly as master. Static configs from m only (BLOCK_T, SPLIT_K, BLOCK_K): m<=16: (16, 32, 128), grid (1, 4, 32) = 128 CTAs, K-slice 256; m<=32: (32, 32, 128), (1, 4, 32); m<=128: (128, 32, 128), (1, 4, 32), each weight tile read once instead of twice; m>128: (128, 16, 128), grid (cdiv(m,128), 4, 16) = 128 CTAs at m=256, K-slice 512, weight read 2x instead of 4x. num_warps 4 (one warpgroup), num_stages 3 (2 if OutOfResources at compile in prepare; recorded as a constant). Keep master's hi/lo split, _router_shape_ok preconditions (adapted to the new K-slice divisibility), the F.linear fp32 fallback and every bf16 layer path. prepare(): warm master's ROUTER_WARM_MS plus every bucket, and print per bucket n_regs, n_spills, PTX counts of wgmma.mma_async and mma.sync for the new kernel AND for master's _router_hilo_kernel at the same m, so the log shows the orientation change is what moved the MMA path.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/router-swapab-wgmma-all-m`, `muse_bf16_proj/submission.py` only (`git diff --stat`: 1 file, +162 / -30). Only the router path and its prepare() warm-up changed. Every bf16 layer path is untouched.

- **New `_router_swapab_kernel`** (do_not_specialize M). Grid `(cdiv(m, BLOCK_T), 256 // 64, SPLIT_K)`, program `(pid_t, pid_w, pid_k)`. The hi and lo tiles `[64, 128]` are loaded as stored (K-contiguous) and used as tl.dot's LHS. The x^T tile `[128, BLOCK_T]` is masked on tokens < m with other=0 and used as the RHS. Each K-step does `acc = tl.dot(hi, xt, acc); acc = tl.dot(lo, xt, acc)`, the same hi-then-lo order as master. The epilogue is one relaxed fp32 `tl.atomic_add` of acc, transposed, into `out[m, 256]`, masked on tokens < m. `out` is still `torch.zeros` right before the launch, as on master.
- **`_router_config(m)` now returns (BLOCK_T, SPLIT_K)**, with BLOCK_K 128 and BLOCK_W 64 fixed:
  - m<=16: (16, 32), grid (1,4,32) = 128 CTAs, K-slice 256. As specced.
  - m<=32: (32, 32), grid (1,4,32), K-slice 256. As specced.
  - m<=128: (64, 16), grid (2,4,16) = 128 CTAs at m=128, K-slice 512. **Deviation**, see below.
  - m>128: (64, 8), grid (4,4,8) = 128 CTAs at m=256, K-slice 1024. **Deviation**, see below.
- **Deviation at m>32.** The brief and review say to keep master's SPLIT_K at m=128/256 so the atomic count stays m*256*SPLIT_K, not doubled. If BLOCK_T stayed at 128, keeping SPLIT_K 16/8 would halve the grid to 64 CTAs. That adds a second confound (parallelism), as large as the atomic one. So these buckets use the review's 128-CTA option: BLOCK_T 64, grid (2,4,16) at m<=128 and (4,4,8) above. Atomic count, CTA count, K-slice and weight re-read factor (2x/4x) are all master's. The only change at m=128/256 is MMA orientation (weight on M, tokens on N). The hypothesis's "weight read once / 2x" claim at m=128/256 is therefore **dropped**, and those rows test orientation alone. Expect them flat unless the print shows master's BLOCK_M=64 tile not on wgmma, or unless register pressure or atomic coalescing differ.
- **`_router_shape_ok`** now checks `k % (128 * SPLIT_K) == 0` with the new SPLIT_K (8192 passes for 32/16/8) and `n % 64 == 0`. The F.linear fp32 fallback is unchanged. kernel_fn also takes it if a bucket has no recorded stage count (never expected).
- **Stage count.** Each bucket tries num_stages 3 in prepare(). On `_OutOfResources` it retries at 2 (`ROUTER_FALLBACK_STAGES`). The result is stored per bucket in `state["router_stages"]` (a constant keyed by the static config tuple), and kernel_fn looks it up from m. The fallback lives in prepare() only.
- **Diagnostics, prepare() only.** `_router_prepare` warms every m in master's `ROUTER_WARM_MS` (1, 8, 16, 32, 64, 128, 256), which covers all four buckets. At each bucket's first m it prints two lines: `[router] swapab ...` for the new kernel, and `[router] master hilo ...`, where `_router_hilo_report` compiles and launches master's `_router_hilo_kernel` with master's config (`_router_hilo_config`) into a dropped scratch output. Each line gives n_regs, n_spills and PTX counts of `wgmma.mma_async`, `mma.sync`, `red.global`, `atom.global` and `cp.async`. The master diagnostic is wrapped in try/except and never runs in kernel_fn. The new kernel's launch sits outside the try, so a real compile failure still surfaces.
- The module docstring router paragraph was updated.
- **Open questions left for the run log.** The four review questions are answered by the print: whether master's BLOCK_M=64 tile already emits wgmma, master's n_regs at BLOCK_M=32, whether the atomics come out as red.global (vectorized or not, which needs a closer read of the PTX if counts look scalar-high), and whether the masked x^T loads are still cp.async.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/router-swapab-wgmma-all-m diff`:

- [x] CUDA-graph capturable: kernel_fn adds no `.item()`, `.cpu()`, `synchronize()` or tensor prints. The router branch uses only shapes (`x.shape[0]` via `_router_config`, cfg n) and a prepare()-time constant dict. The only allocation is the `torch.zeros` output that master already had. Prints happen only in prepare() (report=True and `_router_hilo_report`).
- [x] Nothing derived from `data` is cached across calls: `state["router_stages"]` comes from prepare() on synthetic x. kernel_fn writes nothing to state.
- [x] Weights: the router hi + lo bf16 split is unchanged (the allowed FP32 → BF16 hi+lo with FP32 accumulation exception). The bf16 weights are untouched.
- [x] Activations BF16 where SGLang keeps BF16: x is still loaded as bf16, with fp32 accumulation and fp32 output for the router. n/a for KV cache and queries (none in this problem).
- [x] Attention sees every key: n/a, no attention in this problem. The GEMM covers the full K (SPLIT_K * K_SLICE = 8192) and all 256 columns.
- [x] No timing patches, harness frames, I/O, secrets, module edits, reference import, subprocesses or network: no new imports, and nothing outside the kernel file is touched.
- [x] All work happens inside the timed call, with no output caching or replay. Every call zeroes `out` and recomputes all K-partials. The master-kernel diagnostic in prepare() writes to a dropped scratch tensor.
- [x] No triton.autotune. The `# okc: problem=muse_bf16_proj` first line is kept.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`):

```
CLEAN: muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` passed. The `__pycache__` it left in the worktree was removed.

## Risks / what could fail
medium
LESSONS says the router's small-m grid depth is settled (SPLIT_K 32 and stage counts flat); this changes the MMA orientation and the weight re-read, not the grid depth, but at m<=16 it also moves SPLIT_K 16 -> 32 and K-slice 512 -> 256, which the split-K-32 runs found flat, so a small-m loss would point at the orientation. The 0.7 us vectorized_elementwise memset stays (removing it was flat: router-splitk-semaphore-no-memset). Router accuracy bound is 0.01%: the hi+lo fp32 accumulation is unchanged, only operand order and atomic order differ (atomic order is already nondeterministic on master). At m=128/256 the x^T tile [128, 128] bf16 is 32 KB per stage, acc 64x128 fp32 = 64 regs/thread: watch spills in the print.

## Result (after the run)
- job `587e340476944b46` (submit), status done, passed True, tests 9/9
- speedup 1.1951x, 95% CI [1.1772, 1.2131], score 4042.0790381904117 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1958517819004566, "ci": [1.1779140051719497, 1.2137895586289633], "job": "eeb9985396764ac2"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-swapab-wgmma-all-m.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-swapab-wgmma-all-m.patch`; branch `hyp/muse_bf16_proj/router-swapab-wgmma-all-m`
