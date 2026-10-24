# muse_bf16_proj · 2026-10-24T070715Z · submit · router-triton-hilo-fused-control-splitk-all-m

Hypothesis: `muse_bf16_proj/hypotheses/done/gain/2026-10-23T230544Z-router-triton-hilo-fused-control-splitk-all-m.md` (branch `hyp/muse_bf16_proj/router-triton-hilo-fused-control-splitk-all-m`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `0d3532e`

## Goal
Promote check-proven gain router-triton-hilo-fused-control-splitk-all-m (1.1510x [1.1338,1.1683], check job bd9fa14151644cee) to the board.

Built from hyp/muse_bf16_proj/sglang-gemm-dispatch-parity exactly as the primary (prepare() stores router_hi and router_lo as [256, 8192] bf16 row-major; one Triton kernel for the router at every m; BLOCK_K=128, num_stages=3; acc += tl.dot(x_tile, tl.trans(hi_tile)) + tl.dot(x_tile, tl.trans(lo_tile)) in fp32; tl.atomic_add into an output kernel_fn creates with torch.zeros((m, 256), dtype=torch.float32) immediately before the launch; bf16 layers untouched on _bf16_gemm_dispatch_impl; no triton.autotune; all m-buckets compiled in prepare() on synthetic x), with one difference: the grid is 3-D, grid = (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K) at every m, each program owns one (M-block, N-block, K-slice) and never loops over M, and there is no SPLIT_K=1 plain-store path. Static configs chosen from m at launch: m <= 16: BLOCK_M=16, BLOCK_N=32, SPLIT_K=16 (K-slice 512; 1 x 8 x 16 = 128 CTAs); 16 < m <= 32: BLOCK_M=32, BLOCK_N=32, SPLIT_K=16 (128 CTAs); 32 < m <= 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=16 (2 x 4 x 16 = 128 CTAs at m=128); m > 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=8 (4 x 4 x 8 = 128 CTAs at m=256). Rows beyond m and K beyond the slice are masked, so m=7, 9 and any nearby m are valid. Output dtype/shape as before.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/router-triton-hilo-fused-control-splitk-all-m`, base a392f0b6dc (parent router-triton-hilo-fused). Only `muse_bf16_proj/submission.py` changed, and only in the router area (diff stat: 1 file, +48 / -55).

- `_router_config(m)`: new static buckets, `(BLOCK_M, BLOCK_N, SPLIT_K)`: m<=16 -> (16, 32, 16); 16<m<=32 -> (32, 32, 16); 32<m<=128 -> (64, 64, 16); m>128 -> (64, 64, 8). m=128 lands in the SPLIT_K=16 bucket (`m <= 128`), giving a (2, 4, 16) = 128-CTA grid; m=256 gives (4, 4, 8) = 128 CTAs.
- `_router_hilo_kernel`: the program id is now 3-D (pid_m, pid_n, pid_k). The `for m_start in range(0, M, BLOCK_M)` loop is gone: each program does one M-block (rows masked by `offs_m < M`). The `SPLIT_K` constexpr and the `tl.store` plain-store branch are removed, so the epilogue is always `tl.atomic_add(..., sem="relaxed")`. The K loop, the BLOCK_K=128 tile, and the parent's direct `[BLOCK_K, BLOCK_N]` addressing of the [N, K] hi/lo weights (no `tl.trans`) stay exactly as they were.
- `_router_forward`: always `torch.zeros((m, 256), fp32)` right before the launch (the `torch.empty` branch is removed); `grid = (triton.cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K)`.
- No change to: prepare() (hi/lo split, warm-up on synthetic x over ROUTER_WARM_MS = 1, 8, 16, 32, 64, 128, 256, which covers all four buckets), `_router_shape_ok` (k % (128 * SPLIT_K) still holds: 8192 % 2048 == 0), the F.linear fp32 fallback, num_warps=4, num_stages=3, no triton.autotune, the bf16 layers on `_bf16_gemm_dispatch_impl`.
- Deviation from the change paragraph: none in substance. The paragraph text says `tl.dot(x_tile, tl.trans(hi_tile))`; as the review asked, the parent's direct transposed addressing is kept instead, which computes the same thing.
- Not run on a GPU (this node has none). Untested: the compile of the BLOCK_N=32 buckets and the atomics at m=1000 (16 M-blocks x 4 x 8 = 512 CTAs).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/router-triton-hilo-fused-control-splitk-all-m diff`.

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints are added. The only branching is on `m` (a shape) via `_router_config`, and the grid comes from shapes. The `torch.zeros` output is a per-call allocation sized by shape, as in the parent.
- [x] Nothing derived from `data` is cached across calls: the output is created fresh every call, and state holds only weights and the hi/lo split.
- [x] Weights: prepare() is unchanged. BF16 layers stay BF16. The router FP32 weights run as a BF16 hi + lo pair with fp32 accumulation (`tl.dot` into an fp32 acc and fp32 atomics), which is the allowed exception.
- [x] Activations BF16 where SGLang keeps BF16: x is read as bf16, unchanged. KV cache/queries: n/a, there is no attention in this problem.
- [x] Attention sees every key: n/a, there is no attention in this problem. No skipping of any kind: every K-slice and every N-block is computed.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff only touches the kernel, config, and launch code. Imports are unchanged.
- [x] All work happens inside the timed call; no output caching or replay: the memset and kernel launch happen in kernel_fn every call. prepare() warm-up uses synthetic x only, and its outputs are discarded.
- [x] First line `# okc: problem=muse_bf16_proj` kept.
- [x] `python3 -I -m py_compile` passes.

okc_lint output (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`, exit 0):

```
CLEAN: muse_bf16_proj/submission.py
```

## Risks / what could fail
medium

## Result (after the run)
(pending)
