# muse_bf16_proj · 2026-10-24T163537Z · submit · bf16-m32-swapab-wgmma-triton

Hypothesis: `muse_bf16_proj/hypotheses/done/neutral/2026-10-24T140147Z-bf16-m32-swapab-wgmma-triton.md` (branch `hyp/muse_bf16_proj/bf16-m32-swapab-wgmma-triton`).

- **Job:** `eeb9985396764ac2` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `1afaa45`

## Goal
Promote hand-merged gain bf16-m32-swapab-wgmma-triton (1.1984x [1.1805,1.2164], check job 1602a4e947e64090) to the board (user approved: 'submit all five promotions').

Built on master (658d63b, submission blob 3f76eb8: shape-gated Triton at m<=16 + unpooler slice + 3-D split-K router; cuBLAS everywhere else). Add ONE new swap-AB Triton kernel, _bf16_swapab_kernel, and route the four bf16 layers to it at 17 <= cfg m <= 32 only (the scored m=32 rows of load decode_32); every m<=16 gate, the unpooler slice, the router and every m>=33 shape (including the m=33 o test) keep their master path. Mechanism: every m=32 Triton attempt put the tokens on tl.dot's M side, which Triton lowers to mma.sync (the pooler-splitk2-tma-weight-stream log, 4e89311eefdc4bc3: BLOCK_M=32 compiled to wgmma=0, mma.sync=64, n_regs=255, i.e. at the register cap; BLOCK_M=16 used 111 regs). cuBLAS's m=32 kernels (nvjet_sm90_tst_64x32_*) are the swapped orientation: 64 weight rows on the MMA M side, 32 tokens on N, i.e. wgmma m64n32k16. Swap the operands so Triton can do the same: program (pid_w, pid_k) owns weight rows [pid_w*64, +64) and K [pid_k*K_SLICE, +K_SLICE); in the K loop w_tile = tl.load(w_ptr + offs_w[:, None]*stride_wn + offs_k[None, :]) is [64, BLOCK_K] exactly as stored (K-contiguous, the natural K-major A operand), xt_tile = tl.load(x_ptr + offs_t[None, :]*stride_xm + offs_k[:, None], mask=t_mask[None, :], other=0) is [BLOCK_K, 32] (offs_t = arange(32), t_mask = offs_t < M), acc[64, 32] fp32 = tl.dot(w_tile, xt_tile, acc); num_warps=4 (one warpgroup, BLOCK_W=64 = wgmma M). The accumulator is 16 fp32 registers per thread instead of the weight being staged through registers. Epilogue, SPLIT_K == 1 (constexpr): tl.store(out_ptr + offs_t[None, :]*stride_om + offs_w[:, None], acc.to(bf16), mask=t_mask[None, :]) (a transposed store of a 64x32 tile into out [m, n] row-major). SPLIT_K > 1: master's ticket-semaphore reduce from _bf16_splitk_kernel verbatim except that the fp32 workspace tile is addressed transposed (ws[pid_k] + offs_t[None, :]*N + offs_w[:, None], masked by t_mask) and the counter is per weight tile: masked partial store, tl.debug_barrier(), one gpu-scope acq_rel tl.atomic_add ticket on cnt[pid_w], tl.debug_barrier(), the CTA drawing SPLIT_K-1 reloads all partials with .cg, sums 0..SPLIT_K-1 in order, stores bf16 transposed, tl.atomic_xchg(cnt, 0). Per-layer static configs (from cfg layer only): pooler (n 4096, k 8192) SPLIT_K 2, grid (64, 2) = 128 CTAs, BLOCK_K 256, num_stages 4 (48 KB per stage: 32 KB weight + 16 KB x, 192 KB; retry at 3 stages in prepare() on OutOfResources and record it as a constant); qkv_gate (2304, 8192) SPLIT_K 4, grid (36, 4) = 144 CTAs, BLOCK_K 128, num_stages 4; unpooler full width (8192, 4096) SPLIT_K 1, grid (128, 1), BLOCK_K 256, num_stages 4; o (8192, 1024) SPLIT_K 1, grid (128, 1), BLOCK_K 256, num_stages 4 (4 K-steps, all in flight). The shape gate checks layer, 17 <= cfg m <= 32, cfg n/k equal to those widths, x bf16 [m, k] with stride(1) == 1 and the weight's shape; place it after master's four m<=16 gates and before the _bf16_gemm_dispatch_impl fallback. prepare(): allocate state['swapab_ws_pooler'] fp32 [2, 32, 4096], state['swapab_ws_qkv'] fp32 [4, 32, 2304] and int32 zeroed counters [64] and [36] (own keys, not shared with the m<=16 paths), warm every m in 17..32 for each of the four layers on synthetic x, re-zero the counters afterwards, and print per layer (diagnostic only, try/except) n_regs, n_spills and the PTX counts of 'wgmma.mma_async' and 'mma.sync' from the launch handle's asm['ptx']. No autotune, nothing derived from data kept.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/bf16-m32-swapab-wgmma-triton`, `muse_bf16_proj/submission.py` only (+185 lines, 0 removed; uncommitted).

- New section (5) after the o skinny path: constants `SWAPAB_*`, `SWAPAB_CFG` (layer -> N, K, SPLIT_K, BLOCK_K, stages: pooler 4096/8192/2/256/4, qkv_gate 2304/8192/4/128/4, unpooler 8192/4096/1/256/4, o 8192/1024/1/256/4), BLOCK_W 64, BLOCK_T 32, num_warps 4.
- `_bf16_swapab_kernel` (`do_not_specialize=["M"]`): grid (N // 64, SPLIT_K); w_tile [64, BLOCK_K] loaded as stored, xt_tile [BLOCK_K, 32] = x^T masked by `t_mask[None, :]` (other=0), `acc = tl.dot(w_tile, xt_tile, acc)` into fp32 [64, 32]. `SPLIT_K == 1` (constexpr branch): transposed masked bf16 store into out[m, n]. `SPLIT_K > 1`: master's ticket reduce verbatim except the ws tile is addressed transposed (`offs_t[None, :]*N + offs_w[:, None]`, WS_M = 32) and the counter is per weight tile `cnt[pid_w]` (store, debug_barrier, acq_rel gpu-scope atomic_add, debug_barrier, last ticket reloads .cg and sums 0..SPLIT_K-1 in order, bf16 store, atomic_xchg 0).
- `_swapab_route_ok`: layer in SWAPAB_CFG, 17 <= cfg m <= 32, cfg n/k equal to the layer widths, x 2-D bf16 [m, k] with stride(1) == 1, weight bf16 (n, k) with stride(1) == 1. In kernel_fn it sits after the four master gates and before the `_bf16_gemm_dispatch_impl` fallback, and also requires the layer to have been warmed (`layer in state["swapab_stages"]`). m=33 (o test) and every m <= 16 / m >= 128 shape fall through to master's paths unchanged.
- `_swapab_prepare` (called at the end of prepare(), after master's counter re-zeroing): allocates `swapab_ws_pooler` fp32 [2, 32, 4096], `swapab_ws_qkv` fp32 [4, 32, 2304], `swapab_cnt_pooler` int32 [64], `swapab_cnt_qkv` int32 [36] (own keys); for each layer launches m=32 first with report=True (prints n_regs, n_spills and PTX counts of `wgmma.mma_async`, `mma.sync` and `cp.async` from `compiled.asm["ptx"]`, in try/except), then warms m = 17..31 on synthetic randn x; re-zeroes both counters afterwards. Unpooler/o pass `out` as the unused ws/cnt pointers at SPLIT_K == 1.
- Module docstring gained one paragraph describing the new path.

Deviations / implementation choices:
- The OutOfResources retry at 3 stages applies to every layer, not only the pooler: unpooler and o have the same 48 KB/stage x 4 = 192 KB footprint, so they share the failure mode. The chosen stage count per layer is recorded in `state["swapab_stages"]` (a constant from compilation, nothing from data) and printed when the retry fires. `OutOfResources` is imported from `triton.runtime.errors` with a getattr fallback.
- The PTX printout also counts `cp.async` (adversarial review open question 1: whether the masked transposed x load stays on the async-copy pipeline).
- Review open question 4 (masks at m < 32, every m 17..31 warmed): all of 17..32 are warmed per layer. No correctness test exercises 17..31, so these warm-ups prove compile/launch only, not values.
- Not verifiable here (no GPU/Triton): whether Triton actually emits wgmma for this dot, the register count, and whether the transposed epilogue adds an smem layout conversion that pushes past 227 KB at 4 stages (the OOR fallback covers that).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/bf16-m32-swapab-wgmma-triton diff`.

- [x] CUDA-graph capturable: the timed path (`_swapab_route_ok`, `_swapab_buffers`, `_swapab_forward` with report=False) has no `.item()`/`.cpu()`/`synchronize()`/tensor prints; it allocates only the output via torch.empty (as master's paths do, graph-safe); it branches only on cfg layer/m/n/k, tensor shapes/dtypes/strides and the prepare()-time `swapab_stages` dict. The diagnostic print runs only with report=True from prepare().
- [x] Nothing derived from `data` is cached across calls: the new state keys (workspaces, counters, stage counts) are built in prepare() from weights and synthetic x only; the workspace is overwritten every split-K call, the reducer reads only rows written in that launch (same t_mask), and counters self-reset.
- [x] Weights: no repack; the bf16 weights are read in place as bf16 (state[layer], unchanged). Router untouched.
- [x] Activations BF16 where SGLang keeps BF16: x is read as bf16 with fp32 accumulation (and fp32 split-K partials), and the output is stored bf16, as in master's split-K kernels.
- [x] n/a: attention: this problem has no attention (dense projections only).
- [x] n/a: KV cache / FP8 queries: none in this problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds only a Triton kernel, host wrappers and a `triton.runtime.errors` import.
- [x] All work happens inside the timed call; no output caching or replay: every timed m=17..32 call launches the full GEMM on the fresh x; prepare() warm-up outputs are dropped.
- [x] First-line marker `# okc: problem=muse_bf16_proj` unchanged; no triton.autotune.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/bf16-m32-swapab-wgmma-triton/muse_bf16_proj/submission.py
exit=0
```

`python3 -I -m py_compile` OK; `__pycache__` removed from the worktree.

## Risks / what could fail
medium

## Result (after the run)
(pending)
