---
problem: muse_bf16_proj
slug: bf16-m32-swapab-wgmma-triton
created: 2026-10-24T140147Z
base: master
area: bf16 GEMM dispatch (m=17..32, swap-AB Triton)
change: >
  Built on master (658d63b, submission blob 3f76eb8: shape-gated Triton at m<=16 + unpooler slice
  + 3-D split-K router; cuBLAS everywhere else). Add ONE new swap-AB Triton kernel,
  _bf16_swapab_kernel, and route the four bf16 layers to it at 17 <= cfg m <= 32 only (the scored
  m=32 rows of load decode_32); every m<=16 gate, the unpooler slice, the router and every m>=33
  shape (including the m=33 o test) keep their master path. Mechanism: every m=32 Triton attempt
  put the tokens on tl.dot's M side, which Triton lowers to mma.sync (the
  pooler-splitk2-tma-weight-stream log, 4e89311eefdc4bc3: BLOCK_M=32 compiled to wgmma=0,
  mma.sync=64, n_regs=255, i.e. at the register cap; BLOCK_M=16 used 111 regs). cuBLAS's m=32
  kernels (nvjet_sm90_tst_64x32_*) are the swapped orientation: 64 weight rows on the MMA M side,
  32 tokens on N, i.e. wgmma m64n32k16. Swap the operands so Triton can do the same: program
  (pid_w, pid_k) owns weight rows [pid_w*64, +64) and K [pid_k*K_SLICE, +K_SLICE); in the K loop
  w_tile = tl.load(w_ptr + offs_w[:, None]*stride_wn + offs_k[None, :]) is [64, BLOCK_K] exactly
  as stored (K-contiguous, the natural K-major A operand), xt_tile = tl.load(x_ptr + offs_t[None,
  :]*stride_xm + offs_k[:, None], mask=t_mask[None, :], other=0) is [BLOCK_K, 32] (offs_t =
  arange(32), t_mask = offs_t < M), acc[64, 32] fp32 = tl.dot(w_tile, xt_tile, acc); num_warps=4
  (one warpgroup, BLOCK_W=64 = wgmma M). The accumulator is 16 fp32 registers per thread instead
  of the weight being staged through registers. Epilogue, SPLIT_K == 1 (constexpr):
  tl.store(out_ptr + offs_t[None, :]*stride_om + offs_w[:, None], acc.to(bf16), mask=t_mask[None,
  :]) (a transposed store of a 64x32 tile into out [m, n] row-major). SPLIT_K > 1: master's
  ticket-semaphore reduce from _bf16_splitk_kernel verbatim except that the fp32 workspace tile is
  addressed transposed (ws[pid_k] + offs_t[None, :]*N + offs_w[:, None], masked by t_mask) and the
  counter is per weight tile: masked partial store, tl.debug_barrier(), one gpu-scope acq_rel
  tl.atomic_add ticket on cnt[pid_w], tl.debug_barrier(), the CTA drawing SPLIT_K-1 reloads all
  partials with .cg, sums 0..SPLIT_K-1 in order, stores bf16 transposed, tl.atomic_xchg(cnt, 0).
  Per-layer static configs (from cfg layer only): pooler (n 4096, k 8192) SPLIT_K 2, grid (64, 2)
  = 128 CTAs, BLOCK_K 256, num_stages 4 (48 KB per stage: 32 KB weight + 16 KB x, 192 KB; retry at
  3 stages in prepare() on OutOfResources and record it as a constant); qkv_gate (2304, 8192)
  SPLIT_K 4, grid (36, 4) = 144 CTAs, BLOCK_K 128, num_stages 4; unpooler full width (8192, 4096)
  SPLIT_K 1, grid (128, 1), BLOCK_K 256, num_stages 4; o (8192, 1024) SPLIT_K 1, grid (128, 1),
  BLOCK_K 256, num_stages 4 (4 K-steps, all in flight). The shape gate checks layer, 17 <= cfg m
  <= 32, cfg n/k equal to those widths, x bf16 [m, k] with stride(1) == 1 and the weight's shape;
  place it after master's four m<=16 gates and before the _bf16_gemm_dispatch_impl fallback.
  prepare(): allocate state['swapab_ws_pooler'] fp32 [2, 32, 4096], state['swapab_ws_qkv'] fp32
  [4, 32, 2304] and int32 zeroed counters [64] and [36] (own keys, not shared with the m<=16
  paths), warm every m in 17..32 for each of the four layers on synthetic x, re-zero the counters
  afterwards, and print per layer (diagnostic only, try/except) n_regs, n_spills and the PTX
  counts of 'wgmma.mma_async' and 'mma.sync' from the launch handle's asm['ptx']. No autotune,
  nothing derived from data kept.
expected:
  - shapes: layer=qkv_gate load=decode_32 m=32
    direction: faster
    why: 14.1 us today (nvjet 64x32 splitK 8.8 + splitKreduce 1.8) vs floor 8.0; one launch, wgmma at the 144-CTA split-K grid that ran 12.3 us at m=16; target <= 12.8 us
  - shapes: layer=pooler load=decode_32 m=32
    direction: faster
    why: 21.6 us (nvjet 20.2 + reduce 2.0) vs floor 14.1; the m<=16 Triton grid reached 19.2-19.6 us; with wgmma and no 255-register tile the m=32 row should land there; target <= 20.3 us
  - shapes: layer=unpooler load=decode_32 m=32
    direction: faster
    why: 18.9 us single nvjet vs floor 14.1; mma.sync Triton tiles lost 0.91-0.97x here; target <= 18.0 us
  - shapes: layer=o load=decode_32 m=32
    direction: faster
    why: 7.0 us vs floor 3.6; the m<=16 Triton o kernel ran 4.4-4.7 us of kernel; mma.sync at m=32 ran 5.8; target <= 6.4 us shape time
  - shapes: "* m=1"
    direction: flat
    why: untouched (m<=16 gates run first)
  - shapes: "* m=16"
    direction: flat
    why: untouched
  - shapes: "* m=8"
    direction: flat
    why: untouched
  - shapes: "* m=128"
    direction: flat
    why: untouched (cuBLAS)
  - shapes: "* m=256"
    direction: flat
    why: untouched (cuBLAS)
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T140535Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T140535Z
base_sha: f19e0a9125db56de07444e6a99a642c35003206d
base_blob: 3f76eb805990314f5f7e6de6e9e184464edd6019
branch: hyp/muse_bf16_proj/bf16-m32-swapab-wgmma-triton
worktree: .worktrees/muse_bf16_proj/bf16-m32-swapab-wgmma-triton
mode: check
implemented_at: 2026-10-24T140833Z
impl_commit: a2d2ac55d2c58a0bb3c4fe22aca6cd0a865348d2
submission_sha256: 25743a32c3f31821799b83a7f01c26ffbd2b4dc176506448656f2f22ecafeddc
norm_sha256: a5967dba7107295104e4765ff04ecf29a50657f2390c1346f64e9da131bfc0a9
submission_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
lint_warnings: 0
attempt_file: 2026-10-24T140833Z-check-bf16-m32-swapab-wgmma-triton.md
prepared_at: 2026-10-24T140833Z
job: 1602a4e947e64090
dispatched_at: 2026-10-24T140843Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.1984474737274622
speedup_ci: [1.1804707616215502, 1.216424185833374]
harvested_at: 2026-10-24T144810Z
best_at_harvest: 1.1760470299345995
promotion: submitted
promotion_commit: 8b9ad430b2ed965f732443cb8de2069b58f43366
promotion_attempt: 2026-10-24T163537Z-submit-bf16-m32-swapab-wgmma-triton.md
promotion_job: eeb9985396764ac2
promotion_at: 2026-10-24T163551Z
---

# bf16-m32-swapab-wgmma-triton

## Change

Built on master (658d63b, submission blob 3f76eb8: shape-gated Triton at m<=16 + unpooler slice + 3-D split-K router; cuBLAS everywhere else). Add ONE new swap-AB Triton kernel, _bf16_swapab_kernel, and route the four bf16 layers to it at 17 <= cfg m <= 32 only (the scored m=32 rows of load decode_32); every m<=16 gate, the unpooler slice, the router and every m>=33 shape (including the m=33 o test) keep their master path. Mechanism: every m=32 Triton attempt put the tokens on tl.dot's M side, which Triton lowers to mma.sync (the pooler-splitk2-tma-weight-stream log, 4e89311eefdc4bc3: BLOCK_M=32 compiled to wgmma=0, mma.sync=64, n_regs=255, i.e. at the register cap; BLOCK_M=16 used 111 regs). cuBLAS's m=32 kernels (nvjet_sm90_tst_64x32_*) are the swapped orientation: 64 weight rows on the MMA M side, 32 tokens on N, i.e. wgmma m64n32k16. Swap the operands so Triton can do the same: program (pid_w, pid_k) owns weight rows [pid_w*64, +64) and K [pid_k*K_SLICE, +K_SLICE); in the K loop w_tile = tl.load(w_ptr + offs_w[:, None]*stride_wn + offs_k[None, :]) is [64, BLOCK_K] exactly as stored (K-contiguous, the natural K-major A operand), xt_tile = tl.load(x_ptr + offs_t[None, :]*stride_xm + offs_k[:, None], mask=t_mask[None, :], other=0) is [BLOCK_K, 32] (offs_t = arange(32), t_mask = offs_t < M), acc[64, 32] fp32 = tl.dot(w_tile, xt_tile, acc); num_warps=4 (one warpgroup, BLOCK_W=64 = wgmma M). The accumulator is 16 fp32 registers per thread instead of the weight being staged through registers. Epilogue, SPLIT_K == 1 (constexpr): tl.store(out_ptr + offs_t[None, :]*stride_om + offs_w[:, None], acc.to(bf16), mask=t_mask[None, :]) (a transposed store of a 64x32 tile into out [m, n] row-major). SPLIT_K > 1: master's ticket-semaphore reduce from _bf16_splitk_kernel verbatim except that the fp32 workspace tile is addressed transposed (ws[pid_k] + offs_t[None, :]*N + offs_w[:, None], masked by t_mask) and the counter is per weight tile: masked partial store, tl.debug_barrier(), one gpu-scope acq_rel tl.atomic_add ticket on cnt[pid_w], tl.debug_barrier(), the CTA drawing SPLIT_K-1 reloads all partials with .cg, sums 0..SPLIT_K-1 in order, stores bf16 transposed, tl.atomic_xchg(cnt, 0). Per-layer static configs (from cfg layer only): pooler (n 4096, k 8192) SPLIT_K 2, grid (64, 2) = 128 CTAs, BLOCK_K 256, num_stages 4 (48 KB per stage: 32 KB weight + 16 KB x, 192 KB; retry at 3 stages in prepare() on OutOfResources and record it as a constant); qkv_gate (2304, 8192) SPLIT_K 4, grid (36, 4) = 144 CTAs, BLOCK_K 128, num_stages 4; unpooler full width (8192, 4096) SPLIT_K 1, grid (128, 1), BLOCK_K 256, num_stages 4; o (8192, 1024) SPLIT_K 1, grid (128, 1), BLOCK_K 256, num_stages 4 (4 K-steps, all in flight). The shape gate checks layer, 17 <= cfg m <= 32, cfg n/k equal to those widths, x bf16 [m, k] with stride(1) == 1 and the weight's shape; place it after master's four m<=16 gates and before the _bf16_gemm_dispatch_impl fallback. prepare(): allocate state['swapab_ws_pooler'] fp32 [2, 32, 4096], state['swapab_ws_qkv'] fp32 [4, 32, 2304] and int32 zeroed counters [64] and [36] (own keys, not shared with the m<=16 paths), warm every m in 17..32 for each of the four layers on synthetic x, re-zero the counters afterwards, and print per layer (diagnostic only, try/except) n_regs, n_spills and the PTX counts of 'wgmma.mma_async' and 'mma.sync' from the launch handle's asm['ptx']. No autotune, nothing derived from data kept.

## Expected

- `layer=qkv_gate load=decode_32 m=32` faster: 14.1 us today (nvjet 64x32 splitK 8.8 + splitKreduce 1.8) vs floor 8.0; one launch, wgmma at the 144-CTA split-K grid that ran 12.3 us at m=16; target <= 12.8 us
- `layer=pooler load=decode_32 m=32` faster: 21.6 us (nvjet 20.2 + reduce 2.0) vs floor 14.1; the m<=16 Triton grid reached 19.2-19.6 us; with wgmma and no 255-register tile the m=32 row should land there; target <= 20.3 us
- `layer=unpooler load=decode_32 m=32` faster: 18.9 us single nvjet vs floor 14.1; mma.sync Triton tiles lost 0.91-0.97x here; target <= 18.0 us
- `layer=o load=decode_32 m=32` faster: 7.0 us vs floor 3.6; the m<=16 Triton o kernel ran 4.4-4.7 us of kernel; mma.sync at m=32 ran 5.8; target <= 6.4 us shape time
- `* m=1` flat: untouched (m<=16 gates run first)
- `* m=16` flat: untouched
- `* m=8` flat: untouched
- `* m=128` flat: untouched (cuBLAS)
- `* m=256` flat: untouched (cuBLAS)
- `layer=router *` flat: untouched

## Risk

Every m=32 Triton attempt lost (pooler skinny 0.89x, qkv split-K4 0.858x, o 0.907x, unpooler 0.91-0.97x; only the pooler TMA kernel won, 1.061x). The TMA run's PTX dump says why: tokens on the MMA M side compile to mma.sync (wgmma=0) and BLOCK_M=32 hits 255 registers. Swap-AB (weight as tl.dot LHS, tokens on N) is what nvjet's 64x32 tiles do and is the operand swap that doubled the prefill Triton W4A16 GEMM (LESSONS). Weakest link: Triton must accept the masked, transposed-addressed [BLOCK_K, 32] x tile as an MN-major wgmma B operand; if the printed PTX still shows mma.sync, the run settles that Triton cannot reach wgmma at 32 tokens and the m=32 family should go to the cuBLASLt route instead. decode_32 is 1 of 6 loads and its five shapes sum to ~69 us; saving ~4 us there is ~+1% on the headline, so read the four m=32 rows and kernel times. Overlaps pooler-m32-tma-splitk-extend on the pooler m=32 row only; a later combo takes whichever row is faster. Follow-up if it wins: the same kernel at m<=16 (wgmma n16/n8).

## Adversarial review

Reviewed 2026-10-24T140535Z by an adversarial subagent (model opus).

**Verdict: proceed.** The mechanism is sound and well targeted. Triton's MMAv3 (wgmma) lowering needs the dot's M extent to be a multiple of 64 per warpgroup, so tokens on M (16/32) can only get mma.sync, while a 64-row weight tile on M with 32 tokens on N is exactly nvjet's 64x32 TN orientation. The four m=32 rows share no shapes with anything that has landed, and the prepare() PTX/register printout makes a single run readable whichever way it goes, so no companion is needed.

### Justification
A. Plausible. The archived TMA log (4e89311eefdc4bc3.log) shows BLOCK_M=32 compiled to wgmma=0, mma.sync=64 and n_regs=255, against 111 registers at BLOCK_M=16. The m=128 kernel (d34fd76f9c444728.log, BLOCK_M=128) did get wgmma_in_ptx=True. Together these are consistent with Triton's MMAv3 rule (M % 64 == 0 per warpgroup, N % 8 == 0, num_warps % 4 == 0), and the swapped tile (M=64 weight rows, N=32 tokens, num_warps=4) meets it. Both operands are K-contiguous: w[64, BLOCK_K] is a K-major A, and x loaded as [BLOCK_K, 32] with k on dim 0 is a K-major B. That is wgmma's native TN layout, so the 'MN-major B' worry in the Risk section is mislabelled and less of a risk than stated. Profile support: master's combo run (875f3e01) has m=32 at qkv 14.19 us (floor 8.0), o 7.10 (3.62), pooler 21.66 (14.14) and unpooler 18.82 (14.14). These are the bf16 shapes furthest above their floors after the m<=16 Triton wins, and at m<=16 the same weight streams beat nvjet by 5-8%. The configs fit: pooler/unpooler/o use 4 x 48 KB = 192 KB of shared memory (under 227 KB, with an OutOfResources fallback); qkv uses 96 KB, so two CTAs per SM cover the 144-CTA grid. The x shared-memory footprint per stage at BLOCK_K=256 equals the old BLOCK_M=32 attempts', so bytes in flight do not change and the MMA path and register pressure are the variable under test. The expected directions are consistent: every changed shape is m=32 at decode_32 (weight 62), and every other shape is untouched by construction. WEAKEST LINK: the claim that the MMA instruction and register pressure caused the m=32 losses. These are bandwidth-bound shapes (about 2.1 GFLOP for the pooler is about 2 us at peak, well under the 14 us floor), so even with wgmma the kernel may only tie nvjet, and the 255-register mma.sync tile may not have been the binding cost. Secondary: per-shape gains of about 1-1.5 us on four shapes in one of six loads come to about +1% on the geomean, inside the ±1.5% CI. The headline will not resolve it; the per-shape paired ratios and top_kernels times will, since Triton times reproduce within about 1%.

### Confounders on the current kernel
- bf16-shape-gated-triton-winners-combo (658d63b): its gates (pooler/qkv/o m<=16, unpooler n==1024) run first, and the new gate covers 17<=m<=32 only. The shapes are disjoint and the new path has its own ws/cnt keys, so there is no shared state. Not a confounder; if anything it is the evidence that the same weight streams win at m<=16.
- sglang-gemm-dispatch-parity: the m=32 baseline path being replaced is _bf16_gemm_dispatch_impl (nvjet). The comparison is a paired ratio against SGLang in the same run, so removing parity would not change the answer. Not a confounder.
- router-triton-hilo-fused-control-splitk-all-m / router-hilo-at-m8: router only, a separate graph per shape, cold L2 for every replay. Not a confounder.

### Interference and A/B needs
No companion is warranted. (i) ab-revert: no landed hypothesis touches the m=32 bf16 shapes, the launch sequence of those graphs, or their workspaces, so reverting any of them cannot change what this kernel measures. (ii) ab-tune: per-layer constants (SPLIT_K, BLOCK_K, stages) could be swept, but a sweep is a follow-up to a win, not something needed to read the result. (iii) control: the key disambiguation, whether wgmma was actually emitted and at what register count, comes from the prepare() PTX/n_regs printout, not from a second run. Reading the result: if wgmma was emitted and a row is faster, that is a gain attributable to swap-AB. If wgmma was emitted and a row is slower, the mma.sync explanation is refuted and m=32 goes to cuBLASLt or stays on nvjet. If mma.sync is still emitted, the run says Triton cannot reach wgmma here. Every outcome is interpretable from one run. With bf16's one remaining check today, spending it on companions would be wasteful.

### Open questions
- Will Triton's pipeliner keep the masked transposed x load as cp.async with zero-fill (fine), or fall back to synchronous register loads, which would stall the wgmma pipeline? The printed PTX should also count cp.async.
- What does the transposed epilogue store (a 64x32 tile, w-contiguous in registers, stored into out[m, n] row-major) cost? Triton will insert a layout conversion through shared memory. This is minor, but it could matter on o (7 us shape).
- Overlap with the queued pooler-m32-tma-splitk-extend on the pooler decode_32 row only. Do not combine both before each has been read on its own row.
- No correctness test exercises 17<=m<=32, so only the timed m=32 calls check this path. Make sure the warm-up covers every m from 17 to 31 (nearby shapes in final review) and that the masks are correct at m<32.


## Implementation notes

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

## Rules check

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

## Result

- job `1602a4e947e64090` (check), status done, passed True, tests 9/9
- speedup 1.1984x, 95% CI [1.1805, 1.2164], score 4030.8817081278007 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1760470299345995, "ci": [1.1584063244855805, 1.1936877353836184], "job": "875f3e0172ee48a1"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-shape-gated-triton-winners-combo.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.patch`; branch `hyp/muse_bf16_proj/bf16-m32-swapab-wgmma-triton`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 13.24 | 12.91 | 1.0372 | gained |
| layer=o load=decode_1 m=1 | 62 | 6.58 | 6.32 | 0.9919 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 19.2 | 19.0 | 1.0124 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 5.75 | 5.94 | 0.9843 | flat |
| layer=router load=decode_1 m=1 | 62 | 5.74 | 5.73 | 1.0006 | flat |
| layer=qkv_gate load=decode_16 m=16 | 62 | 14.01 | 13.39 | 1.0357 | gained |
| layer=o load=decode_16 m=16 | 62 | 6.36 | 6.66 | 0.9736 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.51 | 19.85 | 1.0146 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.52 | 18.61 | 0.9962 | flat |
| layer=router load=decode_16 m=16 | 62 | 6.2 | 6.03 | 1.0061 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.13 | 13.69 | 1.0363 | gained |
| layer=o load=decode_32 m=32 | 62 | 7.04 | 6.67 | 1.0412 | gained |
| layer=pooler load=decode_32 m=32 | 62 | 21.55 | 20.0 | 1.0929 | gained |
| layer=unpooler load=decode_32 m=32 | 62 | 18.89 | 18.48 | 1.0135 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.37 | 6.55 | 0.9839 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 13.53 | 13.18 | 1.0735 | gained |
| layer=o load=verify_1 m=8 | 62 | 6.75 | 6.32 | 1.0384 | gained |
| layer=pooler load=verify_1 m=8 | 62 | 19.61 | 19.14 | 1.0341 | gained |
| layer=unpooler load=verify_1 m=8 | 62 | 6.45 | 6.13 | 1.0036 | flat |
| layer=router load=verify_1 m=8 | 62 | 6.2 | 5.91 | 1.0452 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 17.1 | 16.81 | 1.017 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.45 | 7.09 | 1.0324 | gained |
| layer=pooler load=verify_16 m=128 | 62 | 25.95 | 25.55 | 0.9991 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 20.27 | 19.79 | 1.0056 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.68 | 7.35 | 1.0389 | gained |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.62 | 19.62 | 1.0101 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.7 | 9.01 | 1.0253 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 29.17 | 30.53 | 1.0068 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.0 | 24.5 | 0.9889 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.37 | 9.3 | 0.9983 | flat |

gained on qkv_gate decode_1, decode_16, decode_32, verify_1; o decode_32, verify_1, verify_16; pooler decode_32, verify_1; router verify_1, verify_16. Expected hit: 4/10. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: bf16-m32-swapab-wgmma-triton`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

