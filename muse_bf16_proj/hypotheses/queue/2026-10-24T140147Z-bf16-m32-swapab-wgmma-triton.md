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
review: required
status: queued
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
