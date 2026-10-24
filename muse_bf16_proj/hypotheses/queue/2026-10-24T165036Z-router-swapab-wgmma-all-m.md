---
problem: muse_bf16_proj
slug: router-swapab-wgmma-all-m
created: 2026-10-24T165036Z
base: master
area: router GEMM
change: >
  Built on master (8b9ad43). Master's router is _router_hilo_kernel with the tokens on tl.dot's M
  side: grid (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K), (BLOCK_M, BLOCK_N, SPLIT_K) = (16, 32,
  16) at m<=16, (32, 32, 16) at m<=32, (64, 64, 16) at m<=128, (64, 64, 8) above, acc += x_tile @
  hi_tile; acc += x_tile @ lo_tile, fp32 atomic_add into a torch.zeros output. At m<=32 the
  16/32-row token tile cannot lower to wgmma (needs 64 M rows per warpgroup; the bf16 m=32 run
  measured wgmma=0 for tokens-on-M), and at m=128/256 the M-blocking re-reads every hi/lo weight
  tile 2x / 4x (grid M dim 2 / 4). Change: replace the router launch with a new
  _router_swapab_kernel, the router analogue of master's _bf16_swapab_kernel: program (pid_t,
  pid_w, pid_k) loads hi and lo tiles [BLOCK_W=64, BLOCK_K] of the [256, 8192] row-major router_hi
  / router_lo (as stored, K-contiguous) as tl.dot's LHS and the x^T tile [BLOCK_K, BLOCK_T]
  (tokens >= m masked to 0) as the RHS, acc[64, BLOCK_T] fp32 = tl.dot(hi, xt, acc); acc =
  tl.dot(lo, xt, acc), then fp32 tl.atomic_add of acc transposed into out[m, 256] (masked on
  tokens < m), out zeroed right before the launch exactly as master. Static configs from m only
  (BLOCK_T, SPLIT_K, BLOCK_K): m<=16: (16, 32, 128), grid (1, 4, 32) = 128 CTAs, K-slice 256;
  m<=32: (32, 32, 128), (1, 4, 32); m<=128: (128, 32, 128), (1, 4, 32), each weight tile read once
  instead of twice; m>128: (128, 16, 128), grid (cdiv(m,128), 4, 16) = 128 CTAs at m=256, K-slice
  512, weight read 2x instead of 4x. num_warps 4 (one warpgroup), num_stages 3 (2 if
  OutOfResources at compile in prepare; recorded as a constant). Keep master's hi/lo split,
  _router_shape_ok preconditions (adapted to the new K-slice divisibility), the F.linear fp32
  fallback and every bf16 layer path. prepare(): warm master's ROUTER_WARM_MS plus every bucket,
  and print per bucket n_regs, n_spills, PTX counts of wgmma.mma_async and mma.sync for the new
  kernel AND for master's _router_hilo_kernel at the same m, so the log shows the orientation
  change is what moved the MMA path.
expected:
  - shapes: layer=router m=1
    direction: faster
    why: _router_hilo_kernel 3.2 us + 0.7 us memset vs floor 1.8 (8.4 MB hi+lo); wgmma m64n16 instead of mma.sync with 128 CTAs and 2 K-steps each
  - shapes: layer=router m=8
    direction: faster
    why: 3.2 us kernel, same as m=1
  - shapes: layer=router m=16
    direction: faster
    why: 3.5 us kernel vs floor 1.8
  - shapes: layer=router m=32
    direction: faster
    why: 4.1 us kernel; the BLOCK_M=32 tokens-on-M tile is the one the bf16 m=32 run measured as wgmma=0
  - shapes: layer=router m=128
    direction: faster
    why: 4.9 us kernel vs floor 2.2; grid M dim 2 re-reads each weight tile twice; swap-AB with 128 tokens on N reads it once (8.4 MB instead of 16.8 MB of L2/HBM weight traffic)
  - shapes: layer=router m=256
    direction: faster
    why: 6.9 us kernel vs floor 2.7; weight read 4x today, 2x with BLOCK_T 128
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# router-swapab-wgmma-all-m

## Change

Built on master (8b9ad43). Master's router is _router_hilo_kernel with the tokens on tl.dot's M side: grid (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K), (BLOCK_M, BLOCK_N, SPLIT_K) = (16, 32, 16) at m<=16, (32, 32, 16) at m<=32, (64, 64, 16) at m<=128, (64, 64, 8) above, acc += x_tile @ hi_tile; acc += x_tile @ lo_tile, fp32 atomic_add into a torch.zeros output. At m<=32 the 16/32-row token tile cannot lower to wgmma (needs 64 M rows per warpgroup; the bf16 m=32 run measured wgmma=0 for tokens-on-M), and at m=128/256 the M-blocking re-reads every hi/lo weight tile 2x / 4x (grid M dim 2 / 4). Change: replace the router launch with a new _router_swapab_kernel, the router analogue of master's _bf16_swapab_kernel: program (pid_t, pid_w, pid_k) loads hi and lo tiles [BLOCK_W=64, BLOCK_K] of the [256, 8192] row-major router_hi / router_lo (as stored, K-contiguous) as tl.dot's LHS and the x^T tile [BLOCK_K, BLOCK_T] (tokens >= m masked to 0) as the RHS, acc[64, BLOCK_T] fp32 = tl.dot(hi, xt, acc); acc = tl.dot(lo, xt, acc), then fp32 tl.atomic_add of acc transposed into out[m, 256] (masked on tokens < m), out zeroed right before the launch exactly as master. Static configs from m only (BLOCK_T, SPLIT_K, BLOCK_K): m<=16: (16, 32, 128), grid (1, 4, 32) = 128 CTAs, K-slice 256; m<=32: (32, 32, 128), (1, 4, 32); m<=128: (128, 32, 128), (1, 4, 32), each weight tile read once instead of twice; m>128: (128, 16, 128), grid (cdiv(m,128), 4, 16) = 128 CTAs at m=256, K-slice 512, weight read 2x instead of 4x. num_warps 4 (one warpgroup), num_stages 3 (2 if OutOfResources at compile in prepare; recorded as a constant). Keep master's hi/lo split, _router_shape_ok preconditions (adapted to the new K-slice divisibility), the F.linear fp32 fallback and every bf16 layer path. prepare(): warm master's ROUTER_WARM_MS plus every bucket, and print per bucket n_regs, n_spills, PTX counts of wgmma.mma_async and mma.sync for the new kernel AND for master's _router_hilo_kernel at the same m, so the log shows the orientation change is what moved the MMA path.

## Expected

- `layer=router m=1` faster: _router_hilo_kernel 3.2 us + 0.7 us memset vs floor 1.8 (8.4 MB hi+lo); wgmma m64n16 instead of mma.sync with 128 CTAs and 2 K-steps each
- `layer=router m=8` faster: 3.2 us kernel, same as m=1
- `layer=router m=16` faster: 3.5 us kernel vs floor 1.8
- `layer=router m=32` faster: 4.1 us kernel; the BLOCK_M=32 tokens-on-M tile is the one the bf16 m=32 run measured as wgmma=0
- `layer=router m=128` faster: 4.9 us kernel vs floor 2.2; grid M dim 2 re-reads each weight tile twice; swap-AB with 128 tokens on N reads it once (8.4 MB instead of 16.8 MB of L2/HBM weight traffic)
- `layer=router m=256` faster: 6.9 us kernel vs floor 2.7; weight read 4x today, 2x with BLOCK_T 128
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched

## Risk

LESSONS says the router's small-m grid depth is settled (SPLIT_K 32 and stage counts flat); this changes the MMA orientation and the weight re-read, not the grid depth, but at m<=16 it also moves SPLIT_K 16 -> 32 and K-slice 512 -> 256, which the split-K-32 runs found flat, so a small-m loss would point at the orientation. The 0.7 us vectorized_elementwise memset stays (removing it was flat: router-splitk-semaphore-no-memset). Router accuracy bound is 0.01%: the hi+lo fp32 accumulation is unchanged, only operand order and atomic order differ (atomic order is already nondeterministic on master). At m=128/256 the x^T tile [128, 128] bf16 is 32 KB per stage, acc 64x128 fp32 = 64 regs/thread: watch spills in the print.
