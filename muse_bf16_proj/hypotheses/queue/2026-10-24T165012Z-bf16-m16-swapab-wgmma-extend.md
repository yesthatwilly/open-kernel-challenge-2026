---
problem: muse_bf16_proj
slug: bf16-m16-swapab-wgmma-extend
created: 2026-10-24T165012Z
base: master
area: bf16 GEMM dispatch (m<=16, swap-AB Triton)
change: >
  Built on master (8b9ad43, submission blob a8bb42a). Master's m<=16 Triton paths all put the
  tokens on tl.dot's M side with BLOCK_M=16 (pooler _bf16_splitk_kernel, qkv_gate
  _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler slice _bf16_splitk_kernel at n==1024,
  m<=8). Triton's MMAv3 (wgmma) needs a 64-row M tile per warpgroup, so these lower to mma.sync;
  at m=32 the same orientation gave wgmma=0 and swapping it to weight-on-LHS gave wgmma and
  +3.6..9.3% (bf16-m32-swapab-wgmma-triton). Change: route those same m<=16 shapes to master's
  existing _bf16_swapab_kernel (weight rows [BLOCK_W=64, BLOCK_K] on tl.dot's LHS, x^T [BLOCK_K,
  BLOCK_T] on the RHS) with BLOCK_T=16 (wgmma m64n16k16; token columns >= m masked; m=1 and m=8
  run the same BLOCK_T=16 compile, M is not specialized). Per-layer config, same as master's
  swap-AB table except BLOCK_T and the unpooler slice: pooler (N 4096, K 8192) SPLIT_K 2, BLOCK_K
  256, 4 stages, grid (64, 2); qkv_gate (2304, 8192) SPLIT_K 4, BLOCK_K 128, 4 stages, grid (36,
  4); o (8192, 1024) SPLIT_K 1, BLOCK_K 256, 4 stages, grid (128, 1); unpooler slice (n == 1024, K
  4096, w = the first 1024 rows of the full weight read in place, stride_wn 4096) SPLIT_K 8,
  BLOCK_K 128, 3 stages, grid (16, 8). Add SWAPAB16_CFG with those rows and new
  prepare()-allocated workspaces/tickets under own keys (ws fp32 [SPLIT_K, 16, n], cnt int32 [n //
  64] zeroed) so master's m=17..32 buffers are untouched; WS_M=16. Gate in kernel_fn BEFORE
  master's m<=16 branches: layer in {pooler, qkv_gate, o} and 1 <= cfg m <= 16 with master's n/k,
  or layer == unpooler and cfg n == 1024 (m <= 8); x bf16, stride(1) == 1. The unpooler full
  weight at m=16 (cuBLAS nvjet) and every m>=17 shape keep master. prepare(): warm every m in
  1..16 per layer (1..8 for the unpooler slice) on synthetic x, re-zero counters, and print per
  layer the report=True line (n_regs, n_spills, wgmma, mma.sync, cp.async), plus the same PTX
  counts for master's incumbent m<=16 kernels compiled at m=16, so the log confirms the incumbents
  are mma.sync and the new path wgmma. If a 4-stage variant is OutOfResources, step down to 3
  (master's fallback). Master's old m<=16 kernels stay in the file but are no longer routed for
  these shapes.
expected:
  - shapes: layer=pooler load=decode_* m=1
    direction: faster
    why: incumbent _bf16_splitk_kernel 19.0 us (3.5 TB/s) vs floor 14.0; swap-AB wgmma at m=32 took pooler 9.3% faster with the identical grid; target <= 18 us
  - shapes: layer=pooler m=16
    direction: faster
    why: 19.5 us kernel vs 14.1 floor; same mechanism
  - shapes: layer=pooler m=8
    direction: faster
    why: 19.3 us kernel vs 14.0 floor
  - shapes: layer=qkv_gate m=1
    direction: faster
    why: _qkv_splitk_sem_kernel 11.8 us (3.2 TB/s) vs floor 7.9; m=32 swap-AB qkv gained 3.6%
  - shapes: layer=qkv_gate m=8
    direction: faster
    why: 12.0 us kernel vs floor 7.9
  - shapes: layer=qkv_gate m=16
    direction: faster
    why: 12.3 us kernel vs floor 7.9
  - shapes: layer=o m=1
    direction: faster
    why: _pooler_skinny_kernel 4.6 us vs floor 3.5; m=32 swap-AB o gained 4.1%
  - shapes: layer=o m=8
    direction: faster
    why: 5.2 us kernel vs floor 3.5
  - shapes: layer=o m=16
    direction: faster
    why: 4.8 us kernel vs floor 3.6
  - shapes: layer=unpooler m=1
    direction: faster
    why: slice _bf16_splitk_kernel 4.5 us vs floor 1.7 (8.4 MB at 1.9 TB/s), furthest from floor of the four
  - shapes: layer=unpooler m=8
    direction: faster
    why: slice kernel 4.6 us vs floor 1.8
  - shapes: layer=unpooler m=16
    direction: flat
    why: full weight stays cuBLAS nvjet
  - shapes: "* m=32"
    direction: flat
    why: master swap-AB path untouched
  - shapes: "* m=128"
    direction: flat
    why: untouched
  - shapes: "* m=256"
    direction: flat
    why: untouched
  - shapes: layer=router *
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

# bf16-m16-swapab-wgmma-extend

## Change

Built on master (8b9ad43, submission blob a8bb42a). Master's m<=16 Triton paths all put the tokens on tl.dot's M side with BLOCK_M=16 (pooler _bf16_splitk_kernel, qkv_gate _qkv_splitk_sem_kernel, o _pooler_skinny_kernel, unpooler slice _bf16_splitk_kernel at n==1024, m<=8). Triton's MMAv3 (wgmma) needs a 64-row M tile per warpgroup, so these lower to mma.sync; at m=32 the same orientation gave wgmma=0 and swapping it to weight-on-LHS gave wgmma and +3.6..9.3% (bf16-m32-swapab-wgmma-triton). Change: route those same m<=16 shapes to master's existing _bf16_swapab_kernel (weight rows [BLOCK_W=64, BLOCK_K] on tl.dot's LHS, x^T [BLOCK_K, BLOCK_T] on the RHS) with BLOCK_T=16 (wgmma m64n16k16; token columns >= m masked; m=1 and m=8 run the same BLOCK_T=16 compile, M is not specialized). Per-layer config, same as master's swap-AB table except BLOCK_T and the unpooler slice: pooler (N 4096, K 8192) SPLIT_K 2, BLOCK_K 256, 4 stages, grid (64, 2); qkv_gate (2304, 8192) SPLIT_K 4, BLOCK_K 128, 4 stages, grid (36, 4); o (8192, 1024) SPLIT_K 1, BLOCK_K 256, 4 stages, grid (128, 1); unpooler slice (n == 1024, K 4096, w = the first 1024 rows of the full weight read in place, stride_wn 4096) SPLIT_K 8, BLOCK_K 128, 3 stages, grid (16, 8). Add SWAPAB16_CFG with those rows and new prepare()-allocated workspaces/tickets under own keys (ws fp32 [SPLIT_K, 16, n], cnt int32 [n // 64] zeroed) so master's m=17..32 buffers are untouched; WS_M=16. Gate in kernel_fn BEFORE master's m<=16 branches: layer in {pooler, qkv_gate, o} and 1 <= cfg m <= 16 with master's n/k, or layer == unpooler and cfg n == 1024 (m <= 8); x bf16, stride(1) == 1. The unpooler full weight at m=16 (cuBLAS nvjet) and every m>=17 shape keep master. prepare(): warm every m in 1..16 per layer (1..8 for the unpooler slice) on synthetic x, re-zero counters, and print per layer the report=True line (n_regs, n_spills, wgmma, mma.sync, cp.async), plus the same PTX counts for master's incumbent m<=16 kernels compiled at m=16, so the log confirms the incumbents are mma.sync and the new path wgmma. If a 4-stage variant is OutOfResources, step down to 3 (master's fallback). Master's old m<=16 kernels stay in the file but are no longer routed for these shapes.

## Expected

- `layer=pooler load=decode_* m=1` faster: incumbent _bf16_splitk_kernel 19.0 us (3.5 TB/s) vs floor 14.0; swap-AB wgmma at m=32 took pooler 9.3% faster with the identical grid; target <= 18 us
- `layer=pooler m=16` faster: 19.5 us kernel vs 14.1 floor; same mechanism
- `layer=pooler m=8` faster: 19.3 us kernel vs 14.0 floor
- `layer=qkv_gate m=1` faster: _qkv_splitk_sem_kernel 11.8 us (3.2 TB/s) vs floor 7.9; m=32 swap-AB qkv gained 3.6%
- `layer=qkv_gate m=8` faster: 12.0 us kernel vs floor 7.9
- `layer=qkv_gate m=16` faster: 12.3 us kernel vs floor 7.9
- `layer=o m=1` faster: _pooler_skinny_kernel 4.6 us vs floor 3.5; m=32 swap-AB o gained 4.1%
- `layer=o m=8` faster: 5.2 us kernel vs floor 3.5
- `layer=o m=16` faster: 4.8 us kernel vs floor 3.6
- `layer=unpooler m=1` faster: slice _bf16_splitk_kernel 4.5 us vs floor 1.7 (8.4 MB at 1.9 TB/s), furthest from floor of the four
- `layer=unpooler m=8` faster: slice kernel 4.6 us vs floor 1.8
- `layer=unpooler m=16` flat: full weight stays cuBLAS nvjet
- `* m=32` flat: master swap-AB path untouched
- `* m=128` flat: untouched
- `* m=256` flat: untouched
- `layer=router *` flat: untouched

## Risk

At m<=16 these shapes are HBM-bound, not MMA-bound, so the m=32 gain (which removed 255-reg tokens-on-M spills) may not transfer: m=16 incumbents already run 96-ish regs. The plausible win is the async wgmma letting more loads stay in flight and fewer registers per CTA; the prepare() PTX/regs print attributes it either way. wgmma n16 wastes nothing at m=16 but computes 15 dead columns at m=1 (free under the BW bound). Per-shape judging: if only some layers gain, a combo keeps those (as with bf16-shape-gated-triton-winners-combo).
