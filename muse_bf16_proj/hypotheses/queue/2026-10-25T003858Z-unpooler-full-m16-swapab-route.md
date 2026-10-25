---
problem: muse_bf16_proj
slug: unpooler-full-m16-swapab-route
created: 2026-10-25T003858Z
base: master
area: unpooler GEMM
change: >
  Built on master (8b9ad43, blob a8bb42a). On master the full-weight unpooler (n=8192, k=4096, 67
  MB) at 9 <= m <= 16 falls through to _bf16_gemm_dispatch_impl (cuBLAS nvjet_sm90_tst_64x16, 19.3
  us kernel, row 18.5 at decode_16), while at m 17..32 it takes the swap-AB Triton kernel
  _bf16_swapab_kernel (SWAPAB_CFG unpooler: SPLIT_K 1, BLOCK_K 256, 4 stages, grid 128 CTAs; 18.7
  us kernel at m=32, row 18.1). bf16-m16-swapab-wgmma-extend covered pooler/qkv/o/slice at m<=16
  but never the full unpooler. Change exactly one gate: add a per-layer lower bound,
  SWAPAB_MIN_M_BY_LAYER = {"unpooler": 9} (others keep SWAPAB_MIN_M = 17), used by
  _swapab_route_ok, so unpooler with cfg n == 8192 at 9..16 takes the same swap-AB kernel with the
  same constexprs (BLOCK_T 32, token columns >= m masked; M is not specialized so no new compile).
  In _swapab_prepare extend the unpooler warm-up range to 9..32. The slice (cfg n == 1024, m <= 8)
  is still caught earlier by _unpool_shape_ok and is unaffected; m=130 and m=1000 tests
  unaffected. No other layer, constant or kernel changes.
expected:
  - shapes: layer=unpooler load=decode_16 m=16
    direction: faster
    why: nvjet 64x16 19.3 us kernel (row 18.5) -> the swap-AB kernel that runs 18.7 us at m=32 and should be no slower at m=16 (same bytes, fewer token columns); floor 14.1
  - shapes: layer=unpooler load=decode_32 m=32
    direction: flat
    why: same path as master
  - shapes: layer=unpooler load=decode_1 m=1
    direction: flat
    why: slice path untouched
  - shapes: layer=unpooler load=verify_1 m=8
    direction: flat
    why: slice path untouched
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# unpooler-full-m16-swapab-route

## Change

Built on master (8b9ad43, blob a8bb42a). On master the full-weight unpooler (n=8192, k=4096, 67 MB) at 9 <= m <= 16 falls through to _bf16_gemm_dispatch_impl (cuBLAS nvjet_sm90_tst_64x16, 19.3 us kernel, row 18.5 at decode_16), while at m 17..32 it takes the swap-AB Triton kernel _bf16_swapab_kernel (SWAPAB_CFG unpooler: SPLIT_K 1, BLOCK_K 256, 4 stages, grid 128 CTAs; 18.7 us kernel at m=32, row 18.1). bf16-m16-swapab-wgmma-extend covered pooler/qkv/o/slice at m<=16 but never the full unpooler. Change exactly one gate: add a per-layer lower bound, SWAPAB_MIN_M_BY_LAYER = {"unpooler": 9} (others keep SWAPAB_MIN_M = 17), used by _swapab_route_ok, so unpooler with cfg n == 8192 at 9..16 takes the same swap-AB kernel with the same constexprs (BLOCK_T 32, token columns >= m masked; M is not specialized so no new compile). In _swapab_prepare extend the unpooler warm-up range to 9..32. The slice (cfg n == 1024, m <= 8) is still caught earlier by _unpool_shape_ok and is unaffected; m=130 and m=1000 tests unaffected. No other layer, constant or kernel changes.

## Expected

- `layer=unpooler load=decode_16 m=16` faster: nvjet 64x16 19.3 us kernel (row 18.5) -> the swap-AB kernel that runs 18.7 us at m=32 and should be no slower at m=16 (same bytes, fewer token columns); floor 14.1
- `layer=unpooler load=decode_32 m=32` flat: same path as master
- `layer=unpooler load=decode_1 m=1` flat: slice path untouched
- `layer=unpooler load=verify_1 m=8` flat: slice path untouched
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Small, one-shape change (~0.5-1 us on one 62-weight shape, ~+1% on decode_16); attributable from the unpooler decode_16 row alone. Risk: swap-AB at m=16 wastes half the wgmma n32 columns, but the kernel is weight-stream-bound so that is free; if the row is not below 18.5 the gate goes back.
