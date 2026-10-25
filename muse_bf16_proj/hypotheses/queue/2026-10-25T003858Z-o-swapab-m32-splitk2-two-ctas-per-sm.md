---
problem: muse_bf16_proj
slug: o-swapab-m32-splitk2-two-ctas-per-sm
created: 2026-10-25T003858Z
base: master
area: o GEMM (m=17..32, swap-AB)
change: >
  Built on master (8b9ad43, blob a8bb42a). o at 17 <= m <= 32 runs _bf16_swapab_kernel with
  SWAPAB_CFG o = (8192, 1024, SPLIT_K 1, BLOCK_K 256, 4 stages): grid (128, 1) = 128 CTAs, each
  streaming 64 x 1024 x 2 B = 128 KB in 4 K-steps, one CTA per SM (4 x 48 KB = 192 KB smem), 4.8
  us kernel against a 3.6 us floor (row 6.3 at decode_32). With one CTA per SM and only 4 steps,
  the pipeline is mostly fill. Change the tuple to (8192, 1024, 2, 256, 2): grid (128, 2) = 256
  CTAs, K_SLICE 512 (2 K-steps, both in flight with 2 stages), 96 KB smem per CTA so 2 CTAs fit
  per SM (264 slots, one wave); each CTA streams 64 KB and the two CTAs on an SM overlap their
  fill. The existing ticket reduce handles SPLIT_K 2 (sum of 2 fp32 [32, 64] partials, fixed
  order). _swapab_buffers returns state swapab_ws_o / swapab_cnt_o for layer o; _swapab_prepare
  allocates swapab_ws_o = (2, SWAPAB_MAX_M, 8192) fp32 (2 MB) and swapab_cnt_o = zeros(128) int32
  and re-zeroes it after warm-up like the others. Nothing else changes (o at m<=16 keeps
  _pooler_skinny_kernel; m=33 o test and m>=128 keep cuBLAS).
expected:
  - shapes: layer=o load=decode_32 m=32
    direction: faster
    why: 4.8 us kernel, 1 CTA/SM with a 4-step pipeline that is mostly fill; 2 CTAs/SM with 64 KB each overlap fill; floor 3.6; target kernel ~4.2, row ~5.8
  - shapes: layer=o load=decode_1 m=1
    direction: flat
    why: m<=16 skinny path untouched
  - shapes: layer=o load=decode_16 m=16
    direction: flat
    why: untouched
  - shapes: layer=o load=verify_1 m=8
    direction: flat
    why: untouched
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# o-swapab-m32-splitk2-two-ctas-per-sm

## Change

Built on master (8b9ad43, blob a8bb42a). o at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG o = (8192, 1024, SPLIT_K 1, BLOCK_K 256, 4 stages): grid (128, 1) = 128 CTAs, each streaming 64 x 1024 x 2 B = 128 KB in 4 K-steps, one CTA per SM (4 x 48 KB = 192 KB smem), 4.8 us kernel against a 3.6 us floor (row 6.3 at decode_32). With one CTA per SM and only 4 steps, the pipeline is mostly fill. Change the tuple to (8192, 1024, 2, 256, 2): grid (128, 2) = 256 CTAs, K_SLICE 512 (2 K-steps, both in flight with 2 stages), 96 KB smem per CTA so 2 CTAs fit per SM (264 slots, one wave); each CTA streams 64 KB and the two CTAs on an SM overlap their fill. The existing ticket reduce handles SPLIT_K 2 (sum of 2 fp32 [32, 64] partials, fixed order). _swapab_buffers returns state swapab_ws_o / swapab_cnt_o for layer o; _swapab_prepare allocates swapab_ws_o = (2, SWAPAB_MAX_M, 8192) fp32 (2 MB) and swapab_cnt_o = zeros(128) int32 and re-zeroes it after warm-up like the others. Nothing else changes (o at m<=16 keeps _pooler_skinny_kernel; m=33 o test and m>=128 keep cuBLAS).

## Expected

- `layer=o load=decode_32 m=32` faster: 4.8 us kernel, 1 CTA/SM with a 4-step pipeline that is mostly fill; 2 CTAs/SM with 64 KB each overlap fill; floor 3.6; target kernel ~4.2, row ~5.8
- `layer=o load=decode_1 m=1` flat: m<=16 skinny path untouched
- `layer=o load=decode_16 m=16` flat: untouched
- `layer=o load=verify_1 m=8` flat: untouched
- `layer=qkv_gate *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Counter-evidence: the CUDA gemv rewrite ran o with SPLIT_K 2 and lost at m=8 (the m-proportional ticket-reduce tail); here the reducer reads only 2 partials of 8 KB and the tail is one CTA per tile, but if the decode_32 row is not below 6.3 the tail cost is confirmed and o-unpooler-slice-cuda-cluster-dsmem-rewrite (DSMEM reduce) is the route instead. One scored shape, attributable from its row.
