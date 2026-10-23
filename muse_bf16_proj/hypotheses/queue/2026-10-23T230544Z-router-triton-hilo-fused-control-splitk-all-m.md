---
problem: muse_bf16_proj
slug: router-triton-hilo-fused-control-splitk-all-m
created: 2026-10-23T230544Z
base: master
area: router GEMM
change: >
  Built from hyp/muse_bf16_proj/sglang-gemm-dispatch-parity exactly as the primary (prepare()
  stores router_hi and router_lo as [256, 8192] bf16 row-major; one Triton kernel for the router
  at every m; BLOCK_K=128, num_stages=3; acc += tl.dot(x_tile, tl.trans(hi_tile)) + tl.dot(x_tile,
  tl.trans(lo_tile)) in fp32; tl.atomic_add into an output kernel_fn creates with torch.zeros((m,
  256), dtype=torch.float32) immediately before the launch; bf16 layers untouched on
  _bf16_gemm_dispatch_impl; no triton.autotune; all m-buckets compiled in prepare() on synthetic
  x), with one difference: the grid is 3-D, grid = (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K) at
  every m, each program owns one (M-block, N-block, K-slice) and never loops over M, and there is
  no SPLIT_K=1 plain-store path. Static configs chosen from m at launch: m <= 16: BLOCK_M=16,
  BLOCK_N=32, SPLIT_K=16 (K-slice 512; 1 x 8 x 16 = 128 CTAs); 16 < m <= 32: BLOCK_M=32,
  BLOCK_N=32, SPLIT_K=16 (128 CTAs); 32 < m <= 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=16 (2 x 4 x 16
  = 128 CTAs at m=128); m > 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=8 (4 x 4 x 8 = 128 CTAs at
  m=256). Rows beyond m and K beyond the slice are masked, so m=7, 9 and any nearby m are valid.
  Output dtype/shape as before.
expected:
  - shapes: layer=router load=verify_16 *
    direction: faster
    why: "128 CTAs splitting K=8192, like cuBLAS's splitK, instead of the primary's 4-8 CTAs each streaming 2.1 MB of weight plus the full x"
  - shapes: layer=router load=verify_32 *
    direction: faster
    why: "same; x (4.2 MB) is read once per N-block across the grid rather than once per program's M loop"
  - shapes: layer=router load=decode_1 *
    direction: faster
    why: 128 CTAs with 3 stages keep ~12 MB in flight, enough to saturate HBM for the 8.4 MB hi+lo read; one launch plus a memset replaces 3 launches (10.4 us)
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: "leaves the 26 us fp32 SIMT path; compare against router-hilo-at-m8's m=8 number to separate path change from Triton"
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: 5 launches (14.5 us) become 1 + memset with a full grid
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: same at m=32
  - shapes: "*layer=qkv_gate*"
    direction: flat
    why: bf16 layers untouched; control
risk: medium
depends_on: router-triton-hilo-fused
priority: 1
kind: control
controls_for: router-triton-hilo-fused
review: required
status: queued
---

# router-triton-hilo-fused-control-splitk-all-m

## Change

Built from hyp/muse_bf16_proj/sglang-gemm-dispatch-parity exactly as the primary (prepare() stores router_hi and router_lo as [256, 8192] bf16 row-major; one Triton kernel for the router at every m; BLOCK_K=128, num_stages=3; acc += tl.dot(x_tile, tl.trans(hi_tile)) + tl.dot(x_tile, tl.trans(lo_tile)) in fp32; tl.atomic_add into an output kernel_fn creates with torch.zeros((m, 256), dtype=torch.float32) immediately before the launch; bf16 layers untouched on _bf16_gemm_dispatch_impl; no triton.autotune; all m-buckets compiled in prepare() on synthetic x), with one difference: the grid is 3-D, grid = (cdiv(m, BLOCK_M), 256 // BLOCK_N, SPLIT_K) at every m, each program owns one (M-block, N-block, K-slice) and never loops over M, and there is no SPLIT_K=1 plain-store path. Static configs chosen from m at launch: m <= 16: BLOCK_M=16, BLOCK_N=32, SPLIT_K=16 (K-slice 512; 1 x 8 x 16 = 128 CTAs); 16 < m <= 32: BLOCK_M=32, BLOCK_N=32, SPLIT_K=16 (128 CTAs); 32 < m <= 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=16 (2 x 4 x 16 = 128 CTAs at m=128); m > 128: BLOCK_M=64, BLOCK_N=64, SPLIT_K=8 (4 x 4 x 8 = 128 CTAs at m=256). Rows beyond m and K beyond the slice are masked, so m=7, 9 and any nearby m are valid. Output dtype/shape as before.

## Expected

- `layer=router load=verify_16 *` faster: 128 CTAs splitting K=8192, like cuBLAS's splitK, instead of the primary's 4-8 CTAs each streaming 2.1 MB of weight plus the full x
- `layer=router load=verify_32 *` faster: same; x (4.2 MB) is read once per N-block across the grid rather than once per program's M loop
- `layer=router load=decode_1 *` faster: 128 CTAs with 3 stages keep ~12 MB in flight, enough to saturate HBM for the 8.4 MB hi+lo read; one launch plus a memset replaces 3 launches (10.4 us)
- `layer=router load=verify_1 *` faster: leaves the 26 us fp32 SIMT path; compare against router-hilo-at-m8's m=8 number to separate path change from Triton
- `layer=router load=decode_16 *` faster: 5 launches (14.5 us) become 1 + memset with a full grid
- `layer=router load=decode_32 *` faster: same at m=32
- `*layer=qkv_gate*` flat: bf16 layers untouched; control

## Risk

Separates 'a fused Triton hi/lo kernel cannot beat cuBLAS splitK + reduce + add on this shape' from 'the primary's grid under-fills the GPU (32 CTAs at m<=32, 4-8 at m>=128)'. If the primary loses at verify_16/verify_32 and this control gains there, the primary's tiling is the cause and the Triton route stays alive; if both lose, the Triton route for the router at large m goes on the do-not-retry list with evidence. If the primary gains everywhere, this control is the tuning data for the follow-up.
