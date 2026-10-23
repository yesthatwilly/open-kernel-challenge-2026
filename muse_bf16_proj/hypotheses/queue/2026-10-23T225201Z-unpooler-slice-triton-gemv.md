---
problem: muse_bf16_proj
slug: unpooler-slice-triton-gemv
created: 2026-10-23T225201Z
base: master
area: unpooler GEMM
change: >
  For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight
  read), replace the _bf16_gemm_dispatch_impl(x, W[:1024], None) call in kernel_fn with one Triton
  launch and no split-K reduce: grid = (1024/BLOCK_N,) with BLOCK_N=8, i.e. 128 programs; each
  program streams its 8 weight rows over the full K=4096 in BLOCK_K=512 steps (num_stages=4),
  loads the x tile [16, BLOCK_K] with rows >= m masked, accumulates in fp32 with acc +=
  tl.sum(x[:, None, :] * w[None, :, :], axis=2) (FMA broadcast; tl.dot needs BLOCK_N >= 16 and
  this shape is launch-bound, not math-bound), and stores the [m, 8] bf16 slab into a
  torch.empty((m, 1024), bf16). prepare() unchanged; W[:1024] of the row-major weight is already
  contiguous. Kernels moved: nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT (4.0-4.3 us) +
  cublasLt::splitKreduce_kernel (1.5-1.6 us) and the gap between them become one kernel; the
  reported 6.9-7.1 us is 4x the 1.7 us floor, and the goal is 2 launches to 1 on a sub-10 us
  shape. Unpooler at m >= 16 (n=8192) and every other layer keep the cuBLAS dispatch.
expected:
  - shapes: layer=unpooler load=decode_1 *
    direction: faster
    why: 6.9 us for 8.4 MB (floor 1.7); one launch instead of nvjet splitK + splitKreduce; target ~4 us
  - shapes: layer=unpooler load=verify_1 *
    direction: faster
    why: m=8 same sliced shape, 7.1 us
  - shapes: layer=unpooler load=decode_16 *
    direction: flat
    why: m=16 is the full 8192-row weight, keeps cuBLAS; control
  - shapes: "*layer=o*"
    direction: flat
    why: untouched; control
risk: medium
depends_on: sglang-gemm-dispatch-parity
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# unpooler-slice-triton-gemv

## Change

For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight read), replace the _bf16_gemm_dispatch_impl(x, W[:1024], None) call in kernel_fn with one Triton launch and no split-K reduce: grid = (1024/BLOCK_N,) with BLOCK_N=8, i.e. 128 programs; each program streams its 8 weight rows over the full K=4096 in BLOCK_K=512 steps (num_stages=4), loads the x tile [16, BLOCK_K] with rows >= m masked, accumulates in fp32 with acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2) (FMA broadcast; tl.dot needs BLOCK_N >= 16 and this shape is launch-bound, not math-bound), and stores the [m, 8] bf16 slab into a torch.empty((m, 1024), bf16). prepare() unchanged; W[:1024] of the row-major weight is already contiguous. Kernels moved: nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT (4.0-4.3 us) + cublasLt::splitKreduce_kernel (1.5-1.6 us) and the gap between them become one kernel; the reported 6.9-7.1 us is 4x the 1.7 us floor, and the goal is 2 launches to 1 on a sub-10 us shape. Unpooler at m >= 16 (n=8192) and every other layer keep the cuBLAS dispatch.

## Expected

- `layer=unpooler load=decode_1 *` faster: 6.9 us for 8.4 MB (floor 1.7); one launch instead of nvjet splitK + splitKreduce; target ~4 us
- `layer=unpooler load=verify_1 *` faster: m=8 same sliced shape, 7.1 us
- `layer=unpooler load=decode_16 *` flat: m=16 is the full 8192-row weight, keeps cuBLAS; control
- `*layer=o*` flat: untouched; control

## Risk

Risk: 128 programs each reading 64 KB may not pull 8.4 MB faster than cuBLAS's 4-way splitK; the shape is so small that launch latency (~2-3 us per graph node) dominates, so the win mostly comes from dropping the second node, which is why it should still be faster even at equal bandwidth. The FMA-broadcast form keeps register use at 16x8x512 fp32 = 64K values per program if written naively; tile it as BLOCK_K=128 inner chunks or use tl.dot with BLOCK_N=16 (64 programs) if Triton spills; either is a tune, not a new idea. Tests touched: m=5 unpooler and m=130/m=1000 unpooler (the latter two are n=8192, untouched path). Correctness: same bf16-in/fp32-acc/bf16-out as cuBLAS. CUDA graph: fixed-shape torch.empty only.
