---
problem: muse_bf16_proj
slug: unpooler-full-triton-skinny-m32
created: 2026-10-23T225303Z
base: master
area: unpooler GEMM
change: >
  Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in {16,
  32} (the full weight: n=8192, k=4096, 67 MB): in kernel_fn route unpooler at 8 < m <= 32 to the
  same kernel with BLOCK_N=64 (8192/64 = 128 programs, one wave), BLOCK_K=256, num_stages=4,
  BLOCK_M from m, fp32 accumulate, bf16 store into torch.empty((m, 8192), bf16). Unpooler at m <=
  8 (the 1024-row slice) and at m >= 128 keep whatever they have (cuBLAS, or
  unpooler-slice-triton-gemv if it landed); pooler keeps the parent's path. Kernel moved:
  nvjet_sm90_tst_64x16/64x32_64x16_2x1_v_bz_TNT, a single non-splitK kernel at 18.5-18.8 us for 67
  MB (3.6 TB/s) against a 14.1 us floor; unlike pooler there is no reduce node to drop, so this
  run isolates the pure streaming-bandwidth question: does the full-K Triton stream beat nvjet's
  2x1 cluster kernel on an N-heavy, K-short (4096) weight.
expected:
  - shapes: layer=unpooler load=decode_16 *
    direction: faster
    why: 18.5 us single nvjet kernel at 3.6 TB/s; 128 programs streaming 512 KB each, target <= 17 us
  - shapes: layer=unpooler load=decode_32 *
    direction: faster
    why: 18.9 us, same
  - shapes: layer=unpooler load=decode_1 *
    direction: flat
    why: sliced path untouched here; control
  - shapes: layer=unpooler load=verify_16 *
    direction: flat
    why: m=128 keeps cuBLAS; control
risk: medium
depends_on: pooler-triton-skinny-m32
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# unpooler-full-triton-skinny-m32

## Change

Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in {16, 32} (the full weight: n=8192, k=4096, 67 MB): in kernel_fn route unpooler at 8 < m <= 32 to the same kernel with BLOCK_N=64 (8192/64 = 128 programs, one wave), BLOCK_K=256, num_stages=4, BLOCK_M from m, fp32 accumulate, bf16 store into torch.empty((m, 8192), bf16). Unpooler at m <= 8 (the 1024-row slice) and at m >= 128 keep whatever they have (cuBLAS, or unpooler-slice-triton-gemv if it landed); pooler keeps the parent's path. Kernel moved: nvjet_sm90_tst_64x16/64x32_64x16_2x1_v_bz_TNT, a single non-splitK kernel at 18.5-18.8 us for 67 MB (3.6 TB/s) against a 14.1 us floor; unlike pooler there is no reduce node to drop, so this run isolates the pure streaming-bandwidth question: does the full-K Triton stream beat nvjet's 2x1 cluster kernel on an N-heavy, K-short (4096) weight.

## Expected

- `layer=unpooler load=decode_16 *` faster: 18.5 us single nvjet kernel at 3.6 TB/s; 128 programs streaming 512 KB each, target <= 17 us
- `layer=unpooler load=decode_32 *` faster: 18.9 us, same
- `layer=unpooler load=decode_1 *` flat: sliced path untouched here; control
- `layer=unpooler load=verify_16 *` flat: m=128 keeps cuBLAS; control

## Risk

Because the cuBLAS kernel here is already a single launch, a gain can only come from bandwidth, which makes this the cleanest test of the streaming kernel itself: if pooler wins but this is flat, the pooler gain was the dropped reduce node, which should go into FINDINGS. K=4096 means only 16 BLOCK_K steps per program; raise num_stages to 6 if the parent's profile shows the pipeline draining. Tests touched: m=130 unpooler outliers (n=8192, m=130 stays on cuBLAS), m=5 unpooler (slice path, untouched). Same correctness and graph properties as the parent.
