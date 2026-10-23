---
problem: muse_bf16_proj
slug: qkv-gate-triton-skinny-m32
created: 2026-10-23T225317Z
base: master
area: qkv GEMM
change: >
  Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'qkv_gate' at m <= 32
  (n=2304, k=8192, a 37.7 MB weight): in kernel_fn route qkv_gate at m <= 32 to the same kernel
  with BLOCK_N=16 (2304/16 = 144 programs, the nearest power-of-two tile that keeps about one
  wave) and the same BLOCK_K=256, num_stages=4, BLOCK_M from m, fp32 accumulate, bf16 store into
  torch.empty((m, 2304), bf16). The pooler path and everything else stay as the parent left them;
  qkv_gate at m >= 128 keeps cuBLAS. Kernels moved:
  nvjet_sm90_tst_128x8/64x16/64x32_..._splitK_TNT (7.8-9.6 us) + cublasLt::splitKreduce_kernel
  (1.6-1.8 us) become one launch on the four m <= 32 qkv_gate shapes; reported 14.1-14.5 us
  against a 7.9 us floor, so the splitK main kernel is at 3.9 TB/s and the reduce + gap is ~3 us
  of the total.
expected:
  - shapes: layer=qkv_gate load=decode_1 *
    direction: faster
    why: "14.1 us: drop the 1.7 us reduce node and match or beat 3.9 TB/s streaming"
  - shapes: layer=qkv_gate load=decode_16 *
    direction: faster
    why: 14.3 us, same mechanism
  - shapes: layer=qkv_gate load=decode_32 *
    direction: faster
    why: 14.5 us, same
  - shapes: layer=qkv_gate load=verify_1 *
    direction: faster
    why: m=8, 14.3 us, same
  - shapes: layer=qkv_gate load=verify_16 *
    direction: flat
    why: m=128 keeps cuBLAS; control
  - shapes: "*layer=pooler*"
    direction: flat
    why: the pooler path of the parent is unchanged in this run and should reproduce the parent number
risk: medium
depends_on: pooler-triton-skinny-m32
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# qkv-gate-triton-skinny-m32

## Change

Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'qkv_gate' at m <= 32 (n=2304, k=8192, a 37.7 MB weight): in kernel_fn route qkv_gate at m <= 32 to the same kernel with BLOCK_N=16 (2304/16 = 144 programs, the nearest power-of-two tile that keeps about one wave) and the same BLOCK_K=256, num_stages=4, BLOCK_M from m, fp32 accumulate, bf16 store into torch.empty((m, 2304), bf16). The pooler path and everything else stay as the parent left them; qkv_gate at m >= 128 keeps cuBLAS. Kernels moved: nvjet_sm90_tst_128x8/64x16/64x32_..._splitK_TNT (7.8-9.6 us) + cublasLt::splitKreduce_kernel (1.6-1.8 us) become one launch on the four m <= 32 qkv_gate shapes; reported 14.1-14.5 us against a 7.9 us floor, so the splitK main kernel is at 3.9 TB/s and the reduce + gap is ~3 us of the total.

## Expected

- `layer=qkv_gate load=decode_1 *` faster: 14.1 us: drop the 1.7 us reduce node and match or beat 3.9 TB/s streaming
- `layer=qkv_gate load=decode_16 *` faster: 14.3 us, same mechanism
- `layer=qkv_gate load=decode_32 *` faster: 14.5 us, same
- `layer=qkv_gate load=verify_1 *` faster: m=8, 14.3 us, same
- `layer=qkv_gate load=verify_16 *` flat: m=128 keeps cuBLAS; control
- `*layer=pooler*` flat: the pooler path of the parent is unchanged in this run and should reproduce the parent number

## Risk

Only worth running if the parent is faster on pooler; it reuses the parent's kernel with one tile change so a loss here attributes to the 144-program/BLOCK_N=16 tiling of this layer, not to the kernel. 144 programs is 1.09 waves, so a tail of 12 programs runs alone; BLOCK_N=8 (288 programs, 2.2 waves) is the ab-tune if this is flat. Tests touched: m=1 qkv_gate and m=512 qkv_gate outliers (the latter stays on cuBLAS). Same correctness and graph properties as the parent.
