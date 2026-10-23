---
problem: muse_bf16_proj
slug: pooler-triton-skinny-m32
created: 2026-10-23T225201Z
base: master
area: pooler GEMM
change: >
  For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kernel_fn
  with one Triton kernel, a split-K-free skinny GEMM that streams the 67 MB weight once: grid =
  (N/BLOCK_N,) with BLOCK_N=32 so 4096/32 = 128 programs (one wave on 132 SMs); each program owns
  32 weight rows and loops the full K=8192 in BLOCK_K=256 steps with num_stages=4 (tl.load of a
  [BLOCK_N, BLOCK_K] bf16 weight tile from the row-major [N, K] weight as stored, and of the x
  tile [BLOCK_M, BLOCK_K] with BLOCK_M=16 or 32 from m, rows >= m masked to zero), acc +=
  tl.dot(x, w.T) in fp32, then a masked bf16 store of [m, 32] into a torch.empty((m, 4096), bf16)
  output. prepare() stays as in the parity base (no repack). All other layers and pooler at m >=
  128 keep the cuBLAS dispatch. Kernels moved: nvjet_sm90_tst_128x8/16/32_..._splitK_TNT
  (18.8-20.2 us) plus cublasLt::splitKreduce_kernel (1.6-1.9 us) become one launch; the reduce
  launch and its graph-node gap go away and the weight tile reads are fully sequential per
  program. The question the run answers: can a plain Triton full-K stream hit >= 4.0 TB/s (67 MB
  in <= 17 us) where nvjet splitK sits at 3.6 TB/s.
expected:
  - shapes: layer=pooler load=decode_1 *
    direction: faster
    why: 21.5 us (18.8 nvjet + 1.6 reduce + gap) vs 14.0 floor; target <= 18 us
  - shapes: layer=pooler load=decode_16 *
    direction: faster
    why: same shape class, 21.5 us, BLOCK_M=16
  - shapes: layer=pooler load=decode_32 *
    direction: faster
    why: same, 22.1 us, BLOCK_M=32
  - shapes: layer=pooler load=verify_1 *
    direction: faster
    why: m=8, same as m=1 path with 8 live rows
  - shapes: layer=pooler load=verify_16 *
    direction: flat
    why: m=128 keeps cuBLAS; control
  - shapes: "*layer=unpooler*"
    direction: flat
    why: untouched layer; control
risk: medium
depends_on: sglang-gemm-dispatch-parity
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# pooler-triton-skinny-m32

## Change

For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kernel_fn with one Triton kernel, a split-K-free skinny GEMM that streams the 67 MB weight once: grid = (N/BLOCK_N,) with BLOCK_N=32 so 4096/32 = 128 programs (one wave on 132 SMs); each program owns 32 weight rows and loops the full K=8192 in BLOCK_K=256 steps with num_stages=4 (tl.load of a [BLOCK_N, BLOCK_K] bf16 weight tile from the row-major [N, K] weight as stored, and of the x tile [BLOCK_M, BLOCK_K] with BLOCK_M=16 or 32 from m, rows >= m masked to zero), acc += tl.dot(x, w.T) in fp32, then a masked bf16 store of [m, 32] into a torch.empty((m, 4096), bf16) output. prepare() stays as in the parity base (no repack). All other layers and pooler at m >= 128 keep the cuBLAS dispatch. Kernels moved: nvjet_sm90_tst_128x8/16/32_..._splitK_TNT (18.8-20.2 us) plus cublasLt::splitKreduce_kernel (1.6-1.9 us) become one launch; the reduce launch and its graph-node gap go away and the weight tile reads are fully sequential per program. The question the run answers: can a plain Triton full-K stream hit >= 4.0 TB/s (67 MB in <= 17 us) where nvjet splitK sits at 3.6 TB/s.

## Expected

- `layer=pooler load=decode_1 *` faster: 21.5 us (18.8 nvjet + 1.6 reduce + gap) vs 14.0 floor; target <= 18 us
- `layer=pooler load=decode_16 *` faster: same shape class, 21.5 us, BLOCK_M=16
- `layer=pooler load=decode_32 *` faster: same, 22.1 us, BLOCK_M=32
- `layer=pooler load=verify_1 *` faster: m=8, same as m=1 path with 8 live rows
- `layer=pooler load=verify_16 *` flat: m=128 keeps cuBLAS; control
- `*layer=unpooler*` flat: untouched layer; control

## Risk

Risk: a full-K loop has only num_stages x BLOCK_N x BLOCK_K x 2 = 64 KB in flight per SM; if that is not enough memory-level parallelism to beat nvjet's splitK (which uses 4x more CTAs), the pooler shapes get slower, not faster, and the per-shape rows will show it directly (nothing else moves). Correctness: bf16 inputs, fp32 accumulate, one bf16 rounding at the store, same as cuBLAS; the m=7 pooler test exercises the masked-row path. CUDA graph: fixed-shape torch.empty only, no syncs. Compile: 3 BLOCK_M specializations, seconds. If faster, qkv-gate-triton-skinny-m32, unpooler-full-triton-skinny-m32, o-triton-skinny-m32 and pooler-skinny-tiled-weight-layout stack on it; if slower, they should be withdrawn or retuned (BLOCK_N=16 with 256 programs, or num_stages=6) before running.
