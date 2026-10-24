---
problem: muse_bf16_proj
slug: pooler-splitk2-tma-weight-stream
created: 2026-10-24T092648Z
base: master
area: pooler GEMM
change: >
  From branch hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (pooler at m <= 32:
  _bf16_splitk_kernel, grid (64, 2) = 128 CTAs, BLOCK_N=64, SPLIT_K=2, ticket-semaphore reduce;
  harvested neutral at 20.2-20.7 us per pooler row, ~3.3 TB/s, the same rate as every
  plain-tl.load Triton weight stream so far), change only how the weight tile reaches shared
  memory: load it with a Hopper TMA tensor descriptor instead of per-thread tl.load (cp.async).
  Build the descriptor once in prepare() from the pooler weight alone (shape [4096, 8192], strides
  [8192, 1], block [BLOCK_N=64, BLOCK_K=256], bf16): with Triton >= 3.4 use
  triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [64, 256]) passed as a kernel
  argument and w = w_desc.load([n0, k]) in the K loop; on older Triton use
  triton.tools.experimental_descriptor.create_2d_tma_descriptor and
  tl._experimental_descriptor_load(desc, [n0, k], [64, 256], tl.bfloat16).
  print(triton.__version__) and the chosen API in prepare(). The tile arrives [BLOCK_N, BLOCK_K]
  (K-contiguous, as stored); feed acc = tl.dot(x_tile, tl.trans(w), acc) (the transpose is a
  smem-descriptor/ldmatrix.trans view, not a copy). BLOCK_K 256 (16 K-steps per 4096 K-slice) and
  num_stages 4: 32 KB of weight per stage, 128 KB of weight in flight per CTA, moved by one
  elected thread per stage on an mbarrier instead of ~64 cp.async.16 per thread, which leaves the
  warps free to issue the next stage and the MMA. x stays a plain masked tl.load ([BLOCK_M, 256],
  L2-resident). Smem 4 x (32 KB + BLOCK_M x 512 B) = 160 KB at BLOCK_M=16, 192 KB at BLOCK_M=32
  (drop to num_stages 3 for the BLOCK_M=32 specialization only if it does not fit). Everything
  else stays as the parent: routing (pooler m <= 32 only), grid, SPLIT_K, the fp32 workspace,
  barrier + acq_rel ticket + fixed-order last-CTA reduce + atomic_xchg reset, warm-up and counter
  re-zero in prepare(). No autotune; the descriptor depends only on the weight.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: "21.2 us on master (parent 20.2) vs floor 14.0: the largest per-shape gap on the four decode-type loads (pooler is ~38% of each). Every Triton stream so far tops out at 3.3-3.7 TB/s with cp.async; TMA bulk loads with 128 KB in flight per CTA should reach ~4.0-4.2 TB/s. Target <= 17.5 us kernel time"
  - shapes: layer=pooler load=verify_1 m=8
    direction: faster
    why: same stream at m=8 (parent 19.9 us); target <= 17.5 us
  - shapes: layer=pooler load=decode_16 m=16
    direction: faster
    why: same stream at m=16 (parent 20.7 us); target <= 18 us
  - shapes: layer=pooler load=decode_32 m=32
    direction: flat
    why: parent 22.2 us; the m=32 penalty is in the x path and smem, not the weight stream, so TMA may only recover part of it
  - shapes: layer=pooler load=verify_16 m=128
    direction: flat
    why: "control: cuBLAS"
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
risk: high
depends_on: pooler-triton-splitk2-blockn64-semaphore
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# pooler-splitk2-tma-weight-stream

## Change

From branch hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (pooler at m <= 32: _bf16_splitk_kernel, grid (64, 2) = 128 CTAs, BLOCK_N=64, SPLIT_K=2, ticket-semaphore reduce; harvested neutral at 20.2-20.7 us per pooler row, ~3.3 TB/s, the same rate as every plain-tl.load Triton weight stream so far), change only how the weight tile reaches shared memory: load it with a Hopper TMA tensor descriptor instead of per-thread tl.load (cp.async). Build the descriptor once in prepare() from the pooler weight alone (shape [4096, 8192], strides [8192, 1], block [BLOCK_N=64, BLOCK_K=256], bf16): with Triton >= 3.4 use triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [64, 256]) passed as a kernel argument and w = w_desc.load([n0, k]) in the K loop; on older Triton use triton.tools.experimental_descriptor.create_2d_tma_descriptor and tl._experimental_descriptor_load(desc, [n0, k], [64, 256], tl.bfloat16). print(triton.__version__) and the chosen API in prepare(). The tile arrives [BLOCK_N, BLOCK_K] (K-contiguous, as stored); feed acc = tl.dot(x_tile, tl.trans(w), acc) (the transpose is a smem-descriptor/ldmatrix.trans view, not a copy). BLOCK_K 256 (16 K-steps per 4096 K-slice) and num_stages 4: 32 KB of weight per stage, 128 KB of weight in flight per CTA, moved by one elected thread per stage on an mbarrier instead of ~64 cp.async.16 per thread, which leaves the warps free to issue the next stage and the MMA. x stays a plain masked tl.load ([BLOCK_M, 256], L2-resident). Smem 4 x (32 KB + BLOCK_M x 512 B) = 160 KB at BLOCK_M=16, 192 KB at BLOCK_M=32 (drop to num_stages 3 for the BLOCK_M=32 specialization only if it does not fit). Everything else stays as the parent: routing (pooler m <= 32 only), grid, SPLIT_K, the fp32 workspace, barrier + acq_rel ticket + fixed-order last-CTA reduce + atomic_xchg reset, warm-up and counter re-zero in prepare(). No autotune; the descriptor depends only on the weight.

## Expected

- `layer=pooler load=decode_1 m=1` faster: 21.2 us on master (parent 20.2) vs floor 14.0: the largest per-shape gap on the four decode-type loads (pooler is ~38% of each). Every Triton stream so far tops out at 3.3-3.7 TB/s with cp.async; TMA bulk loads with 128 KB in flight per CTA should reach ~4.0-4.2 TB/s. Target <= 17.5 us kernel time
- `layer=pooler load=verify_1 m=8` faster: same stream at m=8 (parent 19.9 us); target <= 17.5 us
- `layer=pooler load=decode_16 m=16` faster: same stream at m=16 (parent 20.7 us); target <= 18 us
- `layer=pooler load=decode_32 m=32` flat: parent 22.2 us; the m=32 penalty is in the x path and smem, not the weight stream, so TMA may only recover part of it
- `layer=pooler load=verify_16 m=128` flat: control: cuBLAS
- `layer=qkv_gate *` flat: untouched

## Risk

High risk: the judge image Triton version is not recorded anywhere in the repo. Both API branches are named; if neither exists the implementer stops and records it (no run). It changes two things relative to the parent (TMA, and BLOCK_K 128 -> 256, which TMA needs for a 32 KB stage); the held pooler-triton-splitk2-blockn64-semaphore-control-blockk256 is the BLOCK_K-only companion if the reviewer wants the split. Device-side TMA with mbarriers is graph-capturable; the host descriptor is built once from the weight pointer (not input-derived). Readout: the kernel us per pooler row against the parent 20.2/19.9/20.7/22.2 us; a stream above 4 TB/s would carry over to qkv_gate and unpooler at m <= 32.
