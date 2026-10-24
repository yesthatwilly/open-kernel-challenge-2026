---
problem: muse_bf16_proj
slug: pooler-triton-splitk2-blockn64-semaphore-control-blockk256
created: 2026-10-24T075347Z
base: master
area: pooler GEMM
change: >
  From the parent's branch (pooler-triton-splitk2-blockn64-semaphore), change exactly one
  constant: the pooler split-K kernel's BLOCK_K from 128 to 256, so each CTA's K-slice of 4096 is
  16 K-steps instead of 32. Keep num_stages=4, num_warps=4, BLOCK_N=64, SPLIT_K=2, grid (64, 2) =
  128 CTAs, BLOCK_M=16 for m<=16 and 32 for 17..32. Keep the direct transposed weight addressing,
  the fp32 workspace, the barrier + acq_rel ticket + last-CTA fixed-order reduce and atomic_xchg
  reset, the routing (pooler m<=32 only), and the warm-up plus re-zero in prepare(). Smem becomes
  4 x (64x256 + BLOCK_M x256) x 2 B = 160 KB at BLOCK_M=16 and 192 KB at BLOCK_M=32. This is
  exactly the unpooler-full-triton-skinny-m32 per-CTA program, and it fits H200's 227 KB. If
  Triton rejects 192 KB at BLOCK_M=32, use num_stages=3 for the BLOCK_M=32 specialization only and
  record that.
expected:
  - shapes: layer=pooler load=decode_1 *
    direction: faster
    why: 128 KB of weight in flight per CTA, the 3.73 TB/s unpooler config. Expect about 18-18.5 us plus the reduce tail, below the parent if the parent is stage-bound at about 20.3 us
  - shapes: layer=pooler load=verify_1 *
    direction: faster
    why: same as decode_1, BLOCK_M=16 binary
  - shapes: layer=pooler load=decode_16 *
    direction: faster
    why: the exact m=16 unpooler program that measured 17.97 us on the same 67.1 MB
  - shapes: layer=pooler load=decode_32 *
    direction: flat
    why: "unpooler BLOCK_M=32 at this tile measured 19.27 us (paired 0.973); expect about 20 us vs nvjet's 20.2+1.9, roughly flat to slightly faster"
  - shapes: layer=pooler load=verify_16 *
    direction: flat
    why: m=128 stays on cuBLAS
  - shapes: layer=pooler load=verify_32 *
    direction: flat
    why: m=256 stays on cuBLAS
  - shapes: "*layer=router*"
    direction: flat
    why: master router untouched
risk: low
depends_on: pooler-triton-splitk2-blockn64-semaphore
priority: 1
kind: control
controls_for: pooler-triton-splitk2-blockn64-semaphore
review: required
status: queued
hold_reason: null
---

# pooler-triton-splitk2-blockn64-semaphore-control-blockk256

## Change

From the parent's branch (pooler-triton-splitk2-blockn64-semaphore), change exactly one constant: the pooler split-K kernel's BLOCK_K from 128 to 256, so each CTA's K-slice of 4096 is 16 K-steps instead of 32. Keep num_stages=4, num_warps=4, BLOCK_N=64, SPLIT_K=2, grid (64, 2) = 128 CTAs, BLOCK_M=16 for m<=16 and 32 for 17..32. Keep the direct transposed weight addressing, the fp32 workspace, the barrier + acq_rel ticket + last-CTA fixed-order reduce and atomic_xchg reset, the routing (pooler m<=32 only), and the warm-up plus re-zero in prepare(). Smem becomes 4 x (64x256 + BLOCK_M x256) x 2 B = 160 KB at BLOCK_M=16 and 192 KB at BLOCK_M=32. This is exactly the unpooler-full-triton-skinny-m32 per-CTA program, and it fits H200's 227 KB. If Triton rejects 192 KB at BLOCK_M=32, use num_stages=3 for the BLOCK_M=32 specialization only and record that.

## Expected

- `layer=pooler load=decode_1 *` faster: 128 KB of weight in flight per CTA, the 3.73 TB/s unpooler config. Expect about 18-18.5 us plus the reduce tail, below the parent if the parent is stage-bound at about 20.3 us
- `layer=pooler load=verify_1 *` faster: same as decode_1, BLOCK_M=16 binary
- `layer=pooler load=decode_16 *` faster: the exact m=16 unpooler program that measured 17.97 us on the same 67.1 MB
- `layer=pooler load=decode_32 *` flat: unpooler BLOCK_M=32 at this tile measured 19.27 us (paired 0.973); expect about 20 us vs nvjet's 20.2+1.9, roughly flat to slightly faster
- `layer=pooler load=verify_16 *` flat: m=128 stays on cuBLAS
- `layer=pooler load=verify_32 *` flat: m=256 stays on cuBLAS
- `*layer=router*` flat: master router untouched

## Risk

Separates stage depth (64 KB vs 128 KB of weight in flight per CTA, the only measured difference between the 3.3 and 3.73 TB/s Triton configs) from the parent's own design (split-K 2 + BLOCK_N=64 + semaphore reduce). Parent about 20.3 us and control about 18.5 us: the parent's spec, not its design, was at fault, and the control is the one to promote. Both about 20.3 us: split-K/semaphore costs as much as it saves. Both about 18.5 us: BLOCK_N=64 + split-K is what matters.
