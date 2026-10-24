---
problem: muse_bf16_proj
slug: unpooler-full-triton-skinny-m32-control-blockm16-all-m
created: 2026-10-24T032454Z
base: master
area: unpooler GEMM
change: >
  From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, grid
  (8192/64 = 128 programs), BLOCK_N=64, BLOCK_K=256, num_stages=4, num_warps=4, fp32 accumulate
  and the bf16 store exactly as the parent, and change only the M tiling: BLOCK_M is 16 for every
  m in 9..32. For m > 16 the kernel keeps two fp32 accumulators [16, 64] and per K-step loads the
  one [64, 256] weight tile once and two [16, 256] x tiles (rows 0..15 and 16..31, rows >= m
  masked to zero), doing two tl.dot(x_i, tl.trans(w_tile), acc_i); for m <= 16 it is the parent's
  single-tile path (so unpooler decode_16 is byte-for-byte the parent's kernel and serves as the
  within-run control). Smem per stage drops to 32 KB weight + 2 x 8 KB x = 48 KB, 192 KB at 4
  stages (same as the parent at m=32; if the parent had to drop to 3 stages, this one may keep 4).
  Two stores, one per 16-row accumulator, masked on rows < m. Warm m in 9..32 in prepare() on
  synthetic zeros. Pooler and every other layer unchanged from the parent.
expected:
  - shapes: layer=unpooler load=decode_32 *
    direction: faster
    why: "than the parent's m=32 row if the pooler's 25.2-vs-20.2 us m=32 penalty is the BLOCK_M=32 tl.dot tile; equal to the parent's row if it is bandwidth"
  - shapes: layer=unpooler load=decode_16 *
    direction: flat
    why: identical kernel binary to the parent at m<=16; within-run control
  - shapes: layer=unpooler load=decode_1 *
    direction: flat
    why: slice path untouched
  - shapes: layer=unpooler load=verify_* *
    direction: flat
    why: m=8 slice and m>=128 stay on cuBLAS
risk: low
depends_on: unpooler-full-triton-skinny-m32
priority: 3
kind: control
controls_for: unpooler-full-triton-skinny-m32
review: required
status: queued
---

# unpooler-full-triton-skinny-m32-control-blockm16-all-m

## Change

From the parent's branch (unpooler-full-triton-skinny-m32), keep the unpooler routing, grid (8192/64 = 128 programs), BLOCK_N=64, BLOCK_K=256, num_stages=4, num_warps=4, fp32 accumulate and the bf16 store exactly as the parent, and change only the M tiling: BLOCK_M is 16 for every m in 9..32. For m > 16 the kernel keeps two fp32 accumulators [16, 64] and per K-step loads the one [64, 256] weight tile once and two [16, 256] x tiles (rows 0..15 and 16..31, rows >= m masked to zero), doing two tl.dot(x_i, tl.trans(w_tile), acc_i); for m <= 16 it is the parent's single-tile path (so unpooler decode_16 is byte-for-byte the parent's kernel and serves as the within-run control). Smem per stage drops to 32 KB weight + 2 x 8 KB x = 48 KB, 192 KB at 4 stages (same as the parent at m=32; if the parent had to drop to 3 stages, this one may keep 4). Two stores, one per 16-row accumulator, masked on rows < m. Warm m in 9..32 in prepare() on synthetic zeros. Pooler and every other layer unchanged from the parent.

## Expected

- `layer=unpooler load=decode_32 *` faster: than the parent's m=32 row if the pooler's 25.2-vs-20.2 us m=32 penalty is the BLOCK_M=32 tl.dot tile; equal to the parent's row if it is bandwidth
- `layer=unpooler load=decode_16 *` flat: identical kernel binary to the parent at m<=16; within-run control
- `layer=unpooler load=decode_1 *` flat: slice path untouched
- `layer=unpooler load=verify_* *` flat: m=8 slice and m>=128 stay on cuBLAS

## Risk

Separates the m=32 row's result into tile (BLOCK_M=32 wgmma tile / smem pipeline) versus streaming bandwidth: if the parent loses at m=32 and this control does not, the family's m=32 problem is the 32-row tile and o-/qkv-gate-triton-skinny should adopt BLOCK_M=16 everywhere; if both lose equally, the plain Triton stream's ceiling (~3.3 TB/s on the pooler) is confirmed on a second shape and the whole skinny-Triton family should be withdrawn.
