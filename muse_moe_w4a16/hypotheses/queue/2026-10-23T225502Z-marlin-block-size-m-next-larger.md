---
problem: muse_moe_w4a16
slug: marlin-block-size-m-next-larger
created: 2026-10-23T225502Z
base: master
area: MoE align/dispatch
change: >
  Change only the block_size_m pick in kernel_fn. SGLang's loop breaks on the first candidate in
  (8, 16, 32, 48, 64) where num_tokens*8/32/block_size_m < 0.9, which yields 8 at bs=1/8/16, 16 at
  bs=32, 48 at bs=128 and 64 at bs=256. Replace it with the next larger candidate (cap 64): 16 at
  bs=1/8/16, 32 at bs=32, 64 at bs=128, 64 at bs=256 (unchanged, a control). The same block size
  feeds moe_align_block_size and the moe_block_size kwarg of both Marlin calls; nothing else
  changes. Mechanism: Marlin MoE re-reads an expert's whole NVFP4 weight for every m-block of that
  expert, so fewer, larger m-blocks mean fewer weight passes (L2 is cold at every replay, so a
  second block re-reads from HBM), at the cost of more padded rows, whose compute is free while
  the kernel is memory-bound.
expected:
  - shapes: bs=32 *
    direction: faster
    why: "bs=32: ~256 routes over ~13 touched experts, ~20 per expert: block 16 gives 2 weight passes for most experts, block 32 gives 1"
  - shapes: bs=128 *
    direction: faster
    why: "bs=128: block 48 -> 64 cuts m-blocks for hot experts (skewed popularity puts 50-100 routes on the hottest)"
  - shapes: bs=16 *
    direction: flat
    why: "bs=16: ~9 touched experts with ~2 routes each, one block either way"
  - shapes: bs=1 *
    direction: flat
    why: "bs=1: one block per touched expert at either size; padding is compute only"
  - shapes: bs=256 *
    direction: flat
    why: "control: 64 before and after"
risk: low
depends_on: sglang-marlin-path-parity
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-block-size-m-next-larger

## Change

Change only the block_size_m pick in kernel_fn. SGLang's loop breaks on the first candidate in (8, 16, 32, 48, 64) where num_tokens*8/32/block_size_m < 0.9, which yields 8 at bs=1/8/16, 16 at bs=32, 48 at bs=128 and 64 at bs=256. Replace it with the next larger candidate (cap 64): 16 at bs=1/8/16, 32 at bs=32, 64 at bs=128, 64 at bs=256 (unchanged, a control). The same block size feeds moe_align_block_size and the moe_block_size kwarg of both Marlin calls; nothing else changes. Mechanism: Marlin MoE re-reads an expert's whole NVFP4 weight for every m-block of that expert, so fewer, larger m-blocks mean fewer weight passes (L2 is cold at every replay, so a second block re-reads from HBM), at the cost of more padded rows, whose compute is free while the kernel is memory-bound.

## Expected

- `bs=32 *` faster: bs=32: ~256 routes over ~13 touched experts, ~20 per expert: block 16 gives 2 weight passes for most experts, block 32 gives 1
- `bs=128 *` faster: bs=128: block 48 -> 64 cuts m-blocks for hot experts (skewed popularity puts 50-100 routes on the hottest)
- `bs=16 *` flat: bs=16: ~9 touched experts with ~2 routes each, one block either way
- `bs=1 *` flat: bs=1: one block per touched expert at either size; padding is compute only
- `bs=256 *` flat: control: 64 before and after

## Risk

Risks: Marlin's thread_m_blocks grows with moe_block_size, so a bigger block changes the kernel config (tile shape, smem) and could lose occupancy at small M; bs=1/8 could regress slightly from doubled A-tile padding. Pure dispatch knob: no correctness or rule exposure. If this is a gain, the obvious follow-up ab-tune is 64 everywhere.
