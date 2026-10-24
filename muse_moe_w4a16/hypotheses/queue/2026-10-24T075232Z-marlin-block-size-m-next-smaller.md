---
problem: muse_moe_w4a16
slug: marlin-block-size-m-next-smaller
created: 2026-10-24T075232Z
base: master
area: w13 marlin
change: >
  Change only the block_size_m pick in kernel_fn: keep SGLang's loop over BLOCK_SIZE_M_CANDIDATES
  = (8, 16, 32, 48, 64) (first candidate with num_tokens*8/32/block_size_m < 0.9) but take the
  candidate one index BELOW it, floored at index 0: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx
  - BLOCK_SIZE_M_STEP_DOWN(num_tokens), 0)] with a module-level BLOCK_SIZE_M_STEP_DOWN = lambda
  num_tokens: 1 so a per-shape partial adoption is a one-line follow-up. Resulting picks, SGLang
  -> new: bs=1/4/8/16: 8 -> 8 (unchanged, already the smallest, the in-run control); bs=32/33: 16
  -> 8; bs=128: 48 -> 32; bs=256/257: 64 -> 48. The same block_size_m feeds moe_align_block_size
  (still over the bank's 32 experts, as on master) and the moe_block_size kwarg of both
  moe_wna16_marlin_gemm calls; nothing else changes. Mechanism: this is the mirror image of
  marlin-block-size-m-next-larger (regression 0.918x), which moved Marlin to the next
  thread_m_blocks config and lost 21% at bs=32 (16 -> 32, thread_m_blocks 1 -> 2), 14% at bs=128
  (48 -> 64, 3 -> 4) and ~2% at bs=1 (8 -> 16, the 8-row special path to the 16-row path; inside
  noise), even though the larger block collapsed m-blocks. So at these M the per-m-block cost
  (tile shape, registers, smem stages) dominates over the number of weight passes; a smaller block
  (8-row path at bs=32; thread_m_blocks 2 at bs=128; 3 at bs=256) may be cheaper per block by more
  than the extra m-blocks cost (experts with more routes than block_size_m now get two m-blocks,
  each streaming that expert's weights, partly from L2).
expected:
  - shapes: bs=32 *
    direction: faster
    why: 16 -> 8 moves to the 8-row Marlin path that bs=1/8/16 use; 16->32 lost 21% here, so the per-block cost gradient points down
  - shapes: bs=128 *
    direction: faster
    why: 48 -> 32 (thread_m_blocks 3 -> 2); 48 -> 64 lost 14% here, Marlin is 96% of the shape
  - shapes: bs=256 *
    direction: faster
    why: 64 -> 48 (thread_m_blocks 4 -> 3); Marlin 96% of 621 us at ~1/6 of SoL
  - shapes: bs=1 *
    direction: flat
    why: "control: 8 before and after"
  - shapes: bs=8 *
    direction: flat
    why: "control: 8 before and after"
  - shapes: bs=16 *
    direction: flat
    why: "control: 8 before and after"
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-block-size-m-next-smaller

## Change

Change only the block_size_m pick in kernel_fn: keep SGLang's loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) (first candidate with num_tokens*8/32/block_size_m < 0.9) but take the candidate one index BELOW it, floored at index 0: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - BLOCK_SIZE_M_STEP_DOWN(num_tokens), 0)] with a module-level BLOCK_SIZE_M_STEP_DOWN = lambda num_tokens: 1 so a per-shape partial adoption is a one-line follow-up. Resulting picks, SGLang -> new: bs=1/4/8/16: 8 -> 8 (unchanged, already the smallest, the in-run control); bs=32/33: 16 -> 8; bs=128: 48 -> 32; bs=256/257: 64 -> 48. The same block_size_m feeds moe_align_block_size (still over the bank's 32 experts, as on master) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls; nothing else changes. Mechanism: this is the mirror image of marlin-block-size-m-next-larger (regression 0.918x), which moved Marlin to the next thread_m_blocks config and lost 21% at bs=32 (16 -> 32, thread_m_blocks 1 -> 2), 14% at bs=128 (48 -> 64, 3 -> 4) and ~2% at bs=1 (8 -> 16, the 8-row special path to the 16-row path; inside noise), even though the larger block collapsed m-blocks. So at these M the per-m-block cost (tile shape, registers, smem stages) dominates over the number of weight passes; a smaller block (8-row path at bs=32; thread_m_blocks 2 at bs=128; 3 at bs=256) may be cheaper per block by more than the extra m-blocks cost (experts with more routes than block_size_m now get two m-blocks, each streaming that expert's weights, partly from L2).

## Expected

- `bs=32 *` faster: 16 -> 8 moves to the 8-row Marlin path that bs=1/8/16 use; 16->32 lost 21% here, so the per-block cost gradient points down
- `bs=128 *` faster: 48 -> 32 (thread_m_blocks 3 -> 2); 48 -> 64 lost 14% here, Marlin is 96% of the shape
- `bs=256 *` faster: 64 -> 48 (thread_m_blocks 4 -> 3); Marlin 96% of 621 us at ~1/6 of SoL
- `bs=1 *` flat: control: 8 before and after
- `bs=8 *` flat: control: 8 before and after
- `bs=16 *` flat: control: 8 before and after

## Risk

Risk: an expert with more local routes than block_size_m now needs an extra m-block, and every m-block streams that expert's full w13 (16 MB) and w2 (8 MB) slice; under skewed popularity at bs=256 the hot experts may cross 48 rows, so a weight-bandwidth loss can outweigh the config gain; read per shape. Correctness is unaffected (pure dispatch knob; the bs=33/257 and skewed bs=128 tests run the new picks). marlin-block-size-m-verify-two-smaller runs the two-step-down picks at bs=128/256 independently from master so the two runs together give the gradient; if all three lose here, SGLang's pick is a local optimum at bs>=32 and the block-size knob is closed. muse_decode_layer is testing a cap of 16 on the same Marlin call; do not merge their result into this one, the shapes and routing differ.
