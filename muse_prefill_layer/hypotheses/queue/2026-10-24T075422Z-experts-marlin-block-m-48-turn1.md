---
problem: muse_prefill_layer
slug: experts-marlin-block-m-48-turn1
created: 2026-10-24T075422Z
base: master
area: w13 marlin + w2 marlin (block_size_m pick)
change: >
  Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 512 <
  num_tokens <= 1024 use block_size_m = 48; every other token count keeps the parity loop's pick
  unchanged (64 at T=3012 and T=16384, and the parity picks at the tests outside the range). In
  the scored set this moves only the two turn_1 shapes (T=753, parity pick 64) and in the tests
  T=600 and T=753. The same block_size_m goes to moe_align_block_size and to the moe_block_size
  kwarg of both moe_wna16_marlin_gemm calls. Nothing else changes. Mechanism: Marlin's per-m-block
  cost grows with thread_m_blocks (muse_moe_w4a16 marlin-block-size-m-next-larger measured +18%
  for 3->4 and +28% for 1->2 at about one block per expert), so 64 -> 48 (thread_m_blocks 4 -> 3)
  should cut each block's cost, while 48 rows still hold, in one block, nearly every local expert
  at T=753 (mean ~23.5 rows per local expert, but routing is skewed by the router's alignment with
  the stream mean and router_bias, so hot experts reach 30-50 rows). It is the sibling of
  experts-marlin-block-m-local-rows (block 32 at T=753): 32 has the larger per-block saving but
  sends hot experts to 2 blocks (2 weight passes); 48 trades a smaller per-block saving for fewer
  passes. Run both; the faster lands and the other is its A/B.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: Marlin pair 730/681 us at thread_m_blocks 4; 48 rows (tmb 3) keeps ~all local experts at one block; sister 3->4 cost was +18%, so ~12-15% off Marlin, ~6-10% per turn_1 shape
  - shapes: "*load=turn_4*"
    direction: flat
    why: T=3012 keeps the parity 64 (control)
  - shapes: "*load=first_turn*"
    direction: flat
    why: T=16384 keeps the parity 64 (control)
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# experts-marlin-block-m-48-turn1

## Change

Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 512 < num_tokens <= 1024 use block_size_m = 48; every other token count keeps the parity loop's pick unchanged (64 at T=3012 and T=16384, and the parity picks at the tests outside the range). In the scored set this moves only the two turn_1 shapes (T=753, parity pick 64) and in the tests T=600 and T=753. The same block_size_m goes to moe_align_block_size and to the moe_block_size kwarg of both moe_wna16_marlin_gemm calls. Nothing else changes. Mechanism: Marlin's per-m-block cost grows with thread_m_blocks (muse_moe_w4a16 marlin-block-size-m-next-larger measured +18% for 3->4 and +28% for 1->2 at about one block per expert), so 64 -> 48 (thread_m_blocks 4 -> 3) should cut each block's cost, while 48 rows still hold, in one block, nearly every local expert at T=753 (mean ~23.5 rows per local expert, but routing is skewed by the router's alignment with the stream mean and router_bias, so hot experts reach 30-50 rows). It is the sibling of experts-marlin-block-m-local-rows (block 32 at T=753): 32 has the larger per-block saving but sends hot experts to 2 blocks (2 weight passes); 48 trades a smaller per-block saving for fewer passes. Run both; the faster lands and the other is its A/B.

## Expected

- `*load=turn_1*` faster: Marlin pair 730/681 us at thread_m_blocks 4; 48 rows (tmb 3) keeps ~all local experts at one block; sister 3->4 cost was +18%, so ~12-15% off Marlin, ~6-10% per turn_1 shape
- `*load=turn_4*` flat: T=3012 keeps the parity 64 (control)
- `*load=first_turn*` flat: T=16384 keeps the parity 64 (control)

## Risk

Same knob as experts-marlin-block-m-local-rows at a different value; the two cannot both land as written (they edit the same pick), so whichever loses becomes the comparison point. 48 is a less common Marlin MoE size (thread_m_blocks 3); if the build's moe_wna16_marlin_gemm has no tmb-3 instantiation for this config it should raise at the first test, which is a clean failure, not a silent fallback.
