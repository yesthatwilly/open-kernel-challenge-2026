---
problem: muse_prefill_layer
slug: experts-marlin-block-m-48-turn4
created: 2026-10-24T075435Z
base: master
area: w13 marlin + w2 marlin (block_size_m pick)
change: >
  Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 2048 <
  num_tokens <= 4095 use block_size_m = 48; every other token count keeps the parity loop's pick
  (64 at T=753 and T=16384, and the parity picks at all tests; T=4096 is excluded by the bound, so
  no test moves). In the scored set this moves only the two turn_4 shapes (T=3012, parity pick
  64). The same block_size_m goes to moe_align_block_size and to the moe_block_size kwarg of both
  moe_wna16_marlin_gemm calls. Nothing else changes. Mechanism: turn_4 has ~94 rows per local
  expert on average (T*8/256), so block 64 needs 2 m-blocks per expert at thread_m_blocks 4 (128
  padded rows) and block 48 needs 2 for experts up to 96 rows and 3 above, at thread_m_blocks 3.
  It is a probe of the Marlin cost curve at real (not padded-tiny) M: in the parity profile Marlin
  is 2248/2201 us of the turn_4 shapes (64%/43%), ~0.3 TFLOP at ~135 TFLOP/s and ~2 weight passes
  of 0.9 GB at ~0.8 TB/s effective, so neither the MMA nor the weight stream is saturated, and a
  per-block-cost model (sister muse_moe_w4a16: +18% for tmb 3->4) predicts the cheaper blocks pay
  for the ~20% more blocks only if the per-block cost dominates the weight passes.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "thread_m_blocks 4 -> 3 at ~94 rows/expert: ~15% cheaper blocks against ~20% more blocks from skewed experts above 96 rows; a gain needs the per-block cost to dominate, a loss says weight passes dominate at this M (both are informative for the Marlin cost model)"
  - shapes: "*load=turn_1*"
    direction: flat
    why: T=753 keeps the parity 64 here (control)
  - shapes: "*load=first_turn*"
    direction: flat
    why: T=16384 keeps 64 (control)
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: hold
hold_reason: block 48 at turn_1 was neutral (1.003x, 0606f849d2f441ad) and local-rows regressed (0.984x); block-m tuning on prefill is settled, and prefill has 3 checks for the rewrite
---

# experts-marlin-block-m-48-turn4

## Change

Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 2048 < num_tokens <= 4095 use block_size_m = 48; every other token count keeps the parity loop's pick (64 at T=753 and T=16384, and the parity picks at all tests; T=4096 is excluded by the bound, so no test moves). In the scored set this moves only the two turn_4 shapes (T=3012, parity pick 64). The same block_size_m goes to moe_align_block_size and to the moe_block_size kwarg of both moe_wna16_marlin_gemm calls. Nothing else changes. Mechanism: turn_4 has ~94 rows per local expert on average (T*8/256), so block 64 needs 2 m-blocks per expert at thread_m_blocks 4 (128 padded rows) and block 48 needs 2 for experts up to 96 rows and 3 above, at thread_m_blocks 3. It is a probe of the Marlin cost curve at real (not padded-tiny) M: in the parity profile Marlin is 2248/2201 us of the turn_4 shapes (64%/43%), ~0.3 TFLOP at ~135 TFLOP/s and ~2 weight passes of 0.9 GB at ~0.8 TB/s effective, so neither the MMA nor the weight stream is saturated, and a per-block-cost model (sister muse_moe_w4a16: +18% for tmb 3->4) predicts the cheaper blocks pay for the ~20% more blocks only if the per-block cost dominates the weight passes.

## Expected

- `*load=turn_4*` faster: thread_m_blocks 4 -> 3 at ~94 rows/expert: ~15% cheaper blocks against ~20% more blocks from skewed experts above 96 rows; a gain needs the per-block cost to dominate, a loss says weight passes dominate at this M (both are informative for the Marlin cost model)
- `*load=turn_1*` flat: T=753 keeps the parity 64 here (control)
- `*load=first_turn*` flat: T=16384 keeps 64 (control)

## Risk

Honest prior: near flat, either sign possible; worth one cheap run because turn_4 Marlin is 2.2 ms (a 10% move is ~6% of a turn_4 shape, ~2% geomean) and it calibrates block size at real M before anyone tunes first_turn. Routing skew (reference.py _router) makes the 2-vs-3 block split data-dependent per shape, so sliding and full turn_4 may move by different amounts. Independent of experts-marlin-block-m-local-rows / -48-turn1 (disjoint token ranges); a later combination is a merge of two picks.
