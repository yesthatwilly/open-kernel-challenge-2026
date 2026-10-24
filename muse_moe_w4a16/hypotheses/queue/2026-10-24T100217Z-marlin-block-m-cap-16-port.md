---
problem: muse_moe_w4a16
slug: marlin-block-m-cap-16-port
created: 2026-10-24T100217Z
base: master
area: w13 marlin
change: >
  Port muse_decode_layer marlin-block-m-cap-16 (landed gain 1.062x, job 0730135cdcac43b7; patch in
  muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.patch) to kernel_fn in
  muse_moe_w4a16/submission.py. Add a module constant MARLIN_BLOCK_SIZE_M_CAP = 16 beside
  BLOCK_SIZE_M_CANDIDATES, and immediately after SGLang unchanged selection loop (first candidate
  in (8, 16, 32, 48, 64) with num_tokens*8/bank["count"]/block_size_m < 0.9, loop exhausting at
  64) add the single line block_size_m = min(block_size_m, MARLIN_BLOCK_SIZE_M_CAP). Master picks,
  verified from the rule with bank count 32: bs=1/4/8/16 -> 8, bs=32/33 -> 16, bs=128 -> 48,
  bs=256/257 -> 64 (not 32 at bs=128). After the cap: bs=128 48 -> 16 (thread_m_blocks 3 -> 1),
  bs=256/257 64 -> 16 (4 -> 1); every bs<=33 is unchanged (8 and 16 are already <= 16) and is the
  in-run control. bs=32 is deliberately left at 16 rather than 8: 8 and 16 both run
  thread_m_blocks=1 (8 only switches to the m_block_size_8 special path), the only evidence on
  that step is next-larger bs=1/8/16 rows at 2-3% (inside noise once extra m-blocks for 9-16-route
  experts are paid), decode_layer kept 16 at T=32 when it gained, and
  marlin-block-size-m-next-smaller already tests bs=32 16 -> 8 from master. The capped value feeds
  moe_align_block_size (still over the bank 32 experts, and moe_align_single_token at bs=1 which
  is unaffected) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls (w13 and w2), as
  it must. Nothing else changes: no input-dependent branch, no sync, the branch is on num_tokens
  only. Mechanism (as measured, not as decode first argued): per-m-block Marlin cost grows with
  thread_m_blocks (next-larger here: 48 -> 64 at bs=128 took Marlin 495 -> 662 us; 16 -> 32 at
  bs=32 lost 21%), while SGLang T*8/32 rule overcounts routes per local expert 8x under EP8 (true
  ~4-8 routes/expert at bs=128, ~8-16 at bs=256 on the busiest GPU), so block 16 adds an extra
  m-block (another weight pass) only for a few hot experts.
expected:
  - shapes: bs=256 *
    direction: faster
    why: 64 -> 16 (thread_m_blocks 4 -> 1), the point neither in-flight sibling tests; decode_layer gained 1.08-1.13x on its T=256 shapes with this exact change; Marlin is 612 of 621 us here at 5.7x SoL
  - shapes: bs=128 *
    direction: faster
    why: 48 -> 16 (3 -> 1); decode_layer gained 1.10-1.15x at T=128 (Marlin ~425 -> 293 us); next-larger showed the opposite step costing 14% here; Marlin 96% of 462 us
  - shapes: bs=32 *
    direction: flat
    why: "control: keeps 16 (bs=32 16 -> 8 is next-smaller own arm)"
  - shapes: bs=16 *
    direction: flat
    why: "control: keeps 8"
  - shapes: bs=8 *
    direction: flat
    why: "control: keeps 8"
  - shapes: bs=1 *
    direction: flat
    why: "control: keeps 8"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-block-m-cap-16-port

## Change

Port muse_decode_layer marlin-block-m-cap-16 (landed gain 1.062x, job 0730135cdcac43b7; patch in muse_decode_layer/hypotheses/done/gain/marlin-block-m-cap-16.patch) to kernel_fn in muse_moe_w4a16/submission.py. Add a module constant MARLIN_BLOCK_SIZE_M_CAP = 16 beside BLOCK_SIZE_M_CANDIDATES, and immediately after SGLang unchanged selection loop (first candidate in (8, 16, 32, 48, 64) with num_tokens*8/bank["count"]/block_size_m < 0.9, loop exhausting at 64) add the single line block_size_m = min(block_size_m, MARLIN_BLOCK_SIZE_M_CAP). Master picks, verified from the rule with bank count 32: bs=1/4/8/16 -> 8, bs=32/33 -> 16, bs=128 -> 48, bs=256/257 -> 64 (not 32 at bs=128). After the cap: bs=128 48 -> 16 (thread_m_blocks 3 -> 1), bs=256/257 64 -> 16 (4 -> 1); every bs<=33 is unchanged (8 and 16 are already <= 16) and is the in-run control. bs=32 is deliberately left at 16 rather than 8: 8 and 16 both run thread_m_blocks=1 (8 only switches to the m_block_size_8 special path), the only evidence on that step is next-larger bs=1/8/16 rows at 2-3% (inside noise once extra m-blocks for 9-16-route experts are paid), decode_layer kept 16 at T=32 when it gained, and marlin-block-size-m-next-smaller already tests bs=32 16 -> 8 from master. The capped value feeds moe_align_block_size (still over the bank 32 experts, and moe_align_single_token at bs=1 which is unaffected) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls (w13 and w2), as it must. Nothing else changes: no input-dependent branch, no sync, the branch is on num_tokens only. Mechanism (as measured, not as decode first argued): per-m-block Marlin cost grows with thread_m_blocks (next-larger here: 48 -> 64 at bs=128 took Marlin 495 -> 662 us; 16 -> 32 at bs=32 lost 21%), while SGLang T*8/32 rule overcounts routes per local expert 8x under EP8 (true ~4-8 routes/expert at bs=128, ~8-16 at bs=256 on the busiest GPU), so block 16 adds an extra m-block (another weight pass) only for a few hot experts.

## Expected

- `bs=256 *` faster: 64 -> 16 (thread_m_blocks 4 -> 1), the point neither in-flight sibling tests; decode_layer gained 1.08-1.13x on its T=256 shapes with this exact change; Marlin is 612 of 621 us here at 5.7x SoL
- `bs=128 *` faster: 48 -> 16 (3 -> 1); decode_layer gained 1.10-1.15x at T=128 (Marlin ~425 -> 293 us); next-larger showed the opposite step costing 14% here; Marlin 96% of 462 us
- `bs=32 *` flat: control: keeps 16 (bs=32 16 -> 8 is next-smaller own arm)
- `bs=16 *` flat: control: keeps 8
- `bs=8 *` flat: control: keeps 8
- `bs=1 *` flat: control: keeps 8

## Risk

Risk: the decode_layer evidence was measured with 256-expert align and dist=soft routing; here the align is over 32 experts and routing is skewed so this GPU is the busiest, so hot experts may hold more than 16 local routes at bs=256 and pay 2+ m-blocks (each a full pass over that expert 16 MB w13 / 8 MB w2 slice, partly from L2). If bs=256 is flat or loses while bs=128 gains, read it as hot-expert re-streaming and prefer the verify-two-smaller pick (32) at bs=256. Correctness unaffected (same fp32 reduce, same math); the bs=257 and skewed bs=128 tests run the new picks. Sweep: with marlin-block-size-m-next-smaller (one step down: bs=128 -> 32, bs=256 -> 48, bs=32 -> 8) and marlin-block-size-m-verify-two-smaller (two steps down: bs=128 -> 16, bs=256 -> 32), this run (cap 16: bs=256 -> 16) completes a 3-point sweep at bs=256 (48/32/16) against SGLang 64. At bs=128 this pick (16) coincides with verify-two-smaller; that overlap is intended for the direct port but means only one of the two needs to spend a check on bs=128, and the bs=128 rows of whichever runs second are a reproducibility read, not new information. All three edit the same line from master, so at most one lands; the others become stacked re-runs or are closed by this result. Run this one first if checks are limited: it is the replicated design.
