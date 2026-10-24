---
problem: muse_moe_w4a16
slug: marlin-block-size-m-verify-two-smaller
created: 2026-10-24T075257Z
base: master
area: w13 marlin
change: >
  Change only the block_size_m pick in kernel_fn, for num_tokens >= 128 only: take SGLang's pick
  (the first of BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) with num_tokens*8/32/block_size_m <
  0.9) and step TWO candidates down: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - 2, 0)].
  Resulting picks, SGLang -> new: bs=128: 48 -> 16 (thread_m_blocks 3 -> 1); bs=256/257: 64 -> 32
  (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps SGLang's pick as the in-run control.
  The same value feeds moe_align_block_size (bank count 32, as on master) and the moe_block_size
  kwarg of both Marlin calls; nothing else changes. Independent of
  marlin-block-size-m-next-smaller (one step down at bs=32/128/256) and run from master in
  parallel with it, so the two give the per-shape gradient of Marlin cost vs block size below
  SGLang's pick, the opposite direction of the next-larger regression (48 -> 64 lost 14% at
  bs=128).
expected:
  - shapes: bs=128 *
    direction: faster
    why: 48 -> 16 puts the verify_16 GEMMs on the thread_m_blocks=1 config; per-block cost fell steeply with smaller blocks in the next-larger run (14% at bs=128, 21% at bs=32), Marlin is 96% of 462 us
  - shapes: bs=256 *
    direction: faster
    why: 64 -> 32 (thread_m_blocks 4 -> 2); Marlin 96% of 621 us, ~6x the SoL floor
  - shapes: bs=1 *
    direction: flat
    why: "control: unchanged pick"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-block-size-m-verify-two-smaller

## Change

Change only the block_size_m pick in kernel_fn, for num_tokens >= 128 only: take SGLang's pick (the first of BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) with num_tokens*8/32/block_size_m < 0.9) and step TWO candidates down: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - 2, 0)]. Resulting picks, SGLang -> new: bs=128: 48 -> 16 (thread_m_blocks 3 -> 1); bs=256/257: 64 -> 32 (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps SGLang's pick as the in-run control. The same value feeds moe_align_block_size (bank count 32, as on master) and the moe_block_size kwarg of both Marlin calls; nothing else changes. Independent of marlin-block-size-m-next-smaller (one step down at bs=32/128/256) and run from master in parallel with it, so the two give the per-shape gradient of Marlin cost vs block size below SGLang's pick, the opposite direction of the next-larger regression (48 -> 64 lost 14% at bs=128).

## Expected

- `bs=128 *` faster: 48 -> 16 puts the verify_16 GEMMs on the thread_m_blocks=1 config; per-block cost fell steeply with smaller blocks in the next-larger run (14% at bs=128, 21% at bs=32), Marlin is 96% of 462 us
- `bs=256 *` faster: 64 -> 32 (thread_m_blocks 4 -> 2); Marlin 96% of 621 us, ~6x the SoL floor
- `bs=1 *` flat: control: unchanged pick
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control

## Risk

Risk: higher than one step down. At bs=128 the average local expert has ~32 routes (more on hot experts under skewed popularity), so 16-row blocks give 2-4+ m-blocks per expert, each streaming the expert's full NVFP4 w13/w2 slice; if Marlin at this M is weight-bandwidth-bound per m-block rather than per-block-latency-bound, this loses, possibly a lot. Read together with marlin-block-size-m-next-smaller: (one-step gain, two-step loss) brackets the optimum; (both lose) closes the knob at verify. Correctness unaffected; the bs=257 and skewed bs=128 tests run the new picks.
