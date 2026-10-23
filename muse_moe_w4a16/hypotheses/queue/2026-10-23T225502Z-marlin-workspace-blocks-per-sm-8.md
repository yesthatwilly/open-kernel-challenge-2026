---
problem: muse_moe_w4a16
slug: marlin-workspace-blocks-per-sm-8
created: 2026-10-23T225502Z
base: master
area: w13 marlin
change: >
  In prepare(), build a second lock workspace marlin_make_workspace(device, max_blocks_per_sm=8)
  and pass it to the w13 moe_wna16_marlin_gemm call only; the w2 call keeps the
  max_blocks_per_sm=4 workspace SGLang's MoE runner uses (its dense runner uses 1). The Marlin
  runner sizes the number of resident CTAs it launches from the workspace's lock count (sms *
  max_blocks_per_sm), capped by what the kernel config's shared memory allows, so a larger
  workspace lets more CTAs be resident and more k-slices run in parallel for the verify shapes
  where the w13 grid (2048 rows x 8192 cols) has work to spare. Nothing else changes.
expected:
  - shapes: bs=256 *
    direction: faster
    why: "bs=256: more resident CTAs on a 620 us kernel that is neither at the 109 us bandwidth floor nor near compute peak (~40 GFLOP), i.e. latency/occupancy-bound"
  - shapes: bs=128 *
    direction: faster
    why: same at bs=128 (467 us vs 101 us floor)
  - shapes: bs=1 *
    direction: flat
    why: bs=1 has too few tiles to fill 132 SMs x 4 either way
  - shapes: bs=32 *
    direction: flat
    why: control
risk: low
depends_on: sglang-marlin-path-parity
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-workspace-blocks-per-sm-8

## Change

In prepare(), build a second lock workspace marlin_make_workspace(device, max_blocks_per_sm=8) and pass it to the w13 moe_wna16_marlin_gemm call only; the w2 call keeps the max_blocks_per_sm=4 workspace SGLang's MoE runner uses (its dense runner uses 1). The Marlin runner sizes the number of resident CTAs it launches from the workspace's lock count (sms * max_blocks_per_sm), capped by what the kernel config's shared memory allows, so a larger workspace lets more CTAs be resident and more k-slices run in parallel for the verify shapes where the w13 grid (2048 rows x 8192 cols) has work to spare. Nothing else changes.

## Expected

- `bs=256 *` faster: bs=256: more resident CTAs on a 620 us kernel that is neither at the 109 us bandwidth floor nor near compute peak (~40 GFLOP), i.e. latency/occupancy-bound
- `bs=128 *` faster: same at bs=128 (467 us vs 101 us floor)
- `bs=1 *` flat: bs=1 has too few tiles to fill 132 SMs x 4 either way
- `bs=32 *` flat: control

## Risk

Risks: the kernel may already cap residency by shared memory (Marlin stages use most of the 228 KB) so the knob is inert and the result flat; a larger lock array is harmless (zeroed once in prepare, reset by the kernel). If flat, record that occupancy is not Marlin's limiter at these shapes; if a gain, the ab-tune is 2 and 16, and the same workspace on w2.
