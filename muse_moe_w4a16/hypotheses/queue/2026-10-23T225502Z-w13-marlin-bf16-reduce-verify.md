---
problem: muse_moe_w4a16
slug: w13-marlin-bf16-reduce-verify
created: 2026-10-23T225502Z
base: master
area: w13 marlin
change: >
  For num_tokens >= 128 (verify_16 and verify_32), pass use_fp32_reduce=False to the w13
  moe_wna16_marlin_gemm call only (use_atomic_add stays False; w2 and all decode shapes keep
  use_fp32_reduce=True). With fp32 reduce, Marlin writes each k-slice's partial C tile to an fp32
  scratch and the last slice reads them back: at bs=256 the w13 output is M*8 x 2I = 2048 x 8192,
  so each slice pass moves 67 MB of fp32 scratch (~14 us at HBM speed per pass) on a 620 us shape;
  without it partials are combined in bf16 in the output tile. Nothing else changes.
expected:
  - shapes: bs=256 *
    direction: faster
    why: 67 MB fp32 scratch per k-slice pass removed from a 620 us shape; small (2-4%) but above the +/-1.5% CI if the kernel is traffic-bound there
  - shapes: bs=128 *
    direction: faster
    why: "bs=128: 33 MB per pass on a 467 us shape; smaller effect"
  - shapes: bs=32 *
    direction: flat
    why: "control: decode shapes keep fp32 reduce"
risk: medium
depends_on: sglang-marlin-path-parity
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# w13-marlin-bf16-reduce-verify

## Change

For num_tokens >= 128 (verify_16 and verify_32), pass use_fp32_reduce=False to the w13 moe_wna16_marlin_gemm call only (use_atomic_add stays False; w2 and all decode shapes keep use_fp32_reduce=True). With fp32 reduce, Marlin writes each k-slice's partial C tile to an fp32 scratch and the last slice reads them back: at bs=256 the w13 output is M*8 x 2I = 2048 x 8192, so each slice pass moves 67 MB of fp32 scratch (~14 us at HBM speed per pass) on a 620 us shape; without it partials are combined in bf16 in the output tile. Nothing else changes.

## Expected

- `bs=256 *` faster: 67 MB fp32 scratch per k-slice pass removed from a 620 us shape; small (2-4%) but above the +/-1.5% CI if the kernel is traffic-bound there
- `bs=128 *` faster: bs=128: 33 MB per pass on a 467 us shape; smaller effect
- `bs=32 *` flat: control: decode shapes keep fp32 reduce

## Risk

Risks: bf16 combination of k-slice partials adds ~2^-8 relative error per slice (2-4 slices): well inside the 1.5% rel-L2 bound, but it is a precision knob, so the attempt file must show the per-row margin from the run; it does not lower activation precision below bf16, which SGLang keeps. The expected gain is small; priority 5. If flat, retire: Marlin is then not traffic-bound at bs=256 and the verify gap (467/620 us vs 101/109 us floors) needs a different kernel, not a knob.
