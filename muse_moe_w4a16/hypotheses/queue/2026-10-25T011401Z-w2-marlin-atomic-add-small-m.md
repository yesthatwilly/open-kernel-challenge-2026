---
problem: muse_moe_w4a16
slug: w2-marlin-atomic-add-small-m
created: 2026-10-25T011401Z
base: master
area: w2 marlin
change: >
  For num_tokens <= 32 only (bs=1, 8, 16, 32 and the bs=4 test), pass use_atomic_add=True and
  use_fp32_reduce=False (Marlin requires the pair) to the SECOND moe_wna16_marlin_gemm call (w2)
  only, by building a separate kwargs dict from master's common for that call; the w13 call and
  every num_tokens > 32 call keep master's common exactly. Reason: w13-marlin-atomic-add-small-m
  tried this on w13 only and was neutral, but w2 is the other shape: size_n = H = 4096 (half of
  w13's 8192) at the same K = I = 4096, so Marlin has half as many N tiles to spread over 132 SMs
  and slices K more, i.e. more of w2's time is the K-slice fp32 global reduce that atomic add
  replaces. y (cache3) is then accumulated in place, so make sure cache3 is zeroed where Marlin
  expects it (Marlin's moe path with use_atomic_add zeroes or requires a zeroed C; read sglang's
  moe_wna16_marlin_gemm wrapper and zero cache3 with one .zero_() only if the wrapper does not,
  inside kernel_fn, graph-safe). Self-check in prepare: at T in {1, 4, 8, 16, 32} compare the
  final output against master's path (rel-L2 <= 5e-3, finite), else keep master and print
  '[w2-atomic] PATH CHOSEN = atomic|master (<reason>)'.
expected:
  - shapes: bs=1 *
    direction: faster
    why: Marlin pair 44.7 us at bs=1 (best profile); w2 is about 1/3 of the bytes; a removed reduce pass is worth a few us if K-slicing is active
  - shapes: bs=8 *
    direction: faster
    why: Marlin pair 114.0 us
  - shapes: bs=16 *
    direction: faster
    why: Marlin pair 176.1 us
  - shapes: bs=32 *
    direction: faster
    why: Marlin pair 207.5 us; decode_32 is the weakest load
  - shapes: bs=128 *
    direction: flat
    why: control
  - shapes: bs=256 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# w2-marlin-atomic-add-small-m

## Change

For num_tokens <= 32 only (bs=1, 8, 16, 32 and the bs=4 test), pass use_atomic_add=True and use_fp32_reduce=False (Marlin requires the pair) to the SECOND moe_wna16_marlin_gemm call (w2) only, by building a separate kwargs dict from master's common for that call; the w13 call and every num_tokens > 32 call keep master's common exactly. Reason: w13-marlin-atomic-add-small-m tried this on w13 only and was neutral, but w2 is the other shape: size_n = H = 4096 (half of w13's 8192) at the same K = I = 4096, so Marlin has half as many N tiles to spread over 132 SMs and slices K more, i.e. more of w2's time is the K-slice fp32 global reduce that atomic add replaces. y (cache3) is then accumulated in place, so make sure cache3 is zeroed where Marlin expects it (Marlin's moe path with use_atomic_add zeroes or requires a zeroed C; read sglang's moe_wna16_marlin_gemm wrapper and zero cache3 with one .zero_() only if the wrapper does not, inside kernel_fn, graph-safe). Self-check in prepare: at T in {1, 4, 8, 16, 32} compare the final output against master's path (rel-L2 <= 5e-3, finite), else keep master and print '[w2-atomic] PATH CHOSEN = atomic|master (<reason>)'.

## Expected

- `bs=1 *` faster: Marlin pair 44.7 us at bs=1 (best profile); w2 is about 1/3 of the bytes; a removed reduce pass is worth a few us if K-slicing is active
- `bs=8 *` faster: Marlin pair 114.0 us
- `bs=16 *` faster: Marlin pair 176.1 us
- `bs=32 *` faster: Marlin pair 207.5 us; decode_32 is the weakest load
- `bs=128 *` flat: control
- `bs=256 *` flat: control

## Risk

LESSONS lists use_atomic_add as a knob that did nothing, but that was measured on w13 only; this is the untested w2 half. If a zero-fill launch is needed it costs ~1-2 us and may cancel the gain; read the Marlin row and the memset row.
