---
problem: muse_moe_w4a16
slug: w2-marlin-atomic-add-verify
created: 2026-10-25T042151Z
base: master
area: w2 marlin
change: >
  On master (9080a71), widen the landed w2 use_atomic_add path from num_tokens <= 32 to all T by
  giving the verify range its own bound constant W2_ATOMIC_VERIFY = True applying at num_tokens >
  32 (bs=128, 256 and the bs=33/257/skewed-128 tests); T <= 32 keeps master's landed
  W2_ATOMIC_MAX_TOKENS path unchanged. Re-run master's _w2_atomic_self_check at T in (33, 128,
  256, 257) with skewed and random routings, its own 'atomic'/'atomic_zero'/'master' decision and
  PATH CHOSEN line; use_fp32_reduce stays True. Note on bs=8/16: master already applies w2 atomic
  at T <= 32, so there is no smaller-M extension left to test. Mechanism: the verify Marlin rows
  swing 391-451 us at bs=256 across identical-code runs today, the same symptom (a lock-serialized
  reduction slow mode) that w2 atomic removed at bs=1. At bs=128/256, block_size_m is 8/16
  (two-smaller pick) so w2 has many slices per output tile and the fp32 reduce takes the lock
  path.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "Marlin pair 264-292 us; if w2's reduce slow mode exists, -5 to -15 us"
  - shapes: bs=256 *
    direction: faster
    why: Marlin pair 391-451 us run to run; target the low end consistently, -10 to -30 us
  - shapes: bs=1 *
    direction: flat
    why: "control: master's landed path"
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
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# w2-marlin-atomic-add-verify

## Change

On master (9080a71), widen the landed w2 use_atomic_add path from num_tokens <= 32 to all T by giving the verify range its own bound constant W2_ATOMIC_VERIFY = True applying at num_tokens > 32 (bs=128, 256 and the bs=33/257/skewed-128 tests); T <= 32 keeps master's landed W2_ATOMIC_MAX_TOKENS path unchanged. Re-run master's _w2_atomic_self_check at T in (33, 128, 256, 257) with skewed and random routings, its own 'atomic'/'atomic_zero'/'master' decision and PATH CHOSEN line; use_fp32_reduce stays True. Note on bs=8/16: master already applies w2 atomic at T <= 32, so there is no smaller-M extension left to test. Mechanism: the verify Marlin rows swing 391-451 us at bs=256 across identical-code runs today, the same symptom (a lock-serialized reduction slow mode) that w2 atomic removed at bs=1. At bs=128/256, block_size_m is 8/16 (two-smaller pick) so w2 has many slices per output tile and the fp32 reduce takes the lock path.

## Expected

- `bs=128 *` faster: Marlin pair 264-292 us; if w2's reduce slow mode exists, -5 to -15 us
- `bs=256 *` faster: Marlin pair 391-451 us run to run; target the low end consistently, -10 to -30 us
- `bs=1 *` flat: control: master's landed path
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control

## Risk

Kill: bs=128 and bs=256 within noise or slower (atomic contention on the 32-expert outputs at 1024-2048 routes can lose). Risk: atomic adds into bf16 at large M may lose precision vs fp32 reduce; the self-check bound 5e-3 vs master catches it.
