---
problem: muse_moe_w4a16
slug: w13-marlin-atomic-add-on-master-small-m
created: 2026-10-25T042150Z
base: master
area: w13 marlin
change: >
  On master (9080a71, w2 atomic add already landed for T <= 32), pass use_atomic_add=True to the
  FIRST moe_wna16_marlin_gemm (w13, top_k=8) for num_tokens <= 32 only, under its own bound
  constant W13_ATOMIC_MAX_TOKENS = 32; keep use_fp32_reduce=True (as the landed w2 change does -
  Marlin accepted it and 5/5 tests passed). Extend master's _w2_atomic_self_check pattern into a
  _w13_atomic_self_check: candidates 'atomic' (no memset) and 'atomic_zero' (cache13/cache1 zeroed
  before the call), each checked against master on the same synthetic T in (1, 4, 8, 16, 32)
  routings with rel-L2 <= 5e-3 and finiteness; print PATH CHOSEN and whether a memset is in the
  path. Nothing else changes. Mechanism: the landed w2 change removed most of bs=1's +23 us judge
  slow mode (44/100 -> 10/100 replays) by replacing the fp32-reduce lock serialization; w13 still
  uses the lock path, the likely source of the remaining 10/100. The earlier w13-only atomic test
  (0.987x) ran on the pre-post-norm base before the slow mode was measured, so this is a new
  measurement.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "remaining slow mode 10/100 -> ~0 and w13 slice reduction cheaper: 44.9 -> ~42-43 us"
  - shapes: bs=8 *
    direction: faster
    why: "small: w13 lock reduction at 8 tokens; 1-3 us at most"
  - shapes: bs=16 *
    direction: flat
    why: w2 atomic did not move it; expected noise
  - shapes: bs=32 *
    direction: flat
    why: noise
  - shapes: bs=128 *
    direction: flat
    why: control, T > 32
  - shapes: bs=256 *
    direction: flat
    why: control
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# w13-marlin-atomic-add-on-master-small-m

## Change

On master (9080a71, w2 atomic add already landed for T <= 32), pass use_atomic_add=True to the FIRST moe_wna16_marlin_gemm (w13, top_k=8) for num_tokens <= 32 only, under its own bound constant W13_ATOMIC_MAX_TOKENS = 32; keep use_fp32_reduce=True (as the landed w2 change does - Marlin accepted it and 5/5 tests passed). Extend master's _w2_atomic_self_check pattern into a _w13_atomic_self_check: candidates 'atomic' (no memset) and 'atomic_zero' (cache13/cache1 zeroed before the call), each checked against master on the same synthetic T in (1, 4, 8, 16, 32) routings with rel-L2 <= 5e-3 and finiteness; print PATH CHOSEN and whether a memset is in the path. Nothing else changes. Mechanism: the landed w2 change removed most of bs=1's +23 us judge slow mode (44/100 -> 10/100 replays) by replacing the fp32-reduce lock serialization; w13 still uses the lock path, the likely source of the remaining 10/100. The earlier w13-only atomic test (0.987x) ran on the pre-post-norm base before the slow mode was measured, so this is a new measurement.

## Expected

- `bs=1 *` faster: remaining slow mode 10/100 -> ~0 and w13 slice reduction cheaper: 44.9 -> ~42-43 us
- `bs=8 *` faster: small: w13 lock reduction at 8 tokens; 1-3 us at most
- `bs=16 *` flat: w2 atomic did not move it; expected noise
- `bs=32 *` flat: noise
- `bs=128 *` flat: control, T > 32
- `bs=256 *` flat: control

## Risk

Kill: bs=1 not below 44.9 us by > 1.5% and its slow-mode count not below 10/100; then the w13 lock path is not the source. Risk: if only 'atomic_zero' passes, the memset costs ~1-2 us and may cancel the gain.
