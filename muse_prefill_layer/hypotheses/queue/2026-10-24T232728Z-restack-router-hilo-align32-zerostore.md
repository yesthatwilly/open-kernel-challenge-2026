---
problem: muse_prefill_layer
slug: restack-router-hilo-align32-zerostore
created: 2026-10-24T232728Z
base: master
area: router GEMM + MoE align (T>4096) + w2 zero-store (T>4096)
change: >
  Built on master. Stack the three shape-disjoint prefill checks of 2026-10-24 23:15Z, each
  applied unchanged from its done/ patch: router-hilo-one-gemm-n512 (one N=512 GEMM for the router
  hi|lo halves, f4e3c5f4c306427b 1.0607), moe-align-32-experts-first-turn (moe_align_block_size
  over LOCAL_EXPERTS in _experts_triton, b5bf94b463d34d21 1.0604) and
  w2-nonlocal-zero-store-off-first-turn (prepare() probe turns the w2 zero-store off when the
  weighted sum skips remote slots, 9b731599dfa24278 1.0609). Each alone is within noise of master
  (1.059); together the router change touches every shape and the other two first_turn only, so
  the stack reads their combined effect.
expected:
  - shapes: "*"
    direction: faster
    why: "sum of three sub-noise gains: router one GEMM at every T, smaller align capacity and no zero-stores at first_turn"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# restack-router-hilo-align32-zerostore

## Change

Built on master. Stack the three shape-disjoint prefill checks of 2026-10-24 23:15Z, each applied unchanged from its done/ patch: router-hilo-one-gemm-n512 (one N=512 GEMM for the router hi|lo halves, f4e3c5f4c306427b 1.0607), moe-align-32-experts-first-turn (moe_align_block_size over LOCAL_EXPERTS in _experts_triton, b5bf94b463d34d21 1.0604) and w2-nonlocal-zero-store-off-first-turn (prepare() probe turns the w2 zero-store off when the weighted sum skips remote slots, 9b731599dfa24278 1.0609). Each alone is within noise of master (1.059); together the router change touches every shape and the other two first_turn only, so the stack reads their combined effect.

## Expected

- `*` faster: sum of three sub-noise gains: router one GEMM at every T, smaller align capacity and no zero-stores at first_turn

## Risk

Restack built in the main thread before the 00:00Z reset with user-approved self-review.
