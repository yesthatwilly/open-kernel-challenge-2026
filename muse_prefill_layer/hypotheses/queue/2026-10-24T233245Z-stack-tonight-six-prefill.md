---
problem: muse_prefill_layer
slug: stack-tonight-six-prefill
created: 2026-10-24T233245Z
base: master
area: router GEMM + MoE align (both paths, parallel) + w2 zero-store + Marlin silu-mul
change: >
  Built on master. stack-tonight-five-prefill plus align-local-parallel-first-turn (the parallel
  local-routes align on the Triton path too, self-checked at T=16384, BLOCK_T 256), each applied
  unchanged. With the parallel align on at first_turn, moe-align-32-experts-first-turn's
  moe_align_block_size(LOCAL_EXPERTS) call becomes the fallback branch only.
expected:
  - shapes: "*"
    direction: faster
    why: the five-stack plus no remote token tiles at first_turn
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# stack-tonight-six-prefill

## Change

Built on master. stack-tonight-five-prefill plus align-local-parallel-first-turn (the parallel local-routes align on the Triton path too, self-checked at T=16384, BLOCK_T 256), each applied unchanged. With the parallel align on at first_turn, moe-align-32-experts-first-turn's moe_align_block_size(LOCAL_EXPERTS) call becomes the fallback branch only.

## Expected

- `*` faster: the five-stack plus no remote token tiles at first_turn

## Risk

Main-thread stack before the 00:00Z reset, user-approved self-review.
