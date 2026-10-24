---
problem: muse_bf16_proj
slug: router-splitk32-small-m-all-loads-in-flight-control-stages3
created: 2026-10-24T103959Z
base: master
area: router GEMM
change: >
  From the parent's branch (router-splitk32-small-m-all-loads-in-flight), keep everything exactly
  as the parent: the new m<=16 bucket (BLOCK_M=16, BLOCK_N=32, SPLIT_K=32) and 17..32 bucket
  (BLOCK_M=32, BLOCK_N=32, SPLIT_K=32), grid (1, 8, 32) = 256 CTAs, K-slice 256, BLOCK_K=128,
  num_warps=4, the torch.zeros output with the relaxed fp32 atomic_add epilogue, the unchanged
  m>32 buckets, and _router_config returning (BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES). Change one
  thing: the two small-m buckets return NUM_STAGES=3 (master's ROUTER_NUM_STAGES) instead of 2, so
  every bucket launches with num_stages=3. ROUTER_WARM_MS is unchanged and still compiles both
  specializations in prepare().
expected:
  - shapes: layer=router load=decode_1 *
    direction: faster
    why: 256 CTAs with both K-steps issued in the prologue (num_stages-1 = 2 >= the 2 K-iterations) give the full grid-depth effect without the 2-stage restructure
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: same bucket as decode_1
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: same bucket
  - shapes: layer=router load=decode_32 *
    direction: flat
    why: doubled atomics (262K fp32) may offset the gain from the deeper grid at m=32
  - shapes: layer=router load=verify_16 *
    direction: flat
    why: m=128 bucket unchanged
  - shapes: layer=router load=verify_32 *
    direction: flat
    why: m=256 bucket unchanged
risk: low
depends_on: router-splitk32-small-m-all-loads-in-flight
priority: 3
kind: control
controls_for: router-splitk32-small-m-all-loads-in-flight
review: required
status: queued
---

# router-splitk32-small-m-all-loads-in-flight-control-stages3

## Change

From the parent's branch (router-splitk32-small-m-all-loads-in-flight), keep everything exactly as the parent: the new m<=16 bucket (BLOCK_M=16, BLOCK_N=32, SPLIT_K=32) and 17..32 bucket (BLOCK_M=32, BLOCK_N=32, SPLIT_K=32), grid (1, 8, 32) = 256 CTAs, K-slice 256, BLOCK_K=128, num_warps=4, the torch.zeros output with the relaxed fp32 atomic_add epilogue, the unchanged m>32 buckets, and _router_config returning (BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES). Change one thing: the two small-m buckets return NUM_STAGES=3 (master's ROUTER_NUM_STAGES) instead of 2, so every bucket launches with num_stages=3. ROUTER_WARM_MS is unchanged and still compiles both specializations in prepare().

## Expected

- `layer=router load=decode_1 *` faster: 256 CTAs with both K-steps issued in the prologue (num_stages-1 = 2 >= the 2 K-iterations) give the full grid-depth effect without the 2-stage restructure
- `layer=router load=verify_1 *` faster: same bucket as decode_1
- `layer=router load=decode_16 *` faster: same bucket
- `layer=router load=decode_32 *` flat: doubled atomics (262K fp32) may offset the gain from the deeper grid at m=32
- `layer=router load=verify_16 *` flat: m=128 bucket unchanged
- `layer=router load=verify_32 *` flat: m=256 bucket unchanged

## Risk

Separates the SPLIT_K 16->32 grid-depth effect (control vs master) from the num_stages 3->2 effect (parent vs control). Without it, a loss could come from either the atomics or the stage change, and a gain cannot tell the semaphore-reduce follow-up which SPLIT_K to build on.
