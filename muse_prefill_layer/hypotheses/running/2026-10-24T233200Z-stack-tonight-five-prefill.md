---
problem: muse_prefill_layer
slug: stack-tonight-five-prefill
created: 2026-10-24T233200Z
base: master
area: router GEMM + MoE align (both paths) + w2 zero-store + Marlin silu-mul
change: >
  Built on master. stack-tonight-four-prefill (router-hilo-one-gemm-n512,
  moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn,
  moe-align-local-routes-parallel-port) plus silu-mul-triton-local-rows-marlin-turn (sorted
  local-rows silu-mul on the Marlin path, prepare() self-checked), each applied unchanged. On the
  Marlin path the silu-mul walks the parallel align's sorted_ids, which hold local routes only.
  prepare() runs _align_par_setup, then _silu_sorted_setup, then _check_experts_e2e.
expected:
  - shapes: "*"
    direction: faster
    why: the four-stack plus silu-mul on local rows at turn_1/turn_4
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: running
stage: claimed
claimed_at: 2026-10-24T233200Z
base_sha: 3b2b91dc85a1be9e6563f3f0987fd41e4d4c0841
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/stack-tonight-five-prefill
worktree: .worktrees/muse_prefill_layer/stack-tonight-five-prefill
mode: null
restack_of: null
stale_base: null
---

# stack-tonight-five-prefill

## Change

Built on master. stack-tonight-four-prefill (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn, moe-align-local-routes-parallel-port) plus silu-mul-triton-local-rows-marlin-turn (sorted local-rows silu-mul on the Marlin path, prepare() self-checked), each applied unchanged. On the Marlin path the silu-mul walks the parallel align's sorted_ids, which hold local routes only. prepare() runs _align_par_setup, then _silu_sorted_setup, then _check_experts_e2e.

## Expected

- `*` faster: the four-stack plus silu-mul on local rows at turn_1/turn_4

## Risk

Main-thread stack before the 00:00Z reset, user-approved self-review.
