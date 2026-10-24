---
problem: muse_prefill_layer
slug: stack-tonight-four-prefill
created: 2026-10-24T232928Z
base: master
area: router GEMM + MoE align (both paths) + w2 zero-store
change: >
  Built on master. Stack all four prefill changes built on 2026-10-24 night, each applied
  unchanged: router-hilo-one-gemm-n512, moe-align-32-experts-first-turn,
  w2-nonlocal-zero-store-off-first-turn (as in restack-router-hilo-align32-zerostore) plus
  moe-align-local-routes-parallel-port (parallel local-routes align on the Marlin path, prepare()
  self-checked). The four touch disjoint lines (router repack and GEMM; the Triton-path align; the
  zero-store probe and w2 flag; the Marlin-path align and its setup), so the run reads their
  combined effect against master and against the two parents.
expected:
  - shapes: "*"
    direction: faster
    why: router one GEMM everywhere, fewer Marlin m-blocks at turn_1/turn_4, smaller align and no zero-stores at first_turn
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: running
stage: claimed
claimed_at: 2026-10-24T232928Z
base_sha: eb35e7e2a89baeba03be2ed19b88daff5128f06a
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/stack-tonight-four-prefill
worktree: .worktrees/muse_prefill_layer/stack-tonight-four-prefill
mode: null
restack_of: null
stale_base: null
---

# stack-tonight-four-prefill

## Change

Built on master. Stack all four prefill changes built on 2026-10-24 night, each applied unchanged: router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn (as in restack-router-hilo-align32-zerostore) plus moe-align-local-routes-parallel-port (parallel local-routes align on the Marlin path, prepare() self-checked). The four touch disjoint lines (router repack and GEMM; the Triton-path align; the zero-store probe and w2 flag; the Marlin-path align and its setup), so the run reads their combined effect against master and against the two parents.

## Expected

- `*` faster: router one GEMM everywhere, fewer Marlin m-blocks at turn_1/turn_4, smaller align and no zero-stores at first_turn

## Risk

Stack built in the main thread before the 00:00Z reset with user-approved self-review.
