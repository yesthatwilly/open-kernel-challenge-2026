---
problem: muse_moe_w4a16
slug: stack-onewarp-hoisted-dispatchcombine
created: 2026-10-25T000436Z
base: master
area: MoE align (bs<=8 one warp; bs>=128 CUDA) + post-expert combine (bs<32 hoisted; bs>=128 CUDA)
change: >
  Built on master. Stack the three moe results of 2026-10-24 that each matched the 1.3278 best
  within noise, applied unchanged from their done/ patches: align-small-one-warp-tiny-m (1-warp
  _align_small_kernel at T<=8, scored 1.3287), post-norm-combine-hoisted-slot-loads (all 8 slot
  loads before the reduction, bs<32, scored 1.3282) and cuda-dispatch-combine-pdl-verify (CUDA
  align + PDL post-norm combine at 32<T<=512, scored 1.3283). Their shapes and kernels are
  disjoint except the post-norm combine, which the hoisted change edits only at bs<32 and the CUDA
  rewrite replaces only above 32 tokens. Tests whether sub-noise wins add up, as prefill's stack
  showed (1.059 -> 1.0744).
expected:
  - shapes: "*"
    direction: faster
    why: three disjoint sub-noise gains combined
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# stack-onewarp-hoisted-dispatchcombine

## Change

Built on master. Stack the three moe results of 2026-10-24 that each matched the 1.3278 best within noise, applied unchanged from their done/ patches: align-small-one-warp-tiny-m (1-warp _align_small_kernel at T<=8, scored 1.3287), post-norm-combine-hoisted-slot-loads (all 8 slot loads before the reduction, bs<32, scored 1.3282) and cuda-dispatch-combine-pdl-verify (CUDA align + PDL post-norm combine at 32<T<=512, scored 1.3283). Their shapes and kernels are disjoint except the post-norm combine, which the hoisted change edits only at bs<32 and the CUDA rewrite replaces only above 32 tokens. Tests whether sub-noise wins add up, as prefill's stack showed (1.059 -> 1.0744).

## Expected

- `*` faster: three disjoint sub-noise gains combined

## Risk

Reset priority 4 (user-requested stack, 2026-10-24 ~23:50Z); claimed with --skip-review on that instruction.
