---
problem: muse_decode_layer
slug: stack-align-par-sidestream-sliding1
created: 2026-10-24T233721Z
base: master
area: T>32 MoE dispatch (parallel align on the side stream) + sliding batch-1 attention
change: >
  Built on master (dense-chain PDL). moe-align-parallel-large-t-port +
  moe-dispatch-prep-on-side-stream + sliding-decode1-fused-restack-on-pdl. The side-stream build
  moves the T>32 expert_map gather and moe_align_block_size into _expert_dispatch on the side
  stream and passes the result to _experts via dispatch=, which would bypass the align port's
  branch; merged so _expert_dispatch calls _align_parallel(shifted ids, 0, LOCAL_EXPERTS, bsm)
  when state['align_par_ok'] (one launch on the side stream), else master's gather + align. The
  sliding batch-1 restack is disjoint.
expected:
  - shapes: "*"
    direction: faster
    why: parallel align off the critical path at T>32, plus sliding decode_1
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# stack-align-par-sidestream-sliding1

## Change

Built on master (dense-chain PDL). moe-align-parallel-large-t-port + moe-dispatch-prep-on-side-stream + sliding-decode1-fused-restack-on-pdl. The side-stream build moves the T>32 expert_map gather and moe_align_block_size into _expert_dispatch on the side stream and passes the result to _experts via dispatch=, which would bypass the align port's branch; merged so _expert_dispatch calls _align_parallel(shifted ids, 0, LOCAL_EXPERTS, bsm) when state['align_par_ok'] (one launch on the side stream), else master's gather + align. The sliding batch-1 restack is disjoint.

## Expected

- `*` faster: parallel align off the critical path at T>32, plus sliding decode_1

## Risk

Main-thread hand merge before the 00:00Z reset, user-approved self-review.
