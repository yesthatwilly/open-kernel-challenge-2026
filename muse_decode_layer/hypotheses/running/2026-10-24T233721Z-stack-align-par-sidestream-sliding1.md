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
status: running
stage: claimed
claimed_at: 2026-10-24T233722Z
base_sha: 90c5720f9cb1f796aeb6a987abdd676cba19ef5f
base_blob: 3d70ae6e2de0d48c4a4f2a469ced8234f5348182
branch: hyp/muse_decode_layer/stack-align-par-sidestream-sliding1
worktree: .worktrees/muse_decode_layer/stack-align-par-sidestream-sliding1
mode: null
restack_of: null
stale_base: null
---

# stack-align-par-sidestream-sliding1

## Change

Built on master (dense-chain PDL). moe-align-parallel-large-t-port + moe-dispatch-prep-on-side-stream + sliding-decode1-fused-restack-on-pdl. The side-stream build moves the T>32 expert_map gather and moe_align_block_size into _expert_dispatch on the side stream and passes the result to _experts via dispatch=, which would bypass the align port's branch; merged so _expert_dispatch calls _align_parallel(shifted ids, 0, LOCAL_EXPERTS, bsm) when state['align_par_ok'] (one launch on the side stream), else master's gather + align. The sliding batch-1 restack is disjoint.

## Expected

- `*` faster: parallel align off the critical path at T>32, plus sliding decode_1

## Risk

Main-thread hand merge before the 00:00Z reset, user-approved self-review.
