---
problem: muse_prefill_layer
slug: silu-mul-sorted-first-turn
created: 2026-10-24T233334Z
base: master
area: silu-mul on the Triton path (T > 4096, first_turn)
change: >
  Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_triton,
  when state['silu_sorted_ok'], replace silu_and_mul(cache1, cache2,
  expert_ids=local_ids.view(-1), expert_step=1) with _silu_mul_sorted(cache1, cache2, sorted_ids,
  n_padded, local_ids.view(-1), size_n) over the Triton path's align (BLOCK_T 256); extend
  _silu_sorted_setup with T=16384 at BLOCK_T 256 so the gate covers it. act_and_mul costs ~297 us
  at first_turn against a ~84 us local-rows floor. experts-triton-w13-fused-silu-epilogue, the
  reason the parent excluded this path, is done (neutral), so nothing is confounded.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: silu-mul over local rows only at first_turn
risk: medium
depends_on: silu-mul-triton-local-rows-marlin-turn
priority: 1
kind: primary
controls_for: null
review: required
status: running
stage: claimed
claimed_at: 2026-10-24T233335Z
base_sha: d4d4c077f75b6689b16c96c62fb5598b1756d3b5
base_blob: 1a06deda403ddee4b4693b34127fd50b8bcb791c
branch: hyp/muse_prefill_layer/silu-mul-sorted-first-turn
worktree: .worktrees/muse_prefill_layer/silu-mul-sorted-first-turn
mode: null
restack_of: null
stale_base: null
---

# silu-mul-sorted-first-turn

## Change

Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_triton, when state['silu_sorted_ok'], replace silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1) with _silu_mul_sorted(cache1, cache2, sorted_ids, n_padded, local_ids.view(-1), size_n) over the Triton path's align (BLOCK_T 256); extend _silu_sorted_setup with T=16384 at BLOCK_T 256 so the gate covers it. act_and_mul costs ~297 us at first_turn against a ~84 us local-rows floor. experts-triton-w13-fused-silu-epilogue, the reason the parent excluded this path, is done (neutral), so nothing is confounded.

## Expected

- `*load=first_turn*` faster: silu-mul over local rows only at first_turn

## Risk

Main-thread build before the 00:00Z reset, user-approved self-review.
