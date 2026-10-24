---
problem: muse_prefill_layer
slug: align-local-parallel-first-turn
created: 2026-10-24T232955Z
base: master
area: MoE align on the Triton path (T > 4096, first_turn)
change: >
  Built on moe-align-local-routes-parallel-port (running, its implementation commit). In
  _experts_triton, when state['align_par_ok'], replace moe_align_block_size(local_ids,
  cfg['BLOCK_T'], N_EXPERTS) with _align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg['BLOCK_T'])
  (local routes only), else keep master's call. Extend _align_par_setup's self-check with T=16384
  at BLOCK_T 256 (the first_turn block) so the gate covers this path. Measures whether the
  kernel's serial per-program chunk loop (n/512 = 256 iterations at 131K routes) beats SGLang's
  count_and_sort + align (87 + 50 us at first_turn) and drops the ~448 remote token tiles and
  their w2 zero-stores.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: no remote token tiles in the Triton grouped GEMMs; one align launch
risk: medium
depends_on: moe-align-local-routes-parallel-port
priority: 2
kind: primary
controls_for: null
review: required
status: running
stage: claimed
claimed_at: 2026-10-24T232955Z
base_sha: 67f7473c299ffe95b9bb8701cdaf2079f00e3cab
base_blob: 8780061c8886bf3d70fe3c433fbbfade9bdf8507
branch: hyp/muse_prefill_layer/align-local-parallel-first-turn
worktree: .worktrees/muse_prefill_layer/align-local-parallel-first-turn
mode: null
restack_of: null
stale_base: null
---

# align-local-parallel-first-turn

## Change

Built on moe-align-local-routes-parallel-port (running, its implementation commit). In _experts_triton, when state['align_par_ok'], replace moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) with _align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg['BLOCK_T']) (local routes only), else keep master's call. Extend _align_par_setup's self-check with T=16384 at BLOCK_T 256 (the first_turn block) so the gate covers this path. Measures whether the kernel's serial per-program chunk loop (n/512 = 256 iterations at 131K routes) beats SGLang's count_and_sort + align (87 + 50 us at first_turn) and drops the ~448 remote token tiles and their w2 zero-stores.

## Expected

- `*load=first_turn*` faster: no remote token tiles in the Triton grouped GEMMs; one align launch

## Risk

Main-thread build before the 00:00Z reset, user-approved self-review.
