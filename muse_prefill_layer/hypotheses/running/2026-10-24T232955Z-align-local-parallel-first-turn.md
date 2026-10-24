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
stage: implemented
claimed_at: 2026-10-24T232955Z
base_sha: 67f7473c299ffe95b9bb8701cdaf2079f00e3cab
base_blob: 8780061c8886bf3d70fe3c433fbbfade9bdf8507
branch: hyp/muse_prefill_layer/align-local-parallel-first-turn
worktree: .worktrees/muse_prefill_layer/align-local-parallel-first-turn
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233023Z
impl_commit: 0b61b2b6f3bb4a07433e7bea0e169e70d46d0894
submission_sha256: 8177399e276e2bedb4909471f4c254855cefe2f038c518e9d9a63b93fc8f34cb
norm_sha256: 35c68a631fd3932f72e0d779eeda2bc5b8a3e654c1334fa62fc41f2fd1058792
submission_blob: 32d9f1e7e5c9c472ea3b8c05d97fa7de566e6135
lint_warnings: 12
attempt_file: 2026-10-24T233023Z-check-align-local-parallel-first-turn.md
prepared_at: 2026-10-24T233023Z
---

# align-local-parallel-first-turn

## Change

Built on moe-align-local-routes-parallel-port (running, its implementation commit). In _experts_triton, when state['align_par_ok'], replace moe_align_block_size(local_ids, cfg['BLOCK_T'], N_EXPERTS) with _align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg['BLOCK_T']) (local routes only), else keep master's call. Extend _align_par_setup's self-check with T=16384 at BLOCK_T 256 (the first_turn block) so the gate covers this path. Measures whether the kernel's serial per-program chunk loop (n/512 = 256 iterations at 131K routes) beats SGLang's count_and_sort + align (87 + 50 us at first_turn) and drops the ~448 remote token tiles and their w2 zero-stores.

## Expected

- `*load=first_turn*` faster: no remote token tiles in the Triton grouped GEMMs; one align launch

## Risk

Main-thread build before the 00:00Z reset, user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:30Z with user-approved self-review, on moe-align-local-routes-parallel-port's implementation commit (claim based it there: parent still running). `_experts_triton`: `_align_parallel(local_ids, 0, LOCAL_EXPERTS, cfg["BLOCK_T"])` when `state["align_par_ok"]`, else master's call. `_align_par_setup` now also tests T=16384 at the Triton path's BLOCK_T (256), which also compiles that variant before capture; a mismatch there turns the whole parallel align OFF (both paths), never just one.

## Rules check

- [x] CUDA-graph capturable: shape-only allocations, prepare()-set gate, no host sync in kernel_fn.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.
