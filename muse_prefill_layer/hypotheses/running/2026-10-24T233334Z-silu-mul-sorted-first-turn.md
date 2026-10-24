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
stage: implemented
claimed_at: 2026-10-24T233335Z
base_sha: d4d4c077f75b6689b16c96c62fb5598b1756d3b5
base_blob: 1a06deda403ddee4b4693b34127fd50b8bcb791c
branch: hyp/muse_prefill_layer/silu-mul-sorted-first-turn
worktree: .worktrees/muse_prefill_layer/silu-mul-sorted-first-turn
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233357Z
impl_commit: a30a516734009c7b29043c582a0e8f006f9dd83e
submission_sha256: 1c7c6b27c04f1afe3fee01ce659e9e3f56704866761be61a98e60a22eadf6234
norm_sha256: 15e47d79d157b9eebd87478eb8529df1ac369836121c3420d8915881b5c28ebe
submission_blob: 9e85c24636d47e2cf91bb76ca793a5a9894ef683
lint_warnings: 7
attempt_file: 2026-10-24T233357Z-check-silu-mul-sorted-first-turn.md
prepared_at: 2026-10-24T233357Z
---

# silu-mul-sorted-first-turn

## Change

Built on silu-mul-triton-local-rows-marlin-turn's implementation commit. In _experts_triton, when state['silu_sorted_ok'], replace silu_and_mul(cache1, cache2, expert_ids=local_ids.view(-1), expert_step=1) with _silu_mul_sorted(cache1, cache2, sorted_ids, n_padded, local_ids.view(-1), size_n) over the Triton path's align (BLOCK_T 256); extend _silu_sorted_setup with T=16384 at BLOCK_T 256 so the gate covers it. act_and_mul costs ~297 us at first_turn against a ~84 us local-rows floor. experts-triton-w13-fused-silu-epilogue, the reason the parent excluded this path, is done (neutral), so nothing is confounded.

## Expected

- `*load=first_turn*` faster: silu-mul over local rows only at first_turn

## Risk

Main-thread build before the 00:00Z reset, user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:35Z with user-approved self-review, on silu-mul-triton-local-rows-marlin-turn's implementation commit (claim based it there: parent still running). `_experts_triton`: `_silu_mul_sorted(cache1, cache2, sorted_ids, n_padded, local_ids.view(-1), size_n)` when `state["silu_sorted_ok"]`, else master's silu_and_mul. `_silu_sorted_setup` now also checks T=16384 at the Triton path's BLOCK_T (256); a miss turns the sorted silu OFF on both paths. Equivalence: master's silu_and_mul(expert_ids=local_ids) also skips -1 slots; the w13 Triton GEMM leaves -1-tile rows unwritten and w2 zero-stores those tiles without reading cache2, so skipping them changes no read value.

## Rules check

- [x] CUDA-graph capturable: one Triton launch, static grid, prepare()-set gate.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.
