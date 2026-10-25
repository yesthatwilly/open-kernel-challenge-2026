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
stage: dispatched
claimed_at: 2026-10-24T233335Z
base_sha: a0eb6398a55163f42feb6a3cdfb8e02b392e090b
base_blob: db8a3344398bca79ea5cafeb580f701c055fd475
branch: hyp/muse_prefill_layer/silu-mul-sorted-first-turn
worktree: .worktrees/muse_prefill_layer/silu-mul-sorted-first-turn
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233357Z
impl_commit: fc7f0a08663ad479ea473c461c282a70c85fd712
submission_sha256: 1c7c6b27c04f1afe3fee01ce659e9e3f56704866761be61a98e60a22eadf6234
norm_sha256: 15e47d79d157b9eebd87478eb8529df1ac369836121c3420d8915881b5c28ebe
submission_blob: 9e85c24636d47e2cf91bb76ca793a5a9894ef683
lint_warnings: 7
attempt_file: 2026-10-25T000329Z-check-silu-mul-sorted-first-turn.md
mode: check
prepared_at: 2026-10-25T000329Z
job: 9528eed511b04f33
dispatched_at: 2026-10-25T000344Z
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

## Re-base (2026-10-25 00:0xZ)

Reset priority 2: re-based onto prefill master after stack-tonight-five-prefill landed (it already carries the Marlin-path sorted silu). The parent diff (8+/2-: the Triton-path gate and the T=16384 self-check) applied cleanly. On first_turn the sorted silu now walks master's moe_align_block_size(LOCAL_EXPERTS) output and still masks local_ids >= 0, so it does not depend on which align ran. Rules check above still holds; lint has no errors.
