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
stage: dispatched
claimed_at: 2026-10-24T232928Z
base_sha: eb35e7e2a89baeba03be2ed19b88daff5128f06a
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/stack-tonight-four-prefill
worktree: .worktrees/muse_prefill_layer/stack-tonight-four-prefill
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T232938Z
impl_commit: ad6a52ef13164f9907dd8a50791705ed06b2989e
submission_sha256: 66c638f476c307b916335c2ae5a65774d6a88d5d32de7feb8307f6e3f66dabf7
norm_sha256: 89579d09c3367258bc7ee538e4ed4b8a1d32d190e9191809c68b49a9956cc518
submission_blob: b0ff525343837b435079497123fefed9afff961c
lint_warnings: 13
attempt_file: 2026-10-24T232938Z-check-stack-tonight-four-prefill.md
prepared_at: 2026-10-24T232938Z
job: 7499d9c71acd4417
dispatched_at: 2026-10-24T232955Z
promotion: submitted
promotion_job: 334e1798db4440da
promotion_at: 2026-10-24T234001Z
---

# stack-tonight-four-prefill

## Change

Built on master. Stack all four prefill changes built on 2026-10-24 night, each applied unchanged: router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn (as in restack-router-hilo-align32-zerostore) plus moe-align-local-routes-parallel-port (parallel local-routes align on the Marlin path, prepare() self-checked). The four touch disjoint lines (router repack and GEMM; the Triton-path align; the zero-store probe and w2 flag; the Marlin-path align and its setup), so the run reads their combined effect against master and against the two parents.

## Expected

- `*` faster: router one GEMM everywhere, fewer Marlin m-blocks at turn_1/turn_4, smaller align and no zero-stores at first_turn

## Risk

Stack built in the main thread before the 00:00Z reset with user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:32Z with user-approved self-review. Worktree on master; `git apply -3` of the three done/neutral patches (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn) and of moe-align-local-routes-parallel-port's implementation diff, all clean, no edits. prepare() order: the zero-store probe and _check_expert_gemms (Triton bank), then the Marlin bank, then _align_par_setup, then _check_experts_e2e, so the end-to-end check covers both changed align paths together.

## Rules check

- [x] CUDA-graph capturable: unchanged from the four parents; kernel_fn gains only prepare()-set flags and shape-only allocations.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged; router hi+lo split as the rules allow.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added (all checks in prepare()).
- [x] okc_lint: no errors.
