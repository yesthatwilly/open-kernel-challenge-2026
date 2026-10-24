---
problem: muse_prefill_layer
slug: stack-tonight-five-prefill
created: 2026-10-24T233200Z
base: master
area: router GEMM + MoE align (both paths) + w2 zero-store + Marlin silu-mul
change: >
  Built on master. stack-tonight-four-prefill (router-hilo-one-gemm-n512,
  moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn,
  moe-align-local-routes-parallel-port) plus silu-mul-triton-local-rows-marlin-turn (sorted
  local-rows silu-mul on the Marlin path, prepare() self-checked), each applied unchanged. On the
  Marlin path the silu-mul walks the parallel align's sorted_ids, which hold local routes only.
  prepare() runs _align_par_setup, then _silu_sorted_setup, then _check_experts_e2e.
expected:
  - shapes: "*"
    direction: faster
    why: the four-stack plus silu-mul on local rows at turn_1/turn_4
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: running
stage: dispatched
claimed_at: 2026-10-24T233200Z
base_sha: 3b2b91dc85a1be9e6563f3f0987fd41e4d4c0841
base_blob: 635cfee3ac2c3f7c66840b8c8a082cad2fa41a68
branch: hyp/muse_prefill_layer/stack-tonight-five-prefill
worktree: .worktrees/muse_prefill_layer/stack-tonight-five-prefill
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-24T233216Z
impl_commit: b9ad6f2d2af0db453232b54cc0dbec7d034c51d9
submission_sha256: d42aba8c748c992af59f39c2879c3adda94cb677341926f1f59fb8d69c985dc5
norm_sha256: 4340f5135c3e58122409931d064ba571c954db3fd4f36921dd730a20c3a3d04b
submission_blob: db8a3344398bca79ea5cafeb580f701c055fd475
lint_warnings: 15
attempt_file: 2026-10-24T233216Z-check-stack-tonight-five-prefill.md
prepared_at: 2026-10-24T233216Z
job: c4d14d62f11e4aaf
dispatched_at: 2026-10-24T233227Z
promotion: submitted
promotion_job: 502f2eab95164781
promotion_at: 2026-10-24T234001Z
---

# stack-tonight-five-prefill

## Change

Built on master. stack-tonight-four-prefill (router-hilo-one-gemm-n512, moe-align-32-experts-first-turn, w2-nonlocal-zero-store-off-first-turn, moe-align-local-routes-parallel-port) plus silu-mul-triton-local-rows-marlin-turn (sorted local-rows silu-mul on the Marlin path, prepare() self-checked), each applied unchanged. On the Marlin path the silu-mul walks the parallel align's sorted_ids, which hold local routes only. prepare() runs _align_par_setup, then _silu_sorted_setup, then _check_experts_e2e.

## Expected

- `*` faster: the four-stack plus silu-mul on local rows at turn_1/turn_4

## Risk

Main-thread stack before the 00:00Z reset, user-approved self-review.

## Implementation notes

Built in the main thread 2026-10-24 ~23:33Z with user-approved self-review. Worktree on master; `git apply -3` of the three done/neutral patches, moe-align-local-routes-parallel-port's implementation diff and silu-mul-triton-local-rows-marlin-turn's implementation diff. Two conflicts, both "both sides added" (the two prepare() setup calls before _check_experts_e2e, and the two new code sections before _experts_marlin), resolved by keeping both, align first; no line edited. On the Marlin path the sorted silu-mul consumes the parallel align's sorted_ids/n_padded (local routes only) and still masks on local_ids >= 0.

## Rules check

- [x] CUDA-graph capturable: unchanged from the five parents.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights unchanged; router hi+lo split as the rules allow.
- [x] Activations BF16; KV FP8; attention untouched.
- [x] Every key attended.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.
