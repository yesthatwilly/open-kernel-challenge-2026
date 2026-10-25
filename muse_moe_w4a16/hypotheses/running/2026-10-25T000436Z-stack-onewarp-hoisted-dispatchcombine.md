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
status: running
stage: dispatched
claimed_at: 2026-10-25T000436Z
base_sha: f540d1c19b5798a2bd1f0421964922a582602231
base_blob: 108d36e9f2266fc88531dd6ff0329a847c882013
branch: hyp/muse_moe_w4a16/stack-onewarp-hoisted-dispatchcombine
worktree: .worktrees/muse_moe_w4a16/stack-onewarp-hoisted-dispatchcombine
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T000449Z
impl_commit: 795e035d270a2616aa6b8daef49ea210d7c1b37e
submission_sha256: 538c39378a082f9a8eafe32e5f8482cd4be0e25182d778bc7ae02666bd2d8404
norm_sha256: 5ad9f9485e682fe07a216033d7f53f05d882f4545de49171d1da17d60602de5d
submission_blob: e9478869b802250d219d7cdefe73e07526622cd7
lint_warnings: 19
attempt_file: 2026-10-25T000449Z-check-stack-onewarp-hoisted-dispatchcombine.md
prepared_at: 2026-10-25T000449Z
job: 7e5ab95a413f4ebe
dispatched_at: 2026-10-25T000511Z
---

# stack-onewarp-hoisted-dispatchcombine

## Change

Built on master. Stack the three moe results of 2026-10-24 that each matched the 1.3278 best within noise, applied unchanged from their done/ patches: align-small-one-warp-tiny-m (1-warp _align_small_kernel at T<=8, scored 1.3287), post-norm-combine-hoisted-slot-loads (all 8 slot loads before the reduction, bs<32, scored 1.3282) and cuda-dispatch-combine-pdl-verify (CUDA align + PDL post-norm combine at 32<T<=512, scored 1.3283). Their shapes and kernels are disjoint except the post-norm combine, which the hoisted change edits only at bs<32 and the CUDA rewrite replaces only above 32 tokens. Tests whether sub-noise wins add up, as prefill's stack showed (1.059 -> 1.0744).

## Expected

- `*` faster: three disjoint sub-noise gains combined

## Risk

Reset priority 4 (user-requested stack, 2026-10-24 ~23:50Z); claimed with --skip-review on that instruction.

## Implementation notes

Built in the main thread 2026-10-25 ~00:10Z (reset priority 4, user-specified stack; self-reviewed). Worktree on master; `git apply -3` of the three done/neutral patches (align-small-one-warp-tiny-m, post-norm-combine-hoisted-slot-loads, cuda-dispatch-combine-pdl-verify), all clean, no edits. Every parent's base already carried align-parallel-per-expert-verify. Run-time disjoint: the one-warp launch is `_align_small` at n <= 64 (T <= 8); the hoisted loads are the Triton post-norm combine at bs < 32; the CUDA TU replaces `_align_parallel` and the SGLang combine only at 32 < M <= 512, behind its own prepare() self-check (combine gate row 8e-3 / global 4e-3, as audited) with fallback to master.

## Rules check

- [x] CUDA-graph capturable: unchanged from the three parents, each passed the judge.
- [x] Fresh inputs: nothing derived from `data` cached.
- [x] Weights: NVFP4 unchanged.
- [x] Activations BF16.
- [x] n/a: no attention in this problem.
- [x] No timing/harness access, processes, network or module edits.
- [x] No work outside timed calls added.
- [x] okc_lint: no errors.
