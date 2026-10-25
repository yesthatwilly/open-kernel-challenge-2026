---
problem: muse_prefill_layer
slug: turn-align-sglang32-prenorm-triton-restack
created: 2026-10-25T054848Z
base: master
area: "MoE align on the Marlin path (T <= 4096) + pre-expert norm: restack of two measured row wins"
change: >
  Built on master (a81d19c). Two row-level wins already measured in the archive, applied together,
  shapes untouched otherwise. (1) ALIGN at 200 < T <= 4096: master's _align_parallel_kernel scales
  with T because every program streams all n route ids: 16.7 us at T=753, 68.3 us at T=3012
  (stack-tonight-five profile) and 394 us at T=16384 (align-local-parallel-first-turn profile),
  while SGLang's moe_align_block_size over the 32 local experts (master's first_turn call) costs
  count_and_sort 87 + align 51 = 139 us at T=16384, and the pre-stack master's 256-expert version
  cost ~17+~10 us at T=3012. In _experts_marlin replace _align_parallel(local_ids, 0,
  LOCAL_EXPERTS, bsm) with moe_align_block_size(local_ids, bsm, LOCAL_EXPERTS) when T > 1024 (and
  at T <= 1024 only if prepare's timing says so), keeping local-routes-only blocks (ids are 0-31 /
  -1). Its sorted_ids/block_experts/n_padded layout is the one _silu_mul_sorted and Marlin already
  accept (the silu setup self-checks against moe_align_block_size output); extend
  _align_par_setup's 15-case check to compare the Marlin pair's output on the new align vs
  master's on the same routing (bit-exact expected) and fall back to _align_parallel on any
  mismatch. Prepare timing in CUDA graphs with a cold-L2 flush at T=753 and T=3012 picks per T
  class, ON only at <= 0.9x of _align_parallel. (2) PRE-EXPERT NORM: take
  moe-pre-post-norm-triton-multitoken's Triton _pre_expert_rmsnorm_kernel (its patch in
  done/neutral) for the pre-expert norm only (its post-expert combine lost and stays master's),
  with its existing per-T prepare A/B (measured 13.0 -> 9.7 us at T=753, 24.8 -> 20.1 us at
  T=3012, ~77 -> 69 us at T=16384; vs fp32 ref error identical to master's). No CUDA extension;
  graph-safe; nothing cached from inputs; no prints in kernel_fn.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "align 68 -> ~25 us and pre-norm 24.8 -> 20.1 us: about -45 us per turn_4 shape (1.4% sliding, 0.9% full)"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "pre-norm -3 us, align -5 to -8 us if its gate picks SGLang at T=753: ~1% of turn_1 sliding"
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: pre-norm -8 us of 14-22 ms; align unchanged there
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# turn-align-sglang32-prenorm-triton-restack

## Change

Built on master (a81d19c). Two row-level wins already measured in the archive, applied together, shapes untouched otherwise. (1) ALIGN at 200 < T <= 4096: master's _align_parallel_kernel scales with T because every program streams all n route ids: 16.7 us at T=753, 68.3 us at T=3012 (stack-tonight-five profile) and 394 us at T=16384 (align-local-parallel-first-turn profile), while SGLang's moe_align_block_size over the 32 local experts (master's first_turn call) costs count_and_sort 87 + align 51 = 139 us at T=16384, and the pre-stack master's 256-expert version cost ~17+~10 us at T=3012. In _experts_marlin replace _align_parallel(local_ids, 0, LOCAL_EXPERTS, bsm) with moe_align_block_size(local_ids, bsm, LOCAL_EXPERTS) when T > 1024 (and at T <= 1024 only if prepare's timing says so), keeping local-routes-only blocks (ids are 0-31 / -1). Its sorted_ids/block_experts/n_padded layout is the one _silu_mul_sorted and Marlin already accept (the silu setup self-checks against moe_align_block_size output); extend _align_par_setup's 15-case check to compare the Marlin pair's output on the new align vs master's on the same routing (bit-exact expected) and fall back to _align_parallel on any mismatch. Prepare timing in CUDA graphs with a cold-L2 flush at T=753 and T=3012 picks per T class, ON only at <= 0.9x of _align_parallel. (2) PRE-EXPERT NORM: take moe-pre-post-norm-triton-multitoken's Triton _pre_expert_rmsnorm_kernel (its patch in done/neutral) for the pre-expert norm only (its post-expert combine lost and stays master's), with its existing per-T prepare A/B (measured 13.0 -> 9.7 us at T=753, 24.8 -> 20.1 us at T=3012, ~77 -> 69 us at T=16384; vs fp32 ref error identical to master's). No CUDA extension; graph-safe; nothing cached from inputs; no prints in kernel_fn.

## Expected

- `*load=turn_4*` faster: align 68 -> ~25 us and pre-norm 24.8 -> 20.1 us: about -45 us per turn_4 shape (1.4% sliding, 0.9% full)
- `*load=turn_1*` faster: pre-norm -3 us, align -5 to -8 us if its gate picks SGLang at T=753: ~1% of turn_1 sliding
- `*load=first_turn*` unchanged: pre-norm -8 us of 14-22 ms; align unchanged there

## Risk

Expected headline +0.3-0.5%, inside the CI: read it on the _align_* and pre-norm kernel rows, not the headline. Cheap insurance: the rewrites' Marlin fallback at turn_4 inherits the faster align.
