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
review: proceed
status: queued
review_at: 2026-10-25T085209Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T085209Z by an adversarial subagent (model opus).

**Verdict: proceed.** Cheap, row-readable restack. The align swap partly undoes the landed parallel align at turn_4, and the pre-norm half reuses an already-checked kernel. One real gap to fix before building: SGLang's 32-expert align does NOT give local-only blocks (remote routes go into -1 blocks), so prepare must time the align together with the Marlin pair and the sorted silu, not the align alone.

### Justification
The mechanism holds up. _align_parallel_kernel runs 33 CTAs on 132 SMs, and each one streams all n = 8T route ids twice in 512-id chunks, so its cost grows linearly with T: 16.7 us at T=753, 68.3 us at T=3012, 394 us at T=16384. SGLang's moe_align_block_size over 32 experts costs 139 us at T=16384, which scales linearly to about 25 us at T=3012, matching the pre-port master's ~27 us. That makes about -40 us at turn_4 plausible. The pre-norm numbers (24.8 -> 20.1 us at T=3012, 13.0 -> 9.7 at T=753) come from moe-pre-post-norm-triton-multitoken's own prepare A/B, which passed correctness, and the -45 us turn_4 estimate is consistent (~1.4% of 3319 us sliding, ~0.9% of 4849 us full). The headline effect, about +0.3-0.5%, sits well inside the ~±1.5% CI. The Marlin rows move ±70 us between identical-code runs (turn_4 sliding: 2093 / 2118 / 2148 / 2223 us across four recent runs), so the result can only be read on the align and pre-norm kernel rows and the prepare logs, as the hypothesis says. WEAKEST LINK: the change says 'keeping local-routes-only blocks', but moe_align_block_size(local_ids, bsm, 32) puts the ~7/8 remote routes into -1 blocks (moe-align-32-experts-first-turn's md says so explicitly). At T=3012 with bsm 64 that is roughly 330 extra -1 m-blocks next to ~80 local ones. Marlin (is_ep) and _silu_mul_sorted then walk a list about 5x longer up to n_padded. The archive suggests Marlin does not care: turn_4 Marlin was 2093/2173 us before the local-only port and 2148/2115 us after, which is within noise. But a prepare gate that times only the align could pick an align that is faster on its own and slower end to end. Gate on the whole _experts_marlin call (align + w13 Marlin + silu + w2 Marlin) in the graph with a cold-L2 flush at T=753 and T=3012, and switch it ON only when the end-to-end time drops by more than the timing noise. Also, 'bit-exact' needs care: Marlin output per route is order-independent with use_atomic_add=False, so bit-exact is a fair expectation. Compare only local rows, since remote cache3 rows are unwritten or garbage in both versions.

### Confounders on the current kernel
- stack-tonight-five-prefill (landed moe-align-local-routes-parallel-port inside it): this hypothesis partly reverts that sub-change at T > 1024, so any turn_4 align gain is that port's own T-scaling being undone. It is still attributable, because the align kernel rows (_align_parallel_kernel vs moe_align/count_and_sort) appear in the profile and prepare logs both timings.
- stack-tonight-five-prefill (landed silu-mul-triton-local-rows-marlin-turn inside it): _silu_mul_sorted consumes the align's sorted_ids/n_padded. With SGLang's layout it walks the -1 blocks too (masked, so correctness is unaffected but there is extra work), so the silu row may grow slightly at turn_4 and offset some of the align saving. Time it inside the end-to-end gate.
- experts-hybrid-marlin-turn-triton-first-turn: it fixes MARLIN_MAX_TOKENS = 4096, which decides that turn_4 (T=3012) is on the Marlin path where this align change applies. If that threshold changes, for example because a queued turn_4 rewrite lands, the align half becomes dead code at turn_4. No confound for this run.
- sglang-pipeline-parity: the pre-norm replaces state['pre_expert_norm'] (MuseSpark1xRMSNorm) from the parity pipeline. The Triton kernel uses the module's _get_effective_weight(), so it does not depend on how parity built the module. No confound.

### Interference and A/B needs
No companions. Each of the two sub-changes affects disjoint kernels with their own profile rows (align vs the pre-expert RMSNorm), and each logs an in-prepare A/B per T class (CUDA graph, cold L2). The two halves are already separable from a single run without a control. Reverting stack-tonight-five's align port (ab-revert) would only reproduce what this hypothesis does at T > 1024. An ab-tune of the gate threshold is not worth a check while prefill has 8 checks left. Make the result unambiguous by changing the prepare gate, not by adding a companion: gate the align on end-to-end _experts_marlin time and print both the align-only and the end-to-end numbers.

### Open questions
- Does sgl_kernel's moe_align_block_size at Marlin's small block sizes (bsm 16-64) put the -1 bucket last? If so, a one-thread fix-up of n_padded to the local total, at about 2 us, would restore local-only blocks for Marlin and silu and remove the weakest link entirely.
- Is the ~25 us SGLang 32-expert align at T=3012 measured or extrapolated? The cited ~17+10 us figure was for the 256-expert call. The prepare log line will answer this, so record it in the attempt file.
- At T=753 _align_parallel is 16.7 us, and SGLang's two-kernel align is probably ~8-12 us there, close to the 0.9x gate. Expect turn_1 align to stay on _align_parallel, which makes turn_1's expected gain pre-norm only (~3 us).

