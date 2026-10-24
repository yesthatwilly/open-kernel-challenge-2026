---
problem: muse_decode_layer
slug: moe-bs1-fused-persistent-port
created: 2026-10-24T184355Z
base: master
area: w13 marlin + silu-mul + w2 marlin + post-expert combine (T=1)
change: >
  Port muse_moe_w4a16 cuda-fused-moe-bs1-persistent-w13-silu-w2 (impl commit 0efab7b, branch
  hyp/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-w13-silu-w2, moe job 1105a02aa3fa4e56) to
  decode at tokens == 1 only. Copy verbatim from 0efab7b:muse_moe_w4a16/submission.py: FUSED_H,
  FUSED_STAGE_CODES, FUSED_STAGE_SCALES, FUSED_PERM, FUSED_SC_PERM, _FUSED_CPP, _FUSED_CUDA,
  _build_fused, _unpack_codes, _frag_codes, _unfrag_codes, _frag_scales, _unfrag_scales,
  _fused_repack, _fused_forward, _fused_case, _fp32_dequant, _fp32_moe, _rel, _fused_setup. Adapt
  only the interfaces: (1) every weights["post_norm_w"] becomes weights["post_expert_norm"] (same
  tensor, BF16 [4096], applied as 1 + w); (2) _fused_setup self-check calls
  _master_kernel_fn(case, state): replace it with a local wrapper that returns
  _experts(case["hidden_states"], case["topk_weights"], case["topk_ids"], state) (decode master
  MoE on already-normed x with ids in the shifted local space, which is what _fused_case builds
  for start 0, count 32); (3) change _fused_forward to take (x, ids, tw, fused) instead of a data
  dict, keeping its 16-byte pointer check, ids.to(int32).contiguous() and
  tw.to(float32).contiguous(). prepare(): state["fused"] = _fused_setup(weights, state, 0, 32)
  after state is built (decode w13/w2 are this GPU 32 experts at index 0..31); it prints "PATH
  CHOSEN = fused|master". kernel_fn: when tokens == 1 and state["fused"] is not None, replace
  `_experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)` with
  `_fused_forward(state["pre_expert_norm"](expert_input), local_ids, topk_weights,
  state["fused"])`, where local_ids is master (topk_ids - local_start) % N_EXPERTS; its [1, 4096]
  bf16 output is the post-norm-combined expert output and feeds the unchanged sliced unpooler.
  Every T > 1 keeps master byte for byte.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "one persistent launch replaces expert_map gather, SGLang align, Marlin w13 + w2 (40.7 us), act_and_mul and _post_norm_combine_kernel (~50 us of GPU time; moe target 20 us): -20..-30 us of 145.6 us, +15-20%"
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "same chain, Marlin 44.9 us: +15-20%"
  - shapes: "*load=decode_16 *"
    direction: flat
    why: T > 1 untouched
  - shapes: "*load=decode_32 *"
    direction: flat
    why: T > 1 untouched
  - shapes: "*load=verify_*"
    direction: flat
    why: T > 1 untouched
risk: high
depends_on: null
priority: 5
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-24T184953Z
review_companions: null
hold_reason: "adversarial review: Hold: the hypothesis gates its own dispatch on the moe parent (1105a02aa3fa4e56) being faster than moe master at bs=1, and it was not (bs=1 50.93 us vs 48.37 us"
---

# moe-bs1-fused-persistent-port

## Change

Port muse_moe_w4a16 cuda-fused-moe-bs1-persistent-w13-silu-w2 (impl commit 0efab7b, branch hyp/muse_moe_w4a16/cuda-fused-moe-bs1-persistent-w13-silu-w2, moe job 1105a02aa3fa4e56) to decode at tokens == 1 only. Copy verbatim from 0efab7b:muse_moe_w4a16/submission.py: FUSED_H, FUSED_STAGE_CODES, FUSED_STAGE_SCALES, FUSED_PERM, FUSED_SC_PERM, _FUSED_CPP, _FUSED_CUDA, _build_fused, _unpack_codes, _frag_codes, _unfrag_codes, _frag_scales, _unfrag_scales, _fused_repack, _fused_forward, _fused_case, _fp32_dequant, _fp32_moe, _rel, _fused_setup. Adapt only the interfaces: (1) every weights["post_norm_w"] becomes weights["post_expert_norm"] (same tensor, BF16 [4096], applied as 1 + w); (2) _fused_setup self-check calls _master_kernel_fn(case, state): replace it with a local wrapper that returns _experts(case["hidden_states"], case["topk_weights"], case["topk_ids"], state) (decode master MoE on already-normed x with ids in the shifted local space, which is what _fused_case builds for start 0, count 32); (3) change _fused_forward to take (x, ids, tw, fused) instead of a data dict, keeping its 16-byte pointer check, ids.to(int32).contiguous() and tw.to(float32).contiguous(). prepare(): state["fused"] = _fused_setup(weights, state, 0, 32) after state is built (decode w13/w2 are this GPU 32 experts at index 0..31); it prints "PATH CHOSEN = fused|master". kernel_fn: when tokens == 1 and state["fused"] is not None, replace `_experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)` with `_fused_forward(state["pre_expert_norm"](expert_input), local_ids, topk_weights, state["fused"])`, where local_ids is master (topk_ids - local_start) % N_EXPERTS; its [1, 4096] bf16 output is the post-norm-combined expert output and feeds the unchanged sliced unpooler. Every T > 1 keeps master byte for byte.

## Expected

- `*layer=sliding load=decode_1 *` faster: one persistent launch replaces expert_map gather, SGLang align, Marlin w13 + w2 (40.7 us), act_and_mul and _post_norm_combine_kernel (~50 us of GPU time; moe target 20 us): -20..-30 us of 145.6 us, +15-20%
- `*layer=full load=decode_1 *` faster: same chain, Marlin 44.9 us: +15-20%
- `*load=decode_16 *` flat: T > 1 untouched
- `*load=decode_32 *` flat: T > 1 untouched
- `*load=verify_*` flat: T > 1 untouched

## Risk

CONDITIONAL: dispatch only after moe job 1105a02aa3fa4e56 harvests with "PATH CHOSEN = fused" in its log and bs=1 faster than moe master; hold otherwise. Judge on the decode_1 rows: Marlin, act_and_mul, align and _post_norm_combine_kernel gone, one moe_bs1 kernel present. Overlap: the in-judge restack (sliding decode_1 attention + T=1 Triton GEMMs) and small-t-launch-minimal-dense-chain-pdl (T<=32 pre-expert chain) touch decode_1 too but none of these kernels; read kernel rows, not the headline. Competes with moe-small-t-cuda-swapab-port at T=1: if both moe runs win, this one takes T=1. Risk: the persistent kernel spins on device flags and needs all 132 CTAs co-resident; inside the decode graph the side-stream router branch has already joined before the experts, so nothing else runs concurrently.

## Adversarial review

Reviewed 2026-10-24T184953Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold: the hypothesis gates its own dispatch on the moe parent (1105a02aa3fa4e56) being faster than moe master at bs=1, and it was not (bs=1 50.93 us vs 48.37 us base, 0.975x), and the decode T=1 MoE chain it would replace is already cheaper than moe master's, so its case does not stand without the parent winning.

### Justification
Mechanism: replacing align + Marlin w13 + act_and_mul + Marlin w2 + post-norm combine at T=1 with one persistent kernel is plausible in principle (it saves 4-5 launches and the intermediate round trips). But the only measurement contradicts it. In muse_moe_w4a16, job 1105a02aa3fa4e56 did choose the fused path (bs=1 profile shows a single moe_bs1_kernel, 37.7 us), yet the scored bs=1 median was 50.93 us against 48.37 us for the moe base (triton-map-align-on-block-m-restack): 0.975x, flat-to-slower, with the outcome neutral within CI. The 37.7 us profiled kernel time against the base's ~51 us kernel sum (Marlin 44.6 + combine 3.4 + act 1.6 + align 1.6) did not turn into a wall-clock win under graph replay, so the persistent kernel's real cost (flag reset, spin-up, all-CTA co-residency, whatever the profile does not capture) eats the fusion saving. Decode makes this worse, not better. Decode's T=1 Marlin is already 40.2 us (FINDINGS, decode_1 sliding and full) against moe master's 44.6, and decode times every replay with a cold L2 (TASK.md). So the incumbent chain the fused kernel must beat is about 4 us cheaper than the one it already failed to beat. The expected '-20..-30 us, +15-20%' assumes the fused kernel takes its own target (~20 us); the measured 37.7 us kernel / ~47 us wall leaves no visible gain. Weakest link: the expected magnitude rests on the moe target time, not the measured moe result. The hypothesis's own Risk section says to hold unless bs=1 was faster than moe master, which it was not.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (base, rewrite stage 1): the T<=32 dense chain runs PDL-launched kernels (cudaLaunchKernelEx with programmatic serialization) around the MoE. A persistent kernel that spins on device flags and assumes all 132 CTAs are co-resident could see its CTAs scheduled late, or contend with a PDL-early-launched neighbour. That would change its time, or hang it, in a way the moe harness never exercised.
- post-expert-norm-triton-small-t: replaced the T=1 post-expert combine with a Triton kernel. The fused kernel also takes over this step, so any decode_1 delta mixes the fusion with removing an already-optimised combine. The saving available from this step is smaller than in the moe base it was compared against.
- marlin-block-m-cap-16: block_m stays 8 at T=1, so it does not change the incumbent the fused kernel replaces at T=1. No real confounding.

### Interference and A/B needs
No companions. A companion only helps when a single result would be ambiguous. Here the decision is upstream: the parent's measured bs=1 result already says the fused kernel does not beat a Marlin chain that is slower than decode's, and the hypothesis's own condition for dispatch is unmet. A revert of small-t-launch-minimal-dense-chain-pdl (the PDL chain) would be the only sensible ab-revert, to test persistent-kernel and PDL interaction, but spending a check on that before the fused kernel is shown to win standalone is not justified. The right next step is a moe-side stage-2 rewrite (cuda-fused-moe-bs1-persistent-w13-silu-w2 follow-up) that shows bs=1 below moe master's 48.4 us. Re-mint the port after that.

### Open questions
- Why did the moe fused kernel's 37.7 us profiled time become a 50.9 us scored median, worse than the base's 48.4 us? If the cause is fixable (flag reset or spin-up overhead, a cold-L2 effect), the fix belongs in a moe stage-2 rewrite before any port.
- Does the decode graph's PDL chain ever early-launch the kernel after the experts, so that it occupies SMs while the persistent kernel's CTAs are still unscheduled?
- Would the port compete with or supersede moe-small-t-cuda-swapab-port at T=1? That parent failed its tests, so neither T=1 MoE port currently has a winning parent.

