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
review: required
status: queued
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
