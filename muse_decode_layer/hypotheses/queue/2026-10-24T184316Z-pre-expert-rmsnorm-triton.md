---
problem: muse_decode_layer
slug: pre-expert-rmsnorm-triton
created: 2026-10-24T184316Z
base: master
area: pre-expert norm
change: >
  Replace state["pre_expert_norm"](expert_input) in kernel_fn (MuseSpark1xRMSNorm over
  D_EXPERT=4096 -> F.rms_norm -> at::native vectorized_layer_norm, 5.3-6.0 us on every shape in
  a017e10aec544c55, far above the ~1 us its 8-16 KB per token need) with one Triton launch. New
  kernel _pre_expert_rmsnorm_kernel(x_ptr, w_ptr, out_ptr, H, eps, BLOCK_H: tl.constexpr), grid
  (T,), one program per token, num_warps=8, BLOCK_H = triton.next_power_of_2(4096) (same shape as
  master _post_norm_combine_kernel): load the bf16 row into fp32, r = tl.rsqrt(tl.sum(x * x) / H +
  eps) with eps = RMS_EPS (1e-5), nw = load(pre_expert_norm weight).to(fp32) + 1.0, store (x * r *
  nw).to(bfloat16) into a new [T, 4096] bf16 tensor. prepare(): keep the raw weight as
  state["pre_expert_norm_w"] = weights["pre_expert_norm"].contiguous(); keep the module as the
  fallback. Self-check in prepare on synthetic bf16 rows (fixed seed, T in 1, 8, 16, 32, 128, 256,
  with a few rows scaled by 1e-3 and 1e3): max per-row rel L2 against state["pre_expert_norm"](x)
  <= 2e-3 and same shape and dtype; on failure set state["pre_norm_triton"] = False and keep the
  module path; print one line "[pre-expert-norm-triton] ON|OFF max_rel=...". Applies at every T
  (config-free); nothing else changes.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "vectorized_layer_norm 5.9 us -> ~2 us Triton row: -3..-4 us of 145-157 us, +2%"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "5.5-5.8 -> ~2 us: +1.5%"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "5.3 -> ~2 us: +1-1.5%"
  - shapes: "*load=decode_32 *"
    direction: faster
    why: "5.7 -> ~2.2 us: +1%"
  - shapes: "*load=verify_16 *"
    direction: faster
    why: "6.0 -> ~3 us: +0.5%"
  - shapes: "*load=verify_32 *"
    direction: faster
    why: "6.1 -> ~3.5 us: +0.4%"
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# pre-expert-rmsnorm-triton

## Change

Replace state["pre_expert_norm"](expert_input) in kernel_fn (MuseSpark1xRMSNorm over D_EXPERT=4096 -> F.rms_norm -> at::native vectorized_layer_norm, 5.3-6.0 us on every shape in a017e10aec544c55, far above the ~1 us its 8-16 KB per token need) with one Triton launch. New kernel _pre_expert_rmsnorm_kernel(x_ptr, w_ptr, out_ptr, H, eps, BLOCK_H: tl.constexpr), grid (T,), one program per token, num_warps=8, BLOCK_H = triton.next_power_of_2(4096) (same shape as master _post_norm_combine_kernel): load the bf16 row into fp32, r = tl.rsqrt(tl.sum(x * x) / H + eps) with eps = RMS_EPS (1e-5), nw = load(pre_expert_norm weight).to(fp32) + 1.0, store (x * r * nw).to(bfloat16) into a new [T, 4096] bf16 tensor. prepare(): keep the raw weight as state["pre_expert_norm_w"] = weights["pre_expert_norm"].contiguous(); keep the module as the fallback. Self-check in prepare on synthetic bf16 rows (fixed seed, T in 1, 8, 16, 32, 128, 256, with a few rows scaled by 1e-3 and 1e3): max per-row rel L2 against state["pre_expert_norm"](x) <= 2e-3 and same shape and dtype; on failure set state["pre_norm_triton"] = False and keep the module path; print one line "[pre-expert-norm-triton] ON|OFF max_rel=...". Applies at every T (config-free); nothing else changes.

## Expected

- `*load=decode_1 *` faster: vectorized_layer_norm 5.9 us -> ~2 us Triton row: -3..-4 us of 145-157 us, +2%
- `*load=verify_1 *` faster: 5.5-5.8 -> ~2 us: +1.5%
- `*load=decode_16 *` faster: 5.3 -> ~2 us: +1-1.5%
- `*load=decode_32 *` faster: 5.7 -> ~2.2 us: +1%
- `*load=verify_16 *` faster: 6.0 -> ~3 us: +0.5%
- `*load=verify_32 *` faster: 6.1 -> ~3.5 us: +0.4%

## Risk

Judge on the vectorized_layer_norm row vanishing and the new row time per shape; the headline gain (~1%) is under one-run resolution. Same pattern as the landed post-expert-norm-triton-small-t (replacing an at::native norm with a one-program-per-token Triton kernel). OVERLAP: small-t-launch-minimal-dense-chain-pdl (in judge) folds this norm into its dl_route_norm_align kernel at T<=32 when ON; T>32 is untouched by every in-judge run (moe-chain stage 1 keeps F.rms_norm). Stacks with moe-dispatch-prep-on-side-stream, which puts this norm on the main-path side of the overlap.
