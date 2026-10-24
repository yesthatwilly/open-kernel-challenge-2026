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
review: hold
status: hold
review_at: 2026-10-24T185010Z
review_companions: null
hold_reason: "adversarial review: Stale premise: master (small-t-launch-minimal-dense-chain-pdl, f1f2f35) already folds the pre-expert RMSNorm into dl_route_norm_align at T <= 32, so 4 of the 6 "
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

## Adversarial review

Reviewed 2026-10-24T185010Z by an adversarial subagent (model opus).

**Verdict: hold.** Stale premise: master (small-t-launch-minimal-dense-chain-pdl, f1f2f35) already folds the pre-expert RMSNorm into dl_route_norm_align at T <= 32, so 4 of the 6 expected 'faster' loads (decode_1/16/32, verify_1) never execute the F.rms_norm this replaces, and the remaining T=128/256 saving (~3-4 us of 440-910 us, ~0.2% headline) cannot be resolved by one run; re-mint scoped to T > 32 with corrected expectations, or fold it into a larger T > 32 change.

### Justification
The mechanism itself is plausible: at::native vectorized_layer_norm over [T, 4096] bf16 costs 5.3-6.1 us in a017e10aec544c55 regardless of T, while a one-program-per-row Triton kernel moving 2-4 MB at T=128/256 (or KB at T<=32) should take ~2-3 us, the same pattern that won in post-expert-norm-triton-small-t. The weakest link is the evidence base: every number in the hypothesis comes from a017e10aec544c55, the pre-dense-chain master. On current master (submission.py kernel_fn, use_dl = dl is not None and tokens <= DL_MAX_TOKENS = 32) the T <= 32 path calls _dl_forward, whose kernel R already does the pre-expert RMSNorm x (1 + w) in fp32 and writes xn; state['pre_expert_norm'] is only called in the else branch (line ~1384), i.e. at verify_16 (T=128) and verify_32 (T=256), plus the fallback when the dl self-check fails. The master profile (81ee6714b7614a99 top kernels) shows dl_pooler_router/dl_route_norm_align rows and no layer-norm row at T<=32. So the expected rows for decode_1, verify_1, decode_16, decode_32 should be 'flat' (dead code there), and the expected verify_16/verify_32 gains (~+0.5%/+0.4% per shape, two of six loads) give a geomean headline of ~0.15-0.25%, far inside the >=±1.5% CI. The kernel-level result (vectorized_layer_norm row gone, new Triton row ~2-3 us at verify_16/32) would be readable from the profile, but that alone is a small payoff for a daily check.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (landed, current base): already replaced this exact norm at T <= 32 inside dl_route_norm_align; any movement measured on decode_1/16/32 or verify_1 cannot come from this change and would be noise (or, if the dl self-check failed at prepare and fell back to master, a different code path entirely - check the '[small-t-dense-chain] ON' line in the log).
- post-expert-norm-triton-small-t (landed): same pattern on the other side of the experts; no numerical or launch interaction with the pre-expert norm, but it set the precedent the hypothesis relies on, and its gains were at T<=16 shapes now owned by the dense chain.
- marlin-block-m-cap-16 (landed): changes Marlin timing at verify_16/32, the only shapes this change still affects; Marlin is 250-450 us there, so run-to-run Marlin variance dwarfs the ~3-4 us norm saving in the per-shape headline.
- sglang-pipeline-parity-router-hilo-m8 (landed): at T>32 the router runs on the side stream overlapping the pooler; the pre-expert norm sits after the join, so it is on the critical path and its saving should be additive - no confound, but moe-dispatch-prep-on-side-stream (queued) would change that placement.

### Interference and A/B needs
No companion would rescue this run: the problem is not ambiguity of attribution between interacting changes but that the change only executes on two of six loads and its effect there is ~1% of shape time. An ab-revert of small-t-launch-minimal-dense-chain-pdl would re-expose the norm at T<=32 but measures a configuration that will never ship. A control is pointless for a single-kernel substitution. The right fix is to re-scope rather than add runs.

### Open questions
- Re-mint as 'pre-expert-rmsnorm-triton-large-t': gate to tokens > DL_MAX_TOKENS (or just the else branch), expected verify_16/verify_32 'faster' by ~3-4 us each and all T<=32 shapes 'flat', judged on the profile row rather than the headline.
- Higher-value alternative at T>32: fuse the pre-expert norm into a step that already touches expert_input (pooler GEMM split-K reduce epilogue, or moe-align/dispatch prep) instead of a standalone launch, or bundle it with other T>32 small-kernel cleanups (local_ids modulo, routing) so one run moves the headline measurably.
- Does any queued T>32 rewrite (moe-align-parallel-large-t-port, moe-dispatch-prep-on-side-stream) already plan to absorb this norm? If so this becomes a duplicate there too.

