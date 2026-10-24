---
problem: muse_prefill_layer
slug: norm-boundaries-persistent-multirow
created: 2026-10-24T224410Z
base: master
area: attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2, persistent multi-row variant)
change: >
  Built on master (635cfee). The multi-row/persistent re-mint the review of the held
  norm-residual-boundaries-triton-row-kernel invited ('same one-row-per-program structure as the
  incumbent ... re-mint as a multi-row/persistent variant if wanted'). Replace both
  state['attn_boundary'](...) and state['ffn_boundary'](...) calls in kernel_fn with one new
  Triton kernel _boundary_persistent_kernel launched twice, at every T. Math per row, fp32
  throughout, exactly as the held hypothesis spells it out (read its change field): b =
  branch[t].float(); FFN boundary only: b *= post_ffn effective weight (1 + w); b *=
  rsqrt(mean(b^2) + 1e-8); s_new = alpha * stream[t] + beta * b; out = s_new * rsqrt(mean(s_new^2)
  + 1e-5) * pre-norm effective weight (pre_ffn for the attention boundary, next_pre_attn for the
  FFN boundary); store s_new fp32 and out bf16 into fresh torch.empty tensors. Take alpha, beta
  and the effective weights from the SGLang modules master already builds in prepare()
  (_residual(...) after precompute_coefficients(), _norm(...) effective weights), as fp32 [8192]
  tensors, rather than re-deriving constants. STRUCTURE (the difference from the incumbent and
  from the held row kernel): grid = NUM_SMS * 2 programs (host int from
  torch.cuda.get_device_properties), each walks rows t = pid, pid + nprog, ... < T; the row is
  processed in two passes over 8192 columns in BLOCK=2048 chunks with 16-byte vector loads (pass 1
  accumulates sum(b^2) while keeping the branch chunk and stream chunk... if 8192 fp32 + 8192 bf16
  per row do not fit in registers at num_warps 8, re-read them in pass 2 from L2, which is where
  they will be), and the next row's first chunk is loaded before the current row's stores
  (software prefetch), so each SM keeps two rows in flight and there are no CTA launch waves (753
  rows was ~3-6 waves of one-row CTAs). prepare(): compare against the SGLang modules on synthetic
  inputs at T = 753 and 3012 for both boundaries (stream rel L2 <= 1e-6, bf16 output max diff <= 1
  ulp); on mismatch log '[boundary] OFF: <reason>' and keep master's modules, else 'ON'. Nothing
  else changes.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "boundary rows 20.5-23.0 us each vs a 15.4 us byte floor (67-75%): target ~17-18 us each, ~-0.8% on turn_1 shapes"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "74.7-79.6 us each vs 61.7 us floor: target ~66 us, ~-0.7%"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "~396-400 us each vs 335 us floor at ~4.0 TB/s achieved: 0-5% on the rows, ~0-0.4% per shape"
risk: medium
depends_on: null
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# norm-boundaries-persistent-multirow

## Change

Built on master (635cfee). The multi-row/persistent re-mint the review of the held norm-residual-boundaries-triton-row-kernel invited ('same one-row-per-program structure as the incumbent ... re-mint as a multi-row/persistent variant if wanted'). Replace both state['attn_boundary'](...) and state['ffn_boundary'](...) calls in kernel_fn with one new Triton kernel _boundary_persistent_kernel launched twice, at every T. Math per row, fp32 throughout, exactly as the held hypothesis spells it out (read its change field): b = branch[t].float(); FFN boundary only: b *= post_ffn effective weight (1 + w); b *= rsqrt(mean(b^2) + 1e-8); s_new = alpha * stream[t] + beta * b; out = s_new * rsqrt(mean(s_new^2) + 1e-5) * pre-norm effective weight (pre_ffn for the attention boundary, next_pre_attn for the FFN boundary); store s_new fp32 and out bf16 into fresh torch.empty tensors. Take alpha, beta and the effective weights from the SGLang modules master already builds in prepare() (_residual(...) after precompute_coefficients(), _norm(...) effective weights), as fp32 [8192] tensors, rather than re-deriving constants. STRUCTURE (the difference from the incumbent and from the held row kernel): grid = NUM_SMS * 2 programs (host int from torch.cuda.get_device_properties), each walks rows t = pid, pid + nprog, ... < T; the row is processed in two passes over 8192 columns in BLOCK=2048 chunks with 16-byte vector loads (pass 1 accumulates sum(b^2) while keeping the branch chunk and stream chunk... if 8192 fp32 + 8192 bf16 per row do not fit in registers at num_warps 8, re-read them in pass 2 from L2, which is where they will be), and the next row's first chunk is loaded before the current row's stores (software prefetch), so each SM keeps two rows in flight and there are no CTA launch waves (753 rows was ~3-6 waves of one-row CTAs). prepare(): compare against the SGLang modules on synthetic inputs at T = 753 and 3012 for both boundaries (stream rel L2 <= 1e-6, bf16 output max diff <= 1 ulp); on mismatch log '[boundary] OFF: <reason>' and keep master's modules, else 'ON'. Nothing else changes.

## Expected

- `*load=turn_1*` faster: boundary rows 20.5-23.0 us each vs a 15.4 us byte floor (67-75%): target ~17-18 us each, ~-0.8% on turn_1 shapes
- `*load=turn_4*` faster: 74.7-79.6 us each vs 61.7 us floor: target ~66 us, ~-0.7%
- `*load=first_turn*` faster: ~396-400 us each vs 335 us floor at ~4.0 TB/s achieved: 0-5% on the rows, ~0-0.4% per shape

## Risk

Small: the review of the held row kernel put the incumbent at ~84% of floor at first_turn; this variant targets the turn shapes' wave and tail loss. Judge on the two muse_spark_norm_residual_kernel rows vs the new kernel's rows (Triton rows reproduce within ~1%), not the headline. No overlap with in-flight work (the attention rewrites end at the o projection's input; the expert work ends before the unpooler).
