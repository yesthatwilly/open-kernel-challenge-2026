---
problem: muse_prefill_layer
slug: norm-residual-boundaries-triton-row-kernel
created: 2026-10-24T165356Z
base: master
area: attention boundary + FFN boundary (FusedPostNormResidualPreNorm x2)
change: >
  Built on master (635cfee). Replace both FusedPostNormResidualPreNorm calls in kernel_fn
  (state['attn_boundary'] after the o projection, state['ffn_boundary'] after the unpooler;
  SGLang's JIT muse_spark_norm_residual_kernel<8192, 512, ...>) with one new Triton kernel
  _norm_residual_boundary_kernel, launched twice, at every T. Per row t (one program per row,
  BLOCK = 8192 = the whole row in registers, num_warps 16; also try 8 in prepare and keep the
  faster by graph-replay timing on synthetic [3012, 8192] inputs), with fp32 math throughout: b =
  branch[t].float() (bf16 [T, 8192]: attn_output or ffn_output); for the FFN boundary first b = b
  * (1 + post_ffn_norm) (MuseSpark1xRMSNormWithInputScale's input scale; the attention boundary's
  post-norm has no weight); b = b * rsqrt(mean(b^2) + 1e-8); s_new = alpha * stream[t] + beta * b
  (stream fp32 [T, 8192]); out = (s_new * rsqrt(mean(s_new^2) + 1e-5)) * (1 + w_pre) with w_pre =
  pre_ffn_norm (attention boundary) or next_pre_attn_norm (FFN boundary), stored bf16; s_new
  stored fp32. alpha, beta (fp32 [8192]) are precomputed in prepare() per
  reference.residual_coefficients: w = gate / 0.3, beta = sigmoid(w), alpha =
  sqrt(clamp(sigmoid(-w) * (1 + beta), min=RESIDUAL_GATE_EPS)); take them, and the effective (1 +
  w) norm weights and eps values, from the SGLang modules master already builds
  (_residual(...).precompute_coefficients(), _norm(...)._get_effective_weight() / effective weight
  attributes) so the arithmetic matches SGLang's rather than re-deriving constants; prepare()
  asserts the Triton outputs match the JIT kernel's on synthetic inputs at T=753 (stream rel L2 <=
  1e-6, bf16 output max ulp diff <= 1) for both boundaries before kernel_fn may use it, and falls
  back to master's modules if not. Outputs are fresh tensors (torch.empty) of the same
  dtypes/shapes; the router_dtype=None / branch_dtype=bf16 options master passes are the only ones
  reproduced. Everything else (GEMMs, attention, experts) is unchanged. Mechanism: each boundary
  moves 12 B per element (fp32 stream in + out, bf16 branch in, bf16 normed out), floor 15.4 us at
  T=753, 61.7 us at 3012, 335 us at 16384 at 4.8 TB/s; master measures 23.0/20.5 us at turn_1
  (67-75%), 78.0/74.7 at turn_4 (~80%), 398/396 at first_turn (84%), and the branch input was just
  written by the GEMM and is partly L2-resident, so the DRAM floor is lower still. A single-pass
  whole-row-in-registers kernel with 16 B vector loads should run at ~90% of bandwidth on all
  three T.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: two boundary kernels 23.0 + 20.5 us vs 2 x 15.4 floor; target ~2 x 17 us, ~8-10 us of 1080-1420 (~0.7%)
  - shapes: "*load=turn_4*"
    direction: faster
    why: 78.0 + 74.7 us vs 2 x 61.7 floor; target ~2 x 66, ~20 us of 3335-4938 (~0.5%)
  - shapes: "*load=first_turn*"
    direction: faster
    why: 398 + 396 us vs 2 x 335 floor; target ~2 x 365, ~60 us of 14275-22333 (~0.3-0.4%)
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# norm-residual-boundaries-triton-row-kernel

## Change

Built on master (635cfee). Replace both FusedPostNormResidualPreNorm calls in kernel_fn (state['attn_boundary'] after the o projection, state['ffn_boundary'] after the unpooler; SGLang's JIT muse_spark_norm_residual_kernel<8192, 512, ...>) with one new Triton kernel _norm_residual_boundary_kernel, launched twice, at every T. Per row t (one program per row, BLOCK = 8192 = the whole row in registers, num_warps 16; also try 8 in prepare and keep the faster by graph-replay timing on synthetic [3012, 8192] inputs), with fp32 math throughout: b = branch[t].float() (bf16 [T, 8192]: attn_output or ffn_output); for the FFN boundary first b = b * (1 + post_ffn_norm) (MuseSpark1xRMSNormWithInputScale's input scale; the attention boundary's post-norm has no weight); b = b * rsqrt(mean(b^2) + 1e-8); s_new = alpha * stream[t] + beta * b (stream fp32 [T, 8192]); out = (s_new * rsqrt(mean(s_new^2) + 1e-5)) * (1 + w_pre) with w_pre = pre_ffn_norm (attention boundary) or next_pre_attn_norm (FFN boundary), stored bf16; s_new stored fp32. alpha, beta (fp32 [8192]) are precomputed in prepare() per reference.residual_coefficients: w = gate / 0.3, beta = sigmoid(w), alpha = sqrt(clamp(sigmoid(-w) * (1 + beta), min=RESIDUAL_GATE_EPS)); take them, and the effective (1 + w) norm weights and eps values, from the SGLang modules master already builds (_residual(...).precompute_coefficients(), _norm(...)._get_effective_weight() / effective weight attributes) so the arithmetic matches SGLang's rather than re-deriving constants; prepare() asserts the Triton outputs match the JIT kernel's on synthetic inputs at T=753 (stream rel L2 <= 1e-6, bf16 output max ulp diff <= 1) for both boundaries before kernel_fn may use it, and falls back to master's modules if not. Outputs are fresh tensors (torch.empty) of the same dtypes/shapes; the router_dtype=None / branch_dtype=bf16 options master passes are the only ones reproduced. Everything else (GEMMs, attention, experts) is unchanged. Mechanism: each boundary moves 12 B per element (fp32 stream in + out, bf16 branch in, bf16 normed out), floor 15.4 us at T=753, 61.7 us at 3012, 335 us at 16384 at 4.8 TB/s; master measures 23.0/20.5 us at turn_1 (67-75%), 78.0/74.7 at turn_4 (~80%), 398/396 at first_turn (84%), and the branch input was just written by the GEMM and is partly L2-resident, so the DRAM floor is lower still. A single-pass whole-row-in-registers kernel with 16 B vector loads should run at ~90% of bandwidth on all three T.

## Expected

- `*load=turn_1*` faster: two boundary kernels 23.0 + 20.5 us vs 2 x 15.4 floor; target ~2 x 17 us, ~8-10 us of 1080-1420 (~0.7%)
- `*load=turn_4*` faster: 78.0 + 74.7 us vs 2 x 61.7 floor; target ~2 x 66, ~20 us of 3335-4938 (~0.5%)
- `*load=first_turn*` faster: 398 + 396 us vs 2 x 335 floor; target ~2 x 365, ~60 us of 14275-22333 (~0.3-0.4%)

## Risk

Small (~0.4-0.7% per load), so judge it on the two muse_spark_norm_residual_kernel rows being replaced by faster _norm_residual_boundary_kernel rows, not the headline. Risks: (1) the JIT kernel may already be near a practical bandwidth ceiling at T=16384, leaving only turn_1 to move; (2) exact semantics of MuseSpark1xRMSNormWithInputScale / NormPreservingResidualConnection (order of the input scale vs the post-norm, eps placement) must match SGLang, which is why the coefficients come from the SGLang modules and prepare() checks against the JIT kernel before use; (3) whole-row registers at 8192 fp32 per row with 16 warps is 16 values per thread for each of stream and branch, fine for occupancy. Correctness bounds are generous (3-15% rel L2) but the residual stream feeds the next layer, so the prepare() parity check is strict. No caching, no syncs in kernel_fn, bf16 where SGLang keeps bf16 and fp32 for the stream.
