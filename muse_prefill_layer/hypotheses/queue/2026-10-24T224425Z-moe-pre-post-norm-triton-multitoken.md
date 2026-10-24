---
problem: muse_prefill_layer
slug: moe-pre-post-norm-triton-multitoken
created: 2026-10-24T224425Z
base: master
area: pre-expert norm + post-expert norm/combine (the MoE elementwise ends)
change: >
  Built on master (635cfee). Two kernels at the ends of the MoE block, at every T. (1) PRE-EXPERT
  NORM: replace state['pre_expert_norm'](expert_input) (MuseSpark1xRMSNorm -> F.rms_norm, profile
  row at::native vectorized_layer_norm: 6.6/17.8/79 us at T=753/3012/16384 vs byte floors
  ~2.6/10/56 us) with a Triton row kernel _pre_expert_rmsnorm_kernel: grid NUM_SMS*4 persistent
  programs over rows, each row [4096] bf16 loaded once, fp32 sum of squares, y = x * rsqrt(mean +
  1e-5) * w_eff, stored bf16 into a fresh [T, 4096] tensor; w_eff is the module's effective (1 +
  w) weight taken from the SGLang module in prepare() (fp32). (2) POST-EXPERT NORM + WEIGHTED SUM:
  replace fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['post_expert_w'],
  eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids) in both _experts_marlin and
  _experts_triton with a Triton kernel modelled on muse_moe_w4a16 master's
  _post_norm_combine_kernel (copy it from muse_moe_w4a16/submission.py, where it gained 1.130x at
  small M) generalized to many tokens per program: grid NUM_SMS*4 persistent programs over tokens;
  per token loop k in 0..7, skip slots whose local id is -1 (never read their y row, which may be
  unwritten), for local slots load y[t*8+k] [4096] bf16, z = y * post_expert_w (weight_first=True:
  multiply by (1 + w) before normalizing, as master's call does), z *= rsqrt(mean(z^2) + 1e-8),
  acc += topk_weight[t, k] * z, all fp32; store acc bf16 into a fresh [T, 4096] tensor. Copy moe's
  kernel's exact order of operations; do not re-derive. prepare(): both kernels vs the SGLang
  calls on synthetic inputs at T = 753 and 16384 (pre-norm max diff <= 1 bf16 ulp; combine rel L2
  <= 1e-3 with NaN in every remote y row, proving they are never read); on failure log '[moe-ends]
  OFF: <reason>' and keep master's calls, else 'ON'.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "pre-norm 79 us -> ~60 us, post-norm/combine 122 us -> ~70 us (16384 local y rows + output, ~268 MB): ~-0.5% per shape"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "17.8 + 24.4 us -> ~12 + ~12 us: ~-0.5%"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "6.6 + 8.3-8.9 us -> ~4 + ~5 us: ~-0.5%"
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# moe-pre-post-norm-triton-multitoken

## Change

Built on master (635cfee). Two kernels at the ends of the MoE block, at every T. (1) PRE-EXPERT NORM: replace state['pre_expert_norm'](expert_input) (MuseSpark1xRMSNorm -> F.rms_norm, profile row at::native vectorized_layer_norm: 6.6/17.8/79 us at T=753/3012/16384 vs byte floors ~2.6/10/56 us) with a Triton row kernel _pre_expert_rmsnorm_kernel: grid NUM_SMS*4 persistent programs over rows, each row [4096] bf16 loaded once, fp32 sum of squares, y = x * rsqrt(mean + 1e-5) * w_eff, stored bf16 into a fresh [T, 4096] tensor; w_eff is the module's effective (1 + w) weight taken from the SGLang module in prepare() (fp32). (2) POST-EXPERT NORM + WEIGHTED SUM: replace fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['post_expert_w'], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids) in both _experts_marlin and _experts_triton with a Triton kernel modelled on muse_moe_w4a16 master's _post_norm_combine_kernel (copy it from muse_moe_w4a16/submission.py, where it gained 1.130x at small M) generalized to many tokens per program: grid NUM_SMS*4 persistent programs over tokens; per token loop k in 0..7, skip slots whose local id is -1 (never read their y row, which may be unwritten), for local slots load y[t*8+k] [4096] bf16, z = y * post_expert_w (weight_first=True: multiply by (1 + w) before normalizing, as master's call does), z *= rsqrt(mean(z^2) + 1e-8), acc += topk_weight[t, k] * z, all fp32; store acc bf16 into a fresh [T, 4096] tensor. Copy moe's kernel's exact order of operations; do not re-derive. prepare(): both kernels vs the SGLang calls on synthetic inputs at T = 753 and 16384 (pre-norm max diff <= 1 bf16 ulp; combine rel L2 <= 1e-3 with NaN in every remote y row, proving they are never read); on failure log '[moe-ends] OFF: <reason>' and keep master's calls, else 'ON'.

## Expected

- `*load=first_turn*` faster: pre-norm 79 us -> ~60 us, post-norm/combine 122 us -> ~70 us (16384 local y rows + output, ~268 MB): ~-0.5% per shape
- `*load=turn_4*` faster: 17.8 + 24.4 us -> ~12 + ~12 us: ~-0.5%
- `*load=turn_1*` faster: 6.6 + 8.3-8.9 us -> ~4 + ~5 us: ~-0.5%

## Risk

Small and below the headline noise floor (~0.5% per shape); judge it on the four replaced rows (Triton rows reproduce within ~1%). Port of moe's post-norm kernel; its all-M variant was neutral at moe's tiny M, but prefill's M is 753-16384, where SGLang's kernel runs at ~2.2 TB/s. Composes with every expert-engine and dispatch hypothesis in the queue (it reads y by slot and skips remote slots, which is also what the local-routes align needs).
