---
problem: muse_decode_layer
slug: post-expert-norm-triton-small-t
created: 2026-10-24T140057Z
base: master
area: post-expert norm+combine
change: >
  Port muse_moe_w4a16's landed post-expert-norm-triton-small-m (1.130x there, job
  bacd1f502f4e46fa; patch
  muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-triton-small-m.patch) into
  muse_decode_layer/submission.py _experts. Today _experts ends with SGLang's
  fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['post_expert_w'],
  eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids). Master's profile
  (fb07a3bd3aa54baf) shows the same signature as moe's: SGLang's Triton
  _post_expert_norm_weighted_sum_kernel is in the top kernels only at T>=32 (3.6 us at decode_32,
  5.7-5.8 us at verify_16/32), while at T=1/8/16 it is absent and at::native reduce_kernel<512,1>
  (5.3-5.5 us at T=1, 8.9-9.0 us at T=8/16), elementwise_kernel<128,2> x3 (4.9 / 6.6-6.7 / 8.6 us)
  and unrolled_elementwise_kernel x2 (5.7-6.6 us) appear instead: the wrapper's eager-PyTorch path
  at small M. Change: add the verbatim _post_norm_combine_kernel from that patch (grid (T,),
  num_warps=8, BLOCK_H=next_power_of_2(4096), fp32 acc, static_range over the 8 slots, scalar 'if
  eid >= 0' so remote slots' uninitialised cache3 rows are never loaded, nw = 1 + w added in fp32
  in-kernel, explicit zero row for tokens with no local slot, bf16 store into a fresh
  torch.empty((T, 4096), bf16)), a constant SMALL_T_POST_NORM = 32, and in prepare()
  state['post_expert_norm_w'] = weights['post_expert_norm'].contiguous() (the original BF16
  weight, no new-precision copy; post_expert_w stays for the SGLang path). In _experts, after the
  w2 Marlin call: if num_tokens < 32 launch the kernel on y (the w2 output viewed [T, 8, 4096],
  .contiguous()), local_ids (bank['expert_map'][topk_ids], int32, -1 for remote), topk_weights
  (fp32) and return its output; else keep SGLang's call unchanged as the in-run control. Shape
  branch only. Everything else (local_ids arithmetic, moe_align_block_size, block_size_m pick with
  cap 16, both Marlin calls, silu_and_mul, pre-expert norm, unpooler, boundaries, attention) is
  untouched.
expected:
  - shapes: "*load=decode_1 *"
    direction: faster
    why: "T=1: ~16-20 us of at::native reduce/elementwise glue (reduce 5.5 + elementwise 4.9 + part of unrolled 6.0) becomes one ~2-3 us launch; moe saved 25 us at bs=1. Expect 12-20 us off 173/182 us: +7-11% on both layers"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "T=8: reduce 9.0 + elementwise 6.7 + unrolled 5.9 us of glue; moe saved 31 us at bs=8. Expect 15-25 us off 238/248 us: +6-10%"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "T=16: reduce 9.0 + elementwise 8.6 + unrolled 6.3 us; moe saved 36 us at bs=16. Expect 15-25 us off 260/407 us: +4-9% (sliding larger)"
  - shapes: "*load=decode_32 *"
    direction: flat
    why: "control: T=32 keeps SGLang call, its Triton kernel already runs (3.6 us)"
  - shapes: "*load=verify_16 *"
    direction: flat
    why: "control: T=128 unchanged"
  - shapes: "*load=verify_32 *"
    direction: flat
    why: "control: T=256 unchanged"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# post-expert-norm-triton-small-t

## Change

Port muse_moe_w4a16's landed post-expert-norm-triton-small-m (1.130x there, job bacd1f502f4e46fa; patch muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-triton-small-m.patch) into muse_decode_layer/submission.py _experts. Today _experts ends with SGLang's fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['post_expert_w'], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids). Master's profile (fb07a3bd3aa54baf) shows the same signature as moe's: SGLang's Triton _post_expert_norm_weighted_sum_kernel is in the top kernels only at T>=32 (3.6 us at decode_32, 5.7-5.8 us at verify_16/32), while at T=1/8/16 it is absent and at::native reduce_kernel<512,1> (5.3-5.5 us at T=1, 8.9-9.0 us at T=8/16), elementwise_kernel<128,2> x3 (4.9 / 6.6-6.7 / 8.6 us) and unrolled_elementwise_kernel x2 (5.7-6.6 us) appear instead: the wrapper's eager-PyTorch path at small M. Change: add the verbatim _post_norm_combine_kernel from that patch (grid (T,), num_warps=8, BLOCK_H=next_power_of_2(4096), fp32 acc, static_range over the 8 slots, scalar 'if eid >= 0' so remote slots' uninitialised cache3 rows are never loaded, nw = 1 + w added in fp32 in-kernel, explicit zero row for tokens with no local slot, bf16 store into a fresh torch.empty((T, 4096), bf16)), a constant SMALL_T_POST_NORM = 32, and in prepare() state['post_expert_norm_w'] = weights['post_expert_norm'].contiguous() (the original BF16 weight, no new-precision copy; post_expert_w stays for the SGLang path). In _experts, after the w2 Marlin call: if num_tokens < 32 launch the kernel on y (the w2 output viewed [T, 8, 4096], .contiguous()), local_ids (bank['expert_map'][topk_ids], int32, -1 for remote), topk_weights (fp32) and return its output; else keep SGLang's call unchanged as the in-run control. Shape branch only. Everything else (local_ids arithmetic, moe_align_block_size, block_size_m pick with cap 16, both Marlin calls, silu_and_mul, pre-expert norm, unpooler, boundaries, attention) is untouched.

## Expected

- `*load=decode_1 *` faster: T=1: ~16-20 us of at::native reduce/elementwise glue (reduce 5.5 + elementwise 4.9 + part of unrolled 6.0) becomes one ~2-3 us launch; moe saved 25 us at bs=1. Expect 12-20 us off 173/182 us: +7-11% on both layers
- `*load=verify_1 *` faster: T=8: reduce 9.0 + elementwise 6.7 + unrolled 5.9 us of glue; moe saved 31 us at bs=8. Expect 15-25 us off 238/248 us: +6-10%
- `*load=decode_16 *` faster: T=16: reduce 9.0 + elementwise 8.6 + unrolled 6.3 us; moe saved 36 us at bs=16. Expect 15-25 us off 260/407 us: +4-9% (sliding larger)
- `*load=decode_32 *` flat: control: T=32 keeps SGLang call, its Triton kernel already runs (3.6 us)
- `*load=verify_16 *` flat: control: T=128 unchanged
- `*load=verify_32 *` flat: control: T=256 unchanged

## Risk

Largest expected lever of the unported sibling wins: 3 of 6 loads gain ~6-10%, ~+4% on the geomean. Attribution is by the same per-shape kernel signature that held in moe (the glue vanished there). Some unrolled_elementwise launches are also the local_ids (topk_ids - local_start) % 256 arithmetic, which stays; record the full T=1/8/16 kernel list in the attempt file. Correctness: fp32 math matches the reference ordering (scale by 1+w, normalise, weight); the 13 tests include T<32 shapes with tokens that have no local slot (zero row written explicitly). Overlap: shape-disjoint in area from any GEMM or attention change; touches the same T<32 shapes as a bf16 GEMM port, so judge per kernel.
