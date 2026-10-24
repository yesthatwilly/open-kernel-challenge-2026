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
review: proceed
status: queued
review_at: 2026-10-24T140432Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T140432Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a straight port of a kernel already proven in muse_moe_w4a16 (1.130x). Master's decode_layer profile shows the same kernels appearing and disappearing at T<32, and the T>=32 shapes act as an in-run control. One run can be read from the per-shape kernel list alone, so no companion is warranted.

### Justification
Plausible, and the profile supports it directly. In fb07a3bd3aa54baf, _post_expert_norm_weighted_sum_kernel appears at T=32/128/256 (3.6 us sliding decode_32, 5.8/5.8/5.7 us verify_16 and sliding verify_32). It is absent at T=1/8/16. At those sizes reduce_kernel<512,1> (5.3-5.5 us at T=1, 8.9-9.0 us at T=8/16) and elementwise_kernel<128,2> x3 (4.9/6.6-6.7/8.6 us) appear instead, plus reduce_kernel<128,4> (4.3-4.9 us at T=8/16), which the hypothesis does not count. None of these show at T>=32, the same signature that moe removed: its post-change bs=1 profile was Marlin + align + _post_norm_combine_kernel 2.8 us + index, with the glue gone. The kernel matches reference.post_expert_norm exactly: scale by (1+w) in fp32, rsqrt(mean+1e-8), weight by g, sum the local slots, and zero tokens with no local slot. That is closer to the reference than SGLang's bf16 effective-weight path, and it is the same code that passed all 5 tests in moe. The post-norm runs on the main stream after w2 with nothing overlapping it, so any glue removed is wall-clock saved, and SGLang's baseline keeps paying it, so the saving lands in full in the paired ratio. The expected directions fit TASK.md. Sliding shapes carry weight 46 and full 16; gains of 6-10% at decode_1, verify_1 and decode_16 with flat T>=32 come to roughly +3-4% on the geomean, above the ~1.5-2.5% per-shape drift in LESSONS. Weakest link: how much glue is really post-norm glue. unrolled_elementwise_kernel x2 (5.7-6.6 us) also comes from kernel_fn's (topk_ids - local_start) % 256 and the expert_map gather, which stay. So at T=1 the safe saving is reduce 5.5 + elementwise 4.9 minus the ~2.8 us new kernel, about 8-14 us. That is the low end of the 12-20 us claimed, about 5-8% on decode_1 rather than 7-11%. A second soft spot: the T<32 boundary is also where block_size_m changes from 8 to 16, but Marlin's output is identical either way and the post-change profile separates the two.

### Confounders on the current kernel
- sglang-pipeline-parity-router-hilo-m8: this is the base, not a confounder. It introduced the fused_post_expert_norm_weighted_sum call being replaced and the bf16 hi/lo router at T=8. The router changes only how topk_weights is produced; _experts already casts it to fp32 and the new kernel reads it as fp32, so verify_1 cannot pick up an interaction.
- marlin-block-m-cap-16: no confounding. The cap clamps block_size_m only at T=128/256 (48/64 to 16). At T<32 the pick is 8, with or without the cap, and Marlin's y is the same tensor either way. The hypothesis's T>=32 controls do include the capped shapes, but they keep SGLang's call unchanged.
- fa3-hdim128-head-pairing-batched: no confounding. It changes only attention on full layers with batch>1 (decode_16 full is the only overlapping shape), and that is upstream of the MoE with no shared memory traffic or numerical path. Full decode_16 may show a smaller relative gain than sliding decode_16 only because its denominator is larger (407 vs 260 us).

### Interference and A/B needs
No companion is warranted. A gain can be read per kernel: reduce_kernel<512,1>, reduce_kernel<128,4> and elementwise_kernel<128,2> should vanish at T=1/8/16, _post_norm_combine_kernel should appear at about 2.8-3.2 us, and T>=32 should stay flat as the in-run drift control. A flat or loss can be read as well. If the glue is still present, it came from elsewhere (align, Marlin wrappers or the local_ids arithmetic). If the glue is gone but the result is flat, the new kernel's own top_kernels row says why. Reverting a landed hypothesis makes no sense here: reverting the base removes the code path being replaced, and the other two landed changes are shape- or area-disjoint, so neither ab-revert nor ab-tune could make either outcome less ambiguous. There are no sub-changes to isolate either, since it is a single launch replacing a single call. Decode has 1 check left today, and it is better spent on this run alone.

### Open questions
- Record the full T=1/8/16 kernel list in the attempt file, including the 'N more' rows. Check which of reduce_kernel<512,1>, reduce_kernel<128,4>, elementwise_kernel<128,2> x3, elementwise_kernel<128,4> and unrolled_elementwise x2 disappear. unrolled_elementwise is expected to stay in part, since it also serves the local_ids % 256 arithmetic.
- Expect T=1 to gain at the low end of the claim (about 8-14 us, 5-8%) rather than 12-20 us, because some of the counted unrolled_elementwise time is not post-norm.
- bf16-shape-gated-triton-gemms-small-t is queued for the same T<32 shapes. If it runs close to this one, attribute each result per kernel, not by headline, and combine them later by 3-way merge as LESSONS records for moe.
- If this gains, a separate stacked hypothesis could test the same kernel against SGLang's Triton kernel at T>=32 (3.6-5.8 us). The gain there would be under 1.5% per shape and hard to resolve in one run.

