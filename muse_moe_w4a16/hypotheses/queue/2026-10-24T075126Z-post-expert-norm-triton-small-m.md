---
problem: muse_moe_w4a16
slug: post-expert-norm-triton-small-m
created: 2026-10-24T075126Z
base: master
area: post-expert norm+combine
change: >
  For num_tokens < 32 only (bs=1, 8, 16; the bs=4 test too), replace the call to SGLang's
  fused_post_expert_norm_weighted_sum at the end of kernel_fn with one launch of a new Triton
  kernel _post_norm_combine_kernel defined in submission.py; bs>=32 keeps SGLang's call untouched
  as the in-run control. Evidence for the target: in master's profile (job fb2cc736613747df)
  SGLang's Triton _post_expert_norm_weighted_sum_kernel is in the top kernels at bs=32/128/256
  (4.1/6.5/6.1 us) but absent at bs=1/8/16, where instead at::native reduce_kernel<512,1>
  (5.6/8.9/8.9 us), elementwise_kernel<128,2> (4.9/6.5/8.6 us) and unrolled_elementwise_kernel
  (5.4/5.8/6.1 us) appear, i.e. ~16 us at bs=1 (20% of decode_1) and ~21-24 us at bs=8/16; they
  are also unchanged in align-single-token-ep's bs=1 profile, so they do not come from the align.
  The reading is that the SGLang wrapper takes an eager-PyTorch path (float cast, broadcast mul by
  norm weight, pow/mean reduce, rsqrt, weighted slot sum, bf16 cast) at small M. Kernel: grid
  (num_tokens,), num_warps=8, BLOCK_H=4096 (= H, constexpr). Program t zeroes an fp32
  acc[BLOCK_H]; loads nw = 1 + post_norm_w (bf16 original kept in state as state['post_norm_w'],
  cast to fp32 in-kernel, so no new-precision copy of a weight is stored); loops s over 8 slots
  (tl.static_range): eid = load(local_ids[t*8+s]); g = load(topk_weights[t*8+s]); a scalar "if eid >= 0:" branch
  (remote slots are skipped entirely, never loaded, because cache3 rows for remote slots are
  uninitialised torch.empty memory and 0*NaN would poison the sum) loads y[t, s, :] (the w2 Marlin
  output viewed [M, 8, H]) as fp32, v = y * nw, r = rsqrt(sum(v*v)/H + 1e-8), acc += g * v * r.
  Store acc as bf16 into a fresh torch.empty((M, H), bf16) output; tokens with no local slot get
  zeros as the reference requires. This matches reference.post_expert_norm exactly (scale by 1+w
  first, then normalise, then weight) with fp32 math. No atomics, no memset, one launch.
  Everything else in kernel_fn (align, both Marlin calls, silu_and_mul, block_size_m pick) is
  unchanged.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "the ~16 us of at::native reduce/elementwise glue (20% of decode_1, 80.2 us) becomes one ~2-3 us Triton launch reading <=64 KB; expect ~10-13 us off, ~1.12-1.18x on decode_1"
  - shapes: bs=8 *
    direction: faster
    why: ~21 us of the same glue at 152 us -> one launch; expect ~8-12%
  - shapes: bs=16 *
    direction: faster
    why: ~24 us of the same glue at 191 us -> one launch; expect ~8-11%
  - shapes: bs=32 *
    direction: flat
    why: "control: SGLang call kept (its Triton kernel already runs here, 4.1 us)"
  - shapes: bs=128 *
    direction: flat
    why: "control: unchanged"
  - shapes: bs=256 *
    direction: flat
    why: "control: unchanged"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T092219Z
review_companions: null
---

# post-expert-norm-triton-small-m

## Change

For num_tokens < 32 only (bs=1, 8, 16; the bs=4 test too), replace the call to SGLang's fused_post_expert_norm_weighted_sum at the end of kernel_fn with one launch of a new Triton kernel _post_norm_combine_kernel defined in submission.py; bs>=32 keeps SGLang's call untouched as the in-run control. Evidence for the target: in master's profile (job fb2cc736613747df) SGLang's Triton _post_expert_norm_weighted_sum_kernel is in the top kernels at bs=32/128/256 (4.1/6.5/6.1 us) but absent at bs=1/8/16, where instead at::native reduce_kernel<512,1> (5.6/8.9/8.9 us), elementwise_kernel<128,2> (4.9/6.5/8.6 us) and unrolled_elementwise_kernel (5.4/5.8/6.1 us) appear, i.e. ~16 us at bs=1 (20% of decode_1) and ~21-24 us at bs=8/16; they are also unchanged in align-single-token-ep's bs=1 profile, so they do not come from the align. The reading is that the SGLang wrapper takes an eager-PyTorch path (float cast, broadcast mul by norm weight, pow/mean reduce, rsqrt, weighted slot sum, bf16 cast) at small M. Kernel: grid (num_tokens,), num_warps=8, BLOCK_H=4096 (= H, constexpr). Program t zeroes an fp32 acc[BLOCK_H]; loads nw = 1 + post_norm_w (bf16 original kept in state as state['post_norm_w'], cast to fp32 in-kernel, so no new-precision copy of a weight is stored); loops s over 8 slots (tl.static_range): eid = load(local_ids[t*8+s]); g = load(topk_weights[t*8+s]); a scalar "if eid >= 0:" branch (remote slots are skipped entirely, never loaded, because cache3 rows for remote slots are uninitialised torch.empty memory and 0*NaN would poison the sum) loads y[t, s, :] (the w2 Marlin output viewed [M, 8, H]) as fp32, v = y * nw, r = rsqrt(sum(v*v)/H + 1e-8), acc += g * v * r. Store acc as bf16 into a fresh torch.empty((M, H), bf16) output; tokens with no local slot get zeros as the reference requires. This matches reference.post_expert_norm exactly (scale by 1+w first, then normalise, then weight) with fp32 math. No atomics, no memset, one launch. Everything else in kernel_fn (align, both Marlin calls, silu_and_mul, block_size_m pick) is unchanged.

## Expected

- `bs=1 *` faster: the ~16 us of at::native reduce/elementwise glue (20% of decode_1, 80.2 us) becomes one ~2-3 us Triton launch reading <=64 KB; expect ~10-13 us off, ~1.12-1.18x on decode_1
- `bs=8 *` faster: ~21 us of the same glue at 152 us -> one launch; expect ~8-12%
- `bs=16 *` faster: ~24 us of the same glue at 191 us -> one launch; expect ~8-11%
- `bs=32 *` flat: control: SGLang call kept (its Triton kernel already runs here, 4.1 us)
- `bs=128 *` flat: control: unchanged
- `bs=256 *` flat: control: unchanged

## Risk

Risk: the attribution of the at::native kernels to the post-norm wrapper is inferred from which kernels appear/disappear per shape (no SGLang source on this machine). If after the change the reduce/elementwise kernels are still in the bs=1/8/16 profile, they come from elsewhere (silu_and_mul's expert_ids masking path or the Marlin wrapper at small M); record that and the remaining kernel names in the attempt file, and the flat result reads as 'not the post-norm'. Correctness: fp32 accumulation matches the FP32 reference more closely than SGLang's bf16-weight path; the bs=1 and bs=4 tests exercise the branch, including tokens with zero local slots. Single CTA at bs=1 reads <=8 rows of 8 KB from L2 (just written by w2), so it is latency-bound at ~2 us; if it measures slow, split H across 2-4 programs per token is the ab-tune, not part of this run.

## Adversarial review

Reviewed 2026-10-24T092219Z by an adversarial subagent (model opus).

**Verdict: proceed.** Well-targeted and self-diagnosing: at bs<32 master's profile has ~9 extra at::native launches (about 20-29 us at bs=1) and no _post_expert_norm_weighted_sum_kernel, while at bs>=32 the Triton kernel runs and the glue is gone. Replacing exactly that call with one Triton launch either removes the glue (a gain) or leaves it in the profile (a flat that is clearly not the post-norm), so a single run can be read without companions.

### Justification
Plausible. The full master profile (fb2cc736613747df) supports it more strongly than the hypothesis says. bs=32 lists only 7 kernels in total: 2 Marlin, align, _post_expert_norm_weighted_sum_kernel 4.1us, index_elementwise (the expert_map gather), act_and_mul, and one unrolled_elementwise. bs=1/8/16 list 16 kernels each ('12 + 4 more'), and _post_expert_norm_weighted_sum_kernel does not appear down to the 1.0-1.2us rows. A Triton kernel that costs 4.1us at bs=32 would not drop below 1.2us at bs=16, so it really is gone. In its place are reduce_kernel<512,1>, reduce_kernel<128,4> (4.3-4.7us, which the hypothesis does not count), elementwise_kernel<128,2> x3, elementwise_kernel<128,4>, unrolled_elementwise x2, two vectorized_elementwise, and more. At bs=1: 78.1us total, minus Marlin 41.1, align 4.4, act 1.5 and index 1.8, leaves about 29us of glue, and some of it is not post-norm (the one unrolled_elementwise that is also present at bs=32). The ~16us estimate is therefore conservative; the real bound is about 20-25us at bs=1. The work is tiny: at most 64 KB at bs=1 and 1 MB at bs=16, still in L2 because w2 just wrote it. One launch per token at 8 warps with BLOCK_H=4096 (16 fp32 acc regs per thread) should take 2-4us. The direction and size fit TASK.md. All six loads have equal weight, so gains of roughly +12-18% at decode_1 and +8-12% at verify_1 and decode_16 would put the geomean near 1.04-1.05x, well outside the ±1.5% drift. SGLang's baseline pays the same eager path, so the paired ratio benefits fully. Weakest link: the attribution is inferred, and the M<32 boundary is also where block_size_m changes (bs=1/8/16 pick block_size_m=8; bs=32 picks 16). Some of the reduce/elementwise launches could come from the Marlin or align wrappers' block-8 path rather than the post-norm wrapper. Against that, those wrappers' own kernels (Marlin x2, small_batch align) appear at every size, and the post-norm kernel's absence is direct evidence. The profile after the change settles it either way. Minor: SGLang's wrapper threshold is unknown (anywhere in 17..32), so the <32 cut is safe but may not be tight.

### Confounders on the current kernel
- sglang-marlin-path-parity: this is the base, not a confounder. It introduced the fused_post_expert_norm_weighted_sum call being replaced and the block_size_m rule that switches from 8 to 16 at the same M=32 boundary. The change touches only the final combine, and Marlin's output y is identical either way, so a gain cannot come from an interaction. The only confusion it adds is in attribution, and the post-change profile resolves that (whether the at::native reduce/elementwise launches remain at bs=1/8/16).

### Interference and A/B needs
No companion warranted. A gain means the replaced call's eager glue is gone, and the profile shows which launches vanished. A flat or loss with the glue still present means it came from elsewhere (Marlin/align block-8 wrapper), and the hypothesis already records that. A flat or loss with the glue gone means the new kernel is slow, and its own top_kernels time shows that directly. bs>=32 is an in-run control for drift. An ab-revert of the only landed hypothesis would remove the Marlin path altogether, which makes no sense. An ab-tune of block_size_m belongs to the queued marlin-block-size-m-* hypotheses. The split-H variant is correctly deferred as a follow-up ab-tune, to run only if the kernel itself measures slow.

### Open questions
- Do the reduce_kernel<512,1>, reduce_kernel<128,4> and elementwise_kernel<128,2> launches vanish at bs=1/8/16 after the change? Record the full kernel list (including the '4 more') in the attempt file. That separates post-norm glue from block-size-8 Marlin/align wrapper glue.
- If it gains: the same kernel could replace SGLang's Triton call at bs>=32 (4.1-6.5us), but only as a separate hypothesis, since the gain would be under 1.5% per shape and hard to resolve.
- The bs=4 test exercises tokens that have no local slot, and bs=1 can have zero local slots. Confirm the zero row is written explicitly (acc is stored even when every slot is skipped), so torch.empty garbage never reaches the output.

