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
review: required
status: queued
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
