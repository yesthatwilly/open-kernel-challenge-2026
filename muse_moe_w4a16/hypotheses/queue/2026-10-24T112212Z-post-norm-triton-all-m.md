---
problem: muse_moe_w4a16
slug: post-norm-triton-all-m
created: 2026-10-24T112212Z
base: master
area: post-expert norm+combine
change: >
  On top of marlin-block-m-two-smaller-on-postnorm, remove the SMALL_M_POST_NORM = 32 split at the
  end of kernel_fn so every num_tokens (bs=32/33/128/256/257 too) combines with master's
  _post_norm_combine_kernel instead of SGLang's fused_post_expert_norm_weighted_sum. The kernel
  itself is unchanged: grid (num_tokens,), num_warps=8, BLOCK_H = next_power_of_2(H) = 4096, one
  program per token looping over its 8 slots with tl.static_range, skipping remote slots (local id
  -1) without loading them, fp32 math, (1 + w) applied in-kernel from the bf16 post_norm_w, bf16
  store into a fresh [M, H] output. y.contiguous() stays (a no-op on Marlin's contiguous output).
  state['norm_w'] and the fused_post_expert_norm_weighted_sum import become unused; leave them or
  delete them, nothing else changes (block_size_m picks, align, both Marlin calls, silu_and_mul as
  in the dependency).
expected:
  - shapes: bs=128 *
    direction: faster
    why: SGLang _post_expert_norm_weighted_sum_kernel is 6.1 us at bs=128 while _post_norm_combine_kernel is 3.1-3.2 us at bs=8/16 (64-128 slots) and is L2-resident on w2 output; expect ~2 us saved, ~0.6% of the ~345 us restacked shape, readable in top_kernels rather than the headline
  - shapes: bs=256 *
    direction: faster
    why: "SGLang kernel 5.2 us; ours ~3.5-4.5 us for 256 programs: ~1 us, ~0.3% of ~470 us"
  - shapes: bs=32 *
    direction: flat
    why: "SGLang kernel 3.7 us vs ours ~3.3 us: under drift, but bs=32 is the weakest shape and every us there counts"
  - shapes: bs=1 *
    direction: flat
    why: "control: same kernel as master"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
risk: low
depends_on: marlin-block-m-two-smaller-on-postnorm
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# post-norm-triton-all-m

## Change

On top of marlin-block-m-two-smaller-on-postnorm, remove the SMALL_M_POST_NORM = 32 split at the end of kernel_fn so every num_tokens (bs=32/33/128/256/257 too) combines with master's _post_norm_combine_kernel instead of SGLang's fused_post_expert_norm_weighted_sum. The kernel itself is unchanged: grid (num_tokens,), num_warps=8, BLOCK_H = next_power_of_2(H) = 4096, one program per token looping over its 8 slots with tl.static_range, skipping remote slots (local id -1) without loading them, fp32 math, (1 + w) applied in-kernel from the bf16 post_norm_w, bf16 store into a fresh [M, H] output. y.contiguous() stays (a no-op on Marlin's contiguous output). state['norm_w'] and the fused_post_expert_norm_weighted_sum import become unused; leave them or delete them, nothing else changes (block_size_m picks, align, both Marlin calls, silu_and_mul as in the dependency).

## Expected

- `bs=128 *` faster: SGLang _post_expert_norm_weighted_sum_kernel is 6.1 us at bs=128 while _post_norm_combine_kernel is 3.1-3.2 us at bs=8/16 (64-128 slots) and is L2-resident on w2 output; expect ~2 us saved, ~0.6% of the ~345 us restacked shape, readable in top_kernels rather than the headline
- `bs=256 *` faster: SGLang kernel 5.2 us; ours ~3.5-4.5 us for 256 programs: ~1 us, ~0.3% of ~470 us
- `bs=32 *` flat: SGLang kernel 3.7 us vs ours ~3.3 us: under drift, but bs=32 is the weakest shape and every us there counts
- `bs=1 *` flat: control: same kernel as master
- `bs=8 *` flat: control
- `bs=16 *` flat: control

## Risk

Expected effect is small (~0.5-2 us per shape, well under the ~1.5% geomean resolution in LESSONS): it is a cleanup-and-measure run whose verdict should come from the replaced kernel's own time (SGLang _post_expert_norm_weighted_sum_kernel 3.7/6.1/5.2 us at bs=32/128/256 vs _post_norm_combine_kernel on the same shapes), not the headline. Run it after the restack so the bs>=128 shapes are smaller and the combine is a larger share. Risk: our kernel's per-token serial 8-row loop could lose at bs=256 if 256 programs x 8 dependent 8 KB loads underfill bandwidth; if _post_norm_combine_kernel exceeds SGLang's time at bs>=128 the right follow-up is the threshold at 128, not a rewrite. Correctness: same kernel already passes the 5 tests at bs<32; bs=257 just adds programs.
