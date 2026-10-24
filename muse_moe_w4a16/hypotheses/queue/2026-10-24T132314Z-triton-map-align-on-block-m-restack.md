---
problem: muse_moe_w4a16
slug: triton-map-align-on-block-m-restack
created: 2026-10-24T132314Z
base: master
area: MoE align/dispatch + block_size_m pick
change: >
  Stack the two measured, shape-disjoint moe wins: start from master
  (marlin-block-m-two-smaller-on-postnorm, 1.254x: block_size_m two candidates down at num_tokens
  >= 128) and add triton-map-align-small-m-port (1.185x on the older post-norm base, gained
  4.5-12.6% at bs=1/8/16/32, flat at bs=128/256) unchanged: the _align_small_kernel single launch
  for expert_map gather + align at num_tokens <= 32, with its prepare() warm-up and self-check.
  The result is the 3-way merge of the map-align branch onto master; the only conflict is the
  block_size_m pick, resolved by keeping map-align's _block_size_m() call and applying master's
  num_tokens >= 128 two-step adjustment right after it. The two changes touch disjoint shapes
  (<=32 vs >=128), so each shape should read as its own parent did.
expected:
  - shapes: bs=1 *
    direction: faster
    why: map-align +12.6% at bs=1
  - shapes: bs=8 *
    direction: faster
    why: map-align +6.5%
  - shapes: bs=16 *
    direction: faster
    why: map-align +6.1%
  - shapes: bs=32 *
    direction: faster
    why: map-align +4.5%
  - shapes: bs=128 *
    direction: flat
    why: master block-m pick unchanged
  - shapes: bs=256 *
    direction: flat
    why: master block-m pick unchanged
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# triton-map-align-on-block-m-restack

## Change

Stack the two measured, shape-disjoint moe wins: start from master (marlin-block-m-two-smaller-on-postnorm, 1.254x: block_size_m two candidates down at num_tokens >= 128) and add triton-map-align-small-m-port (1.185x on the older post-norm base, gained 4.5-12.6% at bs=1/8/16/32, flat at bs=128/256) unchanged: the _align_small_kernel single launch for expert_map gather + align at num_tokens <= 32, with its prepare() warm-up and self-check. The result is the 3-way merge of the map-align branch onto master; the only conflict is the block_size_m pick, resolved by keeping map-align's _block_size_m() call and applying master's num_tokens >= 128 two-step adjustment right after it. The two changes touch disjoint shapes (<=32 vs >=128), so each shape should read as its own parent did.

## Expected

- `bs=1 *` faster: map-align +12.6% at bs=1
- `bs=8 *` faster: map-align +6.5%
- `bs=16 *` faster: map-align +6.1%
- `bs=32 *` faster: map-align +4.5%
- `bs=128 *` flat: master block-m pick unchanged
- `bs=256 *` flat: master block-m pick unchanged

## Risk

Minted by churnsubs under the standing autonomy instruction to restack two harvested wins; the merged file is at /home/user/.claude/jobs/7156a0da/tmp/merge/merged.py (map-align branch + 5-line block-m adjust).
