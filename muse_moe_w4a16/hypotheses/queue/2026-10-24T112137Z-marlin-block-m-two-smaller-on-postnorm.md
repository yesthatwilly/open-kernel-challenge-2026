---
problem: muse_moe_w4a16
slug: marlin-block-m-two-smaller-on-postnorm
created: 2026-10-24T112137Z
base: master
area: w13 marlin
change: >
  Restack the landed-but-unmerged marlin-block-size-m-verify-two-smaller (1.092x on the
  pre-post-norm base, job ed837a4a46d44a95; patch at
  muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.patch) onto current
  master (post-expert-norm-triton-small-m, 192f225d8f). In kernel_fn only, keep SGLang's selection
  loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) unchanged (first candidate with
  num_tokens*8/bank['count']/block_size_m < 0.9, loop exhausting at 64), remember the index idx it
  stopped at, and for num_tokens >= 128 only set block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx -
  2, 0)]. Picks with bank count 32: bs=128 48 -> 16 (thread_m_blocks 3 -> 1), bs=256/257 64 -> 32
  (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps master's pick (8 or 16) and is the
  in-run control. The chosen value feeds moe_align_block_size (bank count 32, as on master;
  bs>=128 does not take the moe_align_single_token branch) and the moe_block_size kwarg of both
  moe_wna16_marlin_gemm calls (w13 and w2). Nothing else changes: the post-norm split at
  SMALL_M_POST_NORM = 32 (Triton _post_norm_combine_kernel below, SGLang's
  fused_post_expert_norm_weighted_sum at bs>=32) stays exactly as on master, as do silu_and_mul,
  the workspace and both Marlin calls' other kwargs. The branch is on num_tokens only (shape), no
  sync, nothing input-dependent.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "same change gained 36% on verify_16 on the old base (468.8 -> 343.7 us, Marlin 435 -> 275 us); post-norm did not touch bs=128 (SGLang combine kept), so the Marlin saving carries over: expect ~340-350 us vs master 475.9"
  - shapes: bs=256 *
    direction: faster
    why: 64 -> 32 gained 32% on verify_32 on the old base (618.6 -> 468.7 us); same untouched combine on master, expect ~470 us vs 622.2
  - shapes: bs=1 *
    direction: flat
    why: "control: unchanged pick (8) and unchanged Triton post-norm"
  - shapes: bs=8 *
    direction: flat
    why: control
  - shapes: bs=16 *
    direction: flat
    why: control
  - shapes: bs=32 *
    direction: flat
    why: "control: pick stays 16 (16 -> 8 lost 3.6% in marlin-block-size-m-next-smaller), SGLang combine kept"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-block-m-two-smaller-on-postnorm

## Change

Restack the landed-but-unmerged marlin-block-size-m-verify-two-smaller (1.092x on the pre-post-norm base, job ed837a4a46d44a95; patch at muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.patch) onto current master (post-expert-norm-triton-small-m, 192f225d8f). In kernel_fn only, keep SGLang's selection loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) unchanged (first candidate with num_tokens*8/bank['count']/block_size_m < 0.9, loop exhausting at 64), remember the index idx it stopped at, and for num_tokens >= 128 only set block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - 2, 0)]. Picks with bank count 32: bs=128 48 -> 16 (thread_m_blocks 3 -> 1), bs=256/257 64 -> 32 (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps master's pick (8 or 16) and is the in-run control. The chosen value feeds moe_align_block_size (bank count 32, as on master; bs>=128 does not take the moe_align_single_token branch) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls (w13 and w2). Nothing else changes: the post-norm split at SMALL_M_POST_NORM = 32 (Triton _post_norm_combine_kernel below, SGLang's fused_post_expert_norm_weighted_sum at bs>=32) stays exactly as on master, as do silu_and_mul, the workspace and both Marlin calls' other kwargs. The branch is on num_tokens only (shape), no sync, nothing input-dependent.

## Expected

- `bs=128 *` faster: same change gained 36% on verify_16 on the old base (468.8 -> 343.7 us, Marlin 435 -> 275 us); post-norm did not touch bs=128 (SGLang combine kept), so the Marlin saving carries over: expect ~340-350 us vs master 475.9
- `bs=256 *` faster: 64 -> 32 gained 32% on verify_32 on the old base (618.6 -> 468.7 us); same untouched combine on master, expect ~470 us vs 622.2
- `bs=1 *` flat: control: unchanged pick (8) and unchanged Triton post-norm
- `bs=8 *` flat: control
- `bs=16 *` flat: control
- `bs=32 *` flat: control: pick stays 16 (16 -> 8 lost 3.6% in marlin-block-size-m-next-smaller), SGLang combine kept

## Risk

Pure restack of two orthogonal measured gains: the block-m change touches only bs>=128, the post-norm change only bs<32, so the expected headline is roughly 1.130 x 1.092 = ~1.23x, and the bs<128 rows must reproduce master's (55/156/207/121 us) within drift. If marlin-block-m-cap-16-port (in the judge, bs=128 48->16 and bs=256 64->16) harvests better than verify-two-smaller at bs=256, the bs=256 pick here should follow it (16 instead of 32) - either retune this before claim or queue an ab-tune companion; bs=128 is 16 in both. Risk is low: the identical selection code already passed all 5 tests (incl. bs=257) on the old base; the only interaction is that bs=128/256 still use SGLang's combine, which is unchanged.
