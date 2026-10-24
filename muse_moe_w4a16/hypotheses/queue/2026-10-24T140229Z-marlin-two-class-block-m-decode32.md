---
problem: muse_moe_w4a16
slug: marlin-two-class-block-m-decode32
created: 2026-10-24T140229Z
base: master
area: MoE align/dispatch + block_size_m pick
change: >
  At num_tokens in (16, 32] only (decode_32 bs=32; no other scored or test shape; bs=33 is above
  32 and stays on master), give cold experts Marlin's 8-row block and keep SGLang's 16 for the
  rest, by splitting each GEMM into two Marlin launches over disjoint expert classes. Add
  _align_small_two_class_kernel, a copy of _align_small_kernel that computes counts as now, then
  small = counts <= 8. Class S (small experts) is padded to a multiple of 8; class L (the rest) is
  padded to a multiple of 16 (SGLang's pick at bs=32). Each class gets its own seg_start/cumsum
  over only its experts, and its own sorted_ids (capacity n + 32*7 for S, n + 32*15 for L,
  sentinel n everywhere unused), block_experts (-1 past used) and n_padded (0 if the class is
  empty). Each local route is stored only in its expert's class buffer; local_ids is written once.
  In kernel_fn at that shape, call moe_wna16_marlin_gemm for w13 twice into the same cache1: first
  moe_block_size=8 with class S's triple, then moe_block_size=16 with class L's. Rows are route
  indices, so the two calls write disjoint rows. Then silu_and_mul once, unchanged. Then w2 twice
  into the same cache3, the same way. Then master's post-norm (fused_post_expert_norm_weighted_sum
  at bs=32). Same workspace, same flags. prepare() compiles the kernel for tokens 17..32 and
  extends _align_self_check: for each class, the per-expert route sets and block counts must equal
  moe_align_block_size(ref_local restricted to that class's experts, bsm, 32). Mechanism:
  per-m-block Marlin cost grows with block_size_m (LESSONS). At bs=32 a uniform 8 lost (16->8
  multiplied the hot experts' blocks: 239 vs ~186 us in next-smaller) and a uniform 32 lost
  (next-larger). Most local experts receive <= 8 routes at bs=32 (~256 routes over 32 heavy-tailed
  experts), so giving only those the cheaper m_block_size_8 tile, while the hot experts keep 16,
  takes the per-expert best of the two measured uniform picks.
expected:
  - shapes: bs=32 *
    direction: faster
    why: "master weakest shape (1.033x, Marlin w13+w2 187.6 us vs a 70.8 us floor): cold experts drop from a 16-row to an 8-row Marlin tile, while hot experts keep 16 and do not multiply blocks"
  - shapes: bs=16 *
    direction: flat
    why: "control: num_tokens <= 16 untouched (SGLang already picks 8 there)"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-two-class-block-m-decode32

## Change

At num_tokens in (16, 32] only (decode_32 bs=32; no other scored or test shape; bs=33 is above 32 and stays on master), give cold experts Marlin's 8-row block and keep SGLang's 16 for the rest, by splitting each GEMM into two Marlin launches over disjoint expert classes. Add _align_small_two_class_kernel, a copy of _align_small_kernel that computes counts as now, then small = counts <= 8. Class S (small experts) is padded to a multiple of 8; class L (the rest) is padded to a multiple of 16 (SGLang's pick at bs=32). Each class gets its own seg_start/cumsum over only its experts, and its own sorted_ids (capacity n + 32*7 for S, n + 32*15 for L, sentinel n everywhere unused), block_experts (-1 past used) and n_padded (0 if the class is empty). Each local route is stored only in its expert's class buffer; local_ids is written once. In kernel_fn at that shape, call moe_wna16_marlin_gemm for w13 twice into the same cache1: first moe_block_size=8 with class S's triple, then moe_block_size=16 with class L's. Rows are route indices, so the two calls write disjoint rows. Then silu_and_mul once, unchanged. Then w2 twice into the same cache3, the same way. Then master's post-norm (fused_post_expert_norm_weighted_sum at bs=32). Same workspace, same flags. prepare() compiles the kernel for tokens 17..32 and extends _align_self_check: for each class, the per-expert route sets and block counts must equal moe_align_block_size(ref_local restricted to that class's experts, bsm, 32). Mechanism: per-m-block Marlin cost grows with block_size_m (LESSONS). At bs=32 a uniform 8 lost (16->8 multiplied the hot experts' blocks: 239 vs ~186 us in next-smaller) and a uniform 32 lost (next-larger). Most local experts receive <= 8 routes at bs=32 (~256 routes over 32 heavy-tailed experts), so giving only those the cheaper m_block_size_8 tile, while the hot experts keep 16, takes the per-expert best of the two measured uniform picks.

## Expected

- `bs=32 *` faster: master weakest shape (1.033x, Marlin w13+w2 187.6 us vs a 70.8 us floor): cold experts drop from a 16-row to an 8-row Marlin tile, while hot experts keep 16 and do not multiply blocks
- `bs=16 *` flat: control: num_tokens <= 16 untouched (SGLang already picks 8 there)
- `bs=128 *` flat: control: untouched

## Risk

Risk: each extra Marlin launch costs a graph node plus a grid that exits fast when its class is empty or small (~2-4 us each, two extra). The saving must exceed ~5-8 us on 187.6 us. The 8-vs-16 per-block cost gap for 1-8 real rows is unmeasured directly. If it wins, extend it to bs=128/256 (S=8, L=master pick 16/32) on top of triton-map-align-verify-chunked, and ab-tune the hot class larger (32 at bs=32). Reviewer: check that Marlin accepts n_padded=0 (an empty class) without writing, and that the shared lock workspace is clean between the back-to-back calls (master already runs w13 then w2 on one workspace).
