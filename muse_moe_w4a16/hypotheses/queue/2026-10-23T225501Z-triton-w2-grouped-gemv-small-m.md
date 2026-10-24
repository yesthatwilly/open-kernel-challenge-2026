---
problem: muse_moe_w4a16
slug: triton-w2-grouped-gemv-small-m
created: 2026-10-23T225501Z
base: master
area: w2 marlin
change: >
  On top of triton-w13-grouped-gemv-small-m, for num_tokens <= 8 replace the second
  moe_wna16_marlin_gemm (w2, top_k=1, size_m=M*8) with the same Triton grouped-GEMV template: grid
  (32, H/BLOCK_N), program (e, n) reads cnt[e] and slot_list[e] from the compact_routes prologue
  already run for w13, early-exits when cnt==0, gathers cache2[slot] rows (the silu_and_mul
  output, [M*8, I] bf16, flat slot order) masked by arange < cnt into [16, BLOCK_K] tiles,
  dequantizes the expert's w2 NVFP4 tile exactly as the w13 kernel does (raw w2/w2_sf/w2_gs copies
  kept in prepare()), tl.dot in bf16 with fp32 accumulation, multiplies by w2_gs[e] after the K
  loop and stores y[slot, n-tile] into the [M, 8, H] buffer the post-norm reads. With both GEMMs
  off Marlin at small M, drop the now-unused moe_align_block_size call and the expert_map gather
  from the small-M branch (the prologue consumes the mapped ids, so keep expert_map[topk_ids];
  drop only the align), leaving 5 launches: gather, compact_routes, w13_gemv, silu_and_mul,
  w2_gemv, post-norm. Shapes with num_tokens > 8 are untouched.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "w2 at bs=1 is the other half of the weight traffic (9.4 MB per touched expert) and Marlin's M=8-row block path is below bandwidth; plus the align kernels (2-3 launches) leave the small-M path"
  - shapes: bs=8 *
    direction: faster
    why: "same at bs=8, 64 slots, cnt[e] <= 8"
  - shapes: bs=32 *
    direction: flat
    why: "control: untouched Marlin path"
  - shapes: bs=256 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: triton-w13-grouped-gemv-small-m
priority: 3
kind: primary
controls_for: null
review: required
status: hold
hold_reason: stacks on triton-w13-grouped-gemv-small-m, which regressed (2-2.6x slower than Marlin at small M)
---

# triton-w2-grouped-gemv-small-m

## Change

On top of triton-w13-grouped-gemv-small-m, for num_tokens <= 8 replace the second moe_wna16_marlin_gemm (w2, top_k=1, size_m=M*8) with the same Triton grouped-GEMV template: grid (32, H/BLOCK_N), program (e, n) reads cnt[e] and slot_list[e] from the compact_routes prologue already run for w13, early-exits when cnt==0, gathers cache2[slot] rows (the silu_and_mul output, [M*8, I] bf16, flat slot order) masked by arange < cnt into [16, BLOCK_K] tiles, dequantizes the expert's w2 NVFP4 tile exactly as the w13 kernel does (raw w2/w2_sf/w2_gs copies kept in prepare()), tl.dot in bf16 with fp32 accumulation, multiplies by w2_gs[e] after the K loop and stores y[slot, n-tile] into the [M, 8, H] buffer the post-norm reads. With both GEMMs off Marlin at small M, drop the now-unused moe_align_block_size call and the expert_map gather from the small-M branch (the prologue consumes the mapped ids, so keep expert_map[topk_ids]; drop only the align), leaving 5 launches: gather, compact_routes, w13_gemv, silu_and_mul, w2_gemv, post-norm. Shapes with num_tokens > 8 are untouched.

## Expected

- `bs=1 *` faster: w2 at bs=1 is the other half of the weight traffic (9.4 MB per touched expert) and Marlin's M=8-row block path is below bandwidth; plus the align kernels (2-3 launches) leave the small-M path
- `bs=8 *` faster: same at bs=8, 64 slots, cnt[e] <= 8
- `bs=32 *` flat: control: untouched Marlin path
- `bs=256 *` flat: control: untouched

## Risk

Risks: y must land in the exact [M, 8, H] slot layout fused_post_expert_norm_weighted_sum expects (rows for remote slots are never written; the post-norm skips them via slot_expert_ids=-1, so leave them uninitialized as Marlin does, or zero them if the judge's NaN check reads them: verify with the bs=1 test). Removing moe_align also removes its sorted_ids fill: confirm no other consumer. Same rule posture as the w13 hypothesis: lossless NVFP4 copy, bf16 activations, fp32 accumulate, nothing cached from data.
