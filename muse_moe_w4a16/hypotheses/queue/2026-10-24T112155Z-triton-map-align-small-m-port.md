---
problem: muse_moe_w4a16
slug: triton-map-align-small-m-port
created: 2026-10-24T112155Z
base: master
area: MoE align/dispatch
change: >
  Port muse_decode_layer moe-align-single-kernel-small-t (patch:
  muse_decode_layer/hypotheses/done/neutral/moe-align-single-kernel-small-t.patch; its
  _align_small_kernel passed its prepare() self-check, 60 cases, 0 mismatches) to
  muse_moe_w4a16/submission.py. For num_tokens <= 32 only (bs=1, 8, 16, 32 and the bs=4 test;
  bs=33/128/256/257 keep master's path), replace the three launches master issues before w13 - the
  expert_map[topk_ids] gather (at::native index_elementwise_kernel, 1.7-2.4 us), the
  unrolled_elementwise_kernel that follows it (1.4-1.9 us, present at every shape, most likely
  inside the moe_align_block_size wrapper) and moe_align_block_size_small_batch_expert_kernel
  (4.4-6.5 us) - with one single-program Triton kernel (grid (1,), num_warps=4) that reads the M*8
  int32 topk_ids and writes everything downstream consumes: local_ids [M, 8] int32 (id -
  local_start when 0 <= id - local_start < local_count, else -1; local_start/local_count are host
  ints from config, no mod-256 wrap since the bank is the contiguous slice [local_start,
  local_start+32)), sorted_ids (route indices grouped by ascending local expert, each expert
  padded to a multiple of block_size_m with the sentinel M*8, every slot past the used length also
  M*8, capacity M*8 + 32*(block_size_m-1) as master's 32-expert align allocates), block_experts
  (each m-block's expert, -1 past the used blocks) and n_padded (int32 [1]). Use the decode
  kernel's counting sort as is: one-hot [BLOCK_N, 32], per-expert counts, padded cumsum for
  segment starts, rank via cumsum of the one-hot, three disjoint store sets into sorted_ids.
  Remote routes get no bucket (as in decode, where the self-check matched moe_align_block_size's
  per-expert blocks and n_padded exactly). block_size_m pick unchanged (master's rule; the
  moe_align_single_token branch is unreachable with an expert_map and stays). local_ids still feed
  silu_and_mul(expert_ids=...), _post_norm_combine_kernel and, at bs=32,
  fused_post_expert_norm_weighted_sum(slot_expert_ids=...). Both Marlin calls unchanged.
  prepare(): warm-compile the kernel for every num_tokens 1..32 on dummy ids and run the decode
  self-check against moe_align_block_size(ref_local, bsm, 32) (32, matching master's call),
  untimed, printing the one-line summary.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "gather 1.8 + unrolled 1.6 + small-batch align 4.4 = 7.8 us of a 55.3 us shape become one Triton launch (~2-3 us; align-single-token-ep showed a 1.2 us align kernel at bs=1) plus two fewer graph nodes: expect ~4-6 us, 7-10% on decode_1"
  - shapes: bs=8 *
    direction: faster
    why: "1.7 + 1.6 + 4.9 = 8.2 us of 121 us -> ~2.5-3 us: ~4%"
  - shapes: bs=16 *
    direction: faster
    why: "2.2 + 1.4 + 5.6 = 9.2 us of 156 us -> ~3-4 us: ~3-4%"
  - shapes: bs=32 *
    direction: faster
    why: "2.4 + 1.9 + 6.5 = 10.8 us of 207 us -> 4.3 us (the decode kernel measured 4.3 us at T=32): ~3%, and bs=32 is master's weakest shape (0.987x)"
  - shapes: bs=128 *
    direction: flat
    why: "control: SGLang gather + align path untouched"
  - shapes: bs=256 *
    direction: flat
    why: control
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# triton-map-align-small-m-port

## Change

Port muse_decode_layer moe-align-single-kernel-small-t (patch: muse_decode_layer/hypotheses/done/neutral/moe-align-single-kernel-small-t.patch; its _align_small_kernel passed its prepare() self-check, 60 cases, 0 mismatches) to muse_moe_w4a16/submission.py. For num_tokens <= 32 only (bs=1, 8, 16, 32 and the bs=4 test; bs=33/128/256/257 keep master's path), replace the three launches master issues before w13 - the expert_map[topk_ids] gather (at::native index_elementwise_kernel, 1.7-2.4 us), the unrolled_elementwise_kernel that follows it (1.4-1.9 us, present at every shape, most likely inside the moe_align_block_size wrapper) and moe_align_block_size_small_batch_expert_kernel (4.4-6.5 us) - with one single-program Triton kernel (grid (1,), num_warps=4) that reads the M*8 int32 topk_ids and writes everything downstream consumes: local_ids [M, 8] int32 (id - local_start when 0 <= id - local_start < local_count, else -1; local_start/local_count are host ints from config, no mod-256 wrap since the bank is the contiguous slice [local_start, local_start+32)), sorted_ids (route indices grouped by ascending local expert, each expert padded to a multiple of block_size_m with the sentinel M*8, every slot past the used length also M*8, capacity M*8 + 32*(block_size_m-1) as master's 32-expert align allocates), block_experts (each m-block's expert, -1 past the used blocks) and n_padded (int32 [1]). Use the decode kernel's counting sort as is: one-hot [BLOCK_N, 32], per-expert counts, padded cumsum for segment starts, rank via cumsum of the one-hot, three disjoint store sets into sorted_ids. Remote routes get no bucket (as in decode, where the self-check matched moe_align_block_size's per-expert blocks and n_padded exactly). block_size_m pick unchanged (master's rule; the moe_align_single_token branch is unreachable with an expert_map and stays). local_ids still feed silu_and_mul(expert_ids=...), _post_norm_combine_kernel and, at bs=32, fused_post_expert_norm_weighted_sum(slot_expert_ids=...). Both Marlin calls unchanged. prepare(): warm-compile the kernel for every num_tokens 1..32 on dummy ids and run the decode self-check against moe_align_block_size(ref_local, bsm, 32) (32, matching master's call), untimed, printing the one-line summary.

## Expected

- `bs=1 *` faster: gather 1.8 + unrolled 1.6 + small-batch align 4.4 = 7.8 us of a 55.3 us shape become one Triton launch (~2-3 us; align-single-token-ep showed a 1.2 us align kernel at bs=1) plus two fewer graph nodes: expect ~4-6 us, 7-10% on decode_1
- `bs=8 *` faster: 1.7 + 1.6 + 4.9 = 8.2 us of 121 us -> ~2.5-3 us: ~4%
- `bs=16 *` faster: 2.2 + 1.4 + 5.6 = 9.2 us of 156 us -> ~3-4 us: ~3-4%
- `bs=32 *` faster: 2.4 + 1.9 + 6.5 = 10.8 us of 207 us -> 4.3 us (the decode kernel measured 4.3 us at T=32): ~3%, and bs=32 is master's weakest shape (0.987x)
- `bs=128 *` flat: control: SGLang gather + align path untouched
- `bs=256 *` flat: control

## Risk

Why this pays here when it was neutral on decode_layer: the saving is the same ~5 us in absolute terms, but decode_layer shapes are 170-620 us (decode_1 still moved 171 -> 166 us, 1.03x, there), while this problem's decode_1 is 55 us, so the same microseconds are 7-10% of a load. Judge it per shape and by the replaced kernels' own times in top_kernels (align/index/unrolled rows gone, _align_small_kernel added), not by Marlin time, which drifts 20-30% on identical code (align-256-experts-below-128 review). If the unrolled_elementwise row stays, it is not from the align wrapper; record which. Independent of marlin-block-m-two-smaller-on-postnorm (disjoint shapes: <=32 here, >=128 there) and can run from master in parallel. Overlap with the held align-remote-sentinel-ids: both drop remote routes from sorted_ids; this one also removes launches, so if both run, read the sentinel probe first for whether -1 handling cost Marlin anything. Risk: Marlin reads sorted_ids/block_experts layout exactly; the self-check guards it, and a wrong layout fails the 5 tests rather than slipping through.
