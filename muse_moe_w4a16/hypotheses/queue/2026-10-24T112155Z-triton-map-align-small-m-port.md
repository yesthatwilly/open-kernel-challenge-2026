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
review: proceed
status: queued
review_at: 2026-10-24T112524Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T112524Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound, well-attributed launch fusion on shapes where this problem is launch-heavy (bs=1 is 55 us with 7.8 us in the three replaced kernels). Run it alone with sharper self-check logging. No companion is needed because the in-run bs=128/256 controls, the per-kernel profile rows and the unheld align-remote-sentinel-ids probe between them separate the possible mechanisms.

### Justification
The mechanism is plausible and measured, not assumed. In bacd1f502f4e46fa the kernels being replaced cost: bs=1 index 1.8 + unrolled 1.6 + small_batch_align 4.4 us (7.8 of 52.0 us summed kernel time, 55.3 us median); bs=8 1.7+1.6+4.9; bs=16 2.2+1.4+5.6; bs=32 2.4+1.9+6.5. The same _align_small_kernel measured 4.3 us at T=32 on decode_layer (0a0af8669ffe4cb0) and passed its 60-case self-check. Net savings are therefore about 6.5 us at bs=32 (~3%, above that shape's ~1.4% err) and probably 5-6 us at bs=1 (~9-10%). Shapes and directions agree with TASK.md: six equal-weight (60) loads, bs<=32 changed and bs=128/256 untouched as controls. A ~10% gain on decode_1 plus 3-4% on verify_1, decode_16 and decode_32 is roughly +2.5-3% on the geomean, about the size of the CI, so judge it per shape. The unrolled_elementwise row is very probably not from the moe_align_block_size wrapper, contrary to the hypothesis. In align-single-token-ep (a7bb0cecaf8d4de1) the align was replaced but a 1.2 us unrolled_elementwise call remained. It most likely comes from PyTorch advanced indexing casting the int32 index to int64 for expert_map[local_ids], and it scales with M (3.2 us at bs=128) like the gather does. Either way it is removed, because the gather goes away. Weakest link: the new kernel's own latency at small T. It is a single program doing a [BLOCK_N,32] one-hot cumsum plus a [MB_BLOCK,32] segment test, and only its T=32 time (4.3 us) is known. If it runs ~4 us even at T=1-8 (fixed cost of the cumsum and reductions on one SM), bs=1 gains ~3.5 us (6%), not 5-6. LESSONS also records that launch fusion was flat on larger layer shapes. Here, though, the shapes are 55-207 us and the replaced kernels are 4-14% of them, so this is the case LESSONS says pays.

### Confounders on the current kernel
- sglang-marlin-path-parity: it sets the 32-expert moe_align_block_size this replaces, and with it how remote (-1) routes are handled. If SGLang's small-batch kernel at num_experts=32 buckets -1 routes into phantom m-blocks (the open question align-remote-sentinel-ids exists to probe), the Triton kernel drops them. Marlin then sees a smaller n_padded and fewer m-blocks, and part of any gain is Marlin work removed, not launches removed. The decode self-check would not catch this: it compares n_pad only against SGLang's local-expert blocks and never against SGLang's own r_pad. Remedy without a companion: in the port's self-check, print per case whether n_pad != r_pad (as info), and read the Marlin row against the sentinel probe.
- post-expert-norm-triton-small-m: no timing confound. It consumes local_ids (skipping -1) at M<32, and SGLang's fused sum consumes slot_expert_ids at bs=32. The new kernel must reproduce local_ids bit-exactly in dtype int32, shape [M,8], contiguous. If it does not, the failure is a correctness failure on the bs=1/4/33 tests, not a silent speed artifact. Its gain does not change what this hypothesis removes.

### Interference and A/B needs
No companion is warranted. A gain or loss is attributable from one run. (1) bs=128/256 keep master's exact path, so they measure drift. (2) The profile shows the replaced rows (index_elementwise, unrolled_elementwise, small_batch_expert_kernel) disappear and _align_small_kernel appears with its own time, and Triton times reproduce within ~1% (LESSONS). (3) The only plausible second mechanism is a change in Marlin's m-block count if master's align buckets -1 routes. The self-check can settle that deterministically and for free in prepare() by logging n_pad vs SGLang's r_pad, and the now-unheld align-remote-sentinel-ids measures its timing effect on master. An ab-revert of either landed hypothesis would not make the result clearer: parity defines the path being replaced, and the post-norm kernel is downstream and indifferent. A capacity control like the one on decode_layer is not needed, because master already allocates the 32-expert capacity M*8+32*(bsm-1), which the port keeps.

### Open questions
- Does SGLang's moe_align_block_size(local_ids, bsm, 32) put the -1 routes into any m-block? Log r_pad vs n_pad per self-check case. If they differ, part of the gain is Marlin work, and the align-remote-sentinel-ids result should be read alongside this one.
- What is _align_small_kernel's own time at T=1, 8 and 16? It is known only at T=32 (4.3 us). Report it from the profile together with the absence of the three replaced rows.
- If unrolled_elementwise survives at bs<=32, record where it comes from. The evidence (it survived align-single-token-ep) suggests the int64 index cast in expert_map[...], which this change removes. A survivor would point to silu_and_mul or the Marlin wrapper.
- Within-expert route order differed from SGLang's at T=16 and T=32 on decode (info only). Confirm the Marlin outputs are row-independent so per-row results are bit-identical; correctness should be unaffected.

