---
problem: muse_moe_w4a16
slug: align-remote-sentinel-ids
created: 2026-10-24T075257Z
base: master
area: MoE align/dispatch
change: >
  Give moe_align_block_size ids in which remote routes are the out-of-range value num_experts (32)
  instead of -1, so its bounds check drops them from sorted_ids entirely instead of whatever it
  does with -1. prepare(): next to expert_map (remote -> -1), build align_map = torch.full((256,),
  32, int32) with align_map[start:start+count] = arange(count), stored in the bank. kernel_fn,
  every shape: align_ids = align_map[topk_ids].contiguous() (one extra small gather launch, ~1-2
  us) is passed ONLY to moe_align_block_size(align_ids, block_size_m, bank['count']); local_ids
  (with -1) still feed silu_and_mul(expert_ids=...) and
  fused_post_expert_norm_weighted_sum(slot_expert_ids=...) and the Marlin topk_weights path
  exactly as on master. Marlin keeps is_ep=True. Purpose: the mechanism probe for why master's
  32-expert align is 3.2 us slower than SGLang's 256-expert align at bs=1 (80.2 vs 77.0 us). If
  the align treats -1 as a valid signed index at 32 experts (e.g. counted into a local or spare
  bucket), remote routes form m-blocks that both Marlin GEMMs either compute against a local
  expert or skip; sentinel ids remove them, shrinking n_padded and Marlin's m-block loop at every
  shape.
expected:
  - shapes: bs=1 *
    direction: faster
    why: if -1 routes form a phantom block at 32 experts, dropping them saves one expert weight pass (~24 MB, ~4-5 us) net of the extra ~1-2 us gather
  - shapes: bs=8 *
    direction: faster
    why: same, ~56 remote routes per call
  - shapes: bs=16 *
    direction: faster
    why: same
  - shapes: bs=32 *
    direction: faster
    why: same
  - shapes: bs=128 *
    direction: flat
    why: remote blocks would be skipped by Marlin (is_ep) at small per-block cost; any gain is a smaller n_padded, under the CI at 462 us
  - shapes: bs=256 *
    direction: flat
    why: as bs=128
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# align-remote-sentinel-ids

## Change

Give moe_align_block_size ids in which remote routes are the out-of-range value num_experts (32) instead of -1, so its bounds check drops them from sorted_ids entirely instead of whatever it does with -1. prepare(): next to expert_map (remote -> -1), build align_map = torch.full((256,), 32, int32) with align_map[start:start+count] = arange(count), stored in the bank. kernel_fn, every shape: align_ids = align_map[topk_ids].contiguous() (one extra small gather launch, ~1-2 us) is passed ONLY to moe_align_block_size(align_ids, block_size_m, bank['count']); local_ids (with -1) still feed silu_and_mul(expert_ids=...) and fused_post_expert_norm_weighted_sum(slot_expert_ids=...) and the Marlin topk_weights path exactly as on master. Marlin keeps is_ep=True. Purpose: the mechanism probe for why master's 32-expert align is 3.2 us slower than SGLang's 256-expert align at bs=1 (80.2 vs 77.0 us). If the align treats -1 as a valid signed index at 32 experts (e.g. counted into a local or spare bucket), remote routes form m-blocks that both Marlin GEMMs either compute against a local expert or skip; sentinel ids remove them, shrinking n_padded and Marlin's m-block loop at every shape.

## Expected

- `bs=1 *` faster: if -1 routes form a phantom block at 32 experts, dropping them saves one expert weight pass (~24 MB, ~4-5 us) net of the extra ~1-2 us gather
- `bs=8 *` faster: same, ~56 remote routes per call
- `bs=16 *` faster: same
- `bs=32 *` faster: same
- `bs=128 *` flat: remote blocks would be skipped by Marlin (is_ep) at small per-block cost; any gain is a smaller n_padded, under the CI at 462 us
- `bs=256 *` flat: as bs=128

## Risk

Risk: if this sgl_kernel align does not bounds-check ids >= num_experts it writes or counts out of range (a spare (num_experts+1)-th bucket would make block_experts = 32 and Marlin would read past the 32-expert bank): expect a test failure or a fault, which reads as 'the align has no >= num_experts guard', not as a Marlin problem. Output correctness otherwise is guarded by the -1 masks in silu_and_mul and the post-norm, which are unchanged. The extra gather is a known ~1-2 us cost on every shape, so a flat bs=1 means the -1 handling is not the 3.2 us. Read beside align-256-experts-below-128: if that one restores decode_1 parity and this one also gains at bs=1, the -1 bucket is the mechanism and this variant (fewer blocks than SGLang) is the one to keep; if only the 256 revert helps, the cost is in the 32-expert kernel itself.
