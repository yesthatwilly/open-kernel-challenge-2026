---
problem: muse_decode_layer
slug: moe-align-local-expert-count
created: 2026-10-23T225545Z
base: master
area: MoE align/dispatch
change: >
  One-argument change in _experts: call moe_align_block_size(local_ids, block_size_m,
  LOCAL_EXPERTS) with 32 instead of N_EXPERTS (256). The ids it receives are already mapped to
  this GPU's 0..31 or -1 (expert_map), so 32 is the true expert count. The padded-capacity buffers
  SGLang sizes from the expert count shrink from T*8 + 256*(block_size_m - 1) to T*8 +
  32*(block_size_m - 1) entries (at T=256, block 64: 18176 -> 4064 sorted-id slots, 284 -> 64
  expert_ids slots), its per-expert cumsum runs over 32 instead of 256 counters, and the fill
  kernels that preset those buffers touch 4-7x fewer bytes. Both moe_wna16_marlin_gemm calls,
  silu_and_mul and the post-expert sum receive the smaller sorted_ids / block_experts tensors
  unchanged in meaning (num_tokens_post_pad is a device scalar either way). Nothing else changes.
  Attribution: if Marlin's m-block loop walks the host-sized capacity (expert_ids.numel()) rather
  than the device count, the large-T shapes move; if it walks the device count, everything is flat
  and only the fill kernels shrink.
expected:
  - shapes: "*load=verify_32*"
    direction: faster
    why: "T=256, block 64: the largest capacity shrink (284 -> 64 m-block slots); visible only if Marlin iterates the padded capacity"
  - shapes: "*load=verify_16*"
    direction: faster
    why: "T=128, block 48: same mechanism, smaller"
  - shapes: "*load=decode_1*"
    direction: flat
    why: "T=1, block 8: capacity 1800 -> 232 ints, the fills are already sub-microsecond; control"
  - shapes: "*load=decode_32*"
    direction: flat
    why: "T=32, block 16: capacity 4096 -> 736 ints; expected inside the CI"
risk: low
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# moe-align-local-expert-count

## Change

One-argument change in _experts: call moe_align_block_size(local_ids, block_size_m, LOCAL_EXPERTS) with 32 instead of N_EXPERTS (256). The ids it receives are already mapped to this GPU's 0..31 or -1 (expert_map), so 32 is the true expert count. The padded-capacity buffers SGLang sizes from the expert count shrink from T*8 + 256*(block_size_m - 1) to T*8 + 32*(block_size_m - 1) entries (at T=256, block 64: 18176 -> 4064 sorted-id slots, 284 -> 64 expert_ids slots), its per-expert cumsum runs over 32 instead of 256 counters, and the fill kernels that preset those buffers touch 4-7x fewer bytes. Both moe_wna16_marlin_gemm calls, silu_and_mul and the post-expert sum receive the smaller sorted_ids / block_experts tensors unchanged in meaning (num_tokens_post_pad is a device scalar either way). Nothing else changes. Attribution: if Marlin's m-block loop walks the host-sized capacity (expert_ids.numel()) rather than the device count, the large-T shapes move; if it walks the device count, everything is flat and only the fill kernels shrink.

## Expected

- `*load=verify_32*` faster: T=256, block 64: the largest capacity shrink (284 -> 64 m-block slots); visible only if Marlin iterates the padded capacity
- `*load=verify_16*` faster: T=128, block 48: same mechanism, smaller
- `*load=decode_1*` flat: T=1, block 8: capacity 1800 -> 232 ints, the fills are already sub-microsecond; control
- `*load=decode_32*` flat: T=32, block 16: capacity 4096 -> 736 ints; expected inside the CI

## Risk

Risk: the sgl moe_align_block_size op may branch on num_experts (a small-expert-count fast path versus the multi-kernel path above some threshold); a different kernel is still the same contract (sorted ids padded per expert to block_size, -1 for empty blocks), but verify on the 13 tests that the sorted_ids sentinel value (numel) and expert_ids layout are what Marlin expects at 32 experts. The expected gain is honestly uncertain and may be nil; the run is cheap and the absence is informative for the stacked moe-align-single-kernel-small-t. Do not combine with that hypothesis in one run.
