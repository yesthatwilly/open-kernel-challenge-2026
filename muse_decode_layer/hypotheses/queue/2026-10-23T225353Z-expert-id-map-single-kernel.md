---
problem: muse_decode_layer
slug: expert-id-map-single-kernel
created: 2026-10-23T225353Z
base: master
area: MoE align/dispatch
change: >
  Replace the three small kernels that turn the routing output into Marlin's local expert ids with
  one Triton kernel. Today kernel_fn computes local_ids = (topk_ids - data['local_start']) %
  N_EXPERTS (a sub and a remainder launch) and _experts then gathers bank['expert_map'][topk_ids]
  (an index launch) to get int32 ids in 0..31 or -1. The new kernel (one program over the T*8
  int32 ids, no host-side values) reads topk_ids and the local_start device scalar and writes a
  single int32 [T, 8] tensor with ((id - local_start) mod 256) where that is < 32 and -1
  otherwise, exactly what expert_map produced; that tensor is passed unchanged to
  moe_align_block_size, silu_and_mul (expert_ids) and fused_post_expert_norm_weighted_sum
  (slot_expert_ids). topk_weights handling, moe_align_block_size, the Marlin calls and the
  returned topk_ids are untouched; the expert_map table is no longer needed in the bank.
  Attribution: 3 launches become 1 on every shape; only measurable on the launch-bound T<=8
  shapes.
expected:
  - shapes: "*load=decode_1*"
    direction: faster
    why: "two ~2-3 us launches removed from a ~172 us launch-bound shape: ~2-3%"
  - shapes: "*load=verify_1*"
    direction: faster
    why: "same two launches on a ~268 us shape: ~1.5-2%, near the CI edge"
  - shapes: "*load=decode_32*"
    direction: flat
    why: the saved launches are under 1% of ~260-620 us; control
  - shapes: "*load=verify_16*"
    direction: flat
    why: same; control
risk: low
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# expert-id-map-single-kernel

## Change

Replace the three small kernels that turn the routing output into Marlin's local expert ids with one Triton kernel. Today kernel_fn computes local_ids = (topk_ids - data['local_start']) % N_EXPERTS (a sub and a remainder launch) and _experts then gathers bank['expert_map'][topk_ids] (an index launch) to get int32 ids in 0..31 or -1. The new kernel (one program over the T*8 int32 ids, no host-side values) reads topk_ids and the local_start device scalar and writes a single int32 [T, 8] tensor with ((id - local_start) mod 256) where that is < 32 and -1 otherwise, exactly what expert_map produced; that tensor is passed unchanged to moe_align_block_size, silu_and_mul (expert_ids) and fused_post_expert_norm_weighted_sum (slot_expert_ids). topk_weights handling, moe_align_block_size, the Marlin calls and the returned topk_ids are untouched; the expert_map table is no longer needed in the bank. Attribution: 3 launches become 1 on every shape; only measurable on the launch-bound T<=8 shapes.

## Expected

- `*load=decode_1*` faster: two ~2-3 us launches removed from a ~172 us launch-bound shape: ~2-3%
- `*load=verify_1*` faster: same two launches on a ~268 us shape: ~1.5-2%, near the CI edge
- `*load=decode_32*` flat: the saved launches are under 1% of ~260-620 us; control
- `*load=verify_16*` flat: same; control

## Risk

Risk: correctness is exact integer arithmetic, so the only failure modes are dtype (moe_align_block_size and silu_and_mul expect int32 ids; FusedMuseSparkRouting already returns int32 since kernel_fn returns topk_ids as the int32 output) and the modulo of a negative difference (use a Python-style non-negative mod, as torch's % does). The effect size is small: if decode_1 moves less than the +-1.5% CI, record it as flat rather than a gain and keep the kernel only if the stacked moe-align-single-kernel-small-t (which subsumes it) lands. Confounder: if the adversarial review finds that torch already fuses sub+remainder into one kernel on this image, the saving is 1 launch, not 2.
