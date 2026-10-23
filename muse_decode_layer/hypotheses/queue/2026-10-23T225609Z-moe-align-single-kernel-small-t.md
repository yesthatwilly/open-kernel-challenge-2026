---
problem: muse_decode_layer
slug: moe-align-single-kernel-small-t
created: 2026-10-23T225609Z
base: master
area: MoE align/dispatch
change: >
  At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call AND the
  expert-id map kernel from expert-id-map-single-kernel with one Triton kernel, a single program,
  that reads the T*8 int32 topk_ids and the local_start device scalar and writes everything Marlin
  consumes: the int32 local_ids [T, 8] (0..31 or -1, for silu_and_mul and the post-expert sum),
  sorted_ids (flat token*8+k indices grouped by local expert, each expert padded up to a multiple
  of block_size_m with the numel sentinel, in the exact layout moe_align_block_size produces),
  block_experts (the expert id of each m-block, -1 for none) and num_tokens_post_pad (an int32 [1]
  device scalar). With at most 256 routes and 32 experts this is a counting sort in
  registers/shared memory, one launch replacing the 3-4 launches moe_align_block_size issues
  (buffer fills plus its align kernel(s)) and the 1 from the id map. The buffers keep the
  host-sized capacity T*8 + 32*(block_size_m - 1) that SGLang would allocate for 32 experts.
  T=128/256 keep SGLang's moe_align_block_size as is. block_size_m selection, both Marlin GEMMs,
  silu_and_mul and the post-expert sum are unchanged. Attribution: the T<=32 shapes only;
  verify_16/verify_32 are the control.
expected:
  - shapes: "*load=decode_1*"
    direction: faster
    why: "3-4 launches (~8-10 us) removed from a ~172 us launch-bound shape: ~4-6%"
  - shapes: "*load=verify_1*"
    direction: faster
    why: "same launches on a ~268 us shape: ~3%"
  - shapes: "*load=decode_16*"
    direction: faster
    why: same launches, ~2-3% of 266/431 us
  - shapes: "*load=decode_32*"
    direction: faster
    why: same launches, ~1.5-2%, near the CI
  - shapes: "*load=verify_16*"
    direction: flat
    why: T=128 keeps the SGLang op; control
  - shapes: "*load=verify_32*"
    direction: flat
    why: T=256 keeps the SGLang op; control
risk: medium
depends_on: expert-id-map-single-kernel
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# moe-align-single-kernel-small-t

## Change

At T <= 32 (every decode shape and verify_1), replace SGLang's moe_align_block_size call AND the expert-id map kernel from expert-id-map-single-kernel with one Triton kernel, a single program, that reads the T*8 int32 topk_ids and the local_start device scalar and writes everything Marlin consumes: the int32 local_ids [T, 8] (0..31 or -1, for silu_and_mul and the post-expert sum), sorted_ids (flat token*8+k indices grouped by local expert, each expert padded up to a multiple of block_size_m with the numel sentinel, in the exact layout moe_align_block_size produces), block_experts (the expert id of each m-block, -1 for none) and num_tokens_post_pad (an int32 [1] device scalar). With at most 256 routes and 32 experts this is a counting sort in registers/shared memory, one launch replacing the 3-4 launches moe_align_block_size issues (buffer fills plus its align kernel(s)) and the 1 from the id map. The buffers keep the host-sized capacity T*8 + 32*(block_size_m - 1) that SGLang would allocate for 32 experts. T=128/256 keep SGLang's moe_align_block_size as is. block_size_m selection, both Marlin GEMMs, silu_and_mul and the post-expert sum are unchanged. Attribution: the T<=32 shapes only; verify_16/verify_32 are the control.

## Expected

- `*load=decode_1*` faster: 3-4 launches (~8-10 us) removed from a ~172 us launch-bound shape: ~4-6%
- `*load=verify_1*` faster: same launches on a ~268 us shape: ~3%
- `*load=decode_16*` faster: same launches, ~2-3% of 266/431 us
- `*load=decode_32*` faster: same launches, ~1.5-2%, near the CI
- `*load=verify_16*` flat: T=128 keeps the SGLang op; control
- `*load=verify_32*` flat: T=256 keeps the SGLang op; control

## Risk

Risk: the output contract of moe_align_block_size (sorted_ids padding value, the order of tokens within an expert, whether block_experts covers the padded capacity or only the used blocks, and whether the two Marlin calls read num_tokens_post_pad from the device) must be reproduced exactly; read sglang's moe_align_block_size.py and the Marlin kernel's expectations in the image before writing it and compare the three outputs tensor-for-tensor against SGLang's op on the 13 tests' routings in prepare() (untimed) as a self-check, printing any mismatch. Token order within an expert does not affect correctness (fp32 reduce per row) but should match anyway for a clean A/B. Depends on expert-id-map-single-kernel having landed so the id-map launch is not double counted; if that one is flat but not harmful, this still stacks on it. Do not run together with moe-align-local-expert-count.
