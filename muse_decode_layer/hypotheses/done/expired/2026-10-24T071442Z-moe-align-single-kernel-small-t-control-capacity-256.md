---
problem: muse_decode_layer
slug: moe-align-single-kernel-small-t-control-capacity-256
created: 2026-10-24T071442Z
base: master
area: MoE align/dispatch
change: >
  Identical to moe-align-single-kernel-small-t (built on the expert-id-map-single-kernel branch:
  the same single-program Triton kernel at T <= 32 writing local_ids, sorted_ids, block_experts
  and num_tokens_post_pad, and the same T=128/256 fallback to SGLang's
  moe_align_block_size(local_ids, block_size_m, N_EXPERTS)) with one difference: at T <= 32
  allocate sorted_ids with SGLang's 256-expert capacity, T*8 + N_EXPERTS*(block_size_m - 1)
  entries (1800/1856/1920 for T=1/8/16 at bsm 8, 4096 for T=32 at bsm 16). Allocate block_experts
  with cdiv(that capacity, block_size_m) entries (225/232/240/256), exactly as
  moe_align_block_size(..., 256) sizes them. The kernel still fills every unused sorted_ids slot
  with the numel sentinel (T*8) and every unused block_experts slot with -1. Only the extent of
  the two buffers grows. The device-side contents of the used prefix and num_tokens_post_pad are
  bit-identical to the primary's. Marlin, silu_and_mul, the post-expert sum and block_size_m
  selection are unchanged.
expected:
  - shapes: "*load=decode_1*"
    direction: faster
    why: "Pure launch removal at SGLang-identical Marlin input shapes: ~3-6 us on ~170-180 us"
  - shapes: "*load=verify_1*"
    direction: faster
    why: Same launches on ~230-250 us, ~1.5-2.5%
  - shapes: "*load=decode_16*"
    direction: flat
    why: ~1% on 260-430 us, inside the noise
  - shapes: "*load=decode_32*"
    direction: flat
    why: Saving is partly offset by the single-program sort at 256 routes; inside the noise
  - shapes: "*load=verify_16*"
    direction: flat
    why: "T=128 keeps SGLang's op; drift control"
  - shapes: "*load=verify_32*"
    direction: flat
    why: "T=256 keeps SGLang's op; drift control"
risk: low
depends_on: moe-align-single-kernel-small-t
priority: 3
kind: control
controls_for: moe-align-single-kernel-small-t
review: required
status: done
hold_reason: control for moe-align-single-kernel-small-t, which was neutral; a capacity control of a non-gain cannot change a decision, and decode has 2 checks left for the attention rewrite and pairing follow-ups
outcome: expired
verdict_detail: "hold expired after 33 h: control for moe-align-single-kernel-small-t, which was neutral; a capacity control of a non-gain cannot change a decisio"
expired_at: 2026-10-25T211328Z
---

# moe-align-single-kernel-small-t-control-capacity-256

## Change

Identical to moe-align-single-kernel-small-t (built on the expert-id-map-single-kernel branch: the same single-program Triton kernel at T <= 32 writing local_ids, sorted_ids, block_experts and num_tokens_post_pad, and the same T=128/256 fallback to SGLang's moe_align_block_size(local_ids, block_size_m, N_EXPERTS)) with one difference: at T <= 32 allocate sorted_ids with SGLang's 256-expert capacity, T*8 + N_EXPERTS*(block_size_m - 1) entries (1800/1856/1920 for T=1/8/16 at bsm 8, 4096 for T=32 at bsm 16). Allocate block_experts with cdiv(that capacity, block_size_m) entries (225/232/240/256), exactly as moe_align_block_size(..., 256) sizes them. The kernel still fills every unused sorted_ids slot with the numel sentinel (T*8) and every unused block_experts slot with -1. Only the extent of the two buffers grows. The device-side contents of the used prefix and num_tokens_post_pad are bit-identical to the primary's. Marlin, silu_and_mul, the post-expert sum and block_size_m selection are unchanged.

## Expected

- `*load=decode_1*` faster: Pure launch removal at SGLang-identical Marlin input shapes: ~3-6 us on ~170-180 us
- `*load=verify_1*` faster: Same launches on ~230-250 us, ~1.5-2.5%
- `*load=decode_16*` flat: ~1% on 260-430 us, inside the noise
- `*load=decode_32*` flat: Saving is partly offset by the single-program sort at 256 routes; inside the noise
- `*load=verify_16*` flat: T=128 keeps SGLang's op; drift control
- `*load=verify_32*` flat: T=256 keeps SGLang's op; drift control

## Risk

Separates the hypothesis's two sub-changes. Control versus the parent (expert-id-map-single-kernel) gives the launch-fusion effect with Marlin seeing exactly SGLang's tensor extents. Primary versus control, in comparable runs, gives the 256->32 capacity effect on Marlin's m-block partition at small T, which the queued moe-align-local-expert-count can only probe at large T on a different base. Without it, any primary gain or loss at T<=32 could be either mechanism.
