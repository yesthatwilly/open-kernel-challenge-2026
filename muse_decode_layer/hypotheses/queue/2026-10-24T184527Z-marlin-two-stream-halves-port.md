---
problem: muse_decode_layer
slug: marlin-two-stream-halves-port
created: 2026-10-24T184527Z
base: master
area: w13 marlin + silu-mul + w2 marlin (8<=T<=32, stream overlap)
change: >
  Port muse_moe_w4a16 marlin-two-stream-expert-halves-small-m (impl commit ea345883, branch
  hyp/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m, moe job 5d630369b1634f77) to decode
  _experts at 8 <= num_tokens <= 32 (verify_1, decode_16, decode_32); T=1 and T > 32 keep master.
  Copy from ea345883:muse_moe_w4a16/submission.py the align kernels it needs (_align_small_kernel
  and _align_small_halves_kernel with their wrappers and the extended _align_self_check, tags as
  there) and the two-stream body of its kernel_fn. Adapt: decode passes ids already shifted by
  (topk_ids - local_start) % 256, so call the halves align with local_start 0 and count 32; half A
  (local experts 0-15) runs on the current stream and half B (16-31) under
  torch.cuda.stream(state["moe_side_stream"]), a NEW stream created in prepare (do not reuse
  state["side_stream"], which carries the router), with its own second Marlin lock workspace
  marlin_make_workspace(device, max_blocks_per_sm=4) from prepare; each half runs w13 Marlin with
  its own (sorted_ids_h, block_experts_h, n_padded_h) into the shared cache1, silu_and_mul with
  expert_ids=local_ids_h, and w2 Marlin into the shared cache3; then current.wait_stream(moe_side)
  and master post-norm on the full local_ids (_post_norm_combine_kernel below 32,
  fused_post_expert_norm_weighted_sum at 32). block_size_m stays master pick (8 at T<=16, 16 at
  T=32); same Marlin flags. When not capturing, record_stream the side-stream outputs on the
  current stream as master does for the router tensors. Gate on the self-check result (route sets
  and block counts per half), never on capacity lines.
expected:
  - shapes: "*load=decode_32 *"
    direction: faster
    why: "Marlin w13+w2 147 us over ~20 local experts: two half-size chains overlap each other tails and the w13 -> silu -> w2 gaps; moe expected 4-8% of the MoE, ~-6..-12 us, +2-4% on the shape"
  - shapes: "*load=decode_16 *"
    direction: faster
    why: "same on 97.5 / 108.9 us of Marlin: +1.5-3%"
  - shapes: "*load=verify_1 *"
    direction: faster
    why: "same on 116 / 96.6 us: +1-2%"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: T=1 outside the gate
  - shapes: "*load=verify_16 *"
    direction: flat
    why: T>32 outside the gate
risk: medium
depends_on: null
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# marlin-two-stream-halves-port

## Change

Port muse_moe_w4a16 marlin-two-stream-expert-halves-small-m (impl commit ea345883, branch hyp/muse_moe_w4a16/marlin-two-stream-expert-halves-small-m, moe job 5d630369b1634f77) to decode _experts at 8 <= num_tokens <= 32 (verify_1, decode_16, decode_32); T=1 and T > 32 keep master. Copy from ea345883:muse_moe_w4a16/submission.py the align kernels it needs (_align_small_kernel and _align_small_halves_kernel with their wrappers and the extended _align_self_check, tags as there) and the two-stream body of its kernel_fn. Adapt: decode passes ids already shifted by (topk_ids - local_start) % 256, so call the halves align with local_start 0 and count 32; half A (local experts 0-15) runs on the current stream and half B (16-31) under torch.cuda.stream(state["moe_side_stream"]), a NEW stream created in prepare (do not reuse state["side_stream"], which carries the router), with its own second Marlin lock workspace marlin_make_workspace(device, max_blocks_per_sm=4) from prepare; each half runs w13 Marlin with its own (sorted_ids_h, block_experts_h, n_padded_h) into the shared cache1, silu_and_mul with expert_ids=local_ids_h, and w2 Marlin into the shared cache3; then current.wait_stream(moe_side) and master post-norm on the full local_ids (_post_norm_combine_kernel below 32, fused_post_expert_norm_weighted_sum at 32). block_size_m stays master pick (8 at T<=16, 16 at T=32); same Marlin flags. When not capturing, record_stream the side-stream outputs on the current stream as master does for the router tensors. Gate on the self-check result (route sets and block counts per half), never on capacity lines.

## Expected

- `*load=decode_32 *` faster: Marlin w13+w2 147 us over ~20 local experts: two half-size chains overlap each other tails and the w13 -> silu -> w2 gaps; moe expected 4-8% of the MoE, ~-6..-12 us, +2-4% on the shape
- `*load=decode_16 *` faster: same on 97.5 / 108.9 us of Marlin: +1.5-3%
- `*load=verify_1 *` faster: same on 116 / 96.6 us: +1-2%
- `*load=decode_1 *` flat: T=1 outside the gate
- `*load=verify_16 *` flat: T>32 outside the gate

## Risk

CONDITIONAL: dispatch only if moe job 5d630369b1634f77 harvests faster than moe master at bs 8/16/32 (its own spec expected flat to +2%). It is the Marlin-only fallback for moe-small-t-cuda-swapab-port: if that CUDA port lands, this is moot (same rows, same shapes); never run both on the same day. Overlap: small-t-launch-minimal-dense-chain-pdl (in judge) replaces the align at T<=32 when ON; this file needs its own halves align, so if PDL lands first the implementer must keep PDL routing outputs and only swap the align for the halves variant.
