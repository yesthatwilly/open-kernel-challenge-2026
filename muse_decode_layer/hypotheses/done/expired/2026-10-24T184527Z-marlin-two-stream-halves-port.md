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
review: hold
status: done
review_at: 2026-10-24T184953Z
review_companions: null
hold_reason: "adversarial review: Hold: the moe parent it ports (marlin-two-stream-expert-halves-small-m, job 5d630369b1634f77) came back neutral, with every scored shape flat and verify_1 1.5% "
outcome: expired
verdict_detail: "hold expired after 26 h: adversarial review: Hold: the moe parent it ports (marlin-two-stream-expert-halves-small-m, job 5d630369b1634f77) came b"
expired_at: 2026-10-25T211328Z
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

## Adversarial review

Reviewed 2026-10-24T184953Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold: the moe parent it ports (marlin-two-stream-expert-halves-small-m, job 5d630369b1634f77) came back neutral, with every scored shape flat and verify_1 1.5% slower. The hypothesis's own Risk section makes dispatch conditional on that parent winning, and nothing in decode_layer gives it a separate case.

### Justification
A. The mechanism is plausible on paper: Marlin is 96-146 us (49-54% of the shape) at verify_1/decode_16/decode_32 sliding, against floors of 62-85 us for the whole layer, and two half-bank chains could fill each other's tails and the w13->silu->w2 gaps. But the isolated MoE test already measured exactly this on H200 at the same bs 8/16/32 with the same block_size_m and Marlin flags. Results: decode_16 148.18 vs 147.87 us (flat), decode_32 188.4 vs 200.7 us (about 1.009 per the attribution table, within noise), verify_1 114.84 vs 107.77 us (worse). Summed Marlin kernel time also grew (verify_1 158.9 us of Marlin inside a 114.8 us median), so the two halves did overlap, but each half-kernel got slower by about as much as the overlap saved. That fits the moe review's open question: Marlin launches with large dynamic shared memory and fills the SMs, so two concurrent instances cannot co-reside. The expected +1-4% per shape here rests entirely on the moe expectation of 4-8% of the MoE, which the harvest refuted. Weakest link: the hypothesis assumes the overlap gain the parent failed to show. In decode_layer, the MoE also competes with the router side stream and the PDL-chained dense kernels, so there is less idle SM capacity to absorb a second Marlin stream, not more. The expected directions are consistent with the TASK.md gating (decode_1 and verify_16 flat, being outside the gate). The magnitudes are not supported.

### Confounders on the current kernel
- small-t-launch-minimal-dense-chain-pdl (now master): at T<=32 it replaces the align with its fused dl_route_norm_align and runs the pooler/router via PDL (cudaLaunchKernelEx with programmatic serialization). Two things follow. Swapping in the halves align breaks that chain. Adding a third stream inside the captured graph can turn PDL edges into full graph dependencies. Either could move the shape by a few us independent of the halves overlap, so a measured result would be ambiguous.
- marlin-block-m-cap-16: sets block_size_m (8 at T<=16, 16 at T=32), which fixes how many CTAs each half launches. With smaller halves, a different bsm would change the wave fill, so a null result does not rule out the idea under another bsm. That alone is not a reason to run a companion.
- post-expert-norm-triton-small-t: the post-norm reads cache3. Master aliases cache3 onto cache1 through cache13. Under two streams, half A's w2 writing cache3 can overwrite rows that half B's w13/silu is still using in cache1. The moe parent had to give each half its own cache1/cache2 and un-alias cache3 to pass review.

### Interference and A/B needs
No companion is proposed, because the primary should not run. The parent's own clean A/B in the isolated MoE problem already answers the question the companion would ask: a two-stream halves split of the Marlin chain at bs 8-32 on H200 does not beat one stream. A decode_layer run could only add noise from the PDL and router-stream interactions. If someone still wants to revive the idea, the right next step is a moe-side stacked hypothesis (depends_on the neutral parent) that first shows real co-residency. Examples: cap Marlin's smem/blocks per SM so both halves fit, or split by device-side balanced counts. That should land in moe before any decode port.

### Open questions
- Does the moe parent's profile show the two Marlin halves actually overlapping in time, or just slowing each other down? Check the nsys timeline for 5d630369b1634f77 before reviving the idea anywhere.
- Can Marlin be launched with fewer blocks per SM or less smem so two instances co-reside? Without that, the overlap mechanism cannot hold at these M.
- Is moe-small-t-cuda-swapab-port, the CUDA alternative over the same rows, still viable now that its moe parent cuda-nvfp4-swapab-mma-persistent-small-m failed tests? If neither is viable, the T<=32 Marlin budget needs a new idea rather than this fallback.

