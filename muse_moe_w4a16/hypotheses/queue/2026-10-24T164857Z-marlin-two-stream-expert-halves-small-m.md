---
problem: muse_moe_w4a16
slug: marlin-two-stream-expert-halves-small-m
created: 2026-10-24T164857Z
base: master
area: launch overhead
change: >
  At 8 <= num_tokens <= 32 only (verify_1 bs=8, decode_16 bs=16, decode_32 bs=32; bs=1, the bs=4
  test and num_tokens > 32 run master unchanged), split the expert work into two independent
  halves by local expert (A = local experts 0-15, B = 16-31) and run them on two CUDA streams so
  one half's Marlin ramp/tail, the silu_and_mul launch and the inter-kernel graph gaps overlap the
  other half's Marlin. Align: launch _align_small_kernel's logic with grid (2,) in a new
  _align_small_halves_kernel (same code; program h restricts the one-hot/counts/segments/blocks to
  experts [16h, 16h+16), every other route is treated like a remote route): each program writes
  its own sorted_ids_h (capacity n + 16*(bsm-1), sentinel n in every unused slot), block_experts_h
  (-1 past its used blocks; expert ids stay global-local, i.e. 16h + j) and n_padded_h, plus
  local_ids_h = local id if in its half else -1; program 0 also writes the full local_ids. Buffers
  are one stacked allocation per kind. block_size_m stays master's pick per M (8 at bs<=16, 16 at
  bs=32). kernel_fn: allocate cache1/cache2/cache13 and out on the current stream as now, then
  side.wait_stream(current); half A on the current stream and half B under torch.cuda.stream(side)
  each run moe_wna16_marlin_gemm w13 (its own triple, writing only its routes' rows of the shared
  cache1), silu_and_mul(cache1, cache2, expert_ids=local_ids_h, expert_step=1) (rows whose id is
  -1 are skipped, so each stream touches only its own rows), and w2 into the shared cache3; then
  current.wait_stream(side) and master's post-norm on the full local_ids
  (_post_norm_combine_kernel below 32 tokens, fused_post_expert_norm_weighted_sum at 32). Half B
  uses a second Marlin lock workspace (marlin_make_workspace(device, max_blocks_per_sm=4) made in
  prepare) so the two concurrent Marlin calls never share locks; the side stream is created once
  in prepare(). Same Marlin flags. prepare() compiles the halves kernel for tokens 8..32 and
  extends _align_self_check: for each half, per-expert route sets and block counts must equal
  moe_align_block_size(ref_local with the other half set to -1, bsm, 32).
expected:
  - shapes: bs=32 *
    direction: faster
    why: "master weakest shape (1.033x; Marlin w13+w2 187.6 us vs a 70.8 us floor, ~20 distinct local experts): two concurrent half-size Marlin chains fill each other's tails and hide the w13->silu->w2 dependency gaps; expect 4-8%"
  - shapes: bs=16 *
    direction: faster
    why: same mechanism on 116 us of Marlin over ~12 experts
  - shapes: bs=8 *
    direction: faster
    why: same mechanism on 105.8 us of Marlin over ~7 experts; smallest expected effect
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched (about one local expert, nothing to overlap)"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T165207Z
review_companions: null
---

# marlin-two-stream-expert-halves-small-m

## Change

At 8 <= num_tokens <= 32 only (verify_1 bs=8, decode_16 bs=16, decode_32 bs=32; bs=1, the bs=4 test and num_tokens > 32 run master unchanged), split the expert work into two independent halves by local expert (A = local experts 0-15, B = 16-31) and run them on two CUDA streams so one half's Marlin ramp/tail, the silu_and_mul launch and the inter-kernel graph gaps overlap the other half's Marlin. Align: launch _align_small_kernel's logic with grid (2,) in a new _align_small_halves_kernel (same code; program h restricts the one-hot/counts/segments/blocks to experts [16h, 16h+16), every other route is treated like a remote route): each program writes its own sorted_ids_h (capacity n + 16*(bsm-1), sentinel n in every unused slot), block_experts_h (-1 past its used blocks; expert ids stay global-local, i.e. 16h + j) and n_padded_h, plus local_ids_h = local id if in its half else -1; program 0 also writes the full local_ids. Buffers are one stacked allocation per kind. block_size_m stays master's pick per M (8 at bs<=16, 16 at bs=32). kernel_fn: allocate cache1/cache2/cache13 and out on the current stream as now, then side.wait_stream(current); half A on the current stream and half B under torch.cuda.stream(side) each run moe_wna16_marlin_gemm w13 (its own triple, writing only its routes' rows of the shared cache1), silu_and_mul(cache1, cache2, expert_ids=local_ids_h, expert_step=1) (rows whose id is -1 are skipped, so each stream touches only its own rows), and w2 into the shared cache3; then current.wait_stream(side) and master's post-norm on the full local_ids (_post_norm_combine_kernel below 32 tokens, fused_post_expert_norm_weighted_sum at 32). Half B uses a second Marlin lock workspace (marlin_make_workspace(device, max_blocks_per_sm=4) made in prepare) so the two concurrent Marlin calls never share locks; the side stream is created once in prepare(). Same Marlin flags. prepare() compiles the halves kernel for tokens 8..32 and extends _align_self_check: for each half, per-expert route sets and block counts must equal moe_align_block_size(ref_local with the other half set to -1, bsm, 32).

## Expected

- `bs=32 *` faster: master weakest shape (1.033x; Marlin w13+w2 187.6 us vs a 70.8 us floor, ~20 distinct local experts): two concurrent half-size Marlin chains fill each other's tails and hide the w13->silu->w2 dependency gaps; expect 4-8%
- `bs=16 *` faster: same mechanism on 116 us of Marlin over ~12 experts
- `bs=8 *` faster: same mechanism on 105.8 us of Marlin over ~7 experts; smallest expected effect
- `bs=1 *` flat: control: untouched (about one local expert, nothing to overlap)
- `bs=128 *` flat: control: untouched

## Risk

High risk. Costs: two extra Marlin launches, one extra silu_and_mul, a stream fork/join (two event nodes) and a 2-CTA align, ~5-8 us in total; LESSONS records side-stream losses elsewhere (FA3 schedule on a side stream, moe-align-side-stream on prefill). It pays only if Marlin at 8-32 tokens leaves SMs idle in ramp/tail long enough for the other stream's CTAs to use them: Marlin's grid is persistent (sms x blocks_per_sm) and smem-heavy, so the second kernel's CTAs land only as the first one's retire. Distinct from the held marlin-two-class-block-m-decode32, which splits by block size on one stream (serial launches, no overlap). Judge on the bs=8/16/32 paired ratios; the per-kernel profile rows will sum higher (four Marlin rows), so read shape time, not the kernel sum. Correctness hazards for the reviewer: silu_and_mul must skip rows whose expert id is -1 (it is SGLang's EP path, which relies on that); each stream writes disjoint rows of cache1/cache2/cache3; cache buffers allocated before the fork. If it loses at bs=8 but wins at 32, an ab-tune gating to bs >= 16 follows.

## Adversarial review

Reviewed 2026-10-24T165207Z by an adversarial subagent (model opus).

**Verdict: proceed.** Speculative but cleanly attributable: the change is gated to bs=8/16/32, and bs=1/128/256 act as untouched in-run controls, so one run reads either way. Fix one build bug first. master's cache1 and cache3 are both views at offset 0 of cache13, so half A's w2 writes would race half B's w13/silu rows. Expect flat to +2%, not 4-8%.

### Justification
Profile: Marlin is 92.8-94.6% of bs=8/16/32 (105.8/116.0/187.6 us vs floors 37.3/52.5/70.8 us), so it is the only lever there. The per-expert cost is far above the weight-streaming floor: about 20 us/expert at bs=1 and about 15 us/expert at bs=8, against about 5 us/expert of NVFP4 bytes. Marlin is latency- or bytes-in-flight-bound at small M, so more concurrent memory traffic could in principle help. The stated mechanism (ramp/tail plus inter-kernel gaps) is mostly the wrong one. Shape time minus kernel sum at bs=32 is only about 3 us (200.7 vs about 197.5), so hiding the silu launch and graph gaps is worth at most about 1.5%. Marlin's stripe partitioning also spreads every m-block across all SMs, so there is little per-expert tail to fill. The gain can only come from the two Marlin kernels' CTAs being co-resident on the same SMs. Weakest link: Marlin launches one CTA per SM and, as far as I recall, requests the device's max dynamic smem (in vLLM's marlin, launched with max_shared_mem). If so, the side stream's CTAs land only as the first kernel's retire, and the overlap collapses to the two kernel boundaries. The cost is certain: two extra Marlin launches (about 2-4 us each per the marlin-two-class review), an extra silu, a fork/join and a 2-CTA align, about 5-8 us. LESSONS also lists several flat or losing side-stream attempts (moe-align-side-stream, fa3-scheduler-metadata-side-stream). Two more weak points. (1) A fixed split by local id 0-15/16-31 under skewed popularity can leave one half with most of the routes, which shrinks any concurrency gain at bs=8 (about 7 experts). (2) Each half now spreads half the work over the same persistent grid, so fixed per-kernel cost (prologue, k-stripe lock reduction) doubles. The expected directions are consistent with the gating: bs=1/128/256 flat (code paths byte-identical), bs=8 the smallest effect. The 4-8% at bs=32 is optimistic; I expect flat to +2% at bs=32 and flat-to-slower at bs=8.

### Confounders on the current kernel
- sglang-marlin-path-parity: supplies the cache13 buffer with cache1 = cache13[:n*8*2I] and cache3 = cache13[:n*8*H], both at offset 0. This is safe serially but a cross-stream data race here: cache3 row r overlaps cache1 row r//2, which may belong to the other half and be unread by its silu. Allocate cache3 (or the y buffer) separately. Otherwise a correctness failure, or a silently different result, comes from the interaction, not the idea.
- triton-map-align-on-block-m-restack: the halves kernel is a re-parameterisation of its _align_small_kernel and keeps SGLang's block_size_m pick at bs<=32 (8 / 16). Any change in align cost (2-CTA grid, two stacked outputs) lands in the same 2.7-4.4 us row and is part of this hypothesis's measured cost, not a separate effect. Read the align row separately.
- post-expert-norm-triton-small-m: runs after the join on the full local_ids, which program 0 must write in full (remote routes as -1, both halves' routes as local ids). If program 0 writes only its half, the post-norm silently drops half B. Not a performance confounder.
- marlin-block-m-two-smaller-on-postnorm: touches only num_tokens >= 128, which this change leaves untouched. No confounding; its bs=128/256 rows are the in-run flat control.

### Interference and A/B needs
No companion warranted. The change is single-area and shape-gated, and bs=1/128/256 run master's code byte-for-byte in the same run, so a per-shape move at bs=8/16/32 beyond the paired-ratio noise is attributable to the two-stream split. A gain is unambiguous: splitting the halves serially on one stream can only add launches, so any net gain must come from overlap. A loss is mildly ambiguous between 'no overlap' and 'overlap smaller than the 5-8 us overhead'. A serialized-halves control (the same split on one stream) would separate the two, but it cannot change the decision: either way the follow-up is to drop the idea or to gate it to bs=32. With 2 moe checks left it is not worth a check. Noise: DECISIONS records unchanged bs=32 rows spanning 147-265 us across runs, so read only the paired ratio against SGLang within the run and the bs=8/16/32 rows, never the absolute Marlin row sum. The four Marlin rows will sum higher by construction, as the hypothesis notes.

### Open questions
- Does SGLang's moe_wna16_marlin_gemm launch with the device's max dynamic smem (one CTA per SM)? If so, two concurrent Marlin kernels cannot co-reside and the expected gain is about the kernel boundaries only. The profile's per-kernel durations will show it: if each half's Marlin row is about half of master's and they do not overlap in time, there was no concurrency.
- Split balance: under skewed routing a fixed 0-15/16-31 split can be lopsided. A follow-up could have the align kernel pick the halves on device: a greedy by-count assignment computed from the one-hot counts, which needs no host branch and so is legal. This is only worth trying if this run shows any overlap gain.
- If bs=32 gains and bs=8 loses, the hypothesis already plans an ab-tune gating to bs >= 16. Gating to bs=32 alone may be the right cut, since bs=16 has only about 12 experts.
- risk: high means this cannot be a scored run, and moe has 2 checks left today. Spend one check on it only if nothing better is queued (swapab-w13-silu-epilogue-verify is queued and unheld; compare priorities).

