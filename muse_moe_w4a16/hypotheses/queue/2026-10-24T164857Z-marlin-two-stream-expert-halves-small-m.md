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
review: required
status: queued
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
