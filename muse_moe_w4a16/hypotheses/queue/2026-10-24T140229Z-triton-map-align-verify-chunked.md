---
problem: muse_moe_w4a16
slug: triton-map-align-verify-chunked
created: 2026-10-24T140229Z
base: master
area: MoE align/dispatch
change: >
  Extend master's single-program map+align (_align_small_kernel, triton-map-align-small-m-port) to
  num_tokens > 32 (bs=128/256, and the bs=33/257/skewed-128 tests) with a chunked variant
  _align_chunked_kernel, so the expert_map gather (at::native index_elementwise ~3.2 us), the
  align wrapper's fill (unrolled_elementwise ~2.8 us), moe_align_block_size_kernel (~2.7 us) and
  count_and_sort_expert_tokens_kernel (1.8-2.6 us) become one launch. One program, num_warps=8,
  CHUNK=512 routes (constexpr), NUM_E=32. Pass 1 loops over chunks of the n=M*8 flat ids: it
  computes local = id - local_start (masked to [0, count), else -1), stores local_ids, and
  accumulates counts[NUM_E] += sum(onehot [CHUNK, NUM_E]). Then compute padded / seg_start / total
  exactly as the small kernel does. Pass 2 loops over the chunks again, keeping a running
  base[NUM_E] (init 0): pos = seg_start[e] + base[e] + (within-chunk cumsum rank), it stores
  sorted_ids[pos] = flat index for local routes, and does base += chunk counts. Then a loop writes
  each expert's padding slots with the sentinel n, a CHUNK-strided loop fills slots [total, cap),
  and a loop over m-blocks in CHUNK-sized strides writes block_experts (-1 past the used blocks)
  and n_padded. n, cap and n_blocks are runtime (do_not_specialize), so the only constexpr per
  call is BSM. prepare() compiles it once per block_size_m candidate (8..64) on dummy ids and
  extends _align_self_check to tokens (33, 128, 256, 257) with the same per-expert comparison
  against moe_align_block_size(ref_local, bsm, 32). kernel_fn: replace num_tokens <=
  ALIGN_FUSED_MAX_TOKENS with expert_map is not None, using _align_small up to 32 tokens
  (unchanged) and _align_chunked above. block_size_m picks, Marlin calls, silu and post-norm are
  unchanged.
expected:
  - shapes: bs=128 *
    direction: faster
    why: 4 align/gather launches totalling ~10.5 us replaced by one ~3-5 us launch, out of 333 us
  - shapes: bs=256 *
    direction: faster
    why: 4 launches totalling ~11.5 us replaced by one, out of 470 us
  - shapes: bs=32 *
    direction: flat
    why: "control: _align_small unchanged"
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched"
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# triton-map-align-verify-chunked

## Change

Extend master's single-program map+align (_align_small_kernel, triton-map-align-small-m-port) to num_tokens > 32 (bs=128/256, and the bs=33/257/skewed-128 tests) with a chunked variant _align_chunked_kernel, so the expert_map gather (at::native index_elementwise ~3.2 us), the align wrapper's fill (unrolled_elementwise ~2.8 us), moe_align_block_size_kernel (~2.7 us) and count_and_sort_expert_tokens_kernel (1.8-2.6 us) become one launch. One program, num_warps=8, CHUNK=512 routes (constexpr), NUM_E=32. Pass 1 loops over chunks of the n=M*8 flat ids: it computes local = id - local_start (masked to [0, count), else -1), stores local_ids, and accumulates counts[NUM_E] += sum(onehot [CHUNK, NUM_E]). Then compute padded / seg_start / total exactly as the small kernel does. Pass 2 loops over the chunks again, keeping a running base[NUM_E] (init 0): pos = seg_start[e] + base[e] + (within-chunk cumsum rank), it stores sorted_ids[pos] = flat index for local routes, and does base += chunk counts. Then a loop writes each expert's padding slots with the sentinel n, a CHUNK-strided loop fills slots [total, cap), and a loop over m-blocks in CHUNK-sized strides writes block_experts (-1 past the used blocks) and n_padded. n, cap and n_blocks are runtime (do_not_specialize), so the only constexpr per call is BSM. prepare() compiles it once per block_size_m candidate (8..64) on dummy ids and extends _align_self_check to tokens (33, 128, 256, 257) with the same per-expert comparison against moe_align_block_size(ref_local, bsm, 32). kernel_fn: replace num_tokens <= ALIGN_FUSED_MAX_TOKENS with expert_map is not None, using _align_small up to 32 tokens (unchanged) and _align_chunked above. block_size_m picks, Marlin calls, silu and post-norm are unchanged.

## Expected

- `bs=128 *` faster: 4 align/gather launches totalling ~10.5 us replaced by one ~3-5 us launch, out of 333 us
- `bs=256 *` faster: 4 launches totalling ~11.5 us replaced by one, out of 470 us
- `bs=32 *` flat: control: _align_small unchanged
- `bs=1 *` flat: control: untouched

## Risk

Expected ~1.5-2% on each verify shape, ~0.5-0.7% headline, inside one run's CI. Judge it by the removed rows in top_kernels and the new kernel's own time. It is a launch-count change on 330-470 us shapes, which LESSONS says rarely moves the headline. It is still worth a run because it also gives the bs>=128 path a Triton align that a two-class block-m (see marlin-two-class-block-m-decode32) or the swap-AB rewrite can reuse. Risk: the serial chunk loops in one CTA (5 chunks at bs=257) could take 5+ us and leave little gain. If the kernel row exceeds ~6 us, the result is flat. Confounding: swapab-nvfp4-gemm-verify-rewrite replaces Marlin at the same shapes and keeps SGLang's align, so if both land the chunked align must be called with BSM=BLOCK_T (64/128): add those to the prepare compile and self-check.
