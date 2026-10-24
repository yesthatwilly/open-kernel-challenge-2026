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
review: proceed
status: queued
review_at: 2026-10-24T140447Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T140447Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed. This is a sound, rules-clean extension of the landed single-program align to bs>32. One run can be read from the per-shape top_kernels rows: 4 rows removed, 1 new row added, and the code at bs<=32 is byte-identical, so those shapes act as a drift control. The real risk is that a serial two-pass loop in one CTA at n=2048 ends up as slow as the ~11.5 us it replaces.

### Justification
The profile of the current best (bc56182a80444d16) confirms the target rows exactly. At bs=128: index_elementwise 3.2, unrolled_elementwise (sorted_ids fill) 2.8, moe_align_block_size_kernel 2.7 and count_and_sort 1.8 us, 10.5 us in all. At bs=256: 3.3 + 2.8 + 2.8 + 2.6 = 11.5 us. Against 333/470 us, a full win is ~2-2.5% per shape and a realistic win is 1.5-2%, ~0.5% on the headline: inside one run's CI, as the spec admits. TASK.md gives every load equal weight, so 2 of the 6 loads moving ~1.5-2% is consistent with the expected directions. bs=1/8/16/32 are correctly predicted flat, because they keep _align_small. Weakest link: the new kernel's own time. The landed _align_small scales with n: 1.6 us (n=8), 2.7 (n=64), 2.9 (n=128), 4.4 (n=256, 4 warps, one [256,32] tile with cumsum). Extrapolating, the per-route cost is ~1-1.5 us per 128 routes, which mostly comes from the onehot cumsum/rank in pass 2. The chunked kernel at n=1024 (bs=128) and n=2048 (bs=256, 4 chunks of 512) runs pass 1 plus pass 2 serially in one CTA, with a loop-carried base[] dependency. 8 warps roughly halves the per-element cost, but a plausible range is ~5-8 us at bs=128 and ~8-14 us at bs=256. bs=128 should gain. bs=256 could come out flat or even slightly slower. The speed-of-light framing does not apply: this is launch and latency work, far from any bandwidth floor. Minor: the graph-node gap saved by removing 3 launches is only ~0.3-1 us in total (LESSONS).

### Confounders on the current kernel
- marlin-block-m-two-smaller-on-postnorm: sets BSM=16 at bs=128 and BSM=32 at bs=256/257, which fixes cap (1504 / 3040), the padding-slot count and n_blocks (94 / 95). The chunked kernel's padding, cap-fill and block loops therefore cost what they cost at those BSMs. At SGLang's 48/64 the padding loops would be ~3x larger. This shifts the new kernel's own time somewhat, but not what Marlin receives, so it does not confound attribution of the align rows.
- triton-map-align-on-block-m-restack / triton-map-align-small-m-port: same area and same kernel design. They gate on ALIGN_FUSED_MAX_TOKENS=32, so the bs>32 path this hypothesis replaces was never touched by them. They are the in-run control (bs<=32 code unchanged), not a confounder.
- post-expert-norm-triton-small-m: no confound at the target shapes. bs>=32 still uses SGLang's fused_post_expert_norm_weighted_sum with slot_expert_ids=local_ids, which needs remote slots to be exactly -1. The new kernel must reproduce that, as _align_small does.
- sglang-marlin-path-parity: the base pipeline. The 32-expert align capacity (M*8 + 32*(bsm-1)) that the chunked kernel must match comes from it. No interaction beyond that.

### Interference and A/B needs
No companion is warranted. Attribution needs no A/B: the change is visible directly in top_kernels at bs=128/256 (the four at::native/align rows disappear and _align_chunked_kernel appears with its own time). The four bs<=32 shapes are byte-identical code and serve as the drift reference. Reverting block-m-two-smaller (ab-revert) would only change BSM, which changes the kernel's padding-loop cost by about a microsecond and none of Marlin's work. It cannot flip the interpretation, and spending a check on it would be wasted with moe at 4 checks left. A CHUNK/num_warps ab-tune is a follow-up, not a companion: if the kernel row shows bs=128 winning and bs=256 losing, the next step is a gate at n<=1024 or a CHUNK=256 / 2-CTA variant, chosen from this run's rows. Read the result per shape: success means kernel row < ~(sum of removed rows - 1 us) at each verify shape. The headline is expected to stay inside the CI either way.

### Open questions
- What is the new kernel's own time at bs=256 (n=2048, 4 serial chunks x 2 passes)? If it exceeds ~10 us, the bs=256 gain disappears. Would CHUNK=256 with num_warps=8 (lower register pressure for the [CHUNK,32] int32 onehot and cumsum: 512x32 = 16K elements, 64 per thread at 8 warps), or recomputing the onehot only in pass 2, be faster?
- Is unrolled_elementwise (2.8 us) really the wrapper's sorted_ids fill and not some other cast? topk_ids is already int32, so .to(int32) is a no-op. If it were something else that stays, the saving is smaller.
- If the swap-AB rewrite later needs BSM=64/128 from this kernel, the [NUM_E, BSM] padding tile grows to 4K int32. That is fine for one CTA, but add those BSMs to the prepare compile list and the self-check, as the spec notes.

