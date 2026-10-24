---
problem: muse_moe_w4a16
slug: w2-l2-prefetch-decode1
created: 2026-10-24T164836Z
base: master
area: w2 marlin
change: >
  At num_tokens == 1 only (decode_1; every other M runs master unchanged), warm L2 with the w2
  weights of this token's local experts before the w13 Marlin call, so the w2
  moe_wna16_marlin_gemm reads them from L2 instead of DRAM. TASK.md: every timed replay starts
  with a cold L2 (the judge streams a buffer larger than L2 first), and at bs=1 the two Marlin
  calls take 44.6 us for roughly one local expert (~28 MB of w13+w2 codes and scales), i.e. ~0.6
  TB/s: latency-bound, not bandwidth-bound. Add a Triton kernel
  _l2_prefetch_w2_kernel(topk_ids_ptr, w2_ptr, w2s_ptr, sink_ptr, local_start, local_count,
  w2_expert_words, w2s_expert_words, TOP_K_C: tl.constexpr, BLOCK: tl.constexpr), grid (P,) with P
  = 2 * SM count (264), num_warps 4, BLOCK = 2048 int32 words. Each program loads the 8 topk_ids,
  maps them to local ids (id - local_start in [0, local_count), else skip), skips a slot whose
  local id equals an earlier slot's (dedupe, static_range over the 8 slots), and for each
  remaining local expert e loops over its own strided share of that expert's w2 slab (int32 view
  of bank["w2"][e], w2_expert_words = bank["w2"][0].numel() * element_size / 4) and of its scale
  slab (int32 view of bank["w2_s"][e]) with tl.load(..., eviction_policy="evict_last"),
  XOR-accumulating the words into one int32, and finally stores that int32 to sink_ptr[program_id]
  (sink = a torch.empty(P, int32) made once in prepare and never read; it only stops the loads
  being eliminated). The int32 views and per-expert word counts are built once in prepare() from
  the bank tensors (both are stacked [32, ...] contiguous banks; assert it). kernel_fn at
  num_tokens == 1 launches it on the current stream right after _align_small and before the w13
  Marlin call; nothing else changes (same block_size_m, align, Marlin flags, silu_and_mul,
  _post_norm_combine_kernel). Only w2 is prefetched (~9.4 MB per local expert, at most ~19 MB for
  the common 1-2 local experts, well under H200's 50 MB L2, and it must survive w13's ~19 MB
  stream through L2, which evict_last favours). prepare() compiles the kernel once on dummy ids. A
  token with no local expert prefetches nothing.
expected:
  - shapes: bs=1 *
    direction: faster
    why: w2 Marlin (roughly a third of the 44.6 us Marlin pair at decode_1) reads its expert from L2 at lower latency; the prefetch costs ~2.5-3 us at ~3.5 TB/s with 264 CTAs plus one graph node, so expect net -3 to -8 us of 48.4 us (6-15% on decode_1, ~1-2% geomean)
  - shapes: bs=16 *
    direction: flat
    why: "control: num_tokens > 1 untouched (w2 bytes for ~12 experts exceed L2)"
  - shapes: bs=8 *
    direction: flat
    why: "control: untouched"
  - shapes: bs=128 *
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

# w2-l2-prefetch-decode1

## Change

At num_tokens == 1 only (decode_1; every other M runs master unchanged), warm L2 with the w2 weights of this token's local experts before the w13 Marlin call, so the w2 moe_wna16_marlin_gemm reads them from L2 instead of DRAM. TASK.md: every timed replay starts with a cold L2 (the judge streams a buffer larger than L2 first), and at bs=1 the two Marlin calls take 44.6 us for roughly one local expert (~28 MB of w13+w2 codes and scales), i.e. ~0.6 TB/s: latency-bound, not bandwidth-bound. Add a Triton kernel _l2_prefetch_w2_kernel(topk_ids_ptr, w2_ptr, w2s_ptr, sink_ptr, local_start, local_count, w2_expert_words, w2s_expert_words, TOP_K_C: tl.constexpr, BLOCK: tl.constexpr), grid (P,) with P = 2 * SM count (264), num_warps 4, BLOCK = 2048 int32 words. Each program loads the 8 topk_ids, maps them to local ids (id - local_start in [0, local_count), else skip), skips a slot whose local id equals an earlier slot's (dedupe, static_range over the 8 slots), and for each remaining local expert e loops over its own strided share of that expert's w2 slab (int32 view of bank["w2"][e], w2_expert_words = bank["w2"][0].numel() * element_size / 4) and of its scale slab (int32 view of bank["w2_s"][e]) with tl.load(..., eviction_policy="evict_last"), XOR-accumulating the words into one int32, and finally stores that int32 to sink_ptr[program_id] (sink = a torch.empty(P, int32) made once in prepare and never read; it only stops the loads being eliminated). The int32 views and per-expert word counts are built once in prepare() from the bank tensors (both are stacked [32, ...] contiguous banks; assert it). kernel_fn at num_tokens == 1 launches it on the current stream right after _align_small and before the w13 Marlin call; nothing else changes (same block_size_m, align, Marlin flags, silu_and_mul, _post_norm_combine_kernel). Only w2 is prefetched (~9.4 MB per local expert, at most ~19 MB for the common 1-2 local experts, well under H200's 50 MB L2, and it must survive w13's ~19 MB stream through L2, which evict_last favours). prepare() compiles the kernel once on dummy ids. A token with no local expert prefetches nothing.

## Expected

- `bs=1 *` faster: w2 Marlin (roughly a third of the 44.6 us Marlin pair at decode_1) reads its expert from L2 at lower latency; the prefetch costs ~2.5-3 us at ~3.5 TB/s with 264 CTAs plus one graph node, so expect net -3 to -8 us of 48.4 us (6-15% on decode_1, ~1-2% geomean)
- `bs=16 *` flat: control: num_tokens > 1 untouched (w2 bytes for ~12 experts exceed L2)
- `bs=8 *` flat: control: untouched
- `bs=128 *` flat: control: untouched

## Risk

Attribution: judge the bs=1 paired ratio and the Marlin row at decode_1 (should fall by more than the new _l2_prefetch_w2_kernel row adds). Not caching across calls: the prefetch runs inside the timed call, reads only prepare()-time weights at expert ids taken from this call's topk_ids, and its sink is never read. Risks: (1) Marlin at bs=1 may be bound by its fp32 stripe reduce/lock handshakes rather than B-load latency, in which case L2 hits buy little and the ~3 us prefetch is a pure loss (decode_1 -5%); (2) w13's stream may evict the prefetched lines despite evict_last (H200 L2 is two ~25 MB partitions). If it wins, follow-ups: also prefetch w13 when exactly one local expert is present, and fold the prefetch programs into the _align_small launch to drop the extra graph node. If flat, an ab-tune with the prefetch placed after w13 (immediately before w2) separates eviction from latency.
