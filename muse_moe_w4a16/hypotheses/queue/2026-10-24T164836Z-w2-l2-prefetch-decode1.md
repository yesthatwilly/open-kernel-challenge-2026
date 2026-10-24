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
review: proceed
status: queued
review_at: 2026-10-24T165124Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T165124Z by an adversarial subagent (model opus).

**Verdict: proceed.** The rules allow it: the prefetch runs inside the timed call, pays the full DRAM read after the judge's L2 flush, and reads only prepare()-time weights at this call's expert ids, so it doesn't get around the cold-L2 method and is a technique that would carry over to production. Proceed as one check, but build it with the prefetch launched right before the w2 Marlin call (after silu_and_mul), not before w13. The two launches run one after the other on the stream, so the early placement gains no overlap and only adds the risk that w13's stream evicts the prefetched lines.

### Justification
The mechanism is plausible. At decode_1, Marlin is 44.6 of 48.4 us (87%). The SoL floor is 11.4 us, which at 4.8 TB/s is about 55 MB, roughly two local experts at about 28 MB each (w13 about 19 MB, w2 about 9.4 MB). So the two Marlin calls average about 1.2 TB/s. They are latency- or parallelism-bound at M=1, and a 264-CTA streaming load at about 3.5 TB/s followed by L2-hit B loads can beat them. Weakest link 1: the cost estimate assumes one local expert. The SoL says about two on the busiest-GPU routing, so the prefetch is about 19 MB and takes about 5-6 us, not 2.5-3 us. That leaves a net gain only if w2 Marlin drops by more than about 6 us, out of a w2 share that the profile cannot isolate (both Marlin calls share one row). Expect net 0 to -5 us (0-10% on decode_1, at most about 1.5% geomean), which may land within CI at the headline. Weakest link 2: Marlin at M=1 may be bound by its fp32 stripe reduce and lock handshakes rather than by B-load latency (the hypothesis's own risk 1). Moving the w2 call's B loads to L2 then buys little. The expected shapes and directions are consistent: only bs=1 changes, on a branch by shape, and bs=8/16/32/128/256 run master byte for byte, so they act as in-run controls and their movement measures drift.

### Confounders on the current kernel
- None of the landed hypotheses confound it. sglang-marlin-path-parity defines the Marlin path the prefetch targets, but the change only adds a launch before an unchanged call. triton-map-align-on-block-m-restack (_align_small) runs before the prefetch and only reads topk_ids, which the prefetch also reads independently. The bs=1 block_size_m stays SGLang's pick, untouched by marlin-block-m-two-smaller-on-postnorm, which applies only at >=128 tokens. post-expert-norm-triton-small-m runs after w2 and touches no weights.

### Interference and A/B needs
No companion is warranted if the prefetch sits right before the w2 Marlin call. Then a gain can only come from w2 B loads hitting L2: the bs=1 shape time falls and the Marlin row falls by more than the new _l2_prefetch_w2_kernel row adds. A loss or flat result means w2 Marlin at M=1 is not bound by B-load latency, because eviction is ruled out (only silu_and_mul, about 1.6 us, sits between the two). With the prefetch before w13 as written, a flat result would be ambiguous between w13 evicting the lines and Marlin not being latency-bound. That would need the before-w2 control the hypothesis itself names, and it would spend moe's second and last check today. Moving the launch removes that ambiguity at no cost, since the stream runs it serially either way. If the builder keeps the before-w13 placement, add a control companion that is identical except that the launch sits immediately after silu_and_mul.

### Open questions
- How many local experts does the judge's bs=1 busiest-GPU routing actually have? The SoL of 11.4 us implies about two, which halves the hypothesis's net-gain estimate. The log could print the local count from prepare()'s dummy run, but not from data (that would be a host sync).
- Follow-up if it wins: fork the prefetch onto a side stream inside the graph so it overlaps the latency-bound w13 Marlin, which uses only about 1.2 TB/s of DRAM. That hides the prefetch's cost instead of paying it serially, and it is the larger lever.
- Is the w2 share of the 44.6 us Marlin row known from any earlier per-launch profile? Without it, the attribution relies on the shape time minus the new kernel's row.

