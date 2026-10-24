---
problem: muse_moe_w4a16
slug: small-m-grouped-gemv-splitk-rewrite
created: 2026-10-24T081339Z
base: master
area: w13 marlin
change: >
  Stage 1 of a small-batch grouped W4A16 GEMV that replaces both moe_wna16_marlin_gemm calls when
  num_tokens <= 16 (decode_1 bs=1, verify_1 bs=8, decode_16 bs=16); num_tokens > 16 keeps the
  parity Marlin path byte for byte. Unchanged in stage 1: the expert_map gather, silu_and_mul
  (cache1 -> cache2, expert_ids=local_ids), fused_post_expert_norm_weighted_sum and the
  cache1/cache2/cache3 row layout (row = flat slot token*8+k, remote slots left unwritten, as
  Marlin leaves them). (1) Routing: on the small path call moe_align_block_size(local_ids, 16, 32)
  (block 16, not Marlin's 8) so every touched local expert is exactly one block of <= 16 slots
  (cnt[e] <= M <= 16); padded entries carry the sentinel M*8 and are masked. The number of live
  blocks is read on device from n_padded; nothing branches on it on the host. (2) Weights:
  prepare() keeps, beside the Marlin bank, a byte-identical contiguous copy of the 32 local
  experts' w13/w13_sf/w13_gs and w2/w2_sf/w2_gs (NVFP4 bytes as stored, optionally a lossless
  permutation of byte order within a row; never a BF16 copy). (3) Parallelism, the first failure
  of triton-w13-grouped-gemv-small-m: that kernel launched a static (32 experts x 256 n-tiles)
  grid in which ~30/32 rows exited, leaving ~512 CTAs each streaming a whole 4096-long K row with
  3 stages. Instead each GEMM is ONE persistent kernel with a fixed grid (e.g. 132*4 CTAs) looping
  over a device-side work list of items = live_block x n_tile x k_split: w13 (N=8192, K=4096)
  BLOCK_N=64 -> 128 n-tiles x SPLIT_K=4 (1024 each) = 512 items per touched expert; w2 (N=4096,
  K=I=4096) BLOCK_N=64 -> 64 n-tiles x SPLIT_K=8 (512 each) = 512 items per expert. So ~2 touched
  experts at bs=1 still give >= 1024 items over all 132 SMs, ~6 at bs=8 ~3k. (4) Reduce: each item
  writes its fp32 [16, 64] partial to a slot-indexed scratch part[split, slot, n] (allocated once
  in prepare at the bs=16 maximum: w13 4x128x8192 fp32 = 16 MB, w2 8x128x4096 fp32 = 16 MB), then
  takes a ticket with an acq_rel gpu-scope atomic_add on counter[block, n_tile]; the CTA that
  draws SPLIT_K-1 sums the SPLIT_K partials, multiplies by the expert's global scale (w13_gs[e] /
  w2_gs[e], applied once here, never folded into the weight tile), casts to bf16, stores
  cache1[slot, n-tile] (w13) or cache3[slot, n-tile] (w2) masked to live slots, and resets the
  counter to 0 so the next graph replay starts clean (counters zeroed once in prepare). fp32
  atomic_add into a zeroed fp32 buffer is the fallback, but it costs a memset node plus a cast
  pass because silu_and_mul and the post-norm read bf16. (5) Dequant, the second failure (the
  failed kernel spent ~15 fp32 ops per weight with a where-based E2M1 decode, an interleave and
  per-element scale gather; the ALU budget at 4.8 TB/s on 132 SMs at ~1.98 GHz is ~36
  weights/clk/SM, i.e. ~3.5 lane-ops per weight): decode E2M1 by bit placement on packed 32-bit
  words (8 nibbles), e.g. place sign at bit 15 and the 3 magnitude bits at bits 11..9 of an fp16
  half (E2M1 e=0 lands as an exact fp16 subnormal), convert half2->float2 and apply ONE fp32
  multiply by s = sf_e4m3 * 2^14 per 16-element block (in range: <= 448*2^14), then cvt to bf16x2
  (exact: e2m1 x e4m3 has <= 6 significant bits); a direct bf16 placement needs a 2^126 fixup that
  overflows once folded with sf > 2, so it would take two bf16x2 multiplies; a 256-entry byte ->
  bf16x2 constant LUT in shared memory is the equally acceptable alternative. Use
  tl.inline_asm_elementwise (lop3/prmt/cvt, pack=4) as triton_kernels' mxfp4 upcast does if plain
  Triton ops do not produce packed code; no tl.where, no exp2. Order the K permutation so the
  low/high nibbles of a byte map to the tile's K order without tl.interleave (permute x's K index
  instead, a cheap gather on an L2-resident [<=16, BLOCK_K] tile). (6) Math: x (or cache2) rows
  gathered by slot into a [16, BLOCK_K] bf16 tile, tl.dot bf16 x bf16 -> fp32 with M padded to 16
  (tensor cores are not the bottleneck at these M). (7) Bytes in flight: BLOCK_K=256 (8 KB of
  packed weight + 1 KB scales per stage per CTA), num_stages=4, ~4 CTAs/SM -> ~96 KB/SM, ~12 MB in
  flight, about 2x the Little's-law ~6 MB needed at 4.8 TB/s and ~1.2 us latency. Fixed constexpr
  configs, no autotune. Launch count on the small path: gather, align, w13 GEMV, silu_and_mul, w2
  GEMV, post-norm.
expected:
  - shapes: bs=1 load=decode_1 sharding=ep8
    direction: faster
    why: "incumbent: Marlin 41.1 us for w13+w2 (w13 ~22.9, w2 ~18.2 standalone, from the triton-w13 run's profile) in an 80.2 us layer; floor: ~1.93 touched experts x 28.3 MB (w13 18.9 + w2 9.4) = ~11.4 us at 4.8 TB/s (w13 alone ~7.6 us). Stage-1 target: both GEMVs <= ~20 us (>= 55% of HBM), layer ~60 us"
  - shapes: bs=8 load=verify_1 sharding=ep8
    direction: faster
    why: "incumbent: Marlin 107.6 us for both GEMMs (w13 ~69.6, w2 ~38.0) in a 152.0 us layer; floor ~37.3 us (~6.3 touched experts). Target: <= ~60 us for both, layer ~105 us"
  - shapes: bs=16 load=decode_16 sharding=ep8
    direction: faster
    why: "incumbent: Marlin 150.9 us for both GEMMs in a 191.4 us layer; floor ~52.5 us (~8.9 touched experts). Target: <= ~85 us, layer ~125 us; block 16 holds cnt[e] <= 16 in one block, so each expert's weights are read once"
  - shapes: bs=32 *
    direction: flat
    why: "control: num_tokens > 16 keeps the parity Marlin path unchanged"
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched Marlin path"
  - shapes: bs=256 *
    direction: flat
    why: "control: untouched Marlin path"
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
stage: 1
replaces: Marlin w13 + silu + w2 at num_tokens <= 16
review: proceed
status: queued
review_at: 2026-10-24T083755Z
review_companions: null
---

# small-m-grouped-gemv-splitk-rewrite

## Change

Stage 1 of a small-batch grouped W4A16 GEMV that replaces both moe_wna16_marlin_gemm calls when num_tokens <= 16 (decode_1 bs=1, verify_1 bs=8, decode_16 bs=16); num_tokens > 16 keeps the parity Marlin path byte for byte. Unchanged in stage 1: the expert_map gather, silu_and_mul (cache1 -> cache2, expert_ids=local_ids), fused_post_expert_norm_weighted_sum and the cache1/cache2/cache3 row layout (row = flat slot token*8+k, remote slots left unwritten, as Marlin leaves them). (1) Routing: on the small path call moe_align_block_size(local_ids, 16, 32) (block 16, not Marlin's 8) so every touched local expert is exactly one block of <= 16 slots (cnt[e] <= M <= 16); padded entries carry the sentinel M*8 and are masked. The number of live blocks is read on device from n_padded; nothing branches on it on the host. (2) Weights: prepare() keeps, beside the Marlin bank, a byte-identical contiguous copy of the 32 local experts' w13/w13_sf/w13_gs and w2/w2_sf/w2_gs (NVFP4 bytes as stored, optionally a lossless permutation of byte order within a row; never a BF16 copy). (3) Parallelism, the first failure of triton-w13-grouped-gemv-small-m: that kernel launched a static (32 experts x 256 n-tiles) grid in which ~30/32 rows exited, leaving ~512 CTAs each streaming a whole 4096-long K row with 3 stages. Instead each GEMM is ONE persistent kernel with a fixed grid (e.g. 132*4 CTAs) looping over a device-side work list of items = live_block x n_tile x k_split: w13 (N=8192, K=4096) BLOCK_N=64 -> 128 n-tiles x SPLIT_K=4 (1024 each) = 512 items per touched expert; w2 (N=4096, K=I=4096) BLOCK_N=64 -> 64 n-tiles x SPLIT_K=8 (512 each) = 512 items per expert. So ~2 touched experts at bs=1 still give >= 1024 items over all 132 SMs, ~6 at bs=8 ~3k. (4) Reduce: each item writes its fp32 [16, 64] partial to a slot-indexed scratch part[split, slot, n] (allocated once in prepare at the bs=16 maximum: w13 4x128x8192 fp32 = 16 MB, w2 8x128x4096 fp32 = 16 MB), then takes a ticket with an acq_rel gpu-scope atomic_add on counter[block, n_tile]; the CTA that draws SPLIT_K-1 sums the SPLIT_K partials, multiplies by the expert's global scale (w13_gs[e] / w2_gs[e], applied once here, never folded into the weight tile), casts to bf16, stores cache1[slot, n-tile] (w13) or cache3[slot, n-tile] (w2) masked to live slots, and resets the counter to 0 so the next graph replay starts clean (counters zeroed once in prepare). fp32 atomic_add into a zeroed fp32 buffer is the fallback, but it costs a memset node plus a cast pass because silu_and_mul and the post-norm read bf16. (5) Dequant, the second failure (the failed kernel spent ~15 fp32 ops per weight with a where-based E2M1 decode, an interleave and per-element scale gather; the ALU budget at 4.8 TB/s on 132 SMs at ~1.98 GHz is ~36 weights/clk/SM, i.e. ~3.5 lane-ops per weight): decode E2M1 by bit placement on packed 32-bit words (8 nibbles), e.g. place sign at bit 15 and the 3 magnitude bits at bits 11..9 of an fp16 half (E2M1 e=0 lands as an exact fp16 subnormal), convert half2->float2 and apply ONE fp32 multiply by s = sf_e4m3 * 2^14 per 16-element block (in range: <= 448*2^14), then cvt to bf16x2 (exact: e2m1 x e4m3 has <= 6 significant bits); a direct bf16 placement needs a 2^126 fixup that overflows once folded with sf > 2, so it would take two bf16x2 multiplies; a 256-entry byte -> bf16x2 constant LUT in shared memory is the equally acceptable alternative. Use tl.inline_asm_elementwise (lop3/prmt/cvt, pack=4) as triton_kernels' mxfp4 upcast does if plain Triton ops do not produce packed code; no tl.where, no exp2. Order the K permutation so the low/high nibbles of a byte map to the tile's K order without tl.interleave (permute x's K index instead, a cheap gather on an L2-resident [<=16, BLOCK_K] tile). (6) Math: x (or cache2) rows gathered by slot into a [16, BLOCK_K] bf16 tile, tl.dot bf16 x bf16 -> fp32 with M padded to 16 (tensor cores are not the bottleneck at these M). (7) Bytes in flight: BLOCK_K=256 (8 KB of packed weight + 1 KB scales per stage per CTA), num_stages=4, ~4 CTAs/SM -> ~96 KB/SM, ~12 MB in flight, about 2x the Little's-law ~6 MB needed at 4.8 TB/s and ~1.2 us latency. Fixed constexpr configs, no autotune. Launch count on the small path: gather, align, w13 GEMV, silu_and_mul, w2 GEMV, post-norm.

## Expected

- `bs=1 load=decode_1 sharding=ep8` faster: incumbent: Marlin 41.1 us for w13+w2 (w13 ~22.9, w2 ~18.2 standalone, from the triton-w13 run's profile) in an 80.2 us layer; floor: ~1.93 touched experts x 28.3 MB (w13 18.9 + w2 9.4) = ~11.4 us at 4.8 TB/s (w13 alone ~7.6 us). Stage-1 target: both GEMVs <= ~20 us (>= 55% of HBM), layer ~60 us
- `bs=8 load=verify_1 sharding=ep8` faster: incumbent: Marlin 107.6 us for both GEMMs (w13 ~69.6, w2 ~38.0) in a 152.0 us layer; floor ~37.3 us (~6.3 touched experts). Target: <= ~60 us for both, layer ~105 us
- `bs=16 load=decode_16 sharding=ep8` faster: incumbent: Marlin 150.9 us for both GEMMs in a 191.4 us layer; floor ~52.5 us (~8.9 touched experts). Target: <= ~85 us, layer ~125 us; block 16 holds cnt[e] <= 16 in one block, so each expert's weights are read once
- `bs=32 *` flat: control: num_tokens > 16 keeps the parity Marlin path unchanged
- `bs=128 *` flat: control: untouched Marlin path
- `bs=256 *` flat: control: untouched Marlin path

## Risk

Risks. (a) Compile features: tl.inline_asm_elementwise with pack=4, gpu-scope acq_rel tl.atomic_add, a runtime loop bound loaded from n_padded and 4-stage pipelining on sm_90 must all compile in the judge's Triton; no local GPU, so the first run is also the compile test; keep a pure-Triton (LUT) dequant variant ready if the inline asm is rejected. (b) Atomics ordering: the partial store must be visible before the ticket (release on the ticket atomic, acquire before the last CTA reads other splits' partials); the last CTA must reset its counter or the next CUDA-graph replay miscounts; counters must be indexed by (block, n_tile) with blocks bounded by the maximum n_padded/16 at bs=16 (M*8 + 32*15 entries -> <= 38 blocks). fp32 sum order differs from Marlin: fine at 1.5% L2 but not bitwise. (c) Correctness gate in prepare(): run both GEMVs on synthetic data made only from the weights and a fixed torch.Generator (deterministic x, a fixed routing touching e.g. experts 0, 7, 31 with 1 and 16 slots), compare to an fp32 reference built with nvfp4.dequantize(w, sf, gs) of those experts, require rel L2 < 1e-2 per row; on failure print it and set state['small_path']=False so kernel_fn falls back to Marlin (a decision fixed in prepare, never on inputs), so a broken kernel shows as flat, not as a wrong answer. (d) A 0.5-code subnormal flushed anywhere (FTZ on a bf16/fp16 op) silently zeroes 1/16 of the codes: the prepare check covers it, since uniform random codes hit it. (e) Rules: weight copy is the same NVFP4 bytes (lossless), activations bf16 in and out, fp32 only in the accumulator and the split-K scratch, nothing derived from data kept between calls (scratch and counters are workspace, overwritten every call). (f) Overlap: the queued triton-w2-grouped-gemv-small-m depends on the regressed triton-w13 kernel; this rewrite supersedes that line and should not be stacked with it. (g) If stage 1 lands but misses target, stage 2 levers are SPLIT_K/BLOCK_N retune and fusing silu_and_mul into the w13 last-CTA epilogue, then the post-norm into the w2 epilogue.

## Milestones

Rewrite stage 1 of `Marlin w13 + silu + w2 at num_tokens <= 16`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T083755Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound stage-1 rewrite. The parity profile shows Marlin at roughly 1-1.6 TB/s on the three small-M shapes (41.1/107.6/150.9 us against 11.4/37.3/52.5 us floors), and the design fixes both measured causes of the triton-w13 failure: the idle grid, and dequant costing about 15 ALU ops per weight. No rules problem. The biggest risk is that each work item's inner K loop is too short to pipeline (4 BLOCK_K steps for w13, only 2 for w2, with num_stages=4), so a persistent loop over items may expose a full memory latency per item.

### Justification
Mechanism: at bs=1 the two Marlin calls move about 1.93 x 28.3 MB = 54.6 MB in 41.1 us (~1.33 TB/s). The triton-w13 run's profile gives the split: Marlin w2 alone was 18.2 us for ~18 MB (~1.0 TB/s), so w13 was ~22.9 us (inferred by subtraction, not measured). At bs=8 Marlin is ~107.6 us for ~178 MB (~1.65 TB/s). That leaves 2.5-3.6x of headroom per shape, and the stage-1 targets (55-65% of HBM) are modest. The floors match TASK.md exactly: w13 is 8192 x 2048 B plus 8192 x 256 B of scales = 18.9 MB, w2 is 9.4 MB. The touched-expert counts are consistent with the 11.4/37.3/52.5 us floors. The expected directions are consistent: bs<=16 is faster and bs=32/128/256 stay on the byte-identical Marlin path as controls. If every target is met, the headline is about (80.2/60 x 152/105 x 191.4/125)^(1/6) = ~1.20x, so it is readable far beyond the 1.5-2.5% run-to-run drift. The milestone is directly readable: the new kernels appear by name in top_kernels per shape, against the Marlin line. Caveat: the incumbent's w13/w2 split at bs=16 is not measured anywhere, so only the combined 150.9 us is a valid yardstick there. Design checks: (1) Parallelism. 512 items per touched expert per GEMM over a 528-CTA persistent grid gives about 2 items per CTA at bs=1 and 9 at bs=16. This fixes the ~30/32-empty static grid. (2) Bit-placement dequant is correct. Codes e=1..3 land on 2^-14 x {1,1.5}, {2,3}, {4,6}, and the 0.5 code lands as the fp16 subnormal 2^-15. One fp32 multiply by sf x 2^14 (at most 7.3e6) is exact, and the cvt to bf16 is exact (product of at most 6 significant bits). nvfp4.py draws block scales with log2 ~ N(7, 0.4), so nothing approaches a range edge. The cost is about 3 lane-ops per weight (place, cvt f16->f32, fmul, half a cvt bf16x2, plus the sf broadcast). That is close to the self-quoted 3.5 ops/weight full-HBM budget but well inside the ~6 available at the 55% target. (3) tl.dot with M=16 uses mma.sync, not wgmma, which is fine at this M. (4) Bytes in flight (~12 MB) are about 2x Little's law if the pipeline actually fills. WEAKEST LINK: Triton software-pipelines only the innermost loop. A w13 item covers K=1024 (4 iterations of 256) and a w2 item covers K=512 (2 iterations) with num_stages=4. Each item therefore runs a cold prologue and drains before the next item's loads are issued, and w2 can never hold more than 2 stages in flight. At bs=16 (~9 items per CTA) this exposes about 9 memory latencies per CTA, and only the 4-CTA/SM co-residency hides them. Fix options: flatten item x k_step into one linear tl.range loop so prefetch crosses item boundaries, or give w2 SPLIT_K=4 (K=1024 per item) with BLOCK_N=32 to keep the item count. The second weakest link is the ticket-counter split-K reduce, which is unproven on this judge's Triton; the in-flight bf16 semaphore kernels are the first test of it.

### Confounders on the current kernel
- sglang-marlin-path-parity (the base and the only landed hypothesis): it cannot be absent. The rewrite reuses its moe_align_block_size(local_ids, ., 32) over the 32-expert bank, at block 16 instead of 8. Align time at block 16 vs 8 moves by under 1 us and shows as its own profile row, so it cannot flip the sign of a GEMV result.
- The at::native glue (reduce_kernel<512> ~5.6 us, elementwise ~5.4 us and others at bs=1) is not part of the w13 Marlin wrapper: it persisted in the triton-w13 run. If some of it belongs to the w2 wrapper, it disappears with this change and inflates the layer-level gain. Judge the milestone on the new kernels' own time, not on the layer delta.

### Interference and A/B needs
No companions. This is a rewrite judged on its replaced kernels' own time per shape, which the profile reports directly. The only landed hypothesis is the base itself, so an ab-revert would return to the starter, and the align knob it carries is not a meaningful tune axis. The three untouched shapes (bs=32/128/256) are built-in controls for drift. The natural follow-up is stage 2: flattened or retuned SPLIT_K, and silu fused into the w13 last-CTA epilogue.

### Open questions
- Memory ordering of the ticket: a scalar tl.atomic_add with sem='acq_rel' orders only the issuing thread's stores. Put tl.debug_barrier() between the partial stores and the ticket. In the last CTA, after the acquire, put another barrier and read the other splits' partials with cache_modifier='.cg' (L2, bypassing L1), or L1-stale data can be summed.
- How does moe_align_block_size(local_ids, 16, 32) lay out remote (-1) slots and blocks? Marlin with is_ep=True skips blocks whose block_experts entry is -1 and masks sorted_ids >= M*8. The work-list decode must replicate both checks, or a remote block will read expert -1's weights.
- w13 gathers x[slot // 8] (top_k=8 semantics) and w2 gathers cache2[slot] (top_k=1). cache3 aliases cache1's storage, as in the base. Confirm that the gate/up row order of the raw w13 (gate rows [0, I), up rows [I, 2I)) matches what silu_and_mul expects of cache1 as Marlin wrote it.
- Several queued hypotheses build on the regressed triton-w13 kernel: triton-w2-grouped-gemv-small-m, triton-w13-silu-epilogue and possibly triton-small-m-threshold-32. Should they be retired or re-based on this line? The rewrite's own risk (f) says not to stack them.
- align-256-experts-below-128 (p1, queued) changes the align call this path reuses. If it lands first, the rewrite's base and its align row move. Decide the order.

