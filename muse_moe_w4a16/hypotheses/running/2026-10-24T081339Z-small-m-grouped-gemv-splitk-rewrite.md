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
stage: dispatched
replaces: Marlin w13 + silu + w2 at num_tokens <= 16
review: proceed
status: running
review_at: 2026-10-24T083755Z
review_companions: null
claimed_at: 2026-10-24T083755Z
base_sha: 713276ec0ed1f2c1c4e621ba4314e63f989307e4
base_blob: 11fbc1957daf7294cd2810ff82a0aad2f25644bd
branch: hyp/muse_moe_w4a16/small-m-grouped-gemv-splitk-rewrite
worktree: .worktrees/muse_moe_w4a16/small-m-grouped-gemv-splitk-rewrite
mode: check
implemented_at: 2026-10-24T084758Z
impl_commit: 0193fee7339d60d90360931ecbc466df3bacedd1
submission_sha256: d32912216cff529a28507bea9d3b0d06086b881e04efd2179e13091b32b1eae8
norm_sha256: 9b40d2120136a0ec1be7e03f95fc13fb0f28065be33c00b5c4c186932dd86291
submission_blob: 644e024fd9b306357107599e01908bc26281ea22
lint_warnings: 2
attempt_file: 2026-10-24T084758Z-check-small-m-grouped-gemv-splitk-rewrite.md
prepared_at: 2026-10-24T084758Z
job: 734a1159ff334c3d
dispatched_at: 2026-10-24T084809Z
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


## Implementation notes

Worktree `.worktrees/muse_moe_w4a16/small-m-grouped-gemv-splitk-rewrite`, only `muse_moe_w4a16/submission.py` changed (uncommitted; 293 insertions, 1 deletion, the docstring's last line). First-line marker kept. Untested: no GPU/Triton on this node, so the first run is also the compile test; `py_compile` and okc_lint pass.

- **Dispatch** (`kernel_fn`): `if bank["small"] is not None and expert_map is not None and num_tokens <= 16` (a shape branch plus a decision fixed in prepare) -> `_small_forward` then the unchanged `fused_post_expert_norm_weighted_sum`. Otherwise the parity Marlin code runs byte for byte (the new branch is inserted before it; no parity line edited).
- **Small path** (`_small_forward`): `moe_align_block_size(local_ids, 16, 32)`; cache2/cache13/cache1/cache3 allocated exactly as parity (cache3 aliases cache1's storage); w13 GEMV (x rows gathered `x[slot // 8]`, top_k=8) -> cache1; unchanged `silu_and_mul(cache1, cache2, expert_ids=local_ids, expert_step=1)`; w2 GEMV (rows `cache2[slot]`, top_k=1) -> cache3. Launches: gather, align, w13 GEMV, silu, w2 GEMV, post-norm. The w13 output column n is w13 weight row n, so gate is cache1[:, :I) and up cache1[:, I:2I), the same order Marlin writes and silu_and_mul (silu(x[:, :d]) * x[:, d:]) and the reference expect.
- **Kernel** `_grouped_gemv_splitk_kernel` (one `@triton.jit`, two constexpr variants, `do_not_specialize` on `n_valid`, `nb_static` so one compile serves every M; fixed configs, no autotune), num_warps=4, num_stages=4, BLOCK_M=16, BLOCK_K=256:
  - **Grid/items**: persistent 1-D grid of `SMs x CTAs/SM`. CTAs/SM is measured once in prepare from the compiled kernel's `metadata.shared` and `n_regs` (capped at 4, default 2 if the metadata is unreadable); it is printed. Items = live local block x n-tile x k-split, strided over CTAs (`item = pid + j*G`). w13: BLOCK_N=64, SPLIT_K=4 -> 128 n-tiles x 4 = 512 items per touched expert, 4 k-steps each. w2: BLOCK_N=64, SPLIT_K=8 -> 64 x 8 = 512 items per touched expert, 2 k-steps each. Approx. 1k items at bs=1 (~2 experts), ~3k at bs=8, ~4.6k at bs=16.
  - **Pipelining choice: FLATTENED.** The (item x k-step) space is ONE linear `range(0, n_iters)` loop (`n_iters = items_of_this_CTA * KSTEPS`, from device values) in the style of Triton's persistent-matmul tutorial: the item decode is scalar math each iteration and the epilogue sits under `if ks == KSTEPS - 1`, so the pipeliner prefetches the next item's tiles across item boundaries. w2 therefore keeps the spec's SPLIT_K=8 (2 steps per item); the review's alternative (w2 SPLIT_K=4/BLOCK_N=32) was not needed.
  - **Live blocks / remote routes**: the live-block count is read on device: `n_blocks = min(n_padded // 16, nb_static)` (no `.item()`); the prologue loads block_experts[0..64) masked to n_blocks, counts the -1 blocks (Marlin's is_ep skip) and finds where their run starts (r0), so live local block lb maps to block `lb` below r0 and `lb + n_neg` from r0 on. The decode re-loads `e = block_experts[blk]` and masks the whole item if `e < 0` (never reads expert -1's weights; clamped to 0 for address math); slots are masked by `sorted_ids < M*8` (the sentinel). Assumption: the -1 blocks form one contiguous run (one align bucket); the prepare self-test runs the real moe_align with remote slots and would fail (fallback) if not.
  - **Dequant (method)**: plain Triton, not inline asm (deviation from the "use inline_asm if plain ops do not produce packed code" clause: nothing can be inspected here, and plain ops avoid a PTX compile risk on the first run; inline asm / LUT is a stage-2 lever). Bytes are loaded as `[BLOCK_N, 16 groups, 8 bytes]` (8 bytes = one 16-element scale group) and widened to uint16; E2M1 -> fp16 by bit placement (`((b&7)<<9)|((b&8)<<12)` low nibble, `((b&0x70)<<5)|((b&0x80)<<8)` high nibble: e2m1 * 2^-14 exactly, the 0.5 code an exact fp16 subnormal), bitcast fp16, cvt fp32 (exact), ONE fp32 multiply by `sf_e4m3 * 2^14` broadcast per group from a `[BLOCK_N, 16]` scale tile (no per-element scale gather), cvt bf16 (exact, <= 6 significant bits). No tl.where, no exp2, no tl.interleave. The x tile `[16, 256]` is loaded contiguously and split with `tl.split(tl.reshape(x, (16, 128, 2)))` into even/odd K (the "permute x's K index instead"), and the tile does two `tl.dot(bf16, bf16 -> fp32)` with K=128: x_even @ W_lo^T + x_odd @ W_hi^T. No fp16 activations; bf16 dot kept.
  - **Reduce / ticket ordering**: at the last k-step of an item: masked fp32 partial store to `part[split, slot, n]` -> `tl.debug_barrier()` -> one scalar `tl.atomic_add(cnt[e, n_tile], 1, sem="acq_rel", scope="gpu")` -> `tl.debug_barrier()` -> the CTA that drew SPLIT_K-1 reloads all SPLIT_K partials with `cache_modifier=".cg"` in fixed order 0..SPLIT_K-1, multiplies by the expert's fp32 global scale once, stores bf16 to `out[slot, n-tile]` masked to live slots, then `atomic_xchg(cnt, 0)` (self-resetting). Counters are keyed by (local expert, n-tile) (each touched expert is exactly one block at M <= 16), separate arrays `cnt13` [32*128] and `cnt2` [32*64], zeroed once in prepare. Scratch `part13` fp32 [4, 128, 8192] and `part2` fp32 [8, 128, 4096] (16 MB each), workspace overwritten every call.
- **Weights** (`_prepare_small`): `w13`, `w13_sf`, `w2`, `w2_sf` of experts [0, 32) are `.clone()`d contiguous (identical NVFP4 bytes as stored, no permutation; fp8 scales as stored), `w13_gs`/`w2_gs` fp32. No BF16 copy.
- **prepare() self-test** (`_small_self_test`): weights-only synthetic data from `torch.Generator(seed 1234)`: three cases in order M=16 (expert 7 with 16 slots, experts 0, 31, 3 with 1 slot each, the remaining 108 slots remote -1), M=1 (experts 31, 3, 12 plus 5 remote slots), M=16 again. Each runs the real `moe_align_block_size(ids, 16, 32)` and both GEMVs into NaN-prefilled outputs and compares each local row to an fp32 reference `x_rows @ nvfp4.dequantize(w, sf, gs).T`; requires max row rel-L2 < 1e-2, remote rows still NaN (never written), and both counter arrays exactly zero after every case (checks self-reset; the repeated M=16 case runs on counters left by the earlier ones). The first case also measures CTAs/SM and sets the final grid, so cases 2-3 test the grid used in timing. Any exception (compile/launch) is caught. The result is printed LOUDLY between `=` banners in both outcomes: PASSED/FAILED with both rel-L2 values, the grid and occupancy, and `PATH CHOSEN ... TRITON split-K grouped GEMV` or `FALLBACK to Marlin (parity path)`. On failure `bank["small"] = None` and every shape runs Marlin (flat, not wrong).
- **Stage-1 readout**: pending the run. Read from `top_kernels`: `_grouped_gemv_splitk_kernel` (both variants share one name; the per-shape row is w13+w2 combined) against Marlin's combined 41.1 (bs=1) / 107.6 (bs=8) / 150.9 (bs=16) us; floors 11.4 / 37.3 / 52.5 us. Also grep the log for the `[small-m-grouped-gemv-splitk]` banner first: if it says FALLBACK the run measured Marlin only.

Uncertainties: (1) compile of the flattened loop with a conditional atomic/barrier epilogue, tl.split/reshape and 3-D tile loads in the judge's Triton (caught -> fallback, shown in the banner); (2) the plain-Triton dequant is ~6-7 lane-ops per weight, above the spec's ~3.5 full-HBM budget, so stage 1 may land at or a bit under the incumbent rather than at target; (3) smem per CTA (4 stages x ~17 KB + dot-operand conversion scratch) probably allows 2 CTAs/SM, i.e. ~9.5 MB in flight vs ~6 MB needed; (4) the contiguous -1 run assumption about moe_align (self-tested).

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/small-m-grouped-gemv-splitk-rewrite diff`.

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints on the timed path (`kernel_fn` -> `_small_forward` -> `_small_gemv`); the live-block count is read on device from n_padded; the only host branches are `num_tokens <= 16` (shape) and `bank["small"] is not None` (fixed in prepare). Allocations (cache2/cache13) are shape-sized as in parity; the grid is a constant from prepare. The syncs and `int(...)`/`float(...)` reads are in `_prepare_small`/`_small_self_test` (prepare only).
- [x] Nothing derived from `data` is cached across calls: `part13/part2` and `cnt13/cnt2` are workspace rewritten every call (counters return to zero by construction); routing is recomputed by moe_align each call.
- [x] Weights: repacked losslessly in `prepare()` only: a `.clone()` of the same NVFP4 uint8 bytes, fp8 e4m3 block scales and fp32 global scales, no BF16 copy; dequantized only in registers inside the kernel. The Marlin bank is unchanged.
- [x] Activations BF16 where SGLang keeps BF16: x, cache1, cache2, cache3 bf16; fp32 only in the dot accumulator and the split-K scratch; bf16 x bf16 tl.dot (no fp16 activations; fp16 appears only as the exact intermediate of the weight decode). KV cache / queries: n/a: no attention in this problem.
- [x] Attention sees every key: n/a: MoE problem, no attention; every local route is computed (remote -1 routes are skipped exactly as Marlin's is_ep path skips them).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds `import nvfp4` (the problem's helper, used only in the prepare self-test), `triton`, `triton.language`, one `@triton.jit` kernel and host helpers.
- [x] All work happens inside the timed call; no output caching or replay: align, both GEMVs, silu and post-norm run in kernel_fn every call; the self-test's outputs are discarded.
- [x] No `triton.autotune`: fixed constexpr configs `SMALL_W13_CFG=(64, 4)`, `SMALL_W2_CFG=(64, 8)` (checked: no `autotune` in the diff).

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`), exit 0:

```
warn  host-sync    L291: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L356: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_moe_w4a16/small-m-grouped-gemv-splitk-rewrite/muse_moe_w4a16/submission.py
```

L291 is the new self-test's sync (prepare only); L356 is the pre-existing parity `torch.cuda.synchronize()` in `prepare()`. `python3 -I -m py_compile` passes; its `__pycache__` was removed from the worktree.
