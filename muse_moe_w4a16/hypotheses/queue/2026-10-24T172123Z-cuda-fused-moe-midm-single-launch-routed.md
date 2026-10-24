---
problem: muse_moe_w4a16
slug: cuda-fused-moe-midm-single-launch-routed
created: 2026-10-24T172123Z
base: master
area: whole-layer fusion
change: >
  Mid shapes only: 2 <= num_tokens <= 32 (verify_1 bs=8, decode_16, decode_32, and the bs=4 test).
  bs=1 (owned by cuda-fused-moe-bs1-persistent-w13-silu-w2) and num_tokens > 32
  (bs=33/128/256/257) keep master byte for byte. Replace the WHOLE mid-shape MoE (master:
  _align_small, Marlin w13, silu_and_mul (act_and_mul_kernel), Marlin w2, and the post-norm tail
  (_post_norm_combine_kernel below 32 tokens, fused_post_expert_norm_weighted_sum at 32): 5
  launches, ~114 / ~125 / ~198 us profiled at bs 8 / 16 / 32) with ONE persistent CUDA kernel
  compiled in prepare() with torch.utils.cpp_extension.load_inline (one TU, -O3 -arch=sm_90a, no
  fast-math; pattern from muse_attention_decode _build_splitkv; print compile seconds). WHY: the
  in-flight rewrites leave this structure in place. cuda-nvfp4-swapab-mma-persistent-small-m
  replaces only the GEMM pair at bs<=32 and keeps the align launch, the post-norm launch, the
  w13->w2 launch boundary (tail drain + ramp, the w2 weight stream cannot start until the w13 grid
  ends) and the cache round trips; cuda-fused-moe-bs1 fuses everything but only for one token,
  where routing is trivial (R <= 8 routes, no expert grouping). At bs 8-32 with M3-level GEMMs
  (~56-106 us), those leftovers are ~10-15 us per shape (align 2.7-4.4 us, post-norm 3.5-4.2 us,
  two extra graph nodes, the phase boundary), i.e. 10-20% of three of the six equally weighted
  loads. STAGES PLANNED: stage 1 (this file) = the single launch with in-kernel routing,
  flag-gated w13->w2 dataflow and a ticketed post-norm combine; stage 2 = tuning from the profile
  (ring depth, SK tables, per-expert NT buckets, prefetch distance across the phase boundary);
  stage 3 = extend the same kernel to bs=1 (merging the bs=1 sibling) and to bs=33 so every M <=
  32 runs one launch. STAGE-1 DESIGN. MAINLOOP AND BANK: reuse the sibling
  cuda-nvfp4-swapab-mma-persistent-small-m's lossless NVFP4 bank (mma m16n8k16 A-fragment nibble
  order, e4m3 scales permuted per lane group, w13 gate/up interleaved per 128-row tile, gs_eff =
  gs * 2^7, round-trip assert) and its exact dequant (v = (q & 0x80008000) | ((q & 0x70007000) >>
  6), q <<= 4, fma.rn.bf16x2 by bf16(e4m3 x 2^119), -0 addend) and swizzled activation slab (XOR
  the 16 B chunk index with row & 7, as that file's build notes do), verbatim if it has harvested
  with SELF-CHECK PASSED, otherwise implemented as specified there; weights as mma A (swap-AB),
  tokens on N in n8 fragments, fp32 accumulation. GRID: 132 CTAs (= SM count, asserted from device
  properties and cudaOccupancyMaxActiveBlocksPerMultiprocessor >= 1), 256 threads = 8 warps,
  launch_bounds(256, 1), dynamic smem <= 180 KB, so all CTAs are co-resident. PHASE 0, ROUTING
  IN-KERNEL (replaces _align_small; every CTA computes the same plan redundantly from the inputs,
  no global align output): load topk_ids [M, 8] (<= 256 int32) and topk_weights into smem; local =
  id - local_start if 0 <= local < local_count else -1 (bank start/count are kernel args from
  prepare); per local expert counts via a deterministic warp-ballot scan over the flat route index
  (no smem atomics, so the order is the same in every CTA and every call): expert lists in
  ascending expert id, routes within an expert in flat-route order; per active expert a its route
  count n_a (<= M, since a token's 8 ids are distinct) and NT_a = ceil(n_a / 8) in {1, 2, 3, 4};
  per token t its local route count R_t. CTA (t mod 132) writes a zero out row for every token
  with R_t = 0. PHASE 1, w13 + SiLU: items = (active expert a, 128-row tile j of 64 tiles, k-split
  s), SK1 = 4 / 2 / 1 for M <= 8 / <= 16 / <= 32 (host constants by shape, as the sibling),
  ordered expert-major; B slab = x rows of the expert's routes (row t = route / 8) gathered by
  cp.async with zero-fill for pad columns; split partials by the measured ticket pattern (fp32
  partial __stcg, __threadfence, thread-0 atomicAdd ticket, last arriver sums in fixed split order
  with __ldcg and resets the ticket); epilogue acc * gs_eff13 -> silu(gate) * up in fp32 -> bf16
  into a prepare-allocated scratch h [256, I] bf16 at the route's expert-sorted position, then
  __threadfence and st.release.gpu add 1 to flag[a_local][j / 8] (one flag per 512-column chunk of
  h, 8 tiles each). PHASE 2, w2: items = (active expert a, 128-row tile n of 32, k-split s over
  I), SK2 = 8 / 4 / 2 by the same M classes; before waiting, the CTA issues the cp.async stages
  for this item's weights (they do not depend on h), so the w2 stream overlaps the end of phase 1;
  then it spins with ld.acquire.gpu on the flags covering its I range until each equals 8 * (epoch
  + 1), loads the h slab, runs the MMA loop; ticket-reduced; the last arriver multiplies by
  gs_eff2, rounds to bf16 (master keeps y in bf16), and for each of its routes r writes y_bf16[r,
  128n..128n+128) to a scratch y [256, H] and the partial sum of squares of v = float(y_bf16) * (1
  + post_norm_w) over its 128 columns to ssq[r][n] (fixed slot, no float atomics, deterministic),
  then __threadfence and atomicAdd(done[t], 1) for each route's token t. PHASE 3, POST-NORM
  COMBINE (replaces the post-norm launch): the arriver whose increment makes done[t] == 32 * R_t
  computes out[t, :] = sum over t's local routes in slot order of g * v * rsqrt(sum_n ssq[r][n] /
  H + 1e-8) in fp32 (the _post_norm_combine_kernel formula, POST_NORM_EPS), writes bf16 out[t] and
  resets done[t] = 0. EPOCHS, NO MEMSETS: device int epoch (prepare-zeroed) read at entry; flags
  count up and are never reset (int32 wrap ~2^28 calls away); a final exit ticket's last CTA
  increments epoch after __threadfence and resets the exit ticket. DEADLOCK FREEDOM: co-residency
  plus every CTA finishing all its phase-1 items before its first phase-2 wait, so every awaited
  tile belongs to a CTA that is not waiting; phase 3 never waits. kernel_fn: one launch on the
  current stream, out = torch.empty((M, H), bf16), no host branch on data (CUDA-graph safe).
  SELF-CHECK (prepare, untimed, fixed torch.Generator, real bank weights, nothing input-derived):
  M in (2, 4, 8, 16, 32) x {random, skewed (3 local slots on the same experts every token),
  all-one-expert (M routes on one expert, NT = 4), all-remote (every R_t = 0), all-local (8 local
  routes per token), mixed R_t including 0}; compare out per row against master's path on the same
  synthetic inputs, per-row rel L2 <= 1e-2 and max printed; one expert against a local fp32
  dequant matmul (e2m1 grid and e4m3 decode written in the file, not the reference); run every
  case 50 times back to back (epoch, ticket, flag and done reuse) and once under torch.cuda.graph
  capture + replay; assert tickets, done counters and exit ticket are 0 afterwards. Print LOUDLY
  in both outcomes: [cuda-moe-midm-fused] SELF-CHECK PASSED/FAILED, max rel-L2, numRegs and
  localSizeBytes (cudaFuncGetAttributes), smem, SK tables, items per phase per shape, compile
  seconds, PATH CHOSEN. FALLBACK: a load_inline exception, cudaFuncSetAttribute failure, failed
  residency assert, failed check or capture error sets state['midm_fused_ok'] = False and every M
  runs master unchanged; kernel_fn branches only on num_tokens and that flag. MILESTONES readable
  from the profile's top_kernels per shape (the single fused row vs the sum of master's replaced
  rows: ~114 / ~125 / ~198 us at bs 8 / 16 / 32): M1 all 5 tests pass with PATH CHOSEN = fused in
  the log; M2 fused row <= the replaced sum at every mid shape; M3 fused row <= 52 / 72 / 98 us
  (within 1.4x of the 37.3 / 52.5 / 70.8 us floors, ~3.4 TB/s), and the shape's time minus the
  fused row <= 2 us (no other node left).
expected:
  - shapes: bs=8 *
    direction: faster
    why: "verify_1: incumbent 107.8 us scored = Marlin pair 105.8 + post-norm 3.5 + align 2.7 + silu, 5 launches, vs a 37.3 us floor. One launch, in-kernel routing, w2 stream prefetched across the phase boundary: target <= 52 us (M3), load ratio 1.34x -> ~2.8x"
  - shapes: bs=16 *
    direction: faster
    why: "decode_16: incumbent 147.9 us scored (Marlin 116.0 + post-norm 4.2 + align 2.9 + silu) vs a 52.5 us floor; target <= 72 us (M3), load ratio 1.27x -> ~2.6x"
  - shapes: bs=32 *
    direction: faster
    why: "decode_32: incumbent 200.7 us scored (Marlin 187.6 + align 4.4 + post-norm 4.0 + silu) vs a 70.8 us floor; NT=4 streams each hot expert once (Marlin restreams per 16-row block); target <= 98 us (M3), load ratio 1.03x -> ~2.1x. All three at M3: headline ~1.31x -> ~1.85x on master; ~+12% over the small-m CUDA sibling if that hits its own M3"
  - shapes: bs=1 *
    direction: flat
    why: "control: bs=1 runs master (owned by cuda-fused-moe-bs1-persistent-w13-silu-w2)"
  - shapes: bs=128 *
    direction: flat
    why: "control: num_tokens > 32 runs master byte for byte"
  - shapes: bs=256 *
    direction: flat
    why: "control: num_tokens > 32 runs master byte for byte"
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: _align_small + Marlin w13 + silu_and_mul + Marlin w2 + post-norm combine (the whole MoE, 5 launches) at 2 <= num_tokens <= 32
review: required
status: queued
---

# cuda-fused-moe-midm-single-launch-routed

## Change

Mid shapes only: 2 <= num_tokens <= 32 (verify_1 bs=8, decode_16, decode_32, and the bs=4 test). bs=1 (owned by cuda-fused-moe-bs1-persistent-w13-silu-w2) and num_tokens > 32 (bs=33/128/256/257) keep master byte for byte. Replace the WHOLE mid-shape MoE (master: _align_small, Marlin w13, silu_and_mul (act_and_mul_kernel), Marlin w2, and the post-norm tail (_post_norm_combine_kernel below 32 tokens, fused_post_expert_norm_weighted_sum at 32): 5 launches, ~114 / ~125 / ~198 us profiled at bs 8 / 16 / 32) with ONE persistent CUDA kernel compiled in prepare() with torch.utils.cpp_extension.load_inline (one TU, -O3 -arch=sm_90a, no fast-math; pattern from muse_attention_decode _build_splitkv; print compile seconds). WHY: the in-flight rewrites leave this structure in place. cuda-nvfp4-swapab-mma-persistent-small-m replaces only the GEMM pair at bs<=32 and keeps the align launch, the post-norm launch, the w13->w2 launch boundary (tail drain + ramp, the w2 weight stream cannot start until the w13 grid ends) and the cache round trips; cuda-fused-moe-bs1 fuses everything but only for one token, where routing is trivial (R <= 8 routes, no expert grouping). At bs 8-32 with M3-level GEMMs (~56-106 us), those leftovers are ~10-15 us per shape (align 2.7-4.4 us, post-norm 3.5-4.2 us, two extra graph nodes, the phase boundary), i.e. 10-20% of three of the six equally weighted loads. STAGES PLANNED: stage 1 (this file) = the single launch with in-kernel routing, flag-gated w13->w2 dataflow and a ticketed post-norm combine; stage 2 = tuning from the profile (ring depth, SK tables, per-expert NT buckets, prefetch distance across the phase boundary); stage 3 = extend the same kernel to bs=1 (merging the bs=1 sibling) and to bs=33 so every M <= 32 runs one launch. STAGE-1 DESIGN. MAINLOOP AND BANK: reuse the sibling cuda-nvfp4-swapab-mma-persistent-small-m's lossless NVFP4 bank (mma m16n8k16 A-fragment nibble order, e4m3 scales permuted per lane group, w13 gate/up interleaved per 128-row tile, gs_eff = gs * 2^7, round-trip assert) and its exact dequant (v = (q & 0x80008000) | ((q & 0x70007000) >> 6), q <<= 4, fma.rn.bf16x2 by bf16(e4m3 x 2^119), -0 addend) and swizzled activation slab (XOR the 16 B chunk index with row & 7, as that file's build notes do), verbatim if it has harvested with SELF-CHECK PASSED, otherwise implemented as specified there; weights as mma A (swap-AB), tokens on N in n8 fragments, fp32 accumulation. GRID: 132 CTAs (= SM count, asserted from device properties and cudaOccupancyMaxActiveBlocksPerMultiprocessor >= 1), 256 threads = 8 warps, launch_bounds(256, 1), dynamic smem <= 180 KB, so all CTAs are co-resident. PHASE 0, ROUTING IN-KERNEL (replaces _align_small; every CTA computes the same plan redundantly from the inputs, no global align output): load topk_ids [M, 8] (<= 256 int32) and topk_weights into smem; local = id - local_start if 0 <= local < local_count else -1 (bank start/count are kernel args from prepare); per local expert counts via a deterministic warp-ballot scan over the flat route index (no smem atomics, so the order is the same in every CTA and every call): expert lists in ascending expert id, routes within an expert in flat-route order; per active expert a its route count n_a (<= M, since a token's 8 ids are distinct) and NT_a = ceil(n_a / 8) in {1, 2, 3, 4}; per token t its local route count R_t. CTA (t mod 132) writes a zero out row for every token with R_t = 0. PHASE 1, w13 + SiLU: items = (active expert a, 128-row tile j of 64 tiles, k-split s), SK1 = 4 / 2 / 1 for M <= 8 / <= 16 / <= 32 (host constants by shape, as the sibling), ordered expert-major; B slab = x rows of the expert's routes (row t = route / 8) gathered by cp.async with zero-fill for pad columns; split partials by the measured ticket pattern (fp32 partial __stcg, __threadfence, thread-0 atomicAdd ticket, last arriver sums in fixed split order with __ldcg and resets the ticket); epilogue acc * gs_eff13 -> silu(gate) * up in fp32 -> bf16 into a prepare-allocated scratch h [256, I] bf16 at the route's expert-sorted position, then __threadfence and st.release.gpu add 1 to flag[a_local][j / 8] (one flag per 512-column chunk of h, 8 tiles each). PHASE 2, w2: items = (active expert a, 128-row tile n of 32, k-split s over I), SK2 = 8 / 4 / 2 by the same M classes; before waiting, the CTA issues the cp.async stages for this item's weights (they do not depend on h), so the w2 stream overlaps the end of phase 1; then it spins with ld.acquire.gpu on the flags covering its I range until each equals 8 * (epoch + 1), loads the h slab, runs the MMA loop; ticket-reduced; the last arriver multiplies by gs_eff2, rounds to bf16 (master keeps y in bf16), and for each of its routes r writes y_bf16[r, 128n..128n+128) to a scratch y [256, H] and the partial sum of squares of v = float(y_bf16) * (1 + post_norm_w) over its 128 columns to ssq[r][n] (fixed slot, no float atomics, deterministic), then __threadfence and atomicAdd(done[t], 1) for each route's token t. PHASE 3, POST-NORM COMBINE (replaces the post-norm launch): the arriver whose increment makes done[t] == 32 * R_t computes out[t, :] = sum over t's local routes in slot order of g * v * rsqrt(sum_n ssq[r][n] / H + 1e-8) in fp32 (the _post_norm_combine_kernel formula, POST_NORM_EPS), writes bf16 out[t] and resets done[t] = 0. EPOCHS, NO MEMSETS: device int epoch (prepare-zeroed) read at entry; flags count up and are never reset (int32 wrap ~2^28 calls away); a final exit ticket's last CTA increments epoch after __threadfence and resets the exit ticket. DEADLOCK FREEDOM: co-residency plus every CTA finishing all its phase-1 items before its first phase-2 wait, so every awaited tile belongs to a CTA that is not waiting; phase 3 never waits. kernel_fn: one launch on the current stream, out = torch.empty((M, H), bf16), no host branch on data (CUDA-graph safe). SELF-CHECK (prepare, untimed, fixed torch.Generator, real bank weights, nothing input-derived): M in (2, 4, 8, 16, 32) x {random, skewed (3 local slots on the same experts every token), all-one-expert (M routes on one expert, NT = 4), all-remote (every R_t = 0), all-local (8 local routes per token), mixed R_t including 0}; compare out per row against master's path on the same synthetic inputs, per-row rel L2 <= 1e-2 and max printed; one expert against a local fp32 dequant matmul (e2m1 grid and e4m3 decode written in the file, not the reference); run every case 50 times back to back (epoch, ticket, flag and done reuse) and once under torch.cuda.graph capture + replay; assert tickets, done counters and exit ticket are 0 afterwards. Print LOUDLY in both outcomes: [cuda-moe-midm-fused] SELF-CHECK PASSED/FAILED, max rel-L2, numRegs and localSizeBytes (cudaFuncGetAttributes), smem, SK tables, items per phase per shape, compile seconds, PATH CHOSEN. FALLBACK: a load_inline exception, cudaFuncSetAttribute failure, failed residency assert, failed check or capture error sets state['midm_fused_ok'] = False and every M runs master unchanged; kernel_fn branches only on num_tokens and that flag. MILESTONES readable from the profile's top_kernels per shape (the single fused row vs the sum of master's replaced rows: ~114 / ~125 / ~198 us at bs 8 / 16 / 32): M1 all 5 tests pass with PATH CHOSEN = fused in the log; M2 fused row <= the replaced sum at every mid shape; M3 fused row <= 52 / 72 / 98 us (within 1.4x of the 37.3 / 52.5 / 70.8 us floors, ~3.4 TB/s), and the shape's time minus the fused row <= 2 us (no other node left).

## Expected

- `bs=8 *` faster: verify_1: incumbent 107.8 us scored = Marlin pair 105.8 + post-norm 3.5 + align 2.7 + silu, 5 launches, vs a 37.3 us floor. One launch, in-kernel routing, w2 stream prefetched across the phase boundary: target <= 52 us (M3), load ratio 1.34x -> ~2.8x
- `bs=16 *` faster: decode_16: incumbent 147.9 us scored (Marlin 116.0 + post-norm 4.2 + align 2.9 + silu) vs a 52.5 us floor; target <= 72 us (M3), load ratio 1.27x -> ~2.6x
- `bs=32 *` faster: decode_32: incumbent 200.7 us scored (Marlin 187.6 + align 4.4 + post-norm 4.0 + silu) vs a 70.8 us floor; NT=4 streams each hot expert once (Marlin restreams per 16-row block); target <= 98 us (M3), load ratio 1.03x -> ~2.1x. All three at M3: headline ~1.31x -> ~1.85x on master; ~+12% over the small-m CUDA sibling if that hits its own M3
- `bs=1 *` flat: control: bs=1 runs master (owned by cuda-fused-moe-bs1-persistent-w13-silu-w2)
- `bs=128 *` flat: control: num_tokens > 32 runs master byte for byte
- `bs=256 *` flat: control: num_tokens > 32 runs master byte for byte

## Risk

Biggest risk: correctness and liveness of a grid-wide dataflow kernel at multi-token: flag release/acquire across 64 w13 tiles per expert, per-token completion counts that span several experts, epoch bookkeeping across CUDA-graph replays, and co-residency (grid == SM count, 1 CTA/SM; the judge's L2-flush read runs before the replay on the same stream, so it cannot overlap). The 50-repeat and capture self-checks cover reuse; a failure falls back to master. Second risk, correlated with the sibling: the mainloop is the sibling's mma.sync swap-AB design, so if that file shows the mainloop is slow (its M2 fails), this one inherits it; implement it after the sibling harvests if possible, and read the sibling's numRegs, spill and per-shape rows first. Third: phase 2 of a hot expert cannot start its MMA until all 8 tiles of each 512-column h chunk it reads are done, so at bs=32 (few, hot experts) the overlap is mostly the weight prefetch (~54-72 KB per SM), and the phase boundary cost shrinks rather than vanishes; budget ~3 us. Smem: the slab at NT=4 is the sibling reviewer's shared-memory-bound case (~2.9 TB/s even conflict-free), so the bs=32 M3 target may need stage 2. Not a duplicate: neither in-flight CUDA rewrite does multi-token single-launch routing, grouping or combine; the bs=1 sibling's own follow-ups are a deeper ring and K splits, not M > 1. If this lands together with cuda-nvfp4-swapab-mma-persistent-small-m, this owns 2 <= M <= 32 and the sibling's two-launch path becomes the fallback; judge stage 1 against master (Marlin), and against the sibling's rows if it has harvested. depends_on is left empty on purpose: the bank and dequant are specified here in full by reference and can be implemented without the sibling's code.

## Milestones

Rewrite stage 1 of `_align_small + Marlin w13 + silu_and_mul + Marlin w2 + post-norm combine (the whole MoE, 5 launches) at 2 <= num_tokens <= 32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
