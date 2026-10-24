---
problem: muse_moe_w4a16
slug: cuda-nvfp4-swapab-mma-persistent-small-m
created: 2026-10-24T165851Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Small-M only: num_tokens <= 32 (decode_1 bs=1, verify_1 bs=8, decode_16, decode_32 and the
  bs=1/bs=4 tests); num_tokens > 32 (bs=33, 128, 256, 257 tests and verify_16/32) keeps master
  byte for byte. Replace the two moe_wna16_marlin_gemm calls AND silu_and_mul with two launches of
  one hand-written CUDA kernel template, compiled in prepare() with
  torch.utils.cpp_extension.load_inline (one translation unit, extra_cuda_cflags -O3 -arch=sm_90a,
  no fast-math; model on muse_attention_decode _build_splitkv, which compiled in 41 s; print the
  compile seconds). master _align_small and the post-norm tail stay as they are
  (_post_norm_combine_kernel below 32 tokens, fused_post_expert_norm_weighted_sum at 32), and the
  expert_map gather/align still feeds sorted_ids, block_experts, n_padded, but with block_size_m =
  BSM = 8*NT where NT = 1 for M <= 8, 2 for M <= 16, 4 for M <= 32 (powers of 2, so _align_small
  compiles; one m-block then holds every route of an expert at bs <= 8/16/32 up to BSM routes, so
  a hot expert streams its weight once instead of once per 8-row Marlin block; experts above BSM
  routes just take 2 blocks). WEIGHT LAYOUT (prepare, lossless, NVFP4 stays NVFP4): from weights
  w13/w13_sf/w2/w2_sf sliced to the bank [start, start+count), build per-expert tiled banks. (a)
  w13 rows are interleaved so each 128-row N-tile j holds gate rows [64j, 64j+64) then up rows
  [I+64j, I+64j+64), so one CTA tile owns matching gate and up columns and can apply silu in its
  epilogue. (b) Codes are permuted so each lane fetches its mma.sync m16n8k16 A fragment (weight
  as operand A = swap-AB: 16 weight rows on M, tokens on N=8) with one 16-byte cp.async per (16
  rows x 64 k) sub-tile: for k16 step s the lane needs rows lane/4 and lane/4+8 at k cols
  2*(lane%4)+{0,1} and +8, i.e. 8 nibbles = one uint32, and 4 steps make one uint4; within each
  uint32 place register-pair j (a0a1, a2a3, a4a5, a6a7) at nibbles (3-j) and (7-j) so the
  extraction loop below yields them in order. (c) Block scales stay e4m3 bytes, permuted per (16
  rows x 64 k) sub-tile into 8 contiguous bytes per lane group (rows lane/4, lane/4+8 x 4 k16
  blocks; 64 B per warp sub-tile). Keep global scales fp32 and fold the dequant bias into them:
  gs_eff = gs * 2^7. Build the permutation with torch indexing and assert a round-trip un-permute
  equals the source bytes exactly. DEQUANT (registers, Marlin bit trick, exact): q = 8 codes; for
  i in 0..3: v_i = (q & 0x80008000) | ((q & 0x70007000) >> 6); q <<= 4. As bf16x2 bits this is
  sign | code3 << 6, i.e. e2m1 * 2^-126 (the 0.5 code is a bf16 subnormal; fma.rn.bf16x2 supports
  subnormals). Scale: cvt.rn.f16x2.e4m3x2 the two row scales -> float2 -> times 2^119 (exact) ->
  __floats2bfloat162_rn (exact, at most 4 significant bits, < 2^128) -> one __hmul2 per register:
  e2m1 * sf * 2^-7, exact in bf16 (at most 6 significant bits). About 1.5 ALU ops per weight,
  versus ~15 in the Triton GEMVs that lost 2x. Accumulate in fp32 via
  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32; the epilogue multiplies by gs_eff. TILES:
  CTA = 256 threads = 8 warps; CTA tile BN = 128 weight rows (16 per warp, warp-private weight
  slices) x BSM tokens (NT n8 fragments, accumulators 4*NT fp32 per thread); BK = 256 per pipeline
  stage. Per stage smem: weight codes 128 x 128 B = 16 KB + scales 2 KB + activation slab BSM x
  256 bf16 (4 KB per n8 fragment, gathered rows: w13 B rows are x[sorted_ids[slot] / 8], w2 B rows
  are cache2[slot]; sentinel slots >= M*8 load zeros), STAGES = 4 -> 88 KB (NT=1) to 136 KB (NT=4)
  dynamic smem, 1 CTA per SM, launch_bounds(256, 1). B fragments come from the slab with
  ldmatrix.x4 (token-major, k-contiguous rows are mma col-major B, no .trans). Bytes in flight: 3
  stages ahead x 18 KB = 54 KB of weight per SM, ~7.1 MB grid-wide, ~1.9x Little-law need (4.8
  TB/s x ~0.8 us). Weights are loaded with cp.async.cg.shared.global 16 B + an L2 evict_first
  cache hint; activations cp.async.ca (L2-resident after first touch). PERSISTENT GRID AND
  PIPELINE ACROSS ITEMS: grid = number of SMs (132), fixed at capture. Work items are (m-block b,
  N-tile j, k-split s), enumerated with b slowest; the kernel reads n_padded on device, n_items =
  (n_padded / BSM) * n_tiles * SK, so only experts with routed tokens are touched and no host sync
  is needed; blocks whose block_experts is -1 are skipped (is_ep semantics, as Marlin). CTA c
  walks items c, c+132, ... as ONE flat sequence of (item, k-stage) steps: the producer issues
  stage t+3 loads (possibly the next item) before consuming stage t, so the ring never drains at
  item boundaries (the earlier Triton split-K GEMV profile showed shared=8192 B, i.e. no
  pipelining; this is the fix). One __syncthreads per stage (the slab is shared),
  cp.async.wait_group(STAGES-2). SPLIT-K (host constants by shape, no autotune): w13 (N = 8192 ->
  64 tiles per expert, K = 4096): SK = 4 at M <= 8, 2 at M = 16, 1 at M = 32; w2 (N = 4096 -> 32
  tiles, K = 4096): SK = 8 / 4 / 2. Partials: fp32 [SK, items_cap, 128, BSM] workspace allocated
  in prepare at the largest M per NT; per-tile int32 tickets (zeroed in prepare). Reduce = the
  measured ticket pattern: store partial, __threadfence, one atomicAdd(ticket) by thread 0,
  __syncthreads; the last arriver sums the SK partials in fixed split order with ld.global.cg
  (deterministic), resets its ticket to 0, and runs the epilogue. SK = 1 runs the epilogue
  directly. EPILOGUES: w13: acc * gs_eff13[e] -> silu(gate) * up in fp32 -> bf16 cache2[slot,
  64j..64j+64) (the silu_and_mul launch and cache1 disappear; sentinel slots not written). w2: acc
  * gs_eff2[e] -> bf16 y[slot, H] into cache3 exactly where master puts it, so the post-norm tail
  and its local_ids masking are unchanged (remote and sentinel slots never written; the tail never
  reads them). CORRECTNESS SELF-CHECK (prepare, untimed, nothing input-derived): with a fixed
  torch.Generator, build synthetic activations x and routings for M in (1, 4, 8, 16, 32) x
  {random, skewed (3 local slots on the same experts every token), all-one-expert (M routes to one
  expert, forces BSM-full blocks), all-remote, all-local} over the real bank weights, run the new
  path and master Marlin path for that M, and compare cache2 and y per row and the final out:
  per-row rel L2 vs Marlin must be <= 1e-2 and the max over cases is printed. Also compare one
  expert against an fp32 dequant (nvfp4 decode written locally from the e2m1 grid and e4m3 scales,
  not importing the reference) to catch errors Marlin would share. Print LOUDLY in both outcomes:
  [cuda-w4a16] SELF-CHECK PASSED/FAILED, max rel-L2, regs/spills (cuobjdump not needed: print
  cudaFuncGetAttributes numRegs and localSizeBytes), smem per NT, SK table, grid, and PATH CHOSEN.
  After the check, assert every ticket is back to 0. FALLBACK: any load_inline exception, a failed
  self-check, numRegs spill (localSizeBytes > 0 is printed, not fatal), or a cudaFuncSetAttribute
  smem failure sets state[cuda_ok] = False and every M runs master (Marlin) unchanged. kernel_fn
  branches only on num_tokens and that prepare-time flag.
expected:
  - shapes: bs=1 *
    direction: faster
    why: incumbent Marlin w13+w2 44.6 us + silu 1.6 us for ~2 local experts (~55 MB, floor 11.4 us, ~1.2 TB/s). Two persistent launches at >=3 TB/s target <= 22 us for the replaced kernels (M2 <= 44, M3 <= 18)
  - shapes: bs=8 *
    direction: faster
    why: incumbent Marlin 105.8 us (~1.7 TB/s) vs floor 37.3 us; BSM=8 covers a whole expert per m-block; target <= 60 us (M3 <= 56)
  - shapes: bs=16 *
    direction: faster
    why: "incumbent Marlin 116.0 us (~2.2 TB/s) vs floor 52.5 us; target <= 80 us (M3 <= 79): smaller margin, Marlin is already closest to the floor here"
  - shapes: bs=32 *
    direction: faster
    why: incumbent Marlin 187.6 us (~1.8 TB/s, hot experts split into 8-row Marlin blocks restream their weight) vs floor 70.8 us; BSM=32 streams each expert once; target <= 110 us (M3 <= 106)
  - shapes: bs=128 *
    direction: flat
    why: "control: num_tokens > 32 runs master Marlin byte for byte"
  - shapes: bs=256 *
    direction: flat
    why: "control: num_tokens > 32 runs master Marlin byte for byte"
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at bs<=32
review: required
status: queued
---

# cuda-nvfp4-swapab-mma-persistent-small-m

## Change

Small-M only: num_tokens <= 32 (decode_1 bs=1, verify_1 bs=8, decode_16, decode_32 and the bs=1/bs=4 tests); num_tokens > 32 (bs=33, 128, 256, 257 tests and verify_16/32) keeps master byte for byte. Replace the two moe_wna16_marlin_gemm calls AND silu_and_mul with two launches of one hand-written CUDA kernel template, compiled in prepare() with torch.utils.cpp_extension.load_inline (one translation unit, extra_cuda_cflags -O3 -arch=sm_90a, no fast-math; model on muse_attention_decode _build_splitkv, which compiled in 41 s; print the compile seconds). master _align_small and the post-norm tail stay as they are (_post_norm_combine_kernel below 32 tokens, fused_post_expert_norm_weighted_sum at 32), and the expert_map gather/align still feeds sorted_ids, block_experts, n_padded, but with block_size_m = BSM = 8*NT where NT = 1 for M <= 8, 2 for M <= 16, 4 for M <= 32 (powers of 2, so _align_small compiles; one m-block then holds every route of an expert at bs <= 8/16/32 up to BSM routes, so a hot expert streams its weight once instead of once per 8-row Marlin block; experts above BSM routes just take 2 blocks).
WEIGHT LAYOUT (prepare, lossless, NVFP4 stays NVFP4): from weights w13/w13_sf/w2/w2_sf sliced to the bank [start, start+count), build per-expert tiled banks. (a) w13 rows are interleaved so each 128-row N-tile j holds gate rows [64j, 64j+64) then up rows [I+64j, I+64j+64), so one CTA tile owns matching gate and up columns and can apply silu in its epilogue. (b) Codes are permuted so each lane fetches its mma.sync m16n8k16 A fragment (weight as operand A = swap-AB: 16 weight rows on M, tokens on N=8) with one 16-byte cp.async per (16 rows x 64 k) sub-tile: for k16 step s the lane needs rows lane/4 and lane/4+8 at k cols 2*(lane%4)+{0,1} and +8, i.e. 8 nibbles = one uint32, and 4 steps make one uint4; within each uint32 place register-pair j (a0a1, a2a3, a4a5, a6a7) at nibbles (3-j) and (7-j) so the extraction loop below yields them in order. (c) Block scales stay e4m3 bytes, permuted per (16 rows x 64 k) sub-tile into 8 contiguous bytes per lane group (rows lane/4, lane/4+8 x 4 k16 blocks; 64 B per warp sub-tile). Keep global scales fp32 and fold the dequant bias into them: gs_eff = gs * 2^7. Build the permutation with torch indexing and assert a round-trip un-permute equals the source bytes exactly.
DEQUANT (registers, Marlin bit trick, exact): q = 8 codes; for i in 0..3: v_i = (q & 0x80008000) | ((q & 0x70007000) >> 6); q <<= 4. As bf16x2 bits this is sign | code3 << 6, i.e. e2m1 * 2^-126 (the 0.5 code is a bf16 subnormal; fma.rn.bf16x2 supports subnormals). Scale: cvt.rn.f16x2.e4m3x2 the two row scales -> float2 -> times 2^119 (exact) -> __floats2bfloat162_rn (exact, at most 4 significant bits, < 2^128) -> one __hmul2 per register: e2m1 * sf * 2^-7, exact in bf16 (at most 6 significant bits). About 1.5 ALU ops per weight, versus ~15 in the Triton GEMVs that lost 2x. Accumulate in fp32 via mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32; the epilogue multiplies by gs_eff.
TILES: CTA = 256 threads = 8 warps; CTA tile BN = 128 weight rows (16 per warp, warp-private weight slices) x BSM tokens (NT n8 fragments, accumulators 4*NT fp32 per thread); BK = 256 per pipeline stage. Per stage smem: weight codes 128 x 128 B = 16 KB + scales 2 KB + activation slab BSM x 256 bf16 (4 KB per n8 fragment, gathered rows: w13 B rows are x[sorted_ids[slot] / 8], w2 B rows are cache2[slot]; sentinel slots >= M*8 load zeros), STAGES = 4 -> 88 KB (NT=1) to 136 KB (NT=4) dynamic smem, 1 CTA per SM, launch_bounds(256, 1). B fragments come from the slab with ldmatrix.x4 (token-major, k-contiguous rows are mma col-major B, no .trans). Bytes in flight: 3 stages ahead x 18 KB = 54 KB of weight per SM, ~7.1 MB grid-wide, ~1.9x Little-law need (4.8 TB/s x ~0.8 us). Weights are loaded with cp.async.cg.shared.global 16 B + an L2 evict_first cache hint; activations cp.async.ca (L2-resident after first touch).
PERSISTENT GRID AND PIPELINE ACROSS ITEMS: grid = number of SMs (132), fixed at capture. Work items are (m-block b, N-tile j, k-split s), enumerated with b slowest; the kernel reads n_padded on device, n_items = (n_padded / BSM) * n_tiles * SK, so only experts with routed tokens are touched and no host sync is needed; blocks whose block_experts is -1 are skipped (is_ep semantics, as Marlin). CTA c walks items c, c+132, ... as ONE flat sequence of (item, k-stage) steps: the producer issues stage t+3 loads (possibly the next item) before consuming stage t, so the ring never drains at item boundaries (the earlier Triton split-K GEMV profile showed shared=8192 B, i.e. no pipelining; this is the fix). One __syncthreads per stage (the slab is shared), cp.async.wait_group(STAGES-2).
SPLIT-K (host constants by shape, no autotune): w13 (N = 8192 -> 64 tiles per expert, K = 4096): SK = 4 at M <= 8, 2 at M = 16, 1 at M = 32; w2 (N = 4096 -> 32 tiles, K = 4096): SK = 8 / 4 / 2. Partials: fp32 [SK, items_cap, 128, BSM] workspace allocated in prepare at the largest M per NT; per-tile int32 tickets (zeroed in prepare). Reduce = the measured ticket pattern: store partial, __threadfence, one atomicAdd(ticket) by thread 0, __syncthreads; the last arriver sums the SK partials in fixed split order with ld.global.cg (deterministic), resets its ticket to 0, and runs the epilogue. SK = 1 runs the epilogue directly.
EPILOGUES: w13: acc * gs_eff13[e] -> silu(gate) * up in fp32 -> bf16 cache2[slot, 64j..64j+64) (the silu_and_mul launch and cache1 disappear; sentinel slots not written). w2: acc * gs_eff2[e] -> bf16 y[slot, H] into cache3 exactly where master puts it, so the post-norm tail and its local_ids masking are unchanged (remote and sentinel slots never written; the tail never reads them).
CORRECTNESS SELF-CHECK (prepare, untimed, nothing input-derived): with a fixed torch.Generator, build synthetic activations x and routings for M in (1, 4, 8, 16, 32) x {random, skewed (3 local slots on the same experts every token), all-one-expert (M routes to one expert, forces BSM-full blocks), all-remote, all-local} over the real bank weights, run the new path and master Marlin path for that M, and compare cache2 and y per row and the final out: per-row rel L2 vs Marlin must be <= 1e-2 and the max over cases is printed. Also compare one expert against an fp32 dequant (nvfp4 decode written locally from the e2m1 grid and e4m3 scales, not importing the reference) to catch errors Marlin would share. Print LOUDLY in both outcomes: [cuda-w4a16] SELF-CHECK PASSED/FAILED, max rel-L2, regs/spills (cuobjdump not needed: print cudaFuncGetAttributes numRegs and localSizeBytes), smem per NT, SK table, grid, and PATH CHOSEN. After the check, assert every ticket is back to 0.
FALLBACK: any load_inline exception, a failed self-check, numRegs spill (localSizeBytes > 0 is printed, not fatal), or a cudaFuncSetAttribute smem failure sets state[cuda_ok] = False and every M runs master (Marlin) unchanged. kernel_fn branches only on num_tokens and that prepare-time flag.

## Expected

- `bs=1 *` faster: incumbent Marlin w13+w2 44.6 us + silu 1.6 us for ~2 local experts (~55 MB, floor 11.4 us, ~1.2 TB/s). Two persistent launches at >=3 TB/s target <= 22 us for the replaced kernels (M2 <= 44, M3 <= 18)
- `bs=8 *` faster: incumbent Marlin 105.8 us (~1.7 TB/s) vs floor 37.3 us; BSM=8 covers a whole expert per m-block; target <= 60 us (M3 <= 56)
- `bs=16 *` faster: incumbent Marlin 116.0 us (~2.2 TB/s) vs floor 52.5 us; target <= 80 us (M3 <= 79): smaller margin, Marlin is already closest to the floor here
- `bs=32 *` faster: incumbent Marlin 187.6 us (~1.8 TB/s, hot experts split into 8-row Marlin blocks restream their weight) vs floor 70.8 us; BSM=32 streams each expert once; target <= 110 us (M3 <= 106)
- `bs=128 *` flat: control: num_tokens > 32 runs master Marlin byte for byte
- `bs=256 *` flat: control: num_tokens > 32 runs master Marlin byte for byte

## Risk

Why CUDA and not a third Triton try: both Triton small-M W4A16 GEMVs (triton-w13-grouped-gemv-small-m 0.751x, small-m-grouped-gemv-splitk-rewrite 0.664x) lost ~2x to Marlin; the first was dequant-ALU-bound (~15 ops/weight) on an idle grid, the second compiled with shared=8192 B (no software pipeline across its 2-4 step per-item K loops). This design fixes both explicitly: ~1.5 ops/weight exact bit-trick dequant feeding mma.sync with the weight as A, and a flat cross-item cp.async ring with 54 KB per SM in flight. In-repo precedent: muse_attention_decode full-decode-cuda-gather-splitkv (load_inline, self-check, fallback) gained where Triton capped at 2.1 TB/s. If every target is met the headline is roughly 1.31 x (2.0 x 1.74 x 1.32 x 1.63)^(1/6) ~ 1.8x. Risks: load_inline compile time inside the 15 min budget (keep one TU, 3 NT x 2 kinds instantiations); fragment/nibble permutation bugs (the self-check vs Marlin and vs an fp32 dequant catches them, and the round-trip assert catches lossy repacks); bank memory +~0.9 GiB for the second layout (Marlin bank stays for M > 32); partial-buffer traffic at SK > 1 is <= 6% of weight bytes at NT=1; unbalanced items when one expert dominates at bs=32 are fine because items are equal-byte and ordered per expert. Milestones on the replaced kernels (new kernel rows vs Marlin pair + act_and_mul in top_kernels): M1 all 5 tests pass with PATH CHOSEN = CUDA in the log; M2 at or below Marlin per shape (46.2 / 107.4 / 117.6 / 189.2 us at bs 1/8/16/32); M3 >= 3 TB/s effective = within 1.6x of the floor (18 / 60 / 84 / 113 us). Stage 2 ideas if M2 fails: register-direct ld.global.nc.v4 weight loads (skip smem for the warp-private weight), BK 512, 2 CTAs per SM at NT=1, or TMA bulk copies.

## Milestones

Rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at bs<=32`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
