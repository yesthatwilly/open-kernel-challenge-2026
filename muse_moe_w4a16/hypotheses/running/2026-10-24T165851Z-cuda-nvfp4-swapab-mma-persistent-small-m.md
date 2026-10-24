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
review: proceed
status: running
review_at: 2026-10-24T170237Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T170237Z
base_sha: 663f5edaa1c1e7a6a50c7a891240deafb601a765
base_blob: d9a9375ef4721ab9b28cc9d6364c92e67c7ccc59
branch: hyp/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m
mode: null
implemented_at: 2026-10-24T171427Z
impl_commit: b65c9aabd73e6480ee59cd3164c307aece6e622d
submission_sha256: e2f8b43f8127ffe67d6835f756cc5b03009f37f999d8f4737df0189a76fee325
norm_sha256: 4d6c01495799a459fbab53472840a1cbe587cd119add48667a6d9b690d4afb5d
submission_blob: 79debc5dc69cde3f15d3087b59ed8f063b9debd9
lint_warnings: 20
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

## Adversarial review

Reviewed 2026-10-24T170237Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound stage-1 rewrite design. The fragment and nibble math, the exact bit-trick dequant, the non-spinning ticket split-K and the self-check plus fallback all hold up, and there is no rules problem. Three build fixes matter before the run: the activation slab layout (bank conflicts, and shared-memory bandwidth at NT=4), the gate/up pairing across warps in the w13 epilogue, and prepare-time warm-up of _align_small at the new BSM values.

### Justification
The mechanism is plausible. FINDINGS shows Marlin taking 87-95% of every small-M shape: 44.6 us at bs=1, 105.8 at bs=8, 116.0 at bs=16 and 187.6 at bs=32, which is roughly 1.2-2.2 TB/s against floors of 11.4/37.3/52.5/70.8 us. That leaves 2-3x of headroom on 60-weight loads. Checked details: (1) The m16n8k16 A fragment layout is right. Lane rows g and g+8 at k cols 2t+{0,1} and +8 give 8 nibbles per k16, so 4 steps fill one 16 B cp.async per (16x64) sub-tile, and 32 lanes x 16 B = 512 B is exactly that sub-tile. (2) The dequant is right and exact. (q&0x80008000)|((q&0x70007000)>>6) puts sign at bit 15 and exp2/man1 at bf16 bits 8..6, giving e2m1*2^-126 (code 0.5 becomes the subnormal 2^-127). The e4m3 scale times 2^119 stays below the bf16 max (448*2^119 < 2^128). The product has at most 6 significant bits, and bf16 mul/fma keep subnormals. The extraction order (nibbles 7-i and 3-i as hi/lo) matches the stated placement. (3) ldmatrix without .trans on a token-major, k-contiguous slab yields the col-major B fragment correctly. (4) Topk ids are distinct per token and BSM = 8*NT >= M, so no expert ever exceeds BSM routes. Every expert is exactly one m-block, the 'experts above BSM take 2 blocks' case cannot happen, and at bs=16/32 the weight restream that Marlin's 8/16-row blocks incur goes away. That is the real lever there. At bs=8 the blocking equals Marlin's (block 8), so any gain there is pure streaming efficiency. (5) The ticket reduce never spins, so a persistent grid of 132 has no co-residency or deadlock risk. (6) Smem is 88-136 KB with 1 CTA/SM, and 54 KB of weight per SM is in flight, about 1.9x the Little's-law need, so latency hiding is adequate. (7) Item balance: bs=1 has about 512 w13 and 512 w2 items over 132 CTAs (~97% last-wave efficiency), and bs=32 SK=1 has about 10 rounds. WEAKEST LINK: the shared activation slab. Its rows are 256 bf16 = 512 B apart, so every ldmatrix 8x8 hits the same 4 banks, an 8-way conflict, and the spec specifies no swizzle or pad. Even conflict-free, all 8 warps re-read the whole slab every stage, and the weight also round-trips through smem. Rough per-stage smem traffic at NT=4 is 128 KB slab reads + 16 KB weight in + 16 KB out + 16 KB slab in + 4 KB scales, about 180 KB, or ~1400 clk (~0.75 us). The DRAM time per 16 KB weight stage is ~0.45 us at 4.8 TB/s/132 SMs. So bs=32 is shared-memory bound at about 2.9 TB/s even when conflict-free, and far lower with 8-way conflicts. NT=2 is borderline and NT=1 is fine. That is still above Marlin's 1.8 TB/s, so M2 at bs=32 is reachable, but M3 (<=106 us) needs a swizzle (XOR the 16 B chunk index with row&7) or a 16 B row pad plus fewer redundant slab reads. Other minor points: the expected directions and the bs>=128 flat controls agree with TASK's weights and floors. The '<=6% partial traffic' claim holds for w13 (SK=4: 4 KB partial vs 64 KB item). For w2 at SK=8 it is ~11% written plus ~11% read, mostly L2-resident (about 2 MB at bs=1), so it is harmless but not 6%. The projected headline (~1.8x) assumes every M3 target is met; read the replaced-kernel rows, not the headline.

### Confounders on the current kernel
- triton-map-align-on-block-m-restack (with triton-map-align-small-m-port): this change feeds _align_small new BSM values (16 at M 9-16, 32 at M 17-32, versus Marlin's 8/8/16), which changes the align kernel's Triton specialization (CAP_BLOCK/MB_BLOCK/BSM) and its 2.7-4.4 us row. That is a small, attributable effect, not a confound for the replaced-kernel milestone. Read the _align_small row separately in top_kernels.
- post-expert-norm-triton-small-m: the consumer of y is unchanged and, like Marlin with is_ep, the new kernel never writes remote or sentinel rows, so behaviour is the same. Not a confound, as long as the w2 epilogue writes y at the flat route index (t*8+s) exactly where Marlin does.
- marlin-block-m-two-smaller-on-postnorm: touches only num_tokens >= 128, which this hypothesis leaves byte for byte, so it is the in-run control. No interaction.
- sglang-marlin-path-parity: this is the incumbent being replaced and the self-check's reference. Marlin's bank stays for M > 32 and for the fallback. No confound.

### Interference and A/B needs
No companions. This is kind: rewrite, judged on its own kernel rows against the Marlin pair plus act_and_mul per shape, and every bs>=128 shape runs master byte for byte as an in-run control. No landed hypothesis shares its launch sequence in a way that a revert would isolate: the align change is upstream and attributable in top_kernels, and the post-norm tail is unchanged. A sub-change control (for example the CUDA kernel at Marlin's BSM, or with silu unfused) would not change the stage-2 decision. The profile already separates the time per launch, and stage 2 is the natural follow-up.

### Open questions
- BUILD FIX: swizzle the activation slab (XOR the 16 B chunk index with token&7) or pad rows by 16 B. As written, ldmatrix on rows 512 B apart is an 8-way bank conflict on every B-fragment load. Also consider having warps share fewer slab reads (for example 4 warps x 32 weight rows), because at NT=4 the slab re-read by 8 warps makes the kernel shared-memory bound at about 2.9 TB/s.
- BUILD FIX: the w13 tile puts gate rows [64j,64j+64) in warps 0-3 and up rows in warps 4-7, so silu(gate)*up at SK=1 (bs=32) needs a smem exchange the spec does not describe. Either interleave gate/up at 16-row granularity per warp (warp w gets gate 8 rows + up 8 rows, so one thread holds a matching pair), or stage the accumulators through smem. At SK>1 the last arriver reads all partials from global anyway.
- BUILD FIX: prepare() warms _align_small only at Marlin's block_size_m (8/8/16). Add warm-up at the new BSM (16 for M 9-16, 32 for M 17-32), and run _align_self_check at those BSM values too, otherwise Triton JIT-compiles on the first timed call or capture.
- 'slot' is used for two things. The gather and epilogue index must be the route id = sorted_ids[pos] (x row = route/8, cache2/cache3 row = route), not the position pos in the sorted list. Sentinel means route >= M*8.
- The spec says a spill is both in the FALLBACK list and 'printed, not fatal'. Pick one; not fatal is the sensible choice.
- Use separate ticket arrays (or offsets) for the w13 and w2 launches, and assert both are zero after the self-check. Call cudaFuncSetAttribute(MaxDynamicSharedMemorySize) in prepare for all 6 instantiations before any capture.
- Self-check coverage: add M=2 and M=12/24 (non-power-of-two M inside an NT class) to exercise partially filled slabs, and one case with local_start != 0 if the judge's bank can start elsewhere (prepare hard-codes (0, 32), so that only matters if the config varies).
- Harvest note (as for splitkv): read the log tail first. 'PATH CHOSEN = MARLIN' or 'SELF-CHECK FAILED' means the run measured master, not the kernel.


## Implementation notes

Implemented 2026-10-24 in the worktree's `muse_moe_w4a16/submission.py` only (uncommitted; diff stat `1 file changed, 763 insertions(+), 54 deletions(-)`).

- **CUDA source** (`_CUDA_CPP` / `_CUDA_SRC`, built by `_build_cuda()` with `load_inline`, `-O3 -arch=sm_90a`, no fast-math, compile seconds printed). One TU, one template `w4a16_kernel<NT, GATED>` with 6 instantiations (NT 1/2/4 x w13+silu / w2). `w4a16_init()` runs `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` + `cudaFuncGetAttributes` for all 6 in prepare (before any capture) and returns ok / numRegs / localSizeBytes / smem. `w4a16_gemm()` is the launcher.
- **Kernel** as specified: 256 threads, 8 warps x 16 weight rows, BN 128, BK 256, 4 stages, 1 CTA/SM, `launch_bounds(256, 1)`. Smem per stage is 16 KB codes + 2 KB scales + BSM x 512 B slab, so 88 / 104 / 136 KB for NT 1 / 2 / 4. Persistent grid = SM count. Items are (b, j, s) with b slowest. `n_items` comes from `n_padded` on device, clamped to `blocks_cap = 32`. One flat (item, stage) cp.async ring, issuing 3 stages ahead across item boundaries, with one `__syncthreads` per stage and `wait_group(2)`. The producer prefetches the next item's expert and route ids into registers one item early. Weights and scales use `cp.async.cg` 16 B with `L2::cache_hint` evict_first. Activations use `cp.async.ca` 16 B and zero-fill sentinel or e<0 rows. Dequant is the exact bit trick, then `fma.rn.bf16x2` with -0 (explicit, so subnormals are kept) by `bf16(e4m3 * 2^119)`. mma is `m16n8k16.row.col.f32.bf16.bf16.f32`. Split-K uses partials stored with `__stcg`, then `__threadfence`, then an `atomicAdd` ticket by thread 0. The last arriver resets the ticket to 0, sums the SK partials in fixed order with `__ldcg` and runs the epilogue. SK tables: w13 {1:4, 2:2, 4:1}, w2 {1:8, 2:4, 4:2}. Epilogues: w13 writes bf16(silu(gate*gs)*(up*gs)) to `cache2[route, n0]`; w2 writes bf16(acc*gs) to `cache3[route, o]`, at the flat route index. Remote and sentinel slots are never written.
- **Review BUILD FIXES applied:** (1) XOR swizzle: slab chunk c of row r is stored at `c ^ (r & 7)`, and ldmatrix x4 reads logical chunks with the same XOR, so it is conflict-free. Shared-memory bandwidth at NT=4 (8 warps re-reading the slab) is NOT addressed; that is stage-2 material per the review. (2) The w13 rows are interleaved **per warp**: warp w row g is gate column 64j+8w+g and row g+8 is the matching up column. This replaces the spec's "gate rows then up rows per 128-row tile", so c0/c2 (and c1/c3) of one thread are a gate/up pair and need no exchange. (3) `_cuda_setup` warm-compiles `_align_small` for tokens 1..32 at BSM 8/16/32 and runs `_align_self_check(..., bsm_of=_cuda_bsm)`. Any hard mismatch disables the CUDA path. (4) All gathers and epilogues index by route id = `sorted_ids[b*BSM + p]`: x row = route >> 3, cache2/cache3 row = route, sentinel = route >= M*8. (5) A spill (`localSizeBytes > 0`) is printed, not fatal. (6) There are separate ticket buffers `cuda_tk13` [32*64] and `cuda_tk2` [32*32], both asserted zero after the self-check. One fp32 partial workspace (32 MiB, the max over NT/SK) is shared by the two stream-ordered launches. (7) The log prints `PATH CHOSEN = CUDA|MARLIN`, per-M self-check results, regs and spills per instantiation, smem, grid and the SK table.
- **Weight retiling** (`_tile_codes` / `_tile_scales`, `_cuda_bank`): a pure view/permute/flip of the same nibbles and E4M3 bytes per expert. Global scales stay fp32 with gs x 128 folded in (exact). Each expert's round trip is asserted to equal the source bytes exactly. Extra bank memory is about 0.84 GiB; the Marlin bank is kept.
- **Self-check** (`_cuda_self_check`) uses a fixed CPU generator and synthetic inputs only. M is in (1, 2, 4, 8, 12, 16, 24, 32), which adds the review's M=2/12/24 partial-slab cases. The cases are random, skewed, all-one-expert, all-remote and all-local. Each compares cache2 and y per local route, and out per row, against master's Marlin path, with a bound of 1e-2 per-row rel-L2, finite outputs and equal local_ids. At M in (1, 8, 16, 32) the all-one-expert case is also checked against a local fp32 NVFP4 decode (it does not import the reference).
- **Refactor:** kernel_fn's body moved into `_moe(x, topk_ids, topk_weights, bank, state, use_cuda)` so the self-check can run both paths and see the intermediates. The `use_cuda=False` branch is master's code verbatim, the same launches in the same order, so num_tokens > 32 and the fallback are byte for byte. kernel_fn branches only on num_tokens <= 32 and the prepare-time `cuda_ok` flag (plus the bank key).
- **Offline verification (no GPU here):** a numpy emulation ran the actual retiling ops, the kernel's per-lane word/nibble extraction with the bit trick, scale bytes, slab swizzle with ldmatrix x4 semantics, mma m16n8k16 fragment definitions, split-K summation and the epilogue row/token mapping. It covered w2 NT1 SK2, w13 NT1 SK1, w13 NT2 and w2 NT4 SK2 with partial m-blocks and sentinels, against a dense dequant matmul. Max rel err was about 1e-15 in every config, which confirms both the indexing and the exactness. Not verifiable offline: PTX syntax and compile, the `cvt.e4m3x2` byte order (taken from the muse_attention_decode precedent), and performance.
- **Deviations:** the gate/up interleave is per warp, not per 128-row half, per review fix (2). The spill is non-fatal per the review. `blocks_cap = 32` local experts is safe because BSM >= M and topk ids are distinct, so each expert takes at most one m-block; the clamp is a guard. No other new ideas were added.

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/cuda-nvfp4-swapab-mma-persistent-small-m diff`:

- [x] CUDA-graph capturable: kernel_fn → `_moe` (CUDA branch) has no `.item()`/`.cpu()`/`synchronize()`/tensor prints. Allocations are `torch.empty` sized from shapes only. The two launches use the current stream. Item counts come from `n_padded` **on device**, with no host branch on values. `cudaFuncSetAttribute` runs only in prepare.
- [x] Nothing derived from `data` is cached across calls. State holds only prepare-time weights, the workspace, tickets (reset to 0 by the kernel itself) and the flag. Partials and tickets are per-launch scratch.
- [x] Weights are repacked losslessly in `prepare()` only. NVFP4 codes and E4M3 scales are the same bytes, permuted, with the round trip asserted exactly. Global scales stay fp32 (x 2^7, exact). Dequant happens in registers, exactly.
- [x] Activations are BF16 where SGLang keeps BF16. x, cache2 and y are bf16 and the fp32 accumulate matches Marlin's fp32 reduce. KV cache / queries: n/a (not attention).
- [x] Attention sees every key: n/a (MoE problem, no attention).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses (only torch's own `load_inline` build, as in the muse_attention_decode precedent) or network. The fp32 self-check decode is written locally.
- [x] All work happens inside the timed call, with no output caching or replay. The self-check in prepare uses only synthetic inputs from a fixed seed.
- [x] First line is still `# okc: problem=muse_moe_w4a16`. There is no `triton.autotune` and no host syncs in kernel_fn.

okc_lint output:

```
warn  host-sync    L216: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L218: [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:5].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L220: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L239: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L244: s_ids, b_exp, n_pad = s_ids.tolist(), b_exp.tolist(), int(n_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L245: r_ids, r_exp, r_pad = r_ids.tolist(), r_exp.tolist(), int(r_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L781: q13 = torch.empty((count, w13[0].numel()), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L782: s13 = torch.empty((count, weights["w13_sf"][0].numel()), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L783: q2 = torch.empty((count, w2[0].numel()), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
warn  dtype        L784: s2 = torch.empty((count, weights["w2_sf"][0].numel()), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
warn  host-sync    L832: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L834: [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:5].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L837: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L849: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L899: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L917: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L931: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L991: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L993: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1004: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .../cuda-nvfp4-swapab-mma-persistent-small-m/muse_moe_w4a16/submission.py
```

The dtype warnings are the uint8 byte buffers that hold the retiled NVFP4 codes and E4M3 scales (prepare, lossless); no activation is cast. `python3 -I -m py_compile` passed and the `__pycache__` was removed.
