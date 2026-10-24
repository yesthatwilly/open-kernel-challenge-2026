---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-grouped-verify
created: 2026-10-24T172038Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Verify shapes only: num_tokens >= 128 (verify_16 bs=128, verify_32 bs=256, and the bs=128-skewed
  and bs=257 tests). num_tokens < 128 (incl. the bs=33 test) keeps master byte for byte. Replace
  the two moe_wna16_marlin_gemm calls AND silu_and_mul at num_tokens >= 128 with a hand-written
  Hopper CUDA grouped W4A16 GEMM that uses wgmma with the dequantized NVFP4 weight as a REGISTER A
  operand (wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16, A from registers, B from shared
  memory: the swap-AB form, weight rows on M, tokens on N), compiled in prepare() with
  torch.utils.cpp_extension.load_inline (one TU, extra_cuda_cflags -O3 -arch=sm_90a, no fast-math;
  pattern from muse_attention_decode _build_splitkv; print compile seconds). WHY THIS AND NOT THE
  HELD TRITON PORT OR THE SMALL-M CUDA KERNEL: master's bs=128/256 time is Marlin at 316.1 / 490.2
  us (94-95% of the shape) against floors of 101.3 / 108.9 us (~1.5 / ~1.1 TB/s effective). Marlin
  is an sm80 mma.sync kernel; at bs=256 the busiest GPU's ~350-450 local routes need ~70-90 us of
  BF16 MMA even at the 989 TF wgmma peak, so an mma.sync engine (~60% of that) plus dequant cannot
  reach the floor there: the tensor-core path has to be wgmma. The sibling
  cuda-nvfp4-swapab-mma-persistent-small-m is mma.sync with N=8..32 and is gated to bs<=32; the
  held Triton swapab-nvfp4-gemm-verify-rewrite reviewer measured its tile at ~25 us vs Marlin's
  11.6-16.7 us per block (Triton dequant cost). This kernel keeps the sibling's exact ~1 op/weight
  dequant and adds wgmma, TMA/bulk-copy producers and warp specialization. STAGES PLANNED: stage 1
  (this file) = two launches (w13 with silu fused in registers, w2) + one tiny gather kernel,
  master align and master post-norm kept; stage 2 = tuning from the stage-1 profile (N=256 bucket
  with setmaxnreg 240, 2-consumer ping-pong so one warpgroup's dequant overlaps the other's wgmma,
  L2-ordered items, split-K for w2 at bs=128); stage 3 = fold the gather into the w13 producer and
  the post-norm sum-of-squares into the w2 epilogue, and an in-kernel align (one launch fewer
  each). STAGE-1 SCOPE IN DETAIL. ALIGN: keep master's expert_map gather for local_ids, then
  sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, 16, bank count 32) for
  BOTH bs=128 and bs=256 (block 16 is the unit; items merge blocks, below). GATHER (new small CUDA
  kernel in the same TU, or Triton): xs[slot, :] = x[sorted_ids[slot] / 8, :] for slot < n_padded
  (read on device), zero rows for sentinel slots (sorted_ids >= M*8); xs is a prepare-allocated
  bf16 scratch [CAP_ROWS = 257*8 + 32*15 rounded up to 128, 4096] (~21 MB, scratch overwritten
  every call, nothing input-derived kept). WEIGHT BANK (prepare, lossless, NVFP4 stays NVFP4):
  from w13/w13_sf/w2/w2_sf sliced to the bank [start, start+count): (a) w13 rows interleaved at
  8-row granularity so that in every 16-row MMA group rows 0-7 are gate rows [c, c+8) and rows
  8-15 are up rows [I+c, I+c+8): the m16 A fragment gives each thread rows lane/4 and lane/4+8, so
  each thread holds gate and up of the same intermediate column and SiLU*up needs no shared-memory
  exchange (128-row tile j covers intermediate cols [64j, 64j+64)); (b) codes permuted to the mma
  m16n8k16 A-fragment nibble order defined in cuda-nvfp4-swapab-mma-persistent-small-m (the wgmma
  m64 register-A fragment is four stacked m16 A fragments, one per warp of the warpgroup, same
  per-thread layout), stored tile-contiguous per (expert, 128-row tile, BK k-block) so one stage
  is ONE contiguous byte range; (c) e4m3 block scales permuted the same way and stored contiguous
  right after their codes in the same stage record; gs_eff = gs * 2^7 fp32. Build with torch
  indexing; assert a round-trip un-permute equals the source bytes exactly for every expert. If
  the sibling has harvested with SELF-CHECK PASSED, reuse its permutation and dequant device
  functions verbatim (then only the 8-row gate/up interleave is new); otherwise implement them as
  specified there. DEQUANT (exact, as the sibling): q = 8 codes; v_i = (q & 0x80008000) | ((q &
  0x70007000) >> 6), q <<= 4; times bf16x2(e4m3 scale via cvt.rn.f16x2.e4m3x2 x 2^119) with
  fma.rn.bf16x2 and a -0 addend; fp32 accumulation; epilogue times gs_eff. KERNEL (one template,
  two instantiation families w13/w2): CTA = 384 threads = 3 warpgroups, launch_bounds(384, 1),
  grid = SM count (132), persistent. WG0 = producer (setmaxnreg.dec 40): one elected thread issues
  per stage (i) cp.async.bulk.shared::cluster.global.mbarrier (1-D bulk copy, no tensor map) of
  the stage's weight record (128 rows x BK codes + scales) with an L2 evict_first policy, and (ii)
  a 2-D TMA (cp.async.bulk.tensor.2d, tensor map encoded in prepare over the persistent xs scratch
  for w13 and over the persistent h_sorted scratch for w2, SWIZZLE_128B, box N x 64 bf16) for the
  token slab, both completing on the stage's full mbarrier. WG1/WG2 = consumers (setmaxnreg.inc
  232), each owns 64 of the tile's 128 weight rows: wait full[s], ld.shared.v4 its codes+scales
  (16 B per lane per 16x64 sub-tile), dequant into A registers, wgmma (B descriptor on the
  swizzled slab), wgmma.commit_group, wgmma.wait_group(1), then arrive on empty[s] for the
  PREVIOUS stage. Token-N buckets N in {16, 32, 64, 128} (template parameter); stage depth by N so
  smem <= 200 KB and >= 48 KB of weight is in flight per SM: N<=32: BK=256, 6 stages (18 KB weight
  + N*512 B slab per stage); N=64: BK=256, 5 stages; N=128: BK=128, 6 stages (9 KB weight + 32 KB
  slab). WORK ITEMS, built on device (no host sync): every CTA loads block_experts (<= CAP_ROWS/16
  entries) and n_padded into smem at entry and forms CHUNKS = maximal runs of consecutive 16-row
  blocks with the same expert, cut every 8 blocks (<= 128 rows), deterministic and identical in
  every CTA; chunk N bucket = smallest of {16,32,64,128} >= its rows (rows past the chunk belong
  to the next expert or are OOB-zero and are masked at store). Items = (chunk c, 128-row tile j,
  k-split s) ordered (expert, j, chunk, s) so the chunks of one hot expert read the same weight
  tile back to back (the second read hits L2). w13: 64 tiles per expert (8192 rows), SK=1; w2: 32
  tiles per expert (4096 rows), SK=2 at bs=128 and SK=1 at bs>=256 (host constants by shape);
  split partials via the measured ticket pattern (store fp32 partial with __stcg, __threadfence,
  thread-0 atomicAdd ticket, last arriver sums in fixed split order with __ldcg, resets the
  ticket, runs the epilogue). EPILOGUES: w13: acc * gs_eff13 -> silu(gate) * up in fp32 per thread
  (gate/up are the thread's two fragment rows) -> bf16 -> staged through smem (stmatrix) ->
  h_sorted[slot, 64j..64j+64) at the SORTED slot (so w2's B operand is a contiguous row range for
  TMA; sentinel slots get silu(0)*0 = 0, harmless); w2: acc * gs_eff2 -> bf16 -> scattered to
  cache3[sorted_ids[slot], :] for real slots only (remote and sentinel never written), so master's
  fused_post_expert_norm_weighted_sum(y.view(M, 8, H), ..., slot_expert_ids=local_ids) runs
  unchanged. SELF-CHECK (prepare, untimed, fixed torch.Generator, real bank weights, nothing
  input-derived): M in (128, 192, 256, 257) x {random, skewed (4 hot local experts take most
  routes), all-one-expert (M routes on one expert: chunks of 8 blocks, many chunks per expert),
  all-remote, all-local}; run the new path and master's Marlin path on the same synthetic
  x/routings, compare h (cache2 equivalent, via the route permutation), y and final out per row:
  per-row rel L2 <= 1e-2; also one expert against a local fp32 dequant matmul (e2m1 grid and e4m3
  decode written in the file, not the reference); run each case under torch.cuda.graph capture + 3
  replays; assert all tickets are 0 afterwards. Print LOUDLY in both outcomes: [cuda-wgmma-verify]
  SELF-CHECK PASSED/FAILED, max rel-L2, numRegs and localSizeBytes per instantiation
  (cudaFuncGetAttributes), smem per N bucket, items per shape, compile seconds, PATH CHOSEN.
  FALLBACK: any load_inline exception, cudaFuncSetAttribute failure, tensor-map encode failure,
  failed check or graph-capture error sets state['wgmma_ok'] = False and every M runs master
  unchanged; kernel_fn branches only on num_tokens and that flag. Memory: second bank ~0.9 GiB +
  xs/h_sorted scratch ~40 MB + partials (prepare). MILESTONES readable from the profile's
  top_kernels per shape: M1 all 5 tests pass with PATH CHOSEN = wgmma in the log and no
  localSizeBytes spill; M2 gather + w13 + w2 rows <= the replaced Marlin pair + act_and_mul (~320
  us at bs=128, ~497 us at bs=256); M3 <= 170 us at bs=128 (~2.9 TB/s) and <= 220 us at bs=256.
expected:
  - shapes: bs=128 *
    direction: faster
    why: "verify_16: incumbent Marlin w13+w2 316.1 us + act_and_mul 3.9 us (shape 333.4 us scored) vs a 101.3 us floor (~486 MB of NVFP4 weight for ~17 touched experts, memory-bound: MMA at peak ~40 us). Stage-1 target for gather+w13+w2 <= 170 us (M3, ~2.9 TB/s), shape ~187 us, load ratio 1.37x -> ~2.4x"
  - shapes: bs=256 *
    direction: faster
    why: "verify_32: incumbent Marlin w13+w2 490.2 us + act_and_mul 6.4 us (shape 469.9 us scored) vs a 108.9 us floor where MMA at the 989 TF wgmma peak is already ~70-90 us, so Marlin's mma.sync cannot get near it; wgmma register-A with one weight stream per expert chunk targets <= 220 us (M3), shape ~245 us, load ratio 1.31x -> ~2.5x. Both shapes at M3: headline ~1.31x -> ~1.6x"
  - shapes: bs=32 *
    direction: flat
    why: "control: num_tokens < 128 runs master byte for byte"
  - shapes: bs=16 *
    direction: flat
    why: "control: num_tokens < 128 runs master byte for byte"
  - shapes: bs=8 *
    direction: flat
    why: "control: num_tokens < 128 runs master byte for byte"
  - shapes: bs=1 *
    direction: flat
    why: "control: untouched"
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-grouped-verify

## Change

Verify shapes only: num_tokens >= 128 (verify_16 bs=128, verify_32 bs=256, and the bs=128-skewed and bs=257 tests). num_tokens < 128 (incl. the bs=33 test) keeps master byte for byte. Replace the two moe_wna16_marlin_gemm calls AND silu_and_mul at num_tokens >= 128 with a hand-written Hopper CUDA grouped W4A16 GEMM that uses wgmma with the dequantized NVFP4 weight as a REGISTER A operand (wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16, A from registers, B from shared memory: the swap-AB form, weight rows on M, tokens on N), compiled in prepare() with torch.utils.cpp_extension.load_inline (one TU, extra_cuda_cflags -O3 -arch=sm_90a, no fast-math; pattern from muse_attention_decode _build_splitkv; print compile seconds). WHY THIS AND NOT THE HELD TRITON PORT OR THE SMALL-M CUDA KERNEL: master's bs=128/256 time is Marlin at 316.1 / 490.2 us (94-95% of the shape) against floors of 101.3 / 108.9 us (~1.5 / ~1.1 TB/s effective). Marlin is an sm80 mma.sync kernel; at bs=256 the busiest GPU's ~350-450 local routes need ~70-90 us of BF16 MMA even at the 989 TF wgmma peak, so an mma.sync engine (~60% of that) plus dequant cannot reach the floor there: the tensor-core path has to be wgmma. The sibling cuda-nvfp4-swapab-mma-persistent-small-m is mma.sync with N=8..32 and is gated to bs<=32; the held Triton swapab-nvfp4-gemm-verify-rewrite reviewer measured its tile at ~25 us vs Marlin's 11.6-16.7 us per block (Triton dequant cost). This kernel keeps the sibling's exact ~1 op/weight dequant and adds wgmma, TMA/bulk-copy producers and warp specialization. STAGES PLANNED: stage 1 (this file) = two launches (w13 with silu fused in registers, w2) + one tiny gather kernel, master align and master post-norm kept; stage 2 = tuning from the stage-1 profile (N=256 bucket with setmaxnreg 240, 2-consumer ping-pong so one warpgroup's dequant overlaps the other's wgmma, L2-ordered items, split-K for w2 at bs=128); stage 3 = fold the gather into the w13 producer and the post-norm sum-of-squares into the w2 epilogue, and an in-kernel align (one launch fewer each). STAGE-1 SCOPE IN DETAIL. ALIGN: keep master's expert_map gather for local_ids, then sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, 16, bank count 32) for BOTH bs=128 and bs=256 (block 16 is the unit; items merge blocks, below). GATHER (new small CUDA kernel in the same TU, or Triton): xs[slot, :] = x[sorted_ids[slot] / 8, :] for slot < n_padded (read on device), zero rows for sentinel slots (sorted_ids >= M*8); xs is a prepare-allocated bf16 scratch [CAP_ROWS = 257*8 + 32*15 rounded up to 128, 4096] (~21 MB, scratch overwritten every call, nothing input-derived kept). WEIGHT BANK (prepare, lossless, NVFP4 stays NVFP4): from w13/w13_sf/w2/w2_sf sliced to the bank [start, start+count): (a) w13 rows interleaved at 8-row granularity so that in every 16-row MMA group rows 0-7 are gate rows [c, c+8) and rows 8-15 are up rows [I+c, I+c+8): the m16 A fragment gives each thread rows lane/4 and lane/4+8, so each thread holds gate and up of the same intermediate column and SiLU*up needs no shared-memory exchange (128-row tile j covers intermediate cols [64j, 64j+64)); (b) codes permuted to the mma m16n8k16 A-fragment nibble order defined in cuda-nvfp4-swapab-mma-persistent-small-m (the wgmma m64 register-A fragment is four stacked m16 A fragments, one per warp of the warpgroup, same per-thread layout), stored tile-contiguous per (expert, 128-row tile, BK k-block) so one stage is ONE contiguous byte range; (c) e4m3 block scales permuted the same way and stored contiguous right after their codes in the same stage record; gs_eff = gs * 2^7 fp32. Build with torch indexing; assert a round-trip un-permute equals the source bytes exactly for every expert. If the sibling has harvested with SELF-CHECK PASSED, reuse its permutation and dequant device functions verbatim (then only the 8-row gate/up interleave is new); otherwise implement them as specified there. DEQUANT (exact, as the sibling): q = 8 codes; v_i = (q & 0x80008000) | ((q & 0x70007000) >> 6), q <<= 4; times bf16x2(e4m3 scale via cvt.rn.f16x2.e4m3x2 x 2^119) with fma.rn.bf16x2 and a -0 addend; fp32 accumulation; epilogue times gs_eff. KERNEL (one template, two instantiation families w13/w2): CTA = 384 threads = 3 warpgroups, launch_bounds(384, 1), grid = SM count (132), persistent. WG0 = producer (setmaxnreg.dec 40): one elected thread issues per stage (i) cp.async.bulk.shared::cluster.global.mbarrier (1-D bulk copy, no tensor map) of the stage's weight record (128 rows x BK codes + scales) with an L2 evict_first policy, and (ii) a 2-D TMA (cp.async.bulk.tensor.2d, tensor map encoded in prepare over the persistent xs scratch for w13 and over the persistent h_sorted scratch for w2, SWIZZLE_128B, box N x 64 bf16) for the token slab, both completing on the stage's full mbarrier. WG1/WG2 = consumers (setmaxnreg.inc 232), each owns 64 of the tile's 128 weight rows: wait full[s], ld.shared.v4 its codes+scales (16 B per lane per 16x64 sub-tile), dequant into A registers, wgmma (B descriptor on the swizzled slab), wgmma.commit_group, wgmma.wait_group(1), then arrive on empty[s] for the PREVIOUS stage. Token-N buckets N in {16, 32, 64, 128} (template parameter); stage depth by N so smem <= 200 KB and >= 48 KB of weight is in flight per SM: N<=32: BK=256, 6 stages (18 KB weight + N*512 B slab per stage); N=64: BK=256, 5 stages; N=128: BK=128, 6 stages (9 KB weight + 32 KB slab). WORK ITEMS, built on device (no host sync): every CTA loads block_experts (<= CAP_ROWS/16 entries) and n_padded into smem at entry and forms CHUNKS = maximal runs of consecutive 16-row blocks with the same expert, cut every 8 blocks (<= 128 rows), deterministic and identical in every CTA; chunk N bucket = smallest of {16,32,64,128} >= its rows (rows past the chunk belong to the next expert or are OOB-zero and are masked at store). Items = (chunk c, 128-row tile j, k-split s) ordered (expert, j, chunk, s) so the chunks of one hot expert read the same weight tile back to back (the second read hits L2). w13: 64 tiles per expert (8192 rows), SK=1; w2: 32 tiles per expert (4096 rows), SK=2 at bs=128 and SK=1 at bs>=256 (host constants by shape); split partials via the measured ticket pattern (store fp32 partial with __stcg, __threadfence, thread-0 atomicAdd ticket, last arriver sums in fixed split order with __ldcg, resets the ticket, runs the epilogue). EPILOGUES: w13: acc * gs_eff13 -> silu(gate) * up in fp32 per thread (gate/up are the thread's two fragment rows) -> bf16 -> staged through smem (stmatrix) -> h_sorted[slot, 64j..64j+64) at the SORTED slot (so w2's B operand is a contiguous row range for TMA; sentinel slots get silu(0)*0 = 0, harmless); w2: acc * gs_eff2 -> bf16 -> scattered to cache3[sorted_ids[slot], :] for real slots only (remote and sentinel never written), so master's fused_post_expert_norm_weighted_sum(y.view(M, 8, H), ..., slot_expert_ids=local_ids) runs unchanged. SELF-CHECK (prepare, untimed, fixed torch.Generator, real bank weights, nothing input-derived): M in (128, 192, 256, 257) x {random, skewed (4 hot local experts take most routes), all-one-expert (M routes on one expert: chunks of 8 blocks, many chunks per expert), all-remote, all-local}; run the new path and master's Marlin path on the same synthetic x/routings, compare h (cache2 equivalent, via the route permutation), y and final out per row: per-row rel L2 <= 1e-2; also one expert against a local fp32 dequant matmul (e2m1 grid and e4m3 decode written in the file, not the reference); run each case under torch.cuda.graph capture + 3 replays; assert all tickets are 0 afterwards. Print LOUDLY in both outcomes: [cuda-wgmma-verify] SELF-CHECK PASSED/FAILED, max rel-L2, numRegs and localSizeBytes per instantiation (cudaFuncGetAttributes), smem per N bucket, items per shape, compile seconds, PATH CHOSEN. FALLBACK: any load_inline exception, cudaFuncSetAttribute failure, tensor-map encode failure, failed check or graph-capture error sets state['wgmma_ok'] = False and every M runs master unchanged; kernel_fn branches only on num_tokens and that flag. Memory: second bank ~0.9 GiB + xs/h_sorted scratch ~40 MB + partials (prepare). MILESTONES readable from the profile's top_kernels per shape: M1 all 5 tests pass with PATH CHOSEN = wgmma in the log and no localSizeBytes spill; M2 gather + w13 + w2 rows <= the replaced Marlin pair + act_and_mul (~320 us at bs=128, ~497 us at bs=256); M3 <= 170 us at bs=128 (~2.9 TB/s) and <= 220 us at bs=256.

## Expected

- `bs=128 *` faster: verify_16: incumbent Marlin w13+w2 316.1 us + act_and_mul 3.9 us (shape 333.4 us scored) vs a 101.3 us floor (~486 MB of NVFP4 weight for ~17 touched experts, memory-bound: MMA at peak ~40 us). Stage-1 target for gather+w13+w2 <= 170 us (M3, ~2.9 TB/s), shape ~187 us, load ratio 1.37x -> ~2.4x
- `bs=256 *` faster: verify_32: incumbent Marlin w13+w2 490.2 us + act_and_mul 6.4 us (shape 469.9 us scored) vs a 108.9 us floor where MMA at the 989 TF wgmma peak is already ~70-90 us, so Marlin's mma.sync cannot get near it; wgmma register-A with one weight stream per expert chunk targets <= 220 us (M3), shape ~245 us, load ratio 1.31x -> ~2.5x. Both shapes at M3: headline ~1.31x -> ~1.6x
- `bs=32 *` flat: control: num_tokens < 128 runs master byte for byte
- `bs=16 *` flat: control: num_tokens < 128 runs master byte for byte
- `bs=8 *` flat: control: num_tokens < 128 runs master byte for byte
- `bs=1 *` flat: control: untouched

## Risk

Biggest risk: complexity of a warp-specialized wgmma register-A kernel built by load_inline with no GPU on the authoring node: the m64 register-A fragment must match the sibling's m16 permutation per warp, mbarrier phase bookkeeping across a persistent flat (item, stage) sequence, TMA tensor-map encoding through the driver API from Python (cuTensorMapEncodeTiled via torch/cuda-python or a tiny host function in the TU), and compile time inside the 15-minute budget (keep 4 N buckets x 2 kinds = 8 instantiations). Performance risk: dequant ALU is ~1 op/weight, i.e. ~30-50 us of issue per SM over the whole bs=256 bank, the same order as the floor, so it must overlap the stream and the MMA; stage 1 uses two consumer warpgroups on disjoint rows, which overlaps only through the async wgmma; if M2 holds but M3 fails the stage-2 ping-pong schedule is the fix. Shared-memory pressure: the N=128 slab is 32 KB per stage at BK=128; if wave stalls show, stage 2 tries BK=64 with 8 stages. Not a duplicate: cuda-nvfp4-swapab-mma-persistent-small-m is mma.sync, bs<=32, never touches num_tokens >= 128; cuda-fused-moe-bs1 is bs=1; swapab-nvfp4-gemm-verify-rewrite (held) is a Triton port of prefill's kernel at the same shapes; swapab-w13-silu-epilogue-verify depends on that Triton port. If both this and the Triton port ever land, route each verify shape to whichever measured faster (prefill stage-4 lesson). Interactions: the running align-parallel-per-expert-verify changes the bs>=128 align; this file needs only (sorted_ids, block_experts, n_padded) at block 16 over 32 experts, so take whatever align master has at implement time and read its row separately. master's two-smaller block_size_m pick (16 at bs=128, 32 at bs=256) is superseded at num_tokens >= 128 by block 16 + on-device chunking. Judge on the replaced kernels' rows (Marlin pair + act_and_mul vs gather + two wgmma kernels), not the headline.

## Milestones

Rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
