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
review: proceed
status: done
review_at: 2026-10-24T172632Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T172705Z
base_sha: 392a74ee4f7b4f552a563e8bf1c922ac4974e1f6
base_blob: d9a9375ef4721ab9b28cc9d6364c92e67c7ccc59
branch: hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-verify
worktree: .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-verify
mode: submit
implemented_at: 2026-10-24T180001Z
impl_commit: f50b333a602cc590f2785a1fdd2171af25fe94bb
submission_sha256: a7fdef83dbf98e25a4d58f6e4e6cccf3a73c2b080e18267c4809f7bdb9c4264d
norm_sha256: 1b43a3f3bea47906633c64e78cbada93a6b6003a3b8bdc838ce21a53b0f64595
submission_blob: 5c34fabe0f05fb53014dd85b66864cd3b38bc673
lint_warnings: 23
attempt_file: 2026-10-24T181733Z-submit-cuda-nvfp4-wgmma-rega-grouped-verify.md
prepared_at: 2026-10-24T181733Z
job: 3bb822f22ebc4a5c
dispatched_at: 2026-10-24T181917Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.3123142455018568
speedup_ci: [1.292629531819329, 1.3319989591843846]
harvested_at: 2026-10-24T184719Z
best_at_harvest: 1.3249153782392282
promotion: null
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

## Adversarial review

Reviewed 2026-10-24T172632Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as rewrite stage 1. The wgmma register-A swap-AB design suits the bs=128/256 verify shapes, where Marlin is 94-95% of the shape at about 3-4.5x the floor, and no landed hypothesis confounds it. One blocking arithmetic error must be fixed before implementing: as written, the N=64 and N=128 stage tables exceed Hopper's 227 KB of shared memory. The kernel would then fall back to master on the judge, and the run would measure nothing.

### Justification
Mechanism is sound. FINDINGS gives Marlin 316.1 us at bs=128 and 490.2 us at bs=256 (94.0% / 95.2% of the shape) against 101.3 / 108.9 us floors, and every lever tried inside Marlin (block_size_m picks, fp32-reduce off, atomic add, workspace) is exhausted or within CI, so replacing the GEMM pair is the remaining lever at these shapes. Specific checks: (1) The wgmma m64nNk16 register-A fragment is four stacked m16n8k16 A fragments, one per warp, and the accumulator gives each thread rows lane/4 and lane/4+8 of its warp's 16 rows. So the 8-row gate/up interleave really does put gate and up of the same intermediate column in one thread, and silu*up fusion needs no exchange. Correct. (2) Register budget: setmaxnreg 40x128 + 232x256 = 64512 <= 65536. OK. (3) Bytes in flight at 48 KB/SM is enough for M3 (2.9 TB/s) but not for ~4.5 TB/s. (4) Dequant ALU at ~1 op/weight is ~30-50 us per pass over the touched bank. That is acceptable only if it overlaps the stream, which stage 1 does only through async wgmma, as the file admits. BLOCKING DEFECT: the smem table. N=128, BK=128 is 9 KB weight + 32 KB slab = 41 KB/stage, and 6 stages = 246 KB. N=64, BK=256 is 18 KB + 32 KB = 50 KB/stage, and 5 stages = 250 KB. Both exceed the 227 KB opt-in limit, not just the self-imposed 200 KB. Only N<=32 fits (N=32: 34 KB x 6 = 204 KB; N=16: 26 KB x 6 = 156 KB). At bs=256 (and for hot experts at bs=128) chunks of 64-128 rows are common, so cudaFuncSetAttribute fails and the whole path falls back to master: a wasted run. Fix: N=128 at 4-5 stages of BK=128, or BK=64 with 8-10 stages, and N=64 at 4 stages of BK=256. Accept less weight in flight at large N, which is fine because the MMA per weight byte is higher there. Add a Python-side assert of per-bucket smem <= 227 KB before compiling. SECOND DESIGN COST (stage-2 note, not blocking): with a 128-row weight tile, the token slab is re-read from L2 once per weight tile: 64 times for w13 and 32 times for w2. At N=128 the slab is ~3.5x the weight bytes per K. Rough L2 activation traffic at bs=256 is ~300 MB (w13) + ~150 MB (w2), the same order as the 523 MB HBM stream. That probably caps a stage-1 result above the ~1.5x-floor target (~165 us) though not above M3. A 256-row weight tile per CTA (each consumer doing two m64 sets), or N=256 for hot experts, halves this. WEAKEST LINK: the compute argument is internally inconsistent. The file says ~350-450 local routes at bs=256 but 70-90 us of MMA at peak. 400 routes x 50.3M MACs x 2 = ~40 GFLOP = ~41 us at 989 TF. 70-90 us implies ~700-900 routes, which matches bs=128's stated ~40 us at peak only if bs=128 has ~400 routes. Either way the case for wgmma stands: even 41-60 us of MMA at mma.sync's ~60% of peak plus serialized dequant eats most of a 109 us budget. But the route count should be pinned from the profile/log so that M3's 220 us target is grounded. Expected directions are consistent with TASK: only bs=128/256 scored shapes change, the bs<128 controls are byte-for-byte master, and the 5 tests include bs=257 and the skewed bs=128 on the new path.

### Confounders on the current kernel
- None from landed hypotheses: the landed list for master's submission.py is empty in this review's input, so no landed patch shares the area.
- Master's block_size_m two-smaller pick at num_tokens >= 128 (16 at bs=128, 32 at bs=256) defines the incumbent Marlin rows (316.1 / 490.2 us). The new path bypasses it with block 16 plus on-device chunking. This is not a confound because the hypothesis is judged on its own kernel rows against those incumbent rows, but the comparison is against the tuned Marlin, not SGLang's pick.
- In-flight align-parallel-per-expert-verify (running) changes the bs>=128 align. If it lands before this is implemented, the align row and its outputs change under this path. The file correctly reads only (sorted_ids, block_experts, n_padded) at block 16 and asks for the align row to be read separately, so no ambiguity in the replaced rows.

### Interference and A/B needs
No companion is warranted. This is kind: rewrite, judged on its own replaced rows (gather + w13 + w2 vs the Marlin pair + act_and_mul) per shape in top_kernels, not on the headline, and no landed hypothesis touches the same kernels, launch sequence or numerical path. An ab-revert has nothing to revert. An ab-tune of master's block_size_m pick is moot because the new path does not use it. A control isolating the fused silu or the split-K for w2 at bs=128 would cost a run to separate a few us out of ~300, and that is exactly what stage 2's profile-driven tuning does. The only thing that would make this single result uninterpretable is a silent fallback to master. The fix is the smem-table correction and the loud PATH CHOSEN / per-bucket smem log already specified, not a companion run.

### Open questions
- Fix the per-bucket smem table: N=64 (5 x 50 KB = 250 KB) and N=128 (6 x 41 KB = 246 KB) exceed 227 KB. What stage counts are chosen, and does the Python side assert the budget before compiling?
- The TMA box dims are fixed per tensor map: is one CUtensorMap encoded per N bucket per scratch (4 x 2 = 8 maps)? Is BK=128/256 at SWIZZLE_128B (64 bf16 inner box) issued as 2-4 boxes per stage, with each slab stage base 1024 B aligned for the wgmma descriptor?
- How is cuTensorMapEncodeTiled reached: a host function in the TU with extra_ldflags=['-lcuda'], or cuda-python? Is it confirmed available in the judge image? muse_attention_decode's _build_splitkv used load_inline but, from its flags, not the driver API.
- The L2 policy contradicts the reuse plan. The weight record is loaded with evict_first, yet items are ordered so a hot expert's second chunk 'hits L2' on the same tile. Use evict_normal (or evict_last) when an expert has more than one chunk, or have one item cover all of an expert's chunks with multiple accumulators so the tile is dequantized once.
- What is the item-to-CTA assignment: static stride or an atomic work counter? Item costs vary by N bucket (16 to 128), so a static stride can leave a tail of several items at ~10-12 items per CTA.
- Chunk formation: 'every CTA loads block_experts and forms chunks' over up to 160 blocks. If one thread scans serially it costs ~2-4 us per launch, twice. Use a warp-ballot/prefix scan.
- The w2 epilogue scatters to cache3[sorted_ids[slot], :]. In swap-AB a thread's fragment is strided along H per token, so this needs the same smem-staged transpose (stmatrix .trans) as w13 to get 256 B coalesced row writes. Please specify it.
- Pin the busiest-GPU local route counts at bs=128 and bs=256 (the file says ~350-450 at bs=256 but 70-90 us of MMA, which implies ~800). Logging items per shape from the profile run would ground M3.


## Implementation notes

Implemented 2026-10-24 in the worktree's muse_moe_w4a16/submission.py only. It is not committed, and nothing has been sent to okc. Diff: 1 file, +1009/-11.

**What changed, and where**
- New section "CUDA wgmma verify path" (constants `WG_*`, `_WG_CPP`, `_WG_SRC_HEAD`/`_WG_SRC_BODY`, `_wg_wgmma_src`, `_wg_build`, `_wg_tile_expert`/`_wg_bank`, `_wg_chunks`, `_wg_decode_nvfp4`, `_moe_wgmma`, `_wg_self_check`, `_wg_setup`). It sits before `_marlin_layer`. One load_inline TU (`-O3 -arch=sm_90a`, no fast-math) has two instantiations, `wg_kernel<GATED>` (w13+silu, w2), plus `wg_gather_kernel`. The compile seconds are printed.
- Master's `kernel_fn` body moved verbatim into `_moe_marlin(x, topk_ids, topk_weights, bank, state)`, which returns (out, local_ids, cache2, y). The only edits are reading the three arguments instead of `data[...]`. The launch sequence is unchanged. The new `kernel_fn` takes the wgmma path only when `state["wgmma_ok"]` is set and 128 <= num_tokens <= 512. It branches on nothing else (shape, flag, config key). Every other M runs `_moe_marlin` byte for byte as master.
- `prepare()` builds the state dict as before, then calls `_wg_setup`. Any exception, a failed build, a failed attribute or occupancy check, a tensor-map encode error, a failed self-check or a set device error word sets wgmma_ok = False, and everything then runs master.
- Kernel, as specified: 384 threads = 3 warpgroups, launch_bounds(384, 1), grid = SM count, persistent. WG0 runs setmaxnreg.dec 40 and one thread issues the stage: one `cp.async.bulk` 1-D copy of the 9216-byte weight range (L2 evict_first, or evict_normal when the expert has more than one chunk) plus two 2-D TMA boxes (SWIZZLE_128B, box N x 64 bf16) of the xs / h_sorted slab, all on the stage's full mbarrier. WG1/WG2 run setmaxnreg.inc 232. Each owns 64 weight rows: it dequantizes exactly into the register A operand (the sibling's bit trick and e4m3 path, verbatim), issues 8 x `wgmma.mma_async m64nNk16.f32.bf16.bf16` (RS form, K-major SW128 descriptor, +32 B per k16), commits, runs wait_group(1), and releases the previous stage (empty barrier count 8, one arrive per consumer warp).
- Align: master's expert_map gather + `moe_align_block_size(local_ids, 16, 32)`. Chunks are formed on device in every CTA with a warp-ballot prefix scan over block_experts: maximal runs of one local expert, cut every 8 blocks. Items are ordered (expert, tile, chunk), so a hot expert's chunks read each weight tile back to back. Item to CTA uses a static stride.
- Epilogues: w13 applies acc * gs_eff, rounds gate and up to bf16, computes silu(gate) * up in fp32, converts to bf16, does stmatrix .x2.trans into a padded staging tile (pitch 144 B, conflict-free), then writes coalesced 16 B rows into h_sorted at the sorted slot. w2 applies acc * gs_eff, converts to bf16, does stmatrix .x4.trans (pitch 272 B), then writes coalesced 256 B rows to cache3[route], real routes only. Master's fused_post_expert_norm_weighted_sum then runs unchanged.
- Weight bank (prepare, lossless): per expert, per 128-row tile, per 64-k block, one 4608 B record (4096 B codes in the sibling's per-warp, per-lane fragment order + 512 B E4M3 scales), so a BK=128 stage is one contiguous range. A round trip of both parts is asserted exact for every expert. gs_eff = gs * 2^7. The second bank is about 0.9 GiB, and xs/h_sorted scratch is 2 x 64 MiB (8192 rows), allocated in prepare and overwritten every call.

**Deviations from the spec, and why**
1. **Smem (BLOCKING fix from the review).** All buckets use one ring geometry: BK=128, 4 stages of (9216 B weight + 2 x 16 KB slab boxes) = 41 KB/stage. The N bucket is chosen per item at run time, and all buckets mix inside one launch, so a per-bucket stage depth would mean per-bucket launches. Dynamic smem is 187,392 B (w13) and 203,776 B (w2), plus about 7.8 KB static. Python asserts dynamic + 8 KB <= 227 KB before compiling. After the build it checks dynamic + cudaFuncGetAttributes' static <= 227 KB and occupancy >= 1. The cost is about 36 KB of weight in flight per SM, which is below the spec's 48 KB.
2. **No split-K (SK=1 everywhere).** The spec lists split-K for w2 at bs=128 in both stage 1 and stage 2. I left it to stage 2 to cut ticket and workspace risk in a blind build. At bs=128 there are about 32 x n_chunks items per launch, so 132 CTAs are still kept busy. The self-check's "tickets are 0" item does not apply.
3. **Buckets as run-time N.** One kernel template per GEMM, with `switch(bucket)` over 4 `consume_item<NB>` bodies. This gives 2 instantiations rather than 8 and keeps the compile shorter.
4. **Tensor maps.** `cuTensorMapEncodeTiled` is reached through `cudaGetDriverEntryPoint(ByVersion)` from the runtime, not `-lcuda`. This needs no link against libcuda.so, which judge containers often expose only as libcuda.so.1. The 8 maps (4 buckets x {xs, h_sorted}) are encoded once in prepare over the persistent scratch and passed to the kernel as a `__grid_constant__` struct. A missing entry point or an encode error goes to the master fallback.
5. **Bounded mbarrier waits.** These were not in the spec. After about 2^31 cycles a wait sets a device error word and stops waiting, so a pipeline bug ends in SELF-CHECK FAILED and the master fallback rather than a 15-minute timeout. The cost is one clock64 read per spin iteration.
6. **Defensive expert range.** Any block expert outside [0, 32) counts as non-local in the chunking and the gather, and the gather skips SGLang's remote (-1) blocks. Master's log shows SGLang's align buckets remote routes into m-blocks of their own (r_pad > n_pad). As a result the gather copies only local slots.
7. **"Items per shape"** is logged per synthetic self-check case (local routes, n_padded, chunks, N-bucket histogram, w13/w2 items). kernel_fn cannot log it without a host sync. The self-check's "skewed" routing gives every token the 4 hot local experts.
8. **Not gated on `_align_self_check`.** As the caller instructed, the path checks only its own outputs (h via the slot permutation, y and out per row, local_ids equality, route coverage, graph replay, fp32 dequant).

**Offline verification (no GPU, nvcc or torch here)**
- A numpy emulation ran the actual `_wg_code_dims`/`_wg_scale_dims` perms (extracted from the file by AST). It covered the round trip, the per-lane word/nibble/scale extraction, the wgmma m64 A and D fragment layout, the w13 silu and w2 epilogues through an exact stmatrix .trans model and the staging tile, against a dense dequant matmul (gated N=16/32/128, w2 N=16/64). Error was exact (0.0). Every staging and A element was written exactly once.
- The device chunk/decode logic (transcribed) was checked against `_wg_chunks` on 300 random block_experts layouts, including -1 and out-of-range blocks at either end: chunks are identical, the item order is (expert, tile, chunk), and every local block is covered exactly once.
- The generated CUDA TU passes `g++ -std=c++17 -fsyntax-only` with CUDA/torch stub headers. This catches C++ errors only, not PTX.

**Uncertainties (unverifiable here)**
- The PTX forms are untested: wgmma RS m64nNk16 operand order and the SW128 K-major descriptor (LBO 1, SBO 1024, +32 B k-advance), TMA 2-D/bulk syntax, stmatrix .trans, setmaxnreg, createpolicy evict_normal. The cvt.e4m3x2 byte order is inherited from the sibling (also unverified on the judge).
- ptxas should serialize or extend liveness for A registers rewritten while the previous wgmma group is in flight (wait_group(1)). That is correct either way, but if ptxas serializes, dequant does not overlap the MMA.
- Producer code at setmaxnreg 40 may spill. The log shows localSizeBytes. This is not fatal.
- The compile time of the 2-instantiation TU with 4 bucket branches is unknown. The precedent was 41 s.
- Performance: the slab is re-read from L2 once per weight tile (the reviewer's note), there is no split-K, and weight in flight is 36 KB/SM. M2 is plausible, M3 is uncertain.

**Harvest note.** Read the log tail first. `[cuda-wgmma-verify] PATH CHOSEN = MARLIN`, `SELF-CHECK FAILED` or `load_inline failed` means the run measured master at bs >= 128.

### Pre-flight fixes (2026-10-24, after an Opus audit)
The audit verified the wgmma RS form, the descriptor, the bulk/TMA copies, the mbarriers, stmatrix, createpolicy and the driver entry-point tensor-map encode. Exact-layout emulation passes. Fixes:
- The self-check reads the device error word after each eager run and after each graph replay, and aborts at the first timeout. Before, 128 launches x ~1.1 s could overrun the 300 s prepare budget.
- The wait bound is lowered from 2^31 to 2^26 cycles (~34 ms).
- attrs_ok requires numRegs x 384 >= 40x128 + 232x256, so setmaxnreg.inc cannot block forever.
Not applied: moving the compile to import time.

## Rules check

Checked against `git -C .worktrees/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-verify diff`:

- [x] CUDA-graph capturable. kernel_fn's new path (`_moe_wgmma`) has no `.item()`, `.cpu()`, `synchronize()` or tensor prints. Its allocations (cache3, align buffers) depend only on shape, like master's. Its control flow branches only on num_tokens, the prepare-time flag and the config key. The kernels read n_padded and block_experts on device. Every host sync in the diff is in `_wg_self_check`/`_wg_setup` (prepare). The self-check itself runs each case under torch.cuda.graph capture + 3 replays.
- [x] Nothing derived from `data` is cached across calls. xs/h_sorted are prepare-allocated scratch, fully overwritten for every slot that is read as a valid token. Stale rows only ever feed masked output columns. The tensor maps describe the scratch, not inputs. Self-check inputs are synthetic (fixed torch.Generator).
- [x] Weights are repacked losslessly in `prepare()` only. NVFP4 codes and E4M3 scales stay the same bytes, retiled, with round trips asserted exact for every expert. Global scales stay fp32 times 2^7 (exact). The dequant to bf16 inside the kernel is exact (at most 6 significant bits). No BF16 weights are involved.
- [x] Activations stay BF16 where SGLang keeps BF16: x/xs, h (cache2-equivalent) and y are bf16. Gate and up are rounded to bf16 before silu as Marlin + silu_and_mul do. Accumulation is fp32.
- [x] n/a: attention. This problem has no attention, KV cache or key skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. load_inline is the same build precedent as muse_attention_decode. The ROUTED experts are computed fully; no route is skipped except remote ones, as master does.
- [x] All work happens inside the timed call: gather + two GEMMs + post-norm per call, with no output caching or replay. The CUDA graph replays in the self-check are prepare-only on synthetic data.
- [x] First line `# okc: problem=muse_moe_w4a16` is kept. No triton.autotune. num_tokens < 128 runs `_moe_marlin`, which is master's launch sequence unchanged.

okc_lint output:
```
warn  host-sync    L216: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L218: [remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:5].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L220: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L239: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L244: s_ids, b_exp, n_pad = s_ids.tolist(), b_exp.tolist(), int(n_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L245: r_ids, r_exp, r_pad = r_ids.tolist(), r_exp.tolist(), int(r_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1019: q13 = torch.empty((count, rec13), dtype=torch.uint8, device=w13.device)  (allowed only where SGLang does the same)
warn  dtype        L1020: q2 = torch.empty((count, rec2), dtype=torch.uint8, device=w2.device)  (allowed only where SGLang does the same)
warn  host-sync    L1110: "random": [torch.randperm(N_EXPERTS, generator=gen)[:TOP_K].tolist() for _ in range(tokens)],  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1112: "skewed": [local_set[:4] + [others[i] for i in torch.randperm(len(others), generator=gen)[:4].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1116: "all-remote": [[remote_set[i] for i in torch.randperm(len(remote_set), generator=gen)[:TOP_K].tolist()]  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1128: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1130: used = int(n_pad.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1144: chunks = _wg_chunks(b_exp[:used // WG_BSM].tolist(), count)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1154: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1233: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1241: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1242: err_word = int(state["wg_err"].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1300: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1302: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1313: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
lint rc=0
```
`python3 -I -m py_compile` OK; __pycache__ removed from the worktree.

## Result

- job `3bb822f22ebc4a5c` (submit), status done, passed True, tests 5/5
- speedup 1.3123x, 95% CI [1.2926, 1.3320], score 10247.926551202687 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3249153782392282, "ci": [1.30504164756564, 1.3447891089128166], "job": "45eacf4ca2e849d2"}; base result: muse_moe_w4a16/hypotheses/done/gain/triton-map-align-on-block-m-restack.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-verify.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/cuda-nvfp4-wgmma-rega-grouped-verify.patch`; branch `hyp/muse_moe_w4a16/cuda-nvfp4-wgmma-rega-grouped-verify`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 48.37 | 50.54 | 0.9847 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 147.87 | 146.42 | 1.0023 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 200.73 | 198.76 | 0.999 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 107.77 | 107.34 | 0.999 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 333.39 | 343.61 | 1.0167 | flat |
| bs=256 load=verify_32 sharding=ep8 | 60 | 469.91 | 471.52 | 1.0015 | flat |

no shape moved outside noise. Expected hit: 4/6. Noise ±1.5%.

## Follow-ups

- rewrite stage 1 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at num_tokens >= 128`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_moe_w4a16 "<what to fix or tune>" --kind rewrite --depends-on cuda-nvfp4-wgmma-rega-grouped-verify --stage 2`; a slow or failed first cut is the expected start, not a do-not-retry.
- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: cuda-nvfp4-wgmma-rega-grouped-verify`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

