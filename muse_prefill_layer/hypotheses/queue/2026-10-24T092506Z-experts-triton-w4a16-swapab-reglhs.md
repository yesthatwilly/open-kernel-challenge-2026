---
problem: muse_prefill_layer
slug: experts-triton-w4a16-swapab-reglhs
created: 2026-10-24T092506Z
base: master
area: "w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, stage 2: swap-AB register-LHS dot)"
change: >
  Stage 2 of the Triton NVFP4 grouped GEMM, built on stage 1 (branch
  hyp/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4, _moe_nvfp4_gemm_kernel in
  muse_prefill_layer/submission.py). The one fix: swap A and B so the in-register dequantized
  weight tile is the LEFT (register-sourced) operand of the tensor-core dot. Stage 1 computes
  acc[tokens, n] += tl.dot(A[tokens, k], tl.trans(tl.interleave(w_lo*sb, w_hi*sb))): the
  dequantized bf16 weight is the RIGHT operand, which sm_90 wgmma can only read from shared memory
  (and at BLOCK_M 16/32 Triton emits mma.sync regardless), so every K step pays interleave + trans
  + a layout conversion / local_store round trip of 8192 bf16 weight elements through smem, while
  the MMA sees only 16-32 token rows at turn_1. That per-tile overhead is what the profile shows
  (2026 us for w13+w2 at turn_1 vs Marlin 730 us; 13.5 ms at first_turn = ~120 TFLOP/s, i.e. no
  effective wgmma). Stage 2 computes the transpose, acc[n, tokens] += tl.dot(W[n_tile, k_tile],
  A_t[k_tile, tokens]): (1) Operands. The weight tile is the MMA M side: BLOCK_W = 128 rows of the
  expert N (wgmma M = 64 per warpgroup); the tokens are the MMA N side (wgmma n = 8..256). The
  dequantized [128, BLOCK_K] bf16 tile feeds tl.dot as the LHS straight from registers (Triton
  MMAv3 LHS-in-registers path, the one its mixed-precision upcast matmuls use); only the bf16
  activation tile, loaded with tl.load as [BLOCK_K, BLOCK_T] from K-contiguous rows (offs_k[:,
  None] + (offs_token[None, :] // TOP_K) * K, the master router orientation trick), is pipelined
  through smem. Load the uint8 code tile and scale tile in their natural blocked layout and do the
  bit ops there; let Triton hoist the convert to the dot-operand layout onto the uint8 codes (4x
  fewer bytes than bf16). (2) No interleave or trans: a lossless nibble reorder in prepare().
  Within each BLOCK_K = 128 chunk of K, repack the uint8 codes so byte i (0..63) holds k0 + i in
  the low nibble and k0 + 64 + i in the high nibble (a permutation of the same E2M1 codes; sf and
  gs untouched; stored as state bank w13/w2 in place of the original packing, same [E, N, K/2]
  uint8 shape). The low nibbles are then the contiguous k-half [k0, k0+64) and the high nibbles
  [k0+64, k0+128), and each K step is two dots: acc = tl.dot(w_lo, a_t_lo, acc); acc =
  tl.dot(w_hi, a_t_hi, acc), with the activation tile loaded as two [64, BLOCK_T] slabs at k0 and
  k0+64 (pointer offsets only, no reshape/split). Scales: low half sf group k0/16 + i//16, high
  half k0/16 + 4 + i//16, loaded as [128, 64] gather tiles (L1 hits) as stage 1 does. Keep stage 1
  bit-placement E2M1 -> bf16, the 2^119 scale bias (sb multiply per element), and gs[e] * 2^7 once
  in the epilogue. (3) Token tile from the local rows (~T/32 per expert), so each expert is one
  m-block at the turn shapes and each weight element is dequantized once per call there: BLOCK_T
  64 for T <= 1024 (turn_1, ~24 rows; MMA padding is free since the shape is
  dequant/memory-bound), 128 for T <= 4096 (turn_4, ~94 rows), 128 above (first_turn, ~512 rows =
  4 token tiles per expert); moe_align_block_size pads to BLOCK_T. Order program ids so the token
  tiles of one (expert, weight tile) run adjacent (token-tile index fastest; drop GROUP_SIZE_M) so
  the dequantized-from tile is re-read from L2 at first_turn. num_warps 4 (one warpgroup) at
  BLOCK_T 64, 8 at BLOCK_T 128; BLOCK_K 128; num_stages 3. Fixed configs chosen from T only; no
  autotune. (4) Epilogue: acc is [BLOCK_W, BLOCK_T]; store (acc * gs) as bf16 through transposed
  pointers c_ptr + offs_token[None, :] * N + offs_w[:, None], masked by token_mask[None, :]. The
  -1 (non-local) guard and ZERO_NONLOCAL_W2 zero-store keep their stage 1 semantics in the
  transposed indexing. Unchanged from stage 1: moe_align_block_size call (with BLOCK_T),
  silu_and_mul(expert_ids=local_ids), cache13/cache2 buffers,
  fused_post_expert_norm_weighted_sum(slot_expert_ids=local_ids), everything outside the expert
  block. prepare(): rewrite _check_expert_gemms for the reordered layout, comparing against
  nvfp4.dequantize of the ORIGINAL weights on a mid-K tile that spans both nibble halves and all 8
  scale groups of one 128-chunk, for every config (this also warm-compiles them). Before the run,
  compile one BLOCK_T 128 config locally if possible and check the TTGIR for ttng.warp_group_dot
  with a register LHS (dot_op opIdx = 0, parent = mma v3) rather than a local_alloc memdesc;
  record the finding in the implementation notes either way.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "compute-bound (~512 rows/expert, 1.65 TFLOP per call). Marlin (mma.sync) is 11.1-12.0 ms and stage 1 is 13.3-14.4 ms (~120 TFLOP/s, no effective wgmma). A 128x128 swap-AB tile with a register LHS lets wgmma run without the per-tile smem round trip. Milestone: _moe_nvfp4_gemm_kernel x2 at or below 9 ms (Marlin parity <= 11.1 ms is the minimum); floor ~2.5 ms at ~660 TFLOP/s"
  - shapes: "*load=turn_4*"
    direction: flat
    why: "~94 rows/expert, one 128-token tile per expert, so weights are dequantized once (864 MiB packed+scales, floor ~190 us). Stage 1 was 3341/3457 us vs Marlin 2248/2201 us; removing the interleave/trans/smem round trip and raising MMA M to 128 should reach Marlin parity. Milestone: <= 2250 us"
  - shapes: "*load=turn_1*"
    direction: slower
    why: "~24 rows/expert: dequant/memory-bound. Stage 1 2026/2008 us (2.8x Marlin 730/681 us); the fix removes the per-tile layout conversion and halves the weight-tile dequant passes (one 64-token tile per expert instead of 1-2 32-row tiles) but leaves the per-element ALU dequant, so it likely stays behind Marlin. Milestone: <= 1000 us (2x better than stage 1); the turn_1 loss shrinking from 0.46x toward ~0.8x is the attribution signal"
risk: high
depends_on: experts-triton-w4a16-inkernel-fp4
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T
review: required
status: queued
---

# experts-triton-w4a16-swapab-reglhs

## Change

Stage 2 of the Triton NVFP4 grouped GEMM, built on stage 1 (branch hyp/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4, _moe_nvfp4_gemm_kernel in muse_prefill_layer/submission.py). The one fix: swap A and B so the in-register dequantized weight tile is the LEFT (register-sourced) operand of the tensor-core dot. Stage 1 computes acc[tokens, n] += tl.dot(A[tokens, k], tl.trans(tl.interleave(w_lo*sb, w_hi*sb))): the dequantized bf16 weight is the RIGHT operand, which sm_90 wgmma can only read from shared memory (and at BLOCK_M 16/32 Triton emits mma.sync regardless), so every K step pays interleave + trans + a layout conversion / local_store round trip of 8192 bf16 weight elements through smem, while the MMA sees only 16-32 token rows at turn_1. That per-tile overhead is what the profile shows (2026 us for w13+w2 at turn_1 vs Marlin 730 us; 13.5 ms at first_turn = ~120 TFLOP/s, i.e. no effective wgmma). Stage 2 computes the transpose, acc[n, tokens] += tl.dot(W[n_tile, k_tile], A_t[k_tile, tokens]):
(1) Operands. The weight tile is the MMA M side: BLOCK_W = 128 rows of the expert N (wgmma M = 64 per warpgroup); the tokens are the MMA N side (wgmma n = 8..256). The dequantized [128, BLOCK_K] bf16 tile feeds tl.dot as the LHS straight from registers (Triton MMAv3 LHS-in-registers path, the one its mixed-precision upcast matmuls use); only the bf16 activation tile, loaded with tl.load as [BLOCK_K, BLOCK_T] from K-contiguous rows (offs_k[:, None] + (offs_token[None, :] // TOP_K) * K, the master router orientation trick), is pipelined through smem. Load the uint8 code tile and scale tile in their natural blocked layout and do the bit ops there; let Triton hoist the convert to the dot-operand layout onto the uint8 codes (4x fewer bytes than bf16).
(2) No interleave or trans: a lossless nibble reorder in prepare(). Within each BLOCK_K = 128 chunk of K, repack the uint8 codes so byte i (0..63) holds k0 + i in the low nibble and k0 + 64 + i in the high nibble (a permutation of the same E2M1 codes; sf and gs untouched; stored as state bank w13/w2 in place of the original packing, same [E, N, K/2] uint8 shape). The low nibbles are then the contiguous k-half [k0, k0+64) and the high nibbles [k0+64, k0+128), and each K step is two dots: acc = tl.dot(w_lo, a_t_lo, acc); acc = tl.dot(w_hi, a_t_hi, acc), with the activation tile loaded as two [64, BLOCK_T] slabs at k0 and k0+64 (pointer offsets only, no reshape/split). Scales: low half sf group k0/16 + i//16, high half k0/16 + 4 + i//16, loaded as [128, 64] gather tiles (L1 hits) as stage 1 does. Keep stage 1 bit-placement E2M1 -> bf16, the 2^119 scale bias (sb multiply per element), and gs[e] * 2^7 once in the epilogue.
(3) Token tile from the local rows (~T/32 per expert), so each expert is one m-block at the turn shapes and each weight element is dequantized once per call there: BLOCK_T 64 for T <= 1024 (turn_1, ~24 rows; MMA padding is free since the shape is dequant/memory-bound), 128 for T <= 4096 (turn_4, ~94 rows), 128 above (first_turn, ~512 rows = 4 token tiles per expert); moe_align_block_size pads to BLOCK_T. Order program ids so the token tiles of one (expert, weight tile) run adjacent (token-tile index fastest; drop GROUP_SIZE_M) so the dequantized-from tile is re-read from L2 at first_turn. num_warps 4 (one warpgroup) at BLOCK_T 64, 8 at BLOCK_T 128; BLOCK_K 128; num_stages 3. Fixed configs chosen from T only; no autotune.
(4) Epilogue: acc is [BLOCK_W, BLOCK_T]; store (acc * gs) as bf16 through transposed pointers c_ptr + offs_token[None, :] * N + offs_w[:, None], masked by token_mask[None, :]. The -1 (non-local) guard and ZERO_NONLOCAL_W2 zero-store keep their stage 1 semantics in the transposed indexing.
Unchanged from stage 1: moe_align_block_size call (with BLOCK_T), silu_and_mul(expert_ids=local_ids), cache13/cache2 buffers, fused_post_expert_norm_weighted_sum(slot_expert_ids=local_ids), everything outside the expert block. prepare(): rewrite _check_expert_gemms for the reordered layout, comparing against nvfp4.dequantize of the ORIGINAL weights on a mid-K tile that spans both nibble halves and all 8 scale groups of one 128-chunk, for every config (this also warm-compiles them). Before the run, compile one BLOCK_T 128 config locally if possible and check the TTGIR for ttng.warp_group_dot with a register LHS (dot_op opIdx = 0, parent = mma v3) rather than a local_alloc memdesc; record the finding in the implementation notes either way.

## Expected

- `*load=first_turn*` faster: compute-bound (~512 rows/expert, 1.65 TFLOP per call). Marlin (mma.sync) is 11.1-12.0 ms and stage 1 is 13.3-14.4 ms (~120 TFLOP/s, no effective wgmma). A 128x128 swap-AB tile with a register LHS lets wgmma run without the per-tile smem round trip. Milestone: _moe_nvfp4_gemm_kernel x2 at or below 9 ms (Marlin parity <= 11.1 ms is the minimum); floor ~2.5 ms at ~660 TFLOP/s
- `*load=turn_4*` flat: ~94 rows/expert, one 128-token tile per expert, so weights are dequantized once (864 MiB packed+scales, floor ~190 us). Stage 1 was 3341/3457 us vs Marlin 2248/2201 us; removing the interleave/trans/smem round trip and raising MMA M to 128 should reach Marlin parity. Milestone: <= 2250 us
- `*load=turn_1*` slower: ~24 rows/expert: dequant/memory-bound. Stage 1 2026/2008 us (2.8x Marlin 730/681 us); the fix removes the per-tile layout conversion and halves the weight-tile dequant passes (one 64-token tile per expert instead of 1-2 32-row tiles) but leaves the per-element ALU dequant, so it likely stays behind Marlin. Milestone: <= 1000 us (2x better than stage 1); the turn_1 loss shrinking from 0.46x toward ~0.8x is the attribution signal

## Risk

High risk. (a) Triton may refuse or silently not use the register-LHS wgmma path for a computed bf16 tile (then it falls back to an smem store of the LHS, still without interleave/trans; the first_turn row tells which). (b) The nibble reorder changes the stored packing: the prepare() check against nvfp4.dequantize of the original weights must cover both halves and the scale-group indexing, or a layout bug fails the tests. (c) Register pressure at num_warps 4 with a [128, 64] bf16 LHS plus a [128, 64] fp32 acc; spills show as a turn_1 time near stage 1. Rules: NVFP4 stays NVFP4 (a lossless code permutation, repacked once in prepare(), as Marlin repacking was), dequant only in registers, no BF16 weight copy, no autotune, no host sync, nothing input-derived cached; configs branch on T only.

## Milestones

Rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
