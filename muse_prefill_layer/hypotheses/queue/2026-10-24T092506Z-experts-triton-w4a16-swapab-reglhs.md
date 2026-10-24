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
review: proceed
status: queued
review_at: 2026-10-24T100201Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T100201Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage-2 rewrite: putting the dequantized weight tile on the register-LHS side of an MMAv3 dot is the standard Hopper mixed-precision pattern, the nibble reorder is a lossless repack, and the profile rows (_moe_nvfp4_gemm_kernel vs Marlin per shape) will show which milestone it reached. There are two sharp notes. The turn_1 premise that the change 'halves the dequant passes' is mostly wrong, and splitting each K step into two halves doubles the per-element scale gather and convert relative to stage 1. Either could hide the layout-conversion savings, so make the cheap fixes below while building.

### Justification
Mechanism. On sm_90, wgmma takes A from registers (opIdx 0, parent mma v3) and B only from smem. Stage 1 put the in-register dequantized weight on the B side (tl.trans(tl.interleave(...))), which forces a local_store/convert of 8192 bf16 per K step. At BLOCK_M 16/32 (turn_1) it was also below wgmma's M=64, so it ran mma.sync. Swapping to acc[W, T] += W[128, k] @ A_t[k, T] makes the expert N the MMA M side (128 = 2 x m64 per warpgroup) and the tokens the MMA N side (n 64/128 is legal). The nibble reorder (byte i = k0+i low and k0+64+i high per 128-chunk) lets the same code byte feed both half-dots at the same layout position, so no interleave or trans is needed. That is the right design, and the scale indexing (k0/16 + i//16 for the low half, +4 for the high half) checks out. Profile support. Stage 1 measured 2026/2008 us at turn_1, 3341/3457 us at turn_4 and 13.3-14.4 ms at first_turn. Marlin is at 730/681, 2248/2201 and 10.2-12.0 ms. The weight-stream floor is ~190 us (864 MiB), and first_turn is ~1.65 TFLOP, so ~1.7 ms at 989 TFLOP/s and ~2.5 ms at a realistic 660. The losses shrinking with T (0.46x to 0.90x) fit a per-tile overhead, so the expected directions (first_turn faster, turn_4 to parity, turn_1 still slower) are consistent with TASK's weights and floors. The 138-weight sliding first_turn shape dominates its load, so first_turn is where the headline can move. Weakest links. (1) turn_1 attribution: at T=753 stage 1 already ran BLOCK_M 32 against ~24 rows per expert, so most experts were already dequantized once. Going to BLOCK_T 64 does not halve the passes there (it does at turn_4, where 64 becomes 128 against ~94 rows). Any turn_1 change will come from the removed convert/interleave/trans and the change of MMA path, not from fewer passes. (2) Stage 1 loaded one [BLOCK_N, 32] scale tile per 64 k, one scale load per byte shared by both nibbles. Stage 2 loads two [128, 64] gather tiles per 128 k, one per element, and repeats fp8->fp32, *2^119 and ->bf16 on each. That doubles the scale-gather LSU work and the scale-convert ALU work per weight element. Also, a 1-byte gather with constancy 16 cannot be staged with cp.async (4/8/16-byte only), so it is probably a synchronous ld.global on every K step in both stages, an unpipelined latency nobody has measured. (3) first_turn ALU vs MMA. Per 128x128 tile per 128-k step, the dequant is ~16K elements at ~8-12 int/bf16 ops each, ~0.6 us of one SM's ALU, about the same as the 4.2 MFLOP MMA at peak (~0.56 us/SM). BLOCK_T 128 re-dequantizes each weight tile 4x per expert. If Triton serializes a register-A wgmma (wait 0 before the next dequant), first_turn lands near (ALU + MMA) rather than max(ALU, MMA). That is still a plausible 6-9 ms, but the 9 ms milestone is not safe and the ~2.5 ms floor is out of reach for this design without BLOCK_T 256 or warp specialization. Cheap fixes while building, all within this hypothesis's area: compute sb once on a compact [128, 8] (8-byte-per-row, cp.async-able) scale tile per 128-chunk and broadcast it to each half, if Triton keeps the broadcast in the dot-op layout (record in the notes whether it does). Do not mutate weights['w13'] / ['w2'] in place: build the reordered bank as a new tensor.

### Confounders on the current kernel
- sglang-pipeline-parity (the only landed hypothesis): it supplies the local-id remap (-1 for non-local), moe_align_block_size, silu_and_mul and the post-expert weighted sum around the expert GEMMs. Stage 2 keeps all of these unchanged from stage 1 and changes only the GEMM kernel, its tile pick and the weight packing. The rewrite is judged on _moe_nvfp4_gemm_kernel's own us against the Marlin rows from the same parity base, so the parity change cannot pass for an effect of the rewrite. The one coupling is moe_align padding to BLOCK_T, which changes the -1/zero-store block count (ZERO_NONLOCAL_W2 writes ~T*7 x 4096 bf16 of zeros, ~200 us at first_turn). It counts inside the kernel's time in both stages, so read it as a fixed cost, not a confounder.

### Interference and A/B needs
No companions. This is a rewrite stage judged on the replaced kernel's own profile time per shape against Marlin and the floor. The per-shape _moe_nvfp4_gemm_kernel rows of this run against stage 1's (2026/3341/13524 us) and Marlin's attribute the result directly. No landed hypothesis tunes the same kernel: the block-m Marlin knobs were neutral or regressed and never landed, and the parity base is common to both sides. A control that isolates the nibble reorder or the swap from the tile change would not change the next-stage decision. With 3 prefill checks left, the next stage is the follow-up.

### Open questions
- Does the judge's Triton version keep a computed bf16 LHS in registers for ttng.warp_group_dot (no local_alloc of the LHS)? This node has no GPU, but triton.compile with a GPUTarget('cuda', 90, 32) may still emit TTGIR offline if ptxas is installed. Worth trying, as the hypothesis asks.
- Is the register-A wgmma issued async with the next K step's dequant overlapping, or does Triton insert warp_group_dot_wait 0 after each of the two half-dots? The first_turn time per tile tells which (near ALU+MMA means serialized).
- Is the [128, 64] i//16 scale gather pipelined or a synchronous per-iteration global load? If the turn_1 row barely moves from 2026 us, suspect the scale path before the MMA path.
- With soft routing, how many experts exceed 64 rows at turn_1 or 128 at turn_4 (a second, mostly-padded token tile that re-dequantizes)? The routing histogram from one test run would settle whether BLOCK_T 64 is right at T=753.
- Next-stage candidates if first_turn misses 9 ms: BLOCK_T 256 at first_turn (acc 128x256 fp32 = 128 regs/thread at 8 warps) to halve redundant dequant, or skipping the W2 zero-store for -1 blocks if the weighted sum can ignore them via slot_expert_ids.

