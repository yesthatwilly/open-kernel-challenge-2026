---
problem: muse_decode_layer
slug: moe-chain-cuda-swapab-expert-once-large-t
created: 2026-10-24T172107Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  WHY: at T=128/256 the two Marlin MoE GEMMs are the single largest block of the whole problem:
  254/271 us (sliding/full verify_16) and 443/427 us (verify_32) in a017e10aec544c55, 46-70% of
  those shapes, at ~1.1-1.2 TB/s, while the MoE-only floor is about 101/109 us (muse_moe_w4a16 SoL
  at bs=128/256, same weights and routing family). Under EP8 a local expert gets only ~4 (T=128)
  to ~8 (T=256) routes on average (hot ones 30-60), so the GEMM is a pure NVFP4 weight stream;
  Marlin re-streams an expert once per 16-row m-block (cap 16) and underfills. Nothing in the
  decode queue or in moe covers T>=128 with CUDA (moe swapab-nvfp4-gemm-verify-rewrite is Triton
  and held; moe cuda-nvfp4-swapab-mma-persistent-small-m, b65c9aabd7, is M<=32 only). STAGE PLAN:
  (1) this file: three new launches replace align + w13 + silu + w2 at tokens > 32; pre-expert
  F.rms_norm and fused_post_expert_norm_weighted_sum stay. (2) fold the pre-expert RMSNorm into
  the w13 B-operand load (per-token rsqrt from the route-list kernel) and the post-expert norm
  sum-of-squares into the w2 epilogue (fp32 atomics into a [T*8] buffer) so the combine becomes
  one cheap pass; and chain w13 -> w2 per expert with a device-side per-expert completion counter
  inside one persistent launch (no host sync) to remove the inter-kernel tail. (3) extend the same
  kernels down to T<=32 (n8 fragments) and retire Marlin, or adopt the moe small-m CUDA kernel if
  it lands. STAGE-1 SCOPE, gate tokens > 32 from config (covers verify_16/32 and the batch 33 and
  batch 5 x q_len 8 tests; T<=32 keeps master byte for byte). PREPARE (lossless, NVFP4 stays
  NVFP4): build a CUDA bank from w13/w13_sf/w13_gs, w2/w2_sf/w2_gs (local experts 0..31). w13 row
  order permuted so every 16-row MMA tile holds 8 gate rows i..i+7 then the 8 matching up rows
  I+i..I+i+7 (I=4096): in the m16n8k16 C fragment a lane then holds gate (c0,c1, row lane/4) and
  up (c2,c3, row lane/4+8) for the same intermediate column, so silu(gate)*up is computed in
  registers with no smem exchange. Codes and e4m3 group scales (16 k per scale) re-tiled so each
  lane fetches its A fragment for a (16 rows x 64 k) sub-tile with one 16-byte cp.async (reuse the
  permutation and dequant helpers from moe b65c9aabd7 _tile_codes/_tile_scales if they compile,
  else write them from this spec); fold w*_gs into the scale at dequant time (fp32 e4m3*gs ->
  bf16x2 multiply), never store a dequantized weight. Compile with
  torch.utils.cpp_extension.load_inline in prepare (one TU, -O3 -arch=sm_90a, print compile
  seconds; attention compiled a similar TU in 41 s). KERNEL 1 route_list (one CTA, 1024 threads):
  from local_ids [T,8] (master expert_map path) count routes per local expert in smem, exclusive
  prefix, write route_idx (t*8+s grouped by expert), expert_off[33], and a compacted work list of
  active (expert, n_tile, token_chunk) items for w13 and for w2, plus their lengths, and reset
  both kernels work counters; fixed grid, data-dependent counts stay on device. KERNEL 2 w13_silu
  (persistent, grid = #SMs, 1 CTA/SM, 8 warps): item = (expert e, 128-row w13 tile j of 64, chunk
  of <=64 routes of e); weight tile 128 rows x 4096 k streamed exactly once per chunk through a
  4-stage cp.async ring of 256-k slices (16 KB codes + 2 KB scales per stage); B operand = the
  chunk routes rows of the normed expert input gathered by route_idx (bf16, 64 x 256 per stage,
  ldmatrix.trans from smem; rows past the chunk zero-filled with cp.async src-size 0); swap-AB
  mma.sync m16n8k16 bf16 (weight dequantized e2m1->bf16 in registers with the Marlin-style
  lop3/prmt conversion, 8 nibbles per u32) with tokens on N, fp32 acc 16 x 64 per warp; epilogue
  silu(g)*u in fp32 -> bf16 into act[route, 4096] (route-indexed rows, 64 cols per tile). Items
  grabbed by atomicAdd on the work counter; hot experts with > 64 routes take extra chunks. KERNEL
  3 w2 (same template, not gated): item = (e, 128-row w2 tile of 32, chunk), K = 4096 from act
  rows, output bf16 y[t*8+s, 4096] written only at local route rows exactly where Marlin wrote
  them, so fused_post_expert_norm_weighted_sum(y.view(T,8,4096), slot_expert_ids=local_ids) is
  unchanged. FALLBACK and SELF-CHECK: in prepare, after compile, run synthetic cases at T = 40,
  128, 256 with skewed routing (one expert with 96 routes to exercise multi-chunk, some experts
  empty, remote slots present) against the master Marlin path built from the same bank: per-row
  rel L2 of y local rows <= 1e-2, global <= 5e-3, and with y pre-filled with NaN the combine
  output must stay finite (no remote row read). Any compile or check failure sets
  state[moe_cuda_large_t]=False and keeps Marlin; print one line [moe-cuda-large-t] ON/OFF with
  compile s, max row rel, and reason. MILESTONES (profile top_kernels at verify_16/32): kernel
  names moe_w13_silu_swapab, moe_w2_swapab, moe_route_list replace both Marlin rows and
  act_and_mul; M1 correct and replaced-kernel sum <= Marlin+act_and_mul (254/443 us sliding); M2
  <= 200 us at T=256 and <= 180 at T=128 (>=2.6 TB/s on touched experts); M3 <= 160 / 140 us
  (>=3.3 TB/s). If M1 fails with the stream below 2 TB/s, the stage-2 fix is the dequant path
  (ALU-bound like moe triton-w13-grouped-gemv-small-m), not more CTAs.
expected:
  - shapes: "*load=verify_32 *"
    direction: faster
    why: "incumbent Marlin w13+w2 443 us (sliding) / 427 us (full) + act_and_mul ~6 us + align ~3 us of 623/906 us; MoE floor ~109 us; target <=190 us at M2: ~250 us off, sliding shape 623 -> ~370 us (+60%), full 906 -> ~660 us; shape floors 143.8 / 267.2 us"
  - shapes: "*load=verify_16 *"
    direction: faster
    why: "incumbent Marlin 254 / 271 us + act_and_mul ~4 us of 439/601 us; MoE floor ~101 us; target <=170 us: ~85-100 us off, sliding 439 -> ~350 us (+25%); shape floors 126.6 / 188.7 us. Headline from both verify loads at M2: 1.166x -> ~1.28-1.31x"
  - shapes: "*load=decode_*"
    direction: flat
    why: "T<=32 outside the gate: control"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "T=8 outside the gate: control"
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: moe_align_block_size + Marlin w13 + silu_and_mul (act_and_mul_kernel) + Marlin w2 at T > 32 (verify_16 T=128, verify_32 T=256)
review: required
status: queued
---

# moe-chain-cuda-swapab-expert-once-large-t

## Change

WHY: at T=128/256 the two Marlin MoE GEMMs are the single largest block of the whole problem: 254/271 us (sliding/full verify_16) and 443/427 us (verify_32) in a017e10aec544c55, 46-70% of those shapes, at ~1.1-1.2 TB/s, while the MoE-only floor is about 101/109 us (muse_moe_w4a16 SoL at bs=128/256, same weights and routing family). Under EP8 a local expert gets only ~4 (T=128) to ~8 (T=256) routes on average (hot ones 30-60), so the GEMM is a pure NVFP4 weight stream; Marlin re-streams an expert once per 16-row m-block (cap 16) and underfills. Nothing in the decode queue or in moe covers T>=128 with CUDA (moe swapab-nvfp4-gemm-verify-rewrite is Triton and held; moe cuda-nvfp4-swapab-mma-persistent-small-m, b65c9aabd7, is M<=32 only). STAGE PLAN: (1) this file: three new launches replace align + w13 + silu + w2 at tokens > 32; pre-expert F.rms_norm and fused_post_expert_norm_weighted_sum stay. (2) fold the pre-expert RMSNorm into the w13 B-operand load (per-token rsqrt from the route-list kernel) and the post-expert norm sum-of-squares into the w2 epilogue (fp32 atomics into a [T*8] buffer) so the combine becomes one cheap pass; and chain w13 -> w2 per expert with a device-side per-expert completion counter inside one persistent launch (no host sync) to remove the inter-kernel tail. (3) extend the same kernels down to T<=32 (n8 fragments) and retire Marlin, or adopt the moe small-m CUDA kernel if it lands. STAGE-1 SCOPE, gate tokens > 32 from config (covers verify_16/32 and the batch 33 and batch 5 x q_len 8 tests; T<=32 keeps master byte for byte). PREPARE (lossless, NVFP4 stays NVFP4): build a CUDA bank from w13/w13_sf/w13_gs, w2/w2_sf/w2_gs (local experts 0..31). w13 row order permuted so every 16-row MMA tile holds 8 gate rows i..i+7 then the 8 matching up rows I+i..I+i+7 (I=4096): in the m16n8k16 C fragment a lane then holds gate (c0,c1, row lane/4) and up (c2,c3, row lane/4+8) for the same intermediate column, so silu(gate)*up is computed in registers with no smem exchange. Codes and e4m3 group scales (16 k per scale) re-tiled so each lane fetches its A fragment for a (16 rows x 64 k) sub-tile with one 16-byte cp.async (reuse the permutation and dequant helpers from moe b65c9aabd7 _tile_codes/_tile_scales if they compile, else write them from this spec); fold w*_gs into the scale at dequant time (fp32 e4m3*gs -> bf16x2 multiply), never store a dequantized weight. Compile with torch.utils.cpp_extension.load_inline in prepare (one TU, -O3 -arch=sm_90a, print compile seconds; attention compiled a similar TU in 41 s). KERNEL 1 route_list (one CTA, 1024 threads): from local_ids [T,8] (master expert_map path) count routes per local expert in smem, exclusive prefix, write route_idx (t*8+s grouped by expert), expert_off[33], and a compacted work list of active (expert, n_tile, token_chunk) items for w13 and for w2, plus their lengths, and reset both kernels work counters; fixed grid, data-dependent counts stay on device. KERNEL 2 w13_silu (persistent, grid = #SMs, 1 CTA/SM, 8 warps): item = (expert e, 128-row w13 tile j of 64, chunk of <=64 routes of e); weight tile 128 rows x 4096 k streamed exactly once per chunk through a 4-stage cp.async ring of 256-k slices (16 KB codes + 2 KB scales per stage); B operand = the chunk routes rows of the normed expert input gathered by route_idx (bf16, 64 x 256 per stage, ldmatrix.trans from smem; rows past the chunk zero-filled with cp.async src-size 0); swap-AB mma.sync m16n8k16 bf16 (weight dequantized e2m1->bf16 in registers with the Marlin-style lop3/prmt conversion, 8 nibbles per u32) with tokens on N, fp32 acc 16 x 64 per warp; epilogue silu(g)*u in fp32 -> bf16 into act[route, 4096] (route-indexed rows, 64 cols per tile). Items grabbed by atomicAdd on the work counter; hot experts with > 64 routes take extra chunks. KERNEL 3 w2 (same template, not gated): item = (e, 128-row w2 tile of 32, chunk), K = 4096 from act rows, output bf16 y[t*8+s, 4096] written only at local route rows exactly where Marlin wrote them, so fused_post_expert_norm_weighted_sum(y.view(T,8,4096), slot_expert_ids=local_ids) is unchanged. FALLBACK and SELF-CHECK: in prepare, after compile, run synthetic cases at T = 40, 128, 256 with skewed routing (one expert with 96 routes to exercise multi-chunk, some experts empty, remote slots present) against the master Marlin path built from the same bank: per-row rel L2 of y local rows <= 1e-2, global <= 5e-3, and with y pre-filled with NaN the combine output must stay finite (no remote row read). Any compile or check failure sets state[moe_cuda_large_t]=False and keeps Marlin; print one line [moe-cuda-large-t] ON/OFF with compile s, max row rel, and reason. MILESTONES (profile top_kernels at verify_16/32): kernel names moe_w13_silu_swapab, moe_w2_swapab, moe_route_list replace both Marlin rows and act_and_mul; M1 correct and replaced-kernel sum <= Marlin+act_and_mul (254/443 us sliding); M2 <= 200 us at T=256 and <= 180 at T=128 (>=2.6 TB/s on touched experts); M3 <= 160 / 140 us (>=3.3 TB/s). If M1 fails with the stream below 2 TB/s, the stage-2 fix is the dequant path (ALU-bound like moe triton-w13-grouped-gemv-small-m), not more CTAs.

## Expected

- `*load=verify_32 *` faster: incumbent Marlin w13+w2 443 us (sliding) / 427 us (full) + act_and_mul ~6 us + align ~3 us of 623/906 us; MoE floor ~109 us; target <=190 us at M2: ~250 us off, sliding shape 623 -> ~370 us (+60%), full 906 -> ~660 us; shape floors 143.8 / 267.2 us
- `*load=verify_16 *` faster: incumbent Marlin 254 / 271 us + act_and_mul ~4 us of 439/601 us; MoE floor ~101 us; target <=170 us: ~85-100 us off, sliding 439 -> ~350 us (+25%); shape floors 126.6 / 188.7 us. Headline from both verify loads at M2: 1.166x -> ~1.28-1.31x
- `*load=decode_*` flat: T<=32 outside the gate: control
- `*load=verify_1 *` flat: T=8 outside the gate: control

## Risk

Biggest risk: the in-register NVFP4 dequant makes the stream ALU-bound below Marlin (Triton versions were 2-2.6x slower at small M), or the mma fragment / scale permutation is wrong; the prepare self-check catches the latter and keeps Marlin. Compile time ~40-90 s fits the 15 min run. Confounding: none of the running decode runs touches T>32 MoE; marlin-block-m-cap-16 (landed) is the incumbent being replaced. Judge on the replaced kernels rows, not the headline.

## Milestones

Rewrite stage 1 of `moe_align_block_size + Marlin w13 + silu_and_mul (act_and_mul_kernel) + Marlin w2 at T > 32 (verify_16 T=128, verify_32 T=256)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
