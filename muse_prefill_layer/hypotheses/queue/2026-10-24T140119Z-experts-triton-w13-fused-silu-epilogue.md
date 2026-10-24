---
problem: muse_prefill_layer
slug: experts-triton-w13-fused-silu-epilogue
created: 2026-10-24T140119Z
base: master
area: silu-mul (fused into the w13 Triton NVFP4 GEMM epilogue, T > 4096)
change: >
  Fuse SGLang's silu_and_mul into the epilogue of the first (w13) Triton NVFP4 GEMM on the T >
  4096 path (_experts_triton, first_turn only), built on master. (1) prepare(): build the Triton
  bank's w13 codes and w13 block scales with their N rows permuted (lossless, a new tensor; the
  weights stay untouched and still NVFP4) so that every 128-row W tile holds 64 gate rows followed
  by the matching 64 up rows: new row 128j + r = old row 64j + r (gate) for r < 64, and old row
  4096 + 64j + (r - 64) (up) for r >= 64, j = 0..63. Apply the same permutation to w13_sf
  (row-wise); w13_gs is per expert and unchanged. The Marlin bank is built from the original
  weights as today, so T <= 4096 is untouched. _check_expert_gemms compares the w13 variants
  against nvfp4.dequantize of the original rows taken in the permuted order (an index_select of
  ref by the same permutation) and keeps every other assertion. (2) _moe_nvfp4_gemm_kernel gets
  constexpr FUSE_SILU (False for w2 and for the check of the unfused output layout): when True,
  after acc * gs, round to bf16 and back to fp32 (so numerics match today's bf16 cache1 feeding
  silu_and_mul), split acc [BLOCK_W=128, BLOCK_T] into g = rows 0..63 and u = rows 64..127
  (tl.reshape to (2, 64, BLOCK_T), tl.permute, tl.split), compute g * sigmoid(g) * u in fp32 as
  SGLang's act_and_mul does, and store bf16 into the [num_slots, 4096] output at columns pid_w *
  64 + arange(64), rows = sorted ids, masked by token_mask as today. Blocks with expert -1 return
  without writing, as the w13 call does today (silu_and_mul also skips -1 slots, so cache2's -1
  rows were never written before either). (3) _experts_triton: allocate cache2 [num_slots, 4096]
  and pass it as the w13 call's output with FUSE_SILU=True; drop cache1 and the silu_and_mul call;
  cache3 for the w2 output is unchanged (it may now be its own [num_slots, 4096] allocation). The
  w2 call, the post-expert norm, EXPERT_CONFIGS, the shape gate and kernel_fn are unchanged. Add a
  FUSE_SILU_W13 = True module constant so a control can restore the unfused path. No BF16 weight
  copy, no autotune, no host sync.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "removes the act_and_mul_kernel (286.6-297.2 us per first_turn shape in the master profile) and the w13 GEMM's bf16 store of gate+up (16384 local rows x 8192 x 2 B = 268 MB written then re-read by silu, ~110 us of traffic) in favour of a 134 MB store of the activated half: ~300-400 us off a 14.3-22.3 ms shape, 2-2.5% on the 138-weight sliding first_turn and the first_turn load"
  - shapes: "*load=turn_*"
    direction: neutral
    why: "T <= 4096 runs Marlin with the unpermuted Marlin bank and SGLang's silu_and_mul; untouched"
risk: low
depends_on: experts-hybrid-marlin-turn-triton-first-turn
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# experts-triton-w13-fused-silu-epilogue

## Change

Fuse SGLang's silu_and_mul into the epilogue of the first (w13) Triton NVFP4 GEMM on the T > 4096 path (_experts_triton, first_turn only), built on master. (1) prepare(): build the Triton bank's w13 codes and w13 block scales with their N rows permuted (lossless, a new tensor; the weights stay untouched and still NVFP4) so that every 128-row W tile holds 64 gate rows followed by the matching 64 up rows: new row 128j + r = old row 64j + r (gate) for r < 64, and old row 4096 + 64j + (r - 64) (up) for r >= 64, j = 0..63. Apply the same permutation to w13_sf (row-wise); w13_gs is per expert and unchanged. The Marlin bank is built from the original weights as today, so T <= 4096 is untouched. _check_expert_gemms compares the w13 variants against nvfp4.dequantize of the original rows taken in the permuted order (an index_select of ref by the same permutation) and keeps every other assertion. (2) _moe_nvfp4_gemm_kernel gets constexpr FUSE_SILU (False for w2 and for the check of the unfused output layout): when True, after acc * gs, round to bf16 and back to fp32 (so numerics match today's bf16 cache1 feeding silu_and_mul), split acc [BLOCK_W=128, BLOCK_T] into g = rows 0..63 and u = rows 64..127 (tl.reshape to (2, 64, BLOCK_T), tl.permute, tl.split), compute g * sigmoid(g) * u in fp32 as SGLang's act_and_mul does, and store bf16 into the [num_slots, 4096] output at columns pid_w * 64 + arange(64), rows = sorted ids, masked by token_mask as today. Blocks with expert -1 return without writing, as the w13 call does today (silu_and_mul also skips -1 slots, so cache2's -1 rows were never written before either). (3) _experts_triton: allocate cache2 [num_slots, 4096] and pass it as the w13 call's output with FUSE_SILU=True; drop cache1 and the silu_and_mul call; cache3 for the w2 output is unchanged (it may now be its own [num_slots, 4096] allocation). The w2 call, the post-expert norm, EXPERT_CONFIGS, the shape gate and kernel_fn are unchanged. Add a FUSE_SILU_W13 = True module constant so a control can restore the unfused path. No BF16 weight copy, no autotune, no host sync.

## Expected

- `*load=first_turn*` faster: removes the act_and_mul_kernel (286.6-297.2 us per first_turn shape in the master profile) and the w13 GEMM's bf16 store of gate+up (16384 local rows x 8192 x 2 B = 268 MB written then re-read by silu, ~110 us of traffic) in favour of a 134 MB store of the activated half: ~300-400 us off a 14.3-22.3 ms shape, 2-2.5% on the 138-weight sliding first_turn and the first_turn load
- `*load=turn_*` neutral: T <= 4096 runs Marlin with the unpermuted Marlin bank and SGLang's silu_and_mul; untouched

## Risk

Risk: the epilogue split doubles the live fp32 values briefly and may raise register pressure at BLOCK_T 256 (8 warps, acc 128x256); check n_spills in the prepare log. Pairing gate/up in one 128-row tile is required (two separate 128-row accumulators would double acc registers). The gain is ~2%, near the ~1.5-2.5% drift floor, so judge it on the act_and_mul row disappearing and the _moe_nvfp4_gemm_kernel row not growing. It touches disjoint lines from experts-triton-nvfp4-packed-asm-dequant (dequant inside the K loop vs epilogue/prepare), so the two should 3-way merge.
