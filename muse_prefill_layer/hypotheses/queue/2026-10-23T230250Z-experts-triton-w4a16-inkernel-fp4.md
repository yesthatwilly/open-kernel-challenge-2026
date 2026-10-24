---
problem: muse_prefill_layer
slug: experts-triton-w4a16-inkernel-fp4
created: 2026-10-23T230250Z
base: master
area: w13 marlin + w2 marlin (W4A16 engine with in-kernel NVFP4 dequant, for the turn shapes)
change: >
  A Triton grouped-GEMM kernel with in-register NVFP4 dequant replaces both moe_wna16_marlin_gemm
  calls wherever master still runs Marlin (if a dequant-to-BF16 path gated on DEQUANT_MIN_TOKENS
  has landed, leave it and gate this kernel to the shapes below that threshold);
  moe_align_block_size (block 128), silu_and_mul and fused_post_expert_norm_weighted_sum stay the
  parity calls. Start from SGLang's fused_moe_kernel (fused_moe_triton/fused_moe.py) in its
  use_int4_w4a16 form and change only the B-tile dequant: B is loaded as uint8 [BLOCK_N, BLOCK_K
  // 2] from the ORIGINAL packed layout kept contiguous in prepare() (w13 [32, 8192, 2048], w2
  [32, 4096, 2048]: N-major rows, K packed two codes per byte along the last dim; the Marlin bank
  is dropped), split into the low nibble (k even) and high nibble (k odd), mapped to E2M1 by
  arithmetic (e = (n >> 1) & 3, m = n & 1, mag = where(e == 0, 0.5 * m, (1 + 0.5 * m) * exp2(e -
  1)), negated when bit 3 is set), multiplied by sf[e, n, k // 16] (e4m3 loaded as fp32, one scale
  per 16 k) and by gs[e], cast to bf16 and fed to tl.dot with the bf16 A tile; fp32 accumulate;
  bf16 output. Tile config: BLOCK_M 128, BLOCK_N 128, BLOCK_K 64 (four scale groups per tile),
  GROUP_SIZE_M 8, num_warps 8, num_stages 3; top_k=8 A-row gather for gemm1 and top_k=1 for gemm2,
  mul_routed_weight False, the off_experts == -1 guard kept. In prepare() (untimed) assert the
  kernel's dequant of one expert row-slice equals nvfp4.dequantize so a layout error fails before
  the tests.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: Marlin re-streams each expert 4-bit weight set once per 64-row m-block (3 passes at ~188 rows per expert) with max 64-row tiles; a 128-row tile halves the passes and the grouped swizzle keeps an expert tiles L2-resident across its m-blocks; the dequant is ALU work hidden under the MMA, and no BF16 copy is materialized, so there is no fixed cost at small T
  - shapes: "*load=turn_4*"
    direction: faster
    why: same mechanism at ~753 rows per expert (12 Marlin passes vs 6)
  - shapes: "*load=first_turn*"
    direction: faster
    why: same mechanism at 4096 rows per expert against Marlin; a landed dequant-to-BF16 path, if any, keeps these shapes and makes this a flat control
risk: high
depends_on: sglang-pipeline-parity
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# experts-triton-w4a16-inkernel-fp4

## Change

A Triton grouped-GEMM kernel with in-register NVFP4 dequant replaces both moe_wna16_marlin_gemm calls wherever master still runs Marlin (if a dequant-to-BF16 path gated on DEQUANT_MIN_TOKENS has landed, leave it and gate this kernel to the shapes below that threshold); moe_align_block_size (block 128), silu_and_mul and fused_post_expert_norm_weighted_sum stay the parity calls. Start from SGLang's fused_moe_kernel (fused_moe_triton/fused_moe.py) in its use_int4_w4a16 form and change only the B-tile dequant: B is loaded as uint8 [BLOCK_N, BLOCK_K // 2] from the ORIGINAL packed layout kept contiguous in prepare() (w13 [32, 8192, 2048], w2 [32, 4096, 2048]: N-major rows, K packed two codes per byte along the last dim; the Marlin bank is dropped), split into the low nibble (k even) and high nibble (k odd), mapped to E2M1 by arithmetic (e = (n >> 1) & 3, m = n & 1, mag = where(e == 0, 0.5 * m, (1 + 0.5 * m) * exp2(e - 1)), negated when bit 3 is set), multiplied by sf[e, n, k // 16] (e4m3 loaded as fp32, one scale per 16 k) and by gs[e], cast to bf16 and fed to tl.dot with the bf16 A tile; fp32 accumulate; bf16 output. Tile config: BLOCK_M 128, BLOCK_N 128, BLOCK_K 64 (four scale groups per tile), GROUP_SIZE_M 8, num_warps 8, num_stages 3; top_k=8 A-row gather for gemm1 and top_k=1 for gemm2, mul_routed_weight False, the off_experts == -1 guard kept. In prepare() (untimed) assert the kernel's dequant of one expert row-slice equals nvfp4.dequantize so a layout error fails before the tests.

## Expected

- `*load=turn_1*` faster: Marlin re-streams each expert 4-bit weight set once per 64-row m-block (3 passes at ~188 rows per expert) with max 64-row tiles; a 128-row tile halves the passes and the grouped swizzle keeps an expert tiles L2-resident across its m-blocks; the dequant is ALU work hidden under the MMA, and no BF16 copy is materialized, so there is no fixed cost at small T
- `*load=turn_4*` faster: same mechanism at ~753 rows per expert (12 Marlin passes vs 6)
- `*load=first_turn*` faster: same mechanism at 4096 rows per expert against Marlin; a landed dequant-to-BF16 path, if any, keeps these shapes and makes this a flat control

## Risk

High risk, large implementation: a Triton W4A16 kernel whose dequant arithmetic, nibble order and scale indexing must match nvfp4.py exactly (the prepare() assert and the five tests catch errors before timing). Performance risk: Triton's register-level dequant may not keep up with Marlin's ldmatrix/async pipeline at these sizes, in which case the turn shapes get slower and the hypothesis attributes a loss to the engine; the kernel must reach tensor-core throughput with the bf16 cast in the K loop (watch register spills at num_stages 3). Rules: weights stay packed NVFP4 in state (the original tensors, a lossless layout), dequant inside the kernel, BF16 A operands and outputs, fp32 accumulate, no host sync, fixed-shape allocation.
