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
review: proceed
status: queued
review_at: 2026-10-24T075347Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T075347Z by an adversarial subagent (model opus).

**Verdict: proceed.** Worth one run. The first-turn shapes (the 138-weight sliding shape dominates that load) are the only plausible win, and the row-count arithmetic behind the turn_1/turn_4 predictions is wrong by 8x. The per-shape result separates the three loads, and no landed hypothesis needs reverting, so no companion run is warranted. Amend the tile config before building, or expect a turn_1 loss that says more about BLOCK_M 128 than about the engine.

### Justification
The 'expected' rows are miscounted. Only routes to this GPU's 32 experts reach the expert GEMMs. That is about T local routes (case_fields picks the busiest 32-expert range, and the FINDINGS figure of ~137 TFLOP/s at 16K only works out with ~T local routes: 16384*2*(8192*4096+4096*4096) = 1.65 TFLOP in 12.0 ms). Rows per expert are therefore ~T/32: ~23.5 at turn_1, ~94 at turn_4 and ~512 at first_turn, not the 188/753/4096 the hypothesis uses (T*8/32 counts all eight routes as local; SGLang's block-size heuristic makes the same assumption, which is why master picks block_size_m 64). Consequences per load. turn_1: each expert is already a single 64-row Marlin m-block, so there are no extra passes to halve, and the GROUP_SIZE_M swizzle finds no same-expert m-blocks to reuse. This shape is memory- and dequant-bound: about 864 MiB of packed weights and scales gives a ~190 us floor, and Marlin reaches ~1.24 TB/s in 730 us. A BLOCK_M 128 tile computes 32x128 padded rows (~18% useful), and the per-element ALU dequant (arithmetic E2M1 with where/exp2, fp8->fp32 scale, multiply by gs, cast) runs once over ~1.6G weight elements, in the same range as Marlin's whole time. Sister moe_w4a16 measured exactly this: Triton W4A16 with in-kernel FP4 was 2-2.6x slower than Marlin at small M. The likely direction here is slower, not faster. turn_4: 94 rows means Marlin does 2 passes and Triton 1, but re-streaming 864 MiB once more costs ~180 us at 4.8 TB/s against Marlin's 2248 us. Marlin is limited by its compute rate (~135 TFLOP/s useful), not by passes, so turn_4 only moves if the engine is faster. first_turn: 512 rows, compute-bound, Marlin at ~137 TFLOP/s (14% of 989) because it uses mma.sync. A Triton Hopper tl.dot with 128x128x64 tiles and 8 warps plausibly beats that even after paying the smem round-trip that wgmma needs for a register-dequantized B, so this is the shape where the mechanism actually applies. With Marlin at 64% of the 138-weight shape, a 2x engine gain there is worth ~1.4x on that load. The weakest link is the claim that 'the dequant is ALU work hidden under the MMA'. Triton has no warp specialization, and at ~10 fp32 ops per element against 2*BLOCK_M MMA flops per element, dequant and MMA throughput are roughly comparable at BLOCK_M 128 on H100/H200. At BLOCK_M 32 dequant dominates. Cheap mitigations that keep the change in scope: (a) bit-trick E2M1->bf16 (place the 3 exp/mantissa bits into the bf16 exponent/mantissa field and multiply by a power-of-two constant, which handles e=0 subnormals; then OR the sign) in place of where/exp2; (b) fold gs[e] into the fp32 accumulator in the epilogue, a per-expert scalar applied once per output, not once per weight; (c) choose BLOCK_M from the real local rows, T/32: BLOCK_M 16-32 for T<=1024, 64-128 for turn_4, 128 for first_turn. Without (c), a turn_1 loss is expected and uninformative.

### Confounders on the current kernel
- sglang-pipeline-parity (the only landed hypothesis) is the base itself. It fixes SGLang's block_size_m heuristic (T*8/32, so 64 at every scored shape), the moe_align_block_size call, the cache1/cache3 torch.empty buffers and the post-expert weighted sum with slot_expert_ids. The new kernel replaces the GEMMs and changes the moe_align block to 128. Measured deltas are therefore 'Triton@128 vs Marlin@64' and include a padding change: at turn_1 the 128-row padding inflates useful-to-padded work about 2x versus 64, which counts against the new kernel and does not come from the engine. That is not an interaction a revert can remove. Reverting the parity port would remove the comparison baseline, so it is not a meaningful A/B.
- Not landed, but a cheaper competitor: Marlin's block_size_m chosen from the true local rows (~T/32, i.e. 16-32 at turn_1; decode_layer measured +8-15% from capping it at 16 at T=128/256). If turn_1 comes out faster here, part of that gain may be the block-size change, not the engine. Keep that in mind before promoting a turn_1 gain.

### Interference and A/B needs
No companion is warranted. The result is reported per shape and the three loads separate cleanly (turn_1, turn_4, first_turn), so a single run gives the engine's direction per regime. A first_turn gain is unambiguous: Marlin is the only thing replaced, nothing landed touches it beyond the parity base, and no dequant-to-BF16 path has landed (the user has also ruled out per-call BF16 copies), so first_turn is a direct engine A/B, not the 'flat control' the hypothesis describes. A turn_1/turn_4 gain is likewise unambiguous. A loss at turn_1 would be ambiguous between the engine and the 128-row tile. That ambiguity runs one way only, so it does not meet the bar for a companion; fix it by building with BLOCK_M chosen from T/32 rather than by spending a second run. An ab-revert of sglang-pipeline-parity is meaningless, because it is the base. If the run shows first_turn faster and the turn shapes slower, the natural follow-up is a stacked hypothesis that gates the Triton kernel to T >= ~8192 and keeps Marlin (possibly with a smaller block_size_m) below that. That is a follow-up hypothesis, not a companion.

### Open questions
- Does Triton (the version in the judge image) lower tl.dot with a register-produced bf16 B to wgmma, after a local_store to smem, on sm_90, or does it fall back to mma v2? If it falls back, the first_turn advantage over Marlin's mma.sync largely disappears. Check the TTGIR/PTX (look for wgmma.mma_async) in a local compile before spending the run.
- What is the actual local-route count per scored shape? It is ~T according to the 137 TFLOP/s reconciliation, but case_fields chooses the busiest range, so it may be 1.1-1.3x T. That decides the BLOCK_M per shape.
- Should the hypothesis be amended now to (a) set BLOCK_M from T/32, (b) use bit-trick E2M1 conversion and (c) apply gs in the epilogue? All three stay inside the stated area and make a turn-shape loss attributable to the engine.
- Is sf in nvfp4's layout linear [e, n, k//16], as the weights dict states, with the low nibble at even k (confirmed in nvfp4.unpack_e2m1)? The prepare() assert covers it, but it should compare a full 128x64 tile including the scale-group boundaries, not one row-slice.

