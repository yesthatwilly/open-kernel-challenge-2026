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
status: running
review_at: 2026-10-24T075347Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T075347Z
base_sha: 79a102b3fd18c6ab5fa1e8d8d7390e6645a26ec3
base_blob: e4931f050c7fb99835fe116d25bb454a5ea731e9
branch: hyp/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4
worktree: .worktrees/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4
mode: check
implemented_at: 2026-10-24T080302Z
impl_commit: 6a70bfaac484039ebab2355adee2262c864eee2d
submission_sha256: b9321c229e90f61f2f88abeda1179554d0d1f875d99604908ea14bd343888af2
norm_sha256: bd6263fe3e5ed19056cb8b1c22554076b1c79600f5588c0dc06b5d7c0365b668
submission_blob: 2e6cb7c89a513e08be4b5e163319fbc34a35f854
lint_warnings: 2
attempt_file: 2026-10-24T080302Z-check-experts-triton-w4a16-inkernel-fp4.md
prepared_at: 2026-10-24T080302Z
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


## Implementation notes

All changes are in `muse_prefill_layer/submission.py` (worktree), confined to the expert block: the Marlin bank, the two `moe_wna16_marlin_gemm` calls and the block-size choice. `moe_align_block_size`, `silu_and_mul(expert_ids=local_ids)`, the `cache13`/`cache2` buffers and `fused_post_expert_norm_weighted_sum(slot_expert_ids=local_ids)` are unchanged.

- **Removed:** `_marlin_bank` (`prepare_moe_nvfp4_layer_for_marlin`, `marlin_make_workspace`), `moe_wna16_marlin_gemm`, `scalar_types`, `types` and `BLOCK_SIZE_M_CANDIDATES`. State now holds `bank = _expert_bank(weights)`: the original `w13`/`w13_sf`/`w13_gs`/`w2`/`w2_sf`/`w2_gs` tensors, `.contiguous()`. They stay as uint8 codes, e4m3 block scales and fp32 global scales, with no repack and no BF16 copy anywhere.
- **Added `_moe_nvfp4_gemm_kernel`:** SGLang's `fused_moe_kernel` structure (GROUP_SIZE_M swizzle, sorted_ids gather with `// TOP_K`, `token_mask = offs_token < num_valid_tokens`, the `off_experts == -1` guard, no routed-weight multiply, fp32 acc, bf16 store). Only the B-tile path differs. B is loaded as uint8 `[BLOCK_N, BLOCK_K//2]` from the stored N-major layout (contiguous along K, so the loads vectorize). The low nibble (k = 2i) and the high nibble (k = 2i+1) are each converted to bf16, multiplied by the block scale, `tl.interleave`d to `[BLOCK_N, BLOCK_K]` and `tl.trans`posed into `tl.dot`. The scale tile is loaded as `[BLOCK_N, BLOCK_K//2]` at index `i // 8` (= k // 16), so one tile serves both nibbles. Strides come from constexpr N and K. `num_valid_tokens` and `EM` are `do_not_specialize`, so the variants compiled in prepare() are the ones kernel_fn hits for every T. The grid is `cdiv(sorted_ids.shape[0], BLOCK_M) * N/BLOCK_N`, the static bound. The early exit compares `pid_m * BLOCK_M` against `num_tokens_post_padded`, loaded on the device. There is no `.item()`.
- **Deviation (a), bit-placement E2M1:** replaces the spec's `where`/`exp2`. `bits = ((c & 7) << 6) | ((c & 8) << 12)`; the high nibble uses `(b & 0x70) << 2 | (b & 0x80) << 8`. Bitcast to bf16, this is exactly E2M1 * 2^-126, with the e = 0 codes on bf16 subnormals (0.5 -> 2^-127). The power-of-two constant is folded into the scale. `sb = bf16(e4m3_scale * 2^119)` is exact, because e4m3 has 4 significant bits and 448 * 2^119 < bf16 max. Then `w = bits_bf16 * sb = E2M1 * scale * 2^-7`, exact in bf16: at most 6 significant bits, and no underflow down to the e4m3 minimum 2^-9. That is one bf16 multiply per element. Correctness depends on the subnormal operand not being flushed to zero. Triton does not emit `.ftz` for plain mul, but I could not verify that here. The prepare() check catches a flush and names it (the mismatches are then only the +-0.5 codes).
- **Deviation (b), gs in the epilogue:** `acc * (gs[e] * 2^7)` once per output, fp32, then cast to bf16. The spec applied it per weight. This is also slightly more accurate: one rounding of the exact product, against the reference's fp32 sf*gs rounding.
- **Deviation (c), BLOCK_M from the local rows (~T/32), as a host-side branch on `tokens`:** spec was BLOCK_M 128 everywhere. `EXPERT_CONFIGS`:
  - T <= 512: BM 16 (BN 128, BK 64, 4 warps, 4 stages)
  - T <= 1024: BM 32 (turn_1, T = 753)
  - T <= 4096: BM 64 (turn_4, T = 3012, and the T = 4096 test)
  - above that: BM 128 (BN 128, BK 64, GROUP_SIZE_M 8, 8 warps, 3 stages: the spec's config, for first_turn)

  `moe_align_block_size` uses the same BLOCK_M. BN 128 and BK 64 (four scale groups per tile) are kept for all of them. The small-M warps and stages are my picks, not tuned. No autotune.
- **Non-local (-1) rows (ZERO_NONLOCAL_W2 = True):** I could not read SGLang's `fused_post_expert_norm_weighted_sum` or Marlin's is_ep handling on this node. So, following the "if unsure, zero-store" rule, the w2 GEMM zero-stores its -1 blocks, as SGLang's Triton `write_zeros_to_output` guard does. The w13 GEMM skips its -1 blocks without storing: those cache1 rows reach only `silu_and_mul` and then only the w2 GEMM's -1 blocks, which never read A, so they cannot reach the sum. The circumstantial evidence that the sum masks anyway: master runs Marlin with is_ep=True over `torch.empty` caches, both downstream ops take the expert ids, the non-local slots carry non-zero topk weights, and master passes. The zero-store cost is about 0.9 GB of writes at first_turn (~110k non-local rows x 4096 bf16, ~190 us, ~3% of the 5.6 ms shape) and ~43 MB at turn_1. If masking is confirmed, setting the flag to False is a one-line follow-up, which this run's result should not be credited with.
- **prepare() check (`_check_expert_gemms`):** runs the real GEMM kernel for every variant kernel_fn can launch (4 configs x {w13 with TOP_K=8, w2 with TOP_K=1, its zero-store flag}), which warm-compiles them all before capture. Each case is synthetic, built from the weights only, on expert 31 (the largest offsets) at k0 = K/2 + 64 (mid K loop). Row r of A is one-hot at k0 + r % 64, so output row r is column k0 + r%64 of the dequantized expert, over every N column. That covers a full 128-row x 64-k tile: all four 16-wide scale groups, both nibbles, the K-loop pointer advance and the epilogue. It is compared with `nvfp4.dequantize` at rel tol 2^-7, and NaN counts as a failure. It also asserts that a -1 block leaves its rows untouched (w13) or zero (w2), and that a block past `num_tokens_post_padded` is never written (canary NaN rows).
- **Open, cannot check here (flag):** whether `tl.dot` with this register-produced bf16 B (interleave + trans) lowers to `wgmma.mma_async` (after a local_store to smem) on sm_90 in the judge's Triton, or falls back to mma v2. Only the BM 64/128 configs are wgmma candidates. BM 16/32 are mma v2 regardless. If BM 128 falls back, the first_turn advantage over Marlin's mma.sync mostly disappears. Check TTGIR/PTX for `wgmma` locally before reading much into first_turn. Also unverified: the `interleave`/`trans` layout conversions may add smem round trips. The kernel needs Triton >= 3.0 for `tl.interleave`.

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4 diff`:

- [x] CUDA-graph capturable: kernel_fn's expert path has no `.item()`/`.cpu()`/`synchronize()`/prints. Config choice branches on `tokens` (a shape). Allocations are the same shape-sized `torch.empty` as before. The grid is from `sorted_ids.shape[0]` (static), and the padded count is read on the device for the early exit. The syncs and `bool(...)` asserts are only in `_check_expert_gemms`, called from prepare().
- [x] Nothing derived from `data` is cached across calls: state holds only the weight tensors. The prepare() check uses synthetic A built from the weights.
- [x] Weights: NVFP4 stays NVFP4 as stored (uint8 codes, e4m3 scales, fp32 global scales, `.contiguous()` only). The Marlin repack is dropped. Dequant happens only in registers inside the GEMM kernel, and no BF16 weight copy is materialized.
- [x] Activations BF16 where SGLang keeps BF16: GEMM A operands and outputs are bf16 with fp32 accumulate, as Marlin did. The KV cache and FP8 query path are untouched.
- [x] Attention sees every key: n/a to this change; the FA3 call is untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The new import is `nvfp4`, the problem's helper (reference.py imports it too; it is not the reference), used only in prepare().
- [x] All work happens inside the timed call: both GEMMs run every call on that call's sorted_ids. No output caching or replay.
- [x] First line `# okc: problem=muse_prefill_layer` kept. No `triton.autotune`: one fixed config per token-count branch.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py --problem muse_prefill_layer`), exit 0:

```
warn  host-sync    L361: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L417: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4/muse_prefill_layer/submission.py
```

Both warnings are prepare()-time syncs (the existing one, and the one in the dequant check). `python3 -I -m py_compile` passes, and its `__pycache__` was removed.
