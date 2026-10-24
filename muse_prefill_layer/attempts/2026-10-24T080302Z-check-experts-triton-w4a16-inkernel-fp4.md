# muse_prefill_layer · 2026-10-24T080302Z · check · experts-triton-w4a16-inkernel-fp4

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-23T230250Z-experts-triton-w4a16-inkernel-fp4.md` (branch `hyp/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4`).

- **Job:** `218515e2d0b14ce7` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `6a70bfa`

## Goal
A Triton grouped-GEMM kernel with in-register NVFP4 dequant replaces both moe_wna16_marlin_gemm calls wherever master still runs Marlin (if a dequant-to-BF16 path gated on DEQUANT_MIN_TOKENS has landed, leave it and gate this kernel to the shapes below that threshold); moe_align_block_size (block 128), silu_and_mul and fused_post_expert_norm_weighted_sum stay the parity calls. Start from SGLang's fused_moe_kernel (fused_moe_triton/fused_moe.py) in its use_int4_w4a16 form and change only the B-tile dequant: B is loaded as uint8 [BLOCK_N, BLOCK_K // 2] from the ORIGINAL packed layout kept contiguous in prepare() (w13 [32, 8192, 2048], w2 [32, 4096, 2048]: N-major rows, K packed two codes per byte along the last dim; the Marlin bank is dropped), split into the low nibble (k even) and high nibble (k odd), mapped to E2M1 by arithmetic (e = (n >> 1) & 3, m = n & 1, mag = where(e == 0, 0.5 * m, (1 + 0.5 * m) * exp2(e - 1)), negated when bit 3 is set), multiplied by sf[e, n, k // 16] (e4m3 loaded as fp32, one scale per 16 k) and by gs[e], cast to bf16 and fed to tl.dot with the bf16 A tile; fp32 accumulate; bf16 output. Tile config: BLOCK_M 128, BLOCK_N 128, BLOCK_K 64 (four scale groups per tile), GROUP_SIZE_M 8, num_warps 8, num_stages 3; top_k=8 A-row gather for gemm1 and top_k=1 for gemm2, mul_routed_weight False, the off_experts == -1 guard kept. In prepare() (untimed) assert the kernel's dequant of one expert row-slice equals nvfp4.dequantize so a layout error fails before the tests.

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
high
High risk, large implementation: a Triton W4A16 kernel whose dequant arithmetic, nibble order and scale indexing must match nvfp4.py exactly (the prepare() assert and the five tests catch errors before timing). Performance risk: Triton's register-level dequant may not keep up with Marlin's ldmatrix/async pipeline at these sizes, in which case the turn shapes get slower and the hypothesis attributes a loss to the engine; the kernel must reach tensor-core throughput with the bf16 cast in the K loop (watch register spills at num_stages 3). Rules: weights stay packed NVFP4 in state (the original tensors, a lossless layout), dequant inside the kernel, BF16 A operands and outputs, fp32 accumulate, no host sync, fixed-shape allocation.

## Result (after the run)
- job `218515e2d0b14ce7` (check), status done, passed True, tests 5/5
- speedup 0.6891x, 95% CI [0.6787, 0.6994], score 562201.8690349252 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.result.json
- **outcome: regression** (regression)
- files: `muse_prefill_layer/hypotheses/done/regression/experts-triton-w4a16-inkernel-fp4.result.json`, `muse_prefill_layer/hypotheses/done/regression/experts-triton-w4a16-inkernel-fp4.patch`; branch `hyp/muse_prefill_layer/experts-triton-w4a16-inkernel-fp4`
