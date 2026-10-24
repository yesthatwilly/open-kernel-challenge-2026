# muse_prefill_layer · 2026-10-24T164944Z · submit · experts-triton-w13-fused-silu-epilogue

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T140119Z-experts-triton-w13-fused-silu-epilogue.md` (branch `hyp/muse_prefill_layer/experts-triton-w13-fused-silu-epilogue`).

- **Job:** `108231906a30407e` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `920e957`

## Goal
Fuse SGLang's silu_and_mul into the epilogue of the first (w13) Triton NVFP4 GEMM on the T > 4096 path (_experts_triton, first_turn only), built on master. (1) prepare(): build the Triton bank's w13 codes and w13 block scales with their N rows permuted (lossless, a new tensor; the weights stay untouched and still NVFP4) so that every 128-row W tile holds 64 gate rows followed by the matching 64 up rows: new row 128j + r = old row 64j + r (gate) for r < 64, and old row 4096 + 64j + (r - 64) (up) for r >= 64, j = 0..63. Apply the same permutation to w13_sf (row-wise); w13_gs is per expert and unchanged. The Marlin bank is built from the original weights as today, so T <= 4096 is untouched. _check_expert_gemms compares the w13 variants against nvfp4.dequantize of the original rows taken in the permuted order (an index_select of ref by the same permutation) and keeps every other assertion. (2) _moe_nvfp4_gemm_kernel gets constexpr FUSE_SILU (False for w2 and for the check of the unfused output layout): when True, after acc * gs, round to bf16 and back to fp32 (so numerics match today's bf16 cache1 feeding silu_and_mul), split acc [BLOCK_W=128, BLOCK_T] into g = rows 0..63 and u = rows 64..127 (tl.reshape to (2, 64, BLOCK_T), tl.permute, tl.split), compute g * sigmoid(g) * u in fp32 as SGLang's act_and_mul does, and store bf16 into the [num_slots, 4096] output at columns pid_w * 64 + arange(64), rows = sorted ids, masked by token_mask as today. Blocks with expert -1 return without writing, as the w13 call does today (silu_and_mul also skips -1 slots, so cache2's -1 rows were never written before either). (3) _experts_triton: allocate cache2 [num_slots, 4096] and pass it as the w13 call's output with FUSE_SILU=True; drop cache1 and the silu_and_mul call; cache3 for the w2 output is unchanged (it may now be its own [num_slots, 4096] allocation). The w2 call, the post-expert norm, EXPERT_CONFIGS, the shape gate and kernel_fn are unchanged. Add a FUSE_SILU_W13 = True module constant so a control can restore the unfused path. No BF16 weight copy, no autotune, no host sync.

## What changed, and where
All changes are in the worktree's `muse_prefill_layer/submission.py` (uncommitted; diff stat: 1 file, +132 / -28).

- **Constants:** `FUSE_SILU_W13 = True` (setting it False restores the unfused bank and the separate `silu_and_mul` launch) and `W13_FUSE_ROWS = 64`.
- **`_moe_nvfp4_gemm_kernel`:** new constexpr `FUSE_SILU`.
  - When it is True, the epilogue computes `h = bf16(acc * gs)` back in fp32, then `g, u = tl.split(tl.permute(tl.reshape(h, (2, 64, BLOCK_T)), (1, 2, 0)))` and `y = g * sigmoid(g) * u` in fp32.
  - `y` is stored as bf16 with **row stride N // 2** and **column base pid_w * (BLOCK_W // 2)**, masked by token_mask, as the review requires.
  - Blocks with expert -1 return without writing (w13 passes ZERO_NONLOCAL=False). A `tl.static_assert` forbids ZERO_NONLOCAL together with FUSE_SILU, because the zero store uses the unfused [N] layout.
  - The unfused branch is unchanged.
- **`_nvfp4_moe_gemm`:** new `fuse_silu=False` argument. It asserts that `out` is N/2 wide when fusing, that fusing is never combined with zero_nonlocal, and that `BLOCK_W == 2 * W13_FUSE_ROWS`. It passes `FUSE_SILU` to the kernel.
- **`_w13_fused_row_perm` (new):** builds `perm = arange(n).view(2, n/2/64, 64).permute(1, 0, 2).reshape(n)`. New row 128j + r is old row 64j + r for r < 64, and old row 4096 + 64j + (r - 64) otherwise. It asserts that the result is a permutation.
- **`_reorder_nibbles(packed, row_perm=None)`:** applies `index_select(0, perm)` to each expert before the nibble reorder. This keeps the extra memory to one expert at a time, and the existing per-expert inversion check now runs against the permuted rows.
- **`_expert_bank`:**
  - Under FUSE_SILU_W13, `w13_sf` is `index_select(1, perm)` (a new tensor). `w13_gs` and the w2 tensors are unchanged.
  - The w13 codes are row-permuted and nibble-reordered.
  - The bank also holds `w13_perm` (None when unfused).
  - The Marlin bank is still built from clones of the original weights.
- **`_check_expert_gemms`:**
  - The reference now uses the **original** `weights[name+"_sf"]`. The bank's `w13_sf` is permuted, so pairing it with the original codes would have been wrong.
  - The reference and codes are then `index_select`ed by `perm` for w13. Every other assertion is unchanged.
  - For w13 it calls a new `_check_fused_silu` on every config, so the fused BLOCK_T 256 variant is compiled before capture and its `[expert gemm] w13 fused-silu BLOCK_T=... n_regs/n_spills/shared` line is printed.
- **`_check_fused_silu` (new):** checks the fused epilogue directly, as the review asked.
  - It takes the unfused variant's output (already checked against nvfp4.dequantize), puts its columns back in original gate|up order with `argsort(perm)`, and runs SGLang's real `silu_and_mul` on it (expert_ids = e, expert_step=1, as in production).
  - It then runs the fused variant on the same synthetic case into a NaN-filled [rows, 4096] buffer and compares.
  - On a mismatch it raises a precise error: how many values are bad, the max error, the first bad [row, col] with the got and want values, and that column's pid_w, gate row and up row.
  - It asserts that the -1 block and the block past n_padded stay unwritten (NaN).
  - **Tolerance:** 2^-6 relative plus 2^-24 of max|want|, about 2 bf16 ulps. This allows for a possibly different fp32 op order inside act_and_mul. A layout, stride or pairing bug is far outside it.
- **`prepare()`:** the check that the Marlin repack did not touch anything now compares `weights[...]` against its own clones (`weight_scales_before`). Before, it compared against the bank's clones, which would fail now that the bank's w13_sf is permuted. `_check_experts_e2e` (T=8192, Triton vs Marlin) goes through `_experts_triton`, so it runs the FUSE_SILU path end to end.
- **`_experts_triton`:** with `bank["w13_perm"]` set, it allocates `cache2 [num_slots, 4096]` as the w13 output (`fuse_silu=True`) and a separate `cache3 [num_slots, 4096]` for w2. cache1 and the `silu_and_mul` call are gone. The else branch is master's code. cache2 is read only by the w2 call (-1 blocks never read A). The post-expert norm reads y (cache3), which w2 zero-stores for -1 blocks.
- **Unchanged:** the w2 call, the post-expert norm, EXPERT_CONFIGS, the shape gate, `_experts_marlin` and kernel_fn.
- **Deviations from the spec:**
  - The tolerance is about 2 ulp rather than 1.
  - The fused check is also run (and compiled) at BLOCK_T 64 and 128, which kernel_fn never runs on the Triton path. This costs a little compile time in prepare.
  - The review's finer 8/8 interleave (its open question) was **not** implemented. It is a new idea: read the BLOCK_T=256 fused n_spills and shared lines in the prepare log first.
- **Not verifiable here (no GPU):** whether Triton accepts the reshape/permute/split of the wgmma accumulator at BLOCK_T 256 without a large smem convert_layout, and the resulting n_regs, n_spills and shared. The prepare log prints them as `[expert gemm] w13 fused-silu BLOCK_T=256 ...`. Compare them with master's `w13 BLOCK_T=256` line (254 regs, 4 spills, 140288 B). Shared memory may grow by the convert_layout scratch. If it goes over the 227 KB limit, the compile fails in prepare (a loud failure, not a silent one).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-triton-w13-fused-silu-epilogue diff`:

- [x] CUDA-graph capturable: the diff adds no `.item()`, `.cpu()`, `synchronize()` or tensor prints to kernel_fn's call graph. The `synchronize()` and `.tolist()` calls are in `_check_fused_silu`, which runs only from prepare(). The new branch in `_experts_triton` is on `bank["w13_perm"]` (prepare state, not input values). The allocations are shape-only `torch.empty`.
- [x] Nothing derived from `data` is cached across calls: state only gains `bank["w13_perm"]` and the permuted w13 and w13_sf, all built from the weights.
- [x] Weights: repacked losslessly in prepare() only. Whole rows of E2M1 codes and e4m3 scales are permuted (a permutation, asserted), and the nibble reorder is checked per expert to invert exactly. NVFP4 stays NVFP4, w13_gs is unchanged, and there is no BF16 weight copy.
- [x] Activations BF16 where SGLang keeps BF16: the gate|up value is rounded to bf16 before silu, matching today's bf16 cache1. silu*mul is computed in fp32 and stored as bf16, as act_and_mul does, and cache2 is bf16. KV cache and query precision are untouched.
- [x] n/a: attention is not touched (the diff is confined to the expert GEMM, its bank, the checks and `_experts_triton`).
- [x] No timing patches, harness frames or I/O, secrets, module edits, reference import, subprocesses or network. The only new import use is the existing `silu_and_mul` in prepare's check.
- [x] All work happens inside the timed call; no output is cached or replayed. The fused epilogue runs in the timed w13 GEMM, and prepare only builds weights-derived state and runs checks.
- [x] No `triton.autotune` was added (grep of the diff is clean). The first line `# okc: problem=muse_prefill_layer` is kept.

okc_lint output:

```
warn  host-sync    L520: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L556: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L564: idx = bad.nonzero()[0].tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L600: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L653: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L660: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L667: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .../experts-triton-w13-fused-silu-epilogue/muse_prefill_layer/submission.py
```

`python3 -I -m py_compile` passes, and the `__pycache__` it left was removed.

## Risks / what could fail
low
Risk: the epilogue split doubles the live fp32 values briefly and may raise register pressure at BLOCK_T 256 (8 warps, acc 128x256); check n_spills in the prepare log. Pairing gate/up in one 128-row tile is required (two separate 128-row accumulators would double acc registers). The gain is ~2%, near the ~1.5-2.5% drift floor, so judge it on the act_and_mul row disappearing and the _moe_nvfp4_gemm_kernel row not growing. It touches disjoint lines from experts-triton-nvfp4-packed-asm-dequant (dequant inside the K loop vs epilogue/prepare), so the two should 3-way merge.

## Result (after the run)
- job `108231906a30407e` (submit), status done, passed True, tests 5/5
- speedup 1.0525x, 95% CI [1.0367, 1.0683], score 368067.7164353343 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0591912383762452, "ci": [1.0433033698006016, 1.0750791069518888], "job": "ba9c16b9cc6b4605"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-hybrid-marlin-turn-triton-first-turn.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-triton-w13-fused-silu-epilogue.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-triton-w13-fused-silu-epilogue.patch`; branch `hyp/muse_prefill_layer/experts-triton-w13-fused-silu-epilogue`
