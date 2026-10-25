# muse_prefill_layer · 2026-10-25T034946Z · check · experts-triton-first-turn-unmasked-loads-l2-hints

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T224437Z-experts-triton-first-turn-unmasked-loads-l2-hints.md` (branch `hyp/muse_prefill_layer/experts-triton-first-turn-unmasked-loads-l2-hints`).

- **Job:** `aeda8b5a523642b1` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `9504288`

## Goal
Built on master (635cfee). Only the K loop's loads in _moe_nvfp4_gemm_kernel change; tiles, configs, grid, dequant, dots and epilogue are unchanged. (1) UNMASKED A LOADS: today a_lo/a_hi are tl.load(a_ptrs, mask=token_mask[None, :], other=0.0) every K step, where padding slots carry the sentinel id num_valid_tokens (row num_valid // TOP_K is out of bounds). Before the loop, compute safe_token = tl.where(token_mask, offs_token, 0) and build a_ptrs from safe_token // TOP_K, then load a_lo/a_hi with no mask (padding columns read row 0, a real row; their products land only in output columns the masked store already drops). This removes per-element predicates from the hot loop so the activation slabs load as unpredicated 16-byte cp.async. The output store keeps mask=token_mask. (2) L2 HINTS: weight codes b and scales s are read once per (expert, W tile) per token-tile group: load them with eviction_policy='evict_first'; the activation slabs are re-read by all N // BLOCK_W weight tiles of the group: load them with eviction_policy='evict_last'. Applies to every config (the kernel is shared), but only the T > 4096 path runs it in the timed call. _check_expert_gemms runs unchanged (it covers padding slots, the -1 block and the past-n_padded block) and still logs regs/spills/smem per variant; compare them with master's log line.

## What changed, and where
Edited only the worktree's muse_prefill_layer/submission.py (worktree HEAD 62d2868, the a81d19c lineage per the review).
- `_moe_nvfp4_gemm_kernel` gets two SEPARATE constexpr switches. `UNMASKED_A`: padding slots (sentinel id) are pointed at row 0 via `tl.where(token_mask, offs_token, 0) // TOP_K` and a_lo/a_hi load with no mask; the store keeps `mask=token_mask`. `L2_HINTS`: `eviction_policy="evict_last"` on the activation slabs only. Per the review, NO evict_first on the weights (b, s): it fights the token-tile-fastest swizzle.
- Variants are (UNMASKED_A, L2_HINTS) pairs: master (F,F), unmasked (T,F), unmasked+hints (T,T). `_nvfp4_moe_gemm` takes `loads=` and an optional `info` dict (n_regs/n_spills).
- `_check_expert_gemms` now iterates `_check_cases()`: every config with master loads, plus the T>4096 config with both new variants, so all are compiled and dequant-checked before capture.
- New `_loads_ab()` in prepare(), after the Marlin bank and before the e2e check (so the e2e check, whose real align has padding slots, runs the chosen variants): synthetic T=16384, real `moe_align_block_size`, first-turn config; for w13 and w2, CUDA-graph replays of the three variants, 7 alternations (plus a dropped warmup) with rotating order, 256 MB L2 flush before each timed replay, event-timed. Prints medians, ratio, n_regs, n_spills per variant and `PATH CHOSEN`. A variant is chosen per GEMM only if best and <= 0.98x master's median; else master. Any exception keeps master.
- `_experts_triton` passes `_loads_for(state, name, cfg)`: chosen variant on the first-turn config only, master loads otherwise. Marlin path untouched, so turn_* is structurally neutral.
- Deviations: decision is per GEMM (w13, w2 independently) rather than one joint pick. Hints are activations-only, as the review requires (spec's evict_first on weights dropped).
- Not run (no GPU): nothing verified at runtime. Padding-column garbage (row 0; for w2, a possibly unwritten cache2 row) stays in dropped output columns since acc[n, t] depends on A[t] only.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds no .item()/.cpu()/sync/allocation in kernel_fn's path; `_loads_for` reads state and cfg identity only; all A/B syncs and graph captures are in prepare().
- [x] Nothing derived from `data` cached across calls: the choice comes from synthetic prepare() data; state holds only variant tuples.
- [x] Weights: unchanged, NVFP4 stays NVFP4 (dequant in kernel untouched).
- [x] Activations BF16 where SGLang keeps BF16: n/a to change, synthetic inputs are bf16; no dtype changes in the timed path.
- [x] Attention sees every key: n/a, no attention change.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff has none.
- [x] All work inside the timed call; no output caching or replay: kernels still run every call; the A/B only picks a constexpr variant in prepare().
- [x] No triton.autotune, no host sync in kernel_fn, first-line marker `# okc: problem=muse_prefill_layer` kept.

okc_lint (host-sync warnings in prepare() filtered; all are outside kernel_fn's call graph):
```
OK with warnings: .../experts-triton-first-turn-unmasked-loads-l2-hints/muse_prefill_layer/submission.py
```
py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
Speculative tuning of the Triton first_turn engine, which experts-cuda-wgmma-mixed-input-grouped-gemm (in the judge) would replace; run only if that rewrite does not land or as a cheap hedge. Orthogonal to experts-triton-persistent-grid-first-turn (grid) and the dispatch hypotheses (what the grid contains), so it can stack with them. Packed asm dequant was flat (the kernel is not dequant-ALU bound), so if this is also flat the remaining cost is tl.dot's synchronous lowering, the rewrite's premise.

## Result (after the run)
- job `aeda8b5a523642b1` (check), status done, passed True, tests 5/5
- speedup 1.0677x, 95% CI [1.0517, 1.0837], score 362837.936834828 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/stack-tonight-five-prefill.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-triton-first-turn-unmasked-loads-l2-hints.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-triton-first-turn-unmasked-loads-l2-hints.patch`; branch `hyp/muse_prefill_layer/experts-triton-first-turn-unmasked-loads-l2-hints`
