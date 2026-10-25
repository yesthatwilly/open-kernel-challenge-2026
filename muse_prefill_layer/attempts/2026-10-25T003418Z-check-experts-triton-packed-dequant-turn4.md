# muse_prefill_layer · 2026-10-25T003418Z · check · experts-triton-packed-dequant-turn4

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T140133Z-experts-triton-packed-dequant-turn4.md` (branch `hyp/muse_prefill_layer/experts-triton-packed-dequant-turn4`).

- **Job:** `ce3cc1a22a244ec7` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `de7af5e`

## Goal
Stage 6 of the Triton NVFP4 expert rewrite, built on stage 5 (branch hyp/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant, whatever its score). Only the shape gate in muse_prefill_layer/submission.py changes: MARLIN_MAX_TOKENS goes from 4096 to 1024, so turn_4 (T=3012) and the T=4096 test run _experts_triton with the existing (4096, BLOCK_T=128, BLOCK_W=128, num_warps=8, num_stages=3) entry of EXPERT_CONFIGS and stage 5's packed inline-PTX dequant, while turn_1 (T=753) and the T<=1024 tests stay on Marlin. That config is already compiled and checked by _check_expert_gemms in prepare(); the end-to-end Triton-vs-Marlin check (E2E_CHECK_TOKENS = 8192) is unchanged. At T=3012 about 3012 of the 24096 routes are local, ~94 rows per expert, so BLOCK_T 128 covers one expert in one token tile and each weight element is dequantized exactly once (no redundant dequant), making this the shape where cheaper dequant should pay most. If stage 5's log showed register spills for the BLOCK_T 128 variant, use num_stages=2 for it and say so in the attempt file. Nothing else changes: kernel, banks, first_turn config, kernel_fn.

## What changed, and where
Edited only `muse_prefill_layer/submission.py` in the worktree (16 insertions, 13 deletions).
- `MARLIN_MAX_TOKENS` 4096 -> 1024 (comment updated). turn_4 (T=3012) and the T=1506 / T=4096 tests now run `_experts_triton` with the existing (4096, BLOCK_T=128, BLOCK_W=128, num_warps=8, num_stages=3) entry and stage 5's packed dequant; turn_1 and T<=1024 stay on Marlin.
- Review fix (open question 1): `E2E_CHECK_TOKENS` is now `(8192, 3012)` and `_check_experts_e2e` takes `t`; prepare() runs both (in both the first attempt and the scalar-fallback retry), so a packed-path wiring bug at BLOCK_T 128 falls back to the scalar dequant instead of failing the judge. Untimed, prepare() only.
- num_stages: kept 3. REWRITE_READ_2026-10-25.md reports a w13 BLOCK_T 256 spill (4 spills, 255 regs) but no spill for the BLOCK_T 128 variant, so the num_stages=2 rule does not trigger.
- STALE BASE: claim reported stale_base: true. The worktree is on stage 5's commit (3c444875ae), which predates prefill master (stack-tonight-five-prefill, 1.0705 check / 1.0744 scored). Built on stage 5 as the spec says; not rebased. Master's parallel align and `_silu_mul_sorted_kernel` are therefore absent here, so the turn_4 comparison against master's Marlin rows (2148/2115 us) is against a different surrounding pipeline.
- Context from REWRITE_READ: stage 5 was judged DEAD (packed dequant ON, mean +1.4% at first_turn, within noise; asm added a smem round trip). The review expects only ~1.6-1.8 ms at turn_4, so a gain here is plausible but modest. Check the log for 'packed dequant ON' and both '[expert e2e]' lines before attributing.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff touches a constant, prepare() and its untimed check only; kernel_fn path unchanged, no new sync/alloc/value-dependent control flow (gate reads x.shape[0] only).
- [x] Nothing derived from `data` is cached across calls: n/a to kernel_fn; the e2e check uses synthetic data inside prepare() only.
- [x] Weights: unchanged (no repacking changes in the diff).
- [x] Activations BF16 / KV FP8 / queries: unchanged by the diff.
- [x] Attention sees every key: attention untouched by the diff.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added.
- [x] All work happens inside the timed call; no output caching or replay: the Triton path runs fully inside kernel_fn; no caching added.

okc_lint output:
```
warn  host-sync    L537, L579, L633, L638, L649, L658, L668: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_prefill_layer/submission.py
```
py_compile passed; __pycache__ removed.

## Risks / what could fail
medium
Risk: if stage 5 did not cut the per-tile dequant cost (its first_turn kernel rows tell), turn_4 will land near stage 2's 2.1 ms, i.e. flat against Marlin; then this stage should wait for a K-loop restructure. The w2 call's -1 blocks zero-store ~21K non-local rows x 4096 x 2 B (~170 MB, ~35 us) at this T, part of the Triton cost Marlin does not pay.

## Result (after the run)
- job `ce3cc1a22a244ec7` (check), status done, passed True, tests 5/5
- speedup 1.0522x, 95% CI [1.0364, 1.0680], score 368194.26402353856 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.074386115190733, "ci": [1.058270323462872, 1.090501906918594], "job": "502f2eab95164781"}; base result: muse_prefill_layer/hypotheses/done/neutral/experts-triton-nvfp4-packed-asm-dequant.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-triton-packed-dequant-turn4.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-triton-packed-dequant-turn4.patch`; branch `hyp/muse_prefill_layer/experts-triton-packed-dequant-turn4`
