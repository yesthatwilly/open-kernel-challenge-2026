# muse_prefill_layer · 2026-10-24T112334Z · check · experts-triton-w4a16-first-turn-blockt256

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T111900Z-experts-triton-w4a16-first-turn-blockt256.md` (branch `hyp/muse_prefill_layer/experts-triton-w4a16-first-turn-blockt256`).

- **Job:** `b33b87b5083b4d1c` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `2226bae`

## Goal
Stage 3 of the Triton NVFP4 grouped GEMM, built on stage 2 (branch hyp/muse_prefill_layer/experts-triton-w4a16-swapab-reglhs, _moe_nvfp4_gemm_kernel and EXPERT_CONFIGS in muse_prefill_layer/submission.py). One change: a wider token tile at first_turn only, so each in-register dequantized weight tile feeds twice as many MMA columns. Today T > 1024 uses BLOCK_T 128 / BLOCK_W 128 / num_warps 8 / num_stages 3; at first_turn (~512 local rows per expert) that is ~4 token tiles per (expert, weight tile), so every weight element is dequantized ~4-5x per call, and the reviewer's per-tile estimate puts the dequant ALU at about the same cost as the MMA (~0.6 us/SM per 128x128x128 step each), serialized if Triton waits on the register-A wgmma before the next dequant. Split the unbounded config entry into (4096, the existing BLOCK_T 128 config, unchanged, so turn_4 runs exactly the stage-2 code) and (None, dict(BLOCK_T=256, BLOCK_W=128, BLOCK_K=EXPERT_BLOCK_K, GROUP_SIZE_T=4, num_warps=8, num_stages=2)). With 8 warps the two warpgroups each own 64 weight rows x 256 tokens (wgmma m64n256, acc 128 fp32 regs/thread); the dequantized [128, 64] bf16 halves stay the register LHS exactly as in stage 2; the activation slabs become two [64, 256] bf16 tiles per K step (64 KB/stage), hence num_stages 2 (128 KB smem) instead of 3. GROUP_SIZE_T 4 keeps the same number of token rows per swizzle group as stage 2's 8 x 128, so the L2 reuse of the activation across weight tiles is unchanged. moe_align_block_size pads to 256 at T > 4096 (it already takes cfg['BLOCK_T']). Nothing else changes: kernel body, nibble-reordered bank, scale path, epilogue, ZERO_NONLOCAL_W2, silu_and_mul, post-expert sum, the T <= 1024 and T <= 4096 configs. _check_expert_gemms already loops over every EXPERT_CONFIGS entry, so the new config is checked and warm-compiled in prepare(); make sure its one-hot check tile and -1 / past-n_padded checks use BLOCK_T 256 blocks. If the judge's compile reports smem over the limit or the run shows heavy spilling (first_turn kernel time at or above stage 2's 9.1-10.0 ms), the fallback for stage 4 is BLOCK_W 64 x BLOCK_T 256 at num_warps 4; record the TTGIR/occupancy if obtainable.

## What changed, and where
Worktree `.worktrees/muse_prefill_layer/experts-triton-w4a16-first-turn-blockt256`, file `muse_prefill_layer/submission.py` only (uncommitted; 1 file, +31 / -18).

- `EXPERT_CONFIGS`: the unbounded BLOCK_T 128 entry is now bounded at 4096 (byte-identical config, so turn_4 T=3012 and the T=4096 test run stage-2 code), and a new `(None, BLOCK_T=256, BLOCK_W=128, BLOCK_K=EXPERT_BLOCK_K, GROUP_SIZE_T=4, num_warps=8, num_stages=2)` entry serves T > 4096 (first_turn only). The comment above it is updated. `_experts` already passes `cfg["BLOCK_T"]` to `moe_align_block_size`, so padding to 256 needs no change. The kernel body, bank, scales, epilogue, ZERO_NONLOCAL_W2, silu_and_mul and post-expert sum are untouched.
- `_check_expert_gemms` (the REQUIRED review fix): per config `check_rows = max(CHECK_ROWS, bt)` (128, 128, 256), and rows / want / want_codes / x one-hot are built per config from it. `expert_ids` is sized `triton.cdiv(total, bt)` with `total = check_rows + 2*bt`, the -1 block sits at `check_rows // bt`, the valid-but-past block right after it, and `n_padded = check_rows + bt`. The got / nonlocal / past slices use `check_rows`. For BLOCK_T 64 and 128 this is the same layout as before (check_rows 128). For BLOCK_T 256 it covers 256 rows (two passes over the 128-wide one-hot chunk), one -1 block and one past block, for both w13 (top_k 8) and w2 (top_k 1, ZERO_NONLOCAL). The loop runs over every EXPERT_CONFIGS entry, so prepare() compiles and checks the new T > 4096 config (no pre-timing test has T > 4096).
- Open-question diagnostic from the review: `_nvfp4_moe_gemm` takes an optional `report=None`. Only `_check_expert_gemms` sets it. When set, it prints the variant's `n_regs`, `n_spills` and `shared` from the CompiledKernel the Triton launch returns, read with `getattr` and a `'?'` fallback so a Triton version difference cannot fail prepare(). This puts the register/spill reading for BLOCK_T 256 in the run's log tail. kernel_fn always passes report=None, so nothing new happens in the timed path (one Python `is not None` check).
- No deviations from the spec. TTGIR async-wait inspection (open question 2) is not implemented: it needs the compiled asm and is not cheap to print within the 16 KB log tail. Next stage if wanted.
- Uncertainties: (a) smem at 2 stages, about 146 KB plus scratch, should fit (per the review), but it is only confirmed when the judge compiles it in prepare(). (b) The `shared` attribute name differs across Triton versions, so it may print `?`. (c) The check's `out` buffer for w13 at BLOCK_T 256 is 768*8 x 8192 bf16, about 100 MB untimed, which is fine.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-triton-w4a16-first-turn-blockt256 diff`:

- [x] CUDA-graph capturable: no new `.item()`/`.cpu()`/`synchronize()` in kernel_fn's path. The only new print is gated by `report`, which kernel_fn never sets, and it prints ints from compiled-kernel metadata, never a tensor. Config choice uses `num_tokens` (a shape) only, with no control flow on input values. The existing `torch.cuda.synchronize()` calls are in `_check_expert_gemms`/prepare() (lint warns, which is fine there).
- [x] Nothing derived from `data` is cached across calls: no new state entries, and the config table is a module constant.
- [x] Weights: no change to the bank (same nibble-reordered NVFP4, same scales and global scales). Dequant stays in registers inside the kernel, with no BF16 weight copy.
- [x] Activations BF16 where SGLang keeps BF16 (the GEMM still takes BF16 A, BF16 out, FP32 acc). KV cache FP8 and queries are untouched.
- [x] n/a: attention is untouched by this diff (expert GEMM tile config and the prepare() check only), so every key is still attended.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff adds only a print to stdout in prepare().
- [x] All work happens inside the timed call. No output caching or replay, and prepare() only runs the synthetic weight-only check.
- [x] No `triton.autotune` (fixed configs chosen from T), and the first line `# okc: problem=muse_prefill_layer` is kept.

okc_lint output:

```
warn  host-sync    L424: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L480: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_prefill_layer/experts-triton-w4a16-first-turn-blockt256/muse_prefill_layer/submission.py
```

`python3 -I -m py_compile` passes, and the __pycache__ was removed from the worktree.

## Risks / what could fail
high
High risk. (a) Register pressure: 128 fp32 acc regs/thread plus the dequantized bf16 LHS halves, scales and codes may spill at num_warps 8; spills show as first_turn at or above stage 2. (b) Padding: ~512 rows/expert padded to 256 wastes ~128 rows/expert on average (vs ~64 at BLOCK_T 128), ~12% extra MMA work, which the halved dequant must outrun. (c) num_stages 2 exposes more activation-load latency than 3. (d) Why not the turn_1 Marlin gate instead: turn_1 weighs 62 of 310 and is ~7% behind SGLang, so a perfect gate is worth ~+1.4% headline; first_turn weighs 186 and the kernel is ~4x its floor, the bigger expected gain with prefill's 2 checks left. A turn_1 shape gate (keep Marlin at T <= 1024) stays a cheap follow-up. Rules: NVFP4 unchanged (same reordered bank), dequant in registers only, no BF16 weight copy, no autotune, config chosen from T only, no host sync, nothing input-derived cached.

## Result (after the run)
- job `b33b87b5083b4d1c` (check), status done, passed True, tests 5/5
- speedup 1.0229x, 95% CI [1.0075, 1.0382], score 378741.61162259744 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/neutral/experts-triton-w4a16-swapab-reglhs.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-triton-w4a16-first-turn-blockt256.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-triton-w4a16-first-turn-blockt256.patch`; branch `hyp/muse_prefill_layer/experts-triton-w4a16-first-turn-blockt256`
