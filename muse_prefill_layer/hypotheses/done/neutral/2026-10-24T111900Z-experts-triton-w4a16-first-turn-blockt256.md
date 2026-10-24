---
problem: muse_prefill_layer
slug: experts-triton-w4a16-first-turn-blockt256
created: 2026-10-24T111900Z
base: master
area: "w13 marlin + w2 marlin (W4A16 Triton engine, stage 3: first_turn token tile 256)"
change: >
  Stage 3 of the Triton NVFP4 grouped GEMM, built on stage 2 (branch
  hyp/muse_prefill_layer/experts-triton-w4a16-swapab-reglhs, _moe_nvfp4_gemm_kernel and
  EXPERT_CONFIGS in muse_prefill_layer/submission.py). One change: a wider token tile at
  first_turn only, so each in-register dequantized weight tile feeds twice as many MMA columns.
  Today T > 1024 uses BLOCK_T 128 / BLOCK_W 128 / num_warps 8 / num_stages 3; at first_turn (~512
  local rows per expert) that is ~4 token tiles per (expert, weight tile), so every weight element
  is dequantized ~4-5x per call, and the reviewer's per-tile estimate puts the dequant ALU at
  about the same cost as the MMA (~0.6 us/SM per 128x128x128 step each), serialized if Triton
  waits on the register-A wgmma before the next dequant. Split the unbounded config entry into
  (4096, the existing BLOCK_T 128 config, unchanged, so turn_4 runs exactly the stage-2 code) and
  (None, dict(BLOCK_T=256, BLOCK_W=128, BLOCK_K=EXPERT_BLOCK_K, GROUP_SIZE_T=4, num_warps=8,
  num_stages=2)). With 8 warps the two warpgroups each own 64 weight rows x 256 tokens (wgmma
  m64n256, acc 128 fp32 regs/thread); the dequantized [128, 64] bf16 halves stay the register LHS
  exactly as in stage 2; the activation slabs become two [64, 256] bf16 tiles per K step (64
  KB/stage), hence num_stages 2 (128 KB smem) instead of 3. GROUP_SIZE_T 4 keeps the same number
  of token rows per swizzle group as stage 2's 8 x 128, so the L2 reuse of the activation across
  weight tiles is unchanged. moe_align_block_size pads to 256 at T > 4096 (it already takes
  cfg['BLOCK_T']). Nothing else changes: kernel body, nibble-reordered bank, scale path, epilogue,
  ZERO_NONLOCAL_W2, silu_and_mul, post-expert sum, the T <= 1024 and T <= 4096 configs.
  _check_expert_gemms already loops over every EXPERT_CONFIGS entry, so the new config is checked
  and warm-compiled in prepare(); make sure its one-hot check tile and -1 / past-n_padded checks
  use BLOCK_T 256 blocks. If the judge's compile reports smem over the limit or the run shows
  heavy spilling (first_turn kernel time at or above stage 2's 9.1-10.0 ms), the fallback for
  stage 4 is BLOCK_W 64 x BLOCK_T 256 at num_warps 4; record the TTGIR/occupancy if obtainable.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "_moe_nvfp4_gemm_kernel x2 is 9.1-10.0 ms (Marlin 10.2-12.0 ms, floor ~2.5 ms at 660 TFLOP/s) and 48-62% of each first_turn shape; BLOCK_T 256 halves the redundant per-token-tile dequant (~4x to ~2x per weight element) and doubles wgmma n per dequantized LHS. If dequant ALU ~= MMA and they serialize, ~25% off the kernel. Milestone: <= 7.5 ms for w13+w2 (sliding first_turn, weight 138, carries most of the headline: ~2 ms off 15.9 ms is ~+6-8% overall)"
  - shapes: "*load=turn_*"
    direction: flat
    why: T=753 and 3012 keep the stage-2 configs byte-identical (BLOCK_T 64 / 128); turn_1 stays ~796/823 us vs Marlin ~730/681 and turn_4 ~2158/2100 us; any move there is drift
risk: high
depends_on: experts-triton-w4a16-swapab-reglhs
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 3
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T
review: proceed
status: done
review_at: 2026-10-24T112151Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T112151Z
base_sha: 5f5af1d3fe4b9b1ef686af0d022040134029ee30
base_blob: 60abe04615110df3e4a9355a4b984b48b68f85e8
branch: hyp/muse_prefill_layer/experts-triton-w4a16-first-turn-blockt256
worktree: .worktrees/muse_prefill_layer/experts-triton-w4a16-first-turn-blockt256
mode: check
implemented_at: 2026-10-24T112334Z
impl_commit: 2226bae0a20f470cfa1bf733c1f71d1a60b1cbdd
submission_sha256: f4fec2126a1c2053d6e81e8428393d17ace96a0ab00fa04c392ea86db5513a28
norm_sha256: 7ffb167fe66532950a1ace39db2aec0f66f08752bbfee33f3655c7993d170fe0
submission_blob: f17d41c9a9b0696f20e6679a493da9d4aa6d2934
lint_warnings: 2
attempt_file: 2026-10-24T112334Z-check-experts-triton-w4a16-first-turn-blockt256.md
prepared_at: 2026-10-24T112335Z
job: b33b87b5083b4d1c
dispatched_at: 2026-10-24T112343Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.0228712349305686
speedup_ci: [1.00752816640661, 1.038214303454527]
harvested_at: 2026-10-24T124011Z
best_at_harvest: 1.0038664293875537
promotion: null
---

# experts-triton-w4a16-first-turn-blockt256

## Change

Stage 3 of the Triton NVFP4 grouped GEMM, built on stage 2 (branch hyp/muse_prefill_layer/experts-triton-w4a16-swapab-reglhs, _moe_nvfp4_gemm_kernel and EXPERT_CONFIGS in muse_prefill_layer/submission.py). One change: a wider token tile at first_turn only, so each in-register dequantized weight tile feeds twice as many MMA columns. Today T > 1024 uses BLOCK_T 128 / BLOCK_W 128 / num_warps 8 / num_stages 3; at first_turn (~512 local rows per expert) that is ~4 token tiles per (expert, weight tile), so every weight element is dequantized ~4-5x per call, and the reviewer's per-tile estimate puts the dequant ALU at about the same cost as the MMA (~0.6 us/SM per 128x128x128 step each), serialized if Triton waits on the register-A wgmma before the next dequant. Split the unbounded config entry into (4096, the existing BLOCK_T 128 config, unchanged, so turn_4 runs exactly the stage-2 code) and (None, dict(BLOCK_T=256, BLOCK_W=128, BLOCK_K=EXPERT_BLOCK_K, GROUP_SIZE_T=4, num_warps=8, num_stages=2)). With 8 warps the two warpgroups each own 64 weight rows x 256 tokens (wgmma m64n256, acc 128 fp32 regs/thread); the dequantized [128, 64] bf16 halves stay the register LHS exactly as in stage 2; the activation slabs become two [64, 256] bf16 tiles per K step (64 KB/stage), hence num_stages 2 (128 KB smem) instead of 3. GROUP_SIZE_T 4 keeps the same number of token rows per swizzle group as stage 2's 8 x 128, so the L2 reuse of the activation across weight tiles is unchanged. moe_align_block_size pads to 256 at T > 4096 (it already takes cfg['BLOCK_T']). Nothing else changes: kernel body, nibble-reordered bank, scale path, epilogue, ZERO_NONLOCAL_W2, silu_and_mul, post-expert sum, the T <= 1024 and T <= 4096 configs. _check_expert_gemms already loops over every EXPERT_CONFIGS entry, so the new config is checked and warm-compiled in prepare(); make sure its one-hot check tile and -1 / past-n_padded checks use BLOCK_T 256 blocks. If the judge's compile reports smem over the limit or the run shows heavy spilling (first_turn kernel time at or above stage 2's 9.1-10.0 ms), the fallback for stage 4 is BLOCK_W 64 x BLOCK_T 256 at num_warps 4; record the TTGIR/occupancy if obtainable.

## Expected

- `*load=first_turn*` faster: _moe_nvfp4_gemm_kernel x2 is 9.1-10.0 ms (Marlin 10.2-12.0 ms, floor ~2.5 ms at 660 TFLOP/s) and 48-62% of each first_turn shape; BLOCK_T 256 halves the redundant per-token-tile dequant (~4x to ~2x per weight element) and doubles wgmma n per dequantized LHS. If dequant ALU ~= MMA and they serialize, ~25% off the kernel. Milestone: <= 7.5 ms for w13+w2 (sliding first_turn, weight 138, carries most of the headline: ~2 ms off 15.9 ms is ~+6-8% overall)
- `*load=turn_*` flat: T=753 and 3012 keep the stage-2 configs byte-identical (BLOCK_T 64 / 128); turn_1 stays ~796/823 us vs Marlin ~730/681 and turn_4 ~2158/2100 us; any move there is drift

## Risk

High risk. (a) Register pressure: 128 fp32 acc regs/thread plus the dequantized bf16 LHS halves, scales and codes may spill at num_warps 8; spills show as first_turn at or above stage 2. (b) Padding: ~512 rows/expert padded to 256 wastes ~128 rows/expert on average (vs ~64 at BLOCK_T 128), ~12% extra MMA work, which the halved dequant must outrun. (c) num_stages 2 exposes more activation-load latency than 3. (d) Why not the turn_1 Marlin gate instead: turn_1 weighs 62 of 310 and is ~7% behind SGLang, so a perfect gate is worth ~+1.4% headline; first_turn weighs 186 and the kernel is ~4x its floor, the bigger expected gain with prefill's 2 checks left. A turn_1 shape gate (keep Marlin at T <= 1024) stays a cheap follow-up. Rules: NVFP4 unchanged (same reordered bank), dequant in registers only, no BF16 weight copy, no autotune, config chosen from T only, no host sync, nothing input-derived cached.

## Milestones

Rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-24T112151Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound, narrow stage-3 step: a BLOCK_T 256 tile at T > 4096 halves the redundant per-token-tile dequant and doubles wgmma N per dequantized LHS, the turn shapes stay byte-identical, and the profile row for _moe_nvfp4_gemm_kernel at first_turn gives a clean milestone reading against stage 2's 9.1-10.0 ms. One concrete bug must be fixed while building: _check_expert_gemms as written breaks at BLOCK_T 256 (CHECK_ROWS 128 < 256), so prepare() would assert and the whole run would fail.

### Justification
Mechanism. With BLOCK_W 128 and num_warps 8, Triton's warpsPerTileV3 gives warpsPerCTA [8,1]. Each warpgroup then owns 64 W rows x all 256 tokens (wgmma m64n256k16, 128 fp32 acc regs/thread), as the hypothesis says. The register-A dequantized LHS is unchanged, so per K step the dequant ALU stays fixed while MMA work per program doubles. At ~512 local rows per expert (16384 local routes over 32 experts), token tiles per (expert, W tile) go from ~4.5 to ~2.5. Under the serialized model (ALU ~= MMA ~0.6 us/SM per 128x128x128), cost per (expert, W tile, K step) goes from ~4.5 x 1.2 to ~2.5 x 1.8 units, about 15-25% off after the extra padding. The <= 7.5 ms milestone is plausible but optimistic. Grid parallelism is fine: ~80 local token tiles x 64 (w13, N 8192) or 32 (w2, N 4096) W tiles is still tens of waves on 132 SMs. Smem: Triton multi-buffers every loop load, i.e. a_lo + a_hi (64 KB) + codes [128,64] u8 (8 KB) + scales (1 KB), about 73 KB per stage. Two stages is ~146 KB plus the epilogue's convert scratch, which fits the 227 KB limit. Three stages (~219 KB plus scratch) likely would not, so num_stages 2 is the right call. Shapes and directions agree with TASK: only T=16384 (> 4096) changes. turn_4 (T=3012) and the T=4096 test stay on the stage-2 BLOCK_T 128 config, and turn_1 on BLOCK_T 64. The headline arithmetic is overstated, though. The score is a geomean over three loads, and first_turn's load is ~3.14 s (16x16.5 + 138x15.9 + 16x19.6 + 16x23.2 ms), so 2 ms off each first_turn shape is ~12% on that load and ~+4% headline, not +6-8%. Weakest link: the premise that stage 2 is dequant-ALU-bound and serialized with the register-A wgmma is a reviewer estimate, not a measurement. If stage 2 is instead bound by the gathered activation loads' latency (row-gathered, no TMA, now only 2 stages deep), by register pressure, or by tail/imbalance (the identical expert math costs 9.15-10.0 ms across the four first_turn shapes, a 9% spread that larger tiles can amplify), BLOCK_T 256 can come out flat or slower. Register pressure is the most likely failure. Counting 128 acc + ~32 LHS + ~32 code + ~32 broadcast-scale regs plus addressing gives ~230-250 of the 255 allowed at 256 threads, so spills are a real possibility. The stated fallback (BLOCK_W 64 x BLOCK_T 256 at 4 warps) is the right next stage if so.

### Confounders on the current kernel
- sglang-pipeline-parity: no confounding. It fixes the modules around the experts (pooler/pre-expert norm feeding x, silu_and_mul, post-expert weighted sum), and this change touches only the expert GEMM tile config at T > 4096 on top of stage 2, which already carries parity. The comparison is the replaced kernel's own time per first_turn shape against stage 2 (00148fbedfb6414f) and Marlin, which parity does not affect.

### Interference and A/B needs
No companion is warranted. This is a rewrite stage judged on _moe_nvfp4_gemm_kernel's own us at the four first_turn shapes against stage 2's 10009/9909/9747/9146 us and Marlin's 10.2-12.0 ms. Triton kernel times reproduce within ~1% (LESSONS), so a single run reads cleanly. The run also carries a built-in drift control: the turn_1/turn_4 kernel rows run byte-identical stage-2 code (796/823 and 2158/2100 us). If they move by x%, discount the first_turn delta by that much. Compare first_turn row for row (same shape key), not as an average, because the kernel's time already varies 9% across those shapes. The only possible control, BLOCK_T 256 at num_stages 3 or GROUP_SIZE_T 8, would not disambiguate anything worth a check with prefill at 2 checks left. That and the BLOCK_W 64 fallback are next-stage material.

### Open questions
- Does Triton actually issue wgmma m64n256 with warpsPerCTA [8,1] here, and does the kernel spill? Record n_regs / n_spills from the compiled kernel's metadata in prepare() and print them, since the 16 KB log tail comes back with the run.
- Is the register-A wgmma async across the next K step's dequant in the TTGIR (warp_group_dot with isAsync and a wait of 1 rather than 0)? If it waits on 0, the serialization premise holds and warp specialization or ping-pong is the real next lever, not more tile-size tuning.
- Why does identical expert math take 9.15 ms at ctx 65536 but 10.0 ms at ctx 0? If it is routing skew per shape, larger token tiles raise tail cost on hot experts. Compare the stage-3 delta per shape to see whether the gain shrinks where stage 2 was slowest.
- Correct the headline expectation to ~+4% (geomean over three loads) so a milestone-2 result is not read as a disappointment.


## Implementation notes

Worktree `.worktrees/muse_prefill_layer/experts-triton-w4a16-first-turn-blockt256`, file `muse_prefill_layer/submission.py` only (uncommitted; 1 file, +31 / -18).

- `EXPERT_CONFIGS`: the unbounded BLOCK_T 128 entry is now bounded at 4096 (byte-identical config, so turn_4 T=3012 and the T=4096 test run stage-2 code), and a new `(None, BLOCK_T=256, BLOCK_W=128, BLOCK_K=EXPERT_BLOCK_K, GROUP_SIZE_T=4, num_warps=8, num_stages=2)` entry serves T > 4096 (first_turn only). The comment above it is updated. `_experts` already passes `cfg["BLOCK_T"]` to `moe_align_block_size`, so padding to 256 needs no change. The kernel body, bank, scales, epilogue, ZERO_NONLOCAL_W2, silu_and_mul and post-expert sum are untouched.
- `_check_expert_gemms` (the REQUIRED review fix): per config `check_rows = max(CHECK_ROWS, bt)` (128, 128, 256), and rows / want / want_codes / x one-hot are built per config from it. `expert_ids` is sized `triton.cdiv(total, bt)` with `total = check_rows + 2*bt`, the -1 block sits at `check_rows // bt`, the valid-but-past block right after it, and `n_padded = check_rows + bt`. The got / nonlocal / past slices use `check_rows`. For BLOCK_T 64 and 128 this is the same layout as before (check_rows 128). For BLOCK_T 256 it covers 256 rows (two passes over the 128-wide one-hot chunk), one -1 block and one past block, for both w13 (top_k 8) and w2 (top_k 1, ZERO_NONLOCAL). The loop runs over every EXPERT_CONFIGS entry, so prepare() compiles and checks the new T > 4096 config (no pre-timing test has T > 4096).
- Open-question diagnostic from the review: `_nvfp4_moe_gemm` takes an optional `report=None`. Only `_check_expert_gemms` sets it. When set, it prints the variant's `n_regs`, `n_spills` and `shared` from the CompiledKernel the Triton launch returns, read with `getattr` and a `'?'` fallback so a Triton version difference cannot fail prepare(). This puts the register/spill reading for BLOCK_T 256 in the run's log tail. kernel_fn always passes report=None, so nothing new happens in the timed path (one Python `is not None` check).
- No deviations from the spec. TTGIR async-wait inspection (open question 2) is not implemented: it needs the compiled asm and is not cheap to print within the 16 KB log tail. Next stage if wanted.
- Uncertainties: (a) smem at 2 stages, about 146 KB plus scratch, should fit (per the review), but it is only confirmed when the judge compiles it in prepare(). (b) The `shared` attribute name differs across Triton versions, so it may print `?`. (c) The check's `out` buffer for w13 at BLOCK_T 256 is 768*8 x 8192 bf16, about 100 MB untimed, which is fine.

## Rules check

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

## Result

- job `b33b87b5083b4d1c` (check), status done, passed True, tests 5/5
- speedup 1.0229x, 95% CI [1.0075, 1.0382], score 378741.61162259744 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/neutral/experts-triton-w4a16-swapab-reglhs.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-triton-w4a16-first-turn-blockt256.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-triton-w4a16-first-turn-blockt256.patch`; branch `hyp/muse_prefill_layer/experts-triton-w4a16-first-turn-blockt256`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1095.07 | 1141.1 | 0.9976 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1497.85 | 1517.93 | 0.9913 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3250.81 | 3435.39 | 0.9613 | lost |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4729.14 | 4975.37 | 0.9825 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 16528.39 | 15029.51 | 1.0984 | gained |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 15920.71 | 14426.71 | 1.0955 | gained |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 19591.67 | 18568.92 | 1.0743 | gained |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 23188.88 | 22521.73 | 1.0512 | gained |

gained on full first_turn; sliding first_turn; lost on sliding turn_4. Expected hit: 2/2. Noise ±1.5%.

## Follow-ups

- rewrite stage 3 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_prefill_layer "<what to fix or tune>" --kind rewrite --depends-on experts-triton-w4a16-first-turn-blockt256 --stage 4`; a slow or failed first cut is the expected start, not a do-not-retry.
- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: experts-triton-w4a16-first-turn-blockt256`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

