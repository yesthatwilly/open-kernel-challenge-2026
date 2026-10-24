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
review: required
status: queued
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
