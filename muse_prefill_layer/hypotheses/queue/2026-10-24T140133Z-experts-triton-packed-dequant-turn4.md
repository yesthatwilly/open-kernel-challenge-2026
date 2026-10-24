---
problem: muse_prefill_layer
slug: experts-triton-packed-dequant-turn4
created: 2026-10-24T140133Z
base: master
area: "w13 marlin + w2 marlin (Triton NVFP4 engine, stage 6: take over turn_4)"
change: >
  Stage 6 of the Triton NVFP4 expert rewrite, built on stage 5 (branch
  hyp/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant, whatever its score). Only the
  shape gate in muse_prefill_layer/submission.py changes: MARLIN_MAX_TOKENS goes from 4096 to
  1024, so turn_4 (T=3012) and the T=4096 test run _experts_triton with the existing (4096,
  BLOCK_T=128, BLOCK_W=128, num_warps=8, num_stages=3) entry of EXPERT_CONFIGS and stage 5's
  packed inline-PTX dequant, while turn_1 (T=753) and the T<=1024 tests stay on Marlin. That
  config is already compiled and checked by _check_expert_gemms in prepare(); the end-to-end
  Triton-vs-Marlin check (E2E_CHECK_TOKENS = 8192) is unchanged. At T=3012 about 3012 of the 24096
  routes are local, ~94 rows per expert, so BLOCK_T 128 covers one expert in one token tile and
  each weight element is dequantized exactly once (no redundant dequant), making this the shape
  where cheaper dequant should pay most. If stage 5's log showed register spills for the BLOCK_T
  128 variant, use num_stages=2 for it and say so in the attempt file. Nothing else changes:
  kernel, banks, first_turn config, kernel_fn.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "replaced kernels at turn_4: Marlin x2 at 2131-2217 us (43-64% of the shape). Stage 2's Triton engine with the old dequant already ran turn_4 w13+w2 at ~2.1 ms (on par with Marlin); w13+w2 move 905 MB of NVFP4 codes+scales (floor 189 us at 4.8 TB/s) and 0.30 TFLOP (300 us at 989), so both engines are ~7x their floor. With stage 5's ~3-4x cheaper dequant and one dequant per weight element, target w13+w2 at ~1.0-1.3 ms (Milestone: under 1.6 ms), 0.9-1.2 ms off a 3.3-4.9 ms shape, 20-35% on the turn_4 load"
  - shapes: "*load=turn_1*"
    direction: neutral
    why: T=753 <= 1024 stays on Marlin
  - shapes: "*load=first_turn*"
    direction: neutral
    why: same code as stage 5 at T=16384
risk: medium
depends_on: experts-triton-nvfp4-packed-asm-dequant
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 6
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T
review: required
status: queued
---

# experts-triton-packed-dequant-turn4

## Change

Stage 6 of the Triton NVFP4 expert rewrite, built on stage 5 (branch hyp/muse_prefill_layer/experts-triton-nvfp4-packed-asm-dequant, whatever its score). Only the shape gate in muse_prefill_layer/submission.py changes: MARLIN_MAX_TOKENS goes from 4096 to 1024, so turn_4 (T=3012) and the T=4096 test run _experts_triton with the existing (4096, BLOCK_T=128, BLOCK_W=128, num_warps=8, num_stages=3) entry of EXPERT_CONFIGS and stage 5's packed inline-PTX dequant, while turn_1 (T=753) and the T<=1024 tests stay on Marlin. That config is already compiled and checked by _check_expert_gemms in prepare(); the end-to-end Triton-vs-Marlin check (E2E_CHECK_TOKENS = 8192) is unchanged. At T=3012 about 3012 of the 24096 routes are local, ~94 rows per expert, so BLOCK_T 128 covers one expert in one token tile and each weight element is dequantized exactly once (no redundant dequant), making this the shape where cheaper dequant should pay most. If stage 5's log showed register spills for the BLOCK_T 128 variant, use num_stages=2 for it and say so in the attempt file. Nothing else changes: kernel, banks, first_turn config, kernel_fn.

## Expected

- `*load=turn_4*` faster: replaced kernels at turn_4: Marlin x2 at 2131-2217 us (43-64% of the shape). Stage 2's Triton engine with the old dequant already ran turn_4 w13+w2 at ~2.1 ms (on par with Marlin); w13+w2 move 905 MB of NVFP4 codes+scales (floor 189 us at 4.8 TB/s) and 0.30 TFLOP (300 us at 989), so both engines are ~7x their floor. With stage 5's ~3-4x cheaper dequant and one dequant per weight element, target w13+w2 at ~1.0-1.3 ms (Milestone: under 1.6 ms), 0.9-1.2 ms off a 3.3-4.9 ms shape, 20-35% on the turn_4 load
- `*load=turn_1*` neutral: T=753 <= 1024 stays on Marlin
- `*load=first_turn*` neutral: same code as stage 5 at T=16384

## Risk

Risk: if stage 5 did not cut the per-tile dequant cost (its first_turn kernel rows tell), turn_4 will land near stage 2's 2.1 ms, i.e. flat against Marlin; then this stage should wait for a K-loop restructure. The w2 call's -1 blocks zero-store ~21K non-local rows x 4096 x 2 B (~170 MB, ~35 us) at this T, part of the Triton cost Marlin does not pay.

## Milestones

Rewrite stage 6 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
