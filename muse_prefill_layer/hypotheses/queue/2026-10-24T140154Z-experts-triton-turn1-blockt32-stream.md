---
problem: muse_prefill_layer
slug: experts-triton-turn1-blockt32-stream
created: 2026-10-24T140154Z
base: master
area: "w13 marlin + w2 marlin (Triton NVFP4 engine, stage 7: take over turn_1 with a weight-stream config)"
change: >
  Stage 7 of the Triton NVFP4 expert rewrite, built on stage 6 (branch
  hyp/muse_prefill_layer/experts-triton-packed-dequant-turn4, whatever its score). Two edits in
  muse_prefill_layer/submission.py. (1) MARLIN_MAX_TOKENS from 1024 to 0 (the gate stays, so a
  control can restore it): every prefill shape and test (all T > 8) runs _experts_triton with
  stage 5's packed inline-PTX dequant; the Marlin bank and the e2e Triton-vs-Marlin check in
  prepare() stay as they are (Marlin becomes the check's reference only). (2) Replace the first
  EXPERT_CONFIGS entry (T <= 1024: turn_1 T=753 and the T=200/300/753 tests) with a
  bandwidth-shaped config: BLOCK_T=32 (turn_1 puts ~24 local rows on each expert, so one 32-wide
  token tile per expert and wgmma n32 instead of today's 64 with ~60% padding), BLOCK_W=128,
  BLOCK_K=128 (fixed by the nibble order), GROUP_SIZE_T=8, num_warps=4, num_stages=4 (~17 KB of
  codes+scales+activations per stage, ~68 KB in flight per CTA, so 3 CTAs per SM keep ~200 KB of
  weight bytes in flight per SM to stream HBM). At this T each weight element is dequantized once
  and the work is a weight stream: w13+w2 read 905 MB of NVFP4 codes and e4m3 scales (189 us at
  4.8 TB/s). _check_expert_gemms already sizes its check to max(CHECK_ROWS, BLOCK_T) rows and
  compiles every entry, so the new config is checked in prepare(); the 4096 and first_turn entries
  are unchanged. If the prepare log shows spills or smem over budget, fall back to num_stages=3
  and say so in the attempt file.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "replaced kernels at turn_1: Marlin x2 at 653.9 us (full) / 752.4 us (sliding), 45-67% of the shape, ~3.5-4x the 189 us weight-stream floor. Stage 2's Triton engine (old dequant, BLOCK_T 64) ran 0.80 ms here; with packed dequant, no token padding beyond 32 and 4 stages in flight, target w13+w2 at ~300-400 us (Milestone: under 500 us on both turn_1 shapes), 250-450 us off a 1.08-1.42 ms shape, 20-35% on the turn_1 load"
  - shapes: "*load=turn_4*"
    direction: neutral
    why: "T=3012 keeps stage 6's BLOCK_T 128 config"
  - shapes: "*load=first_turn*"
    direction: neutral
    why: T=16384 keeps the BLOCK_T 256 config
risk: medium
depends_on: experts-triton-packed-dequant-turn4
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 7
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T
review: required
status: queued
---

# experts-triton-turn1-blockt32-stream

## Change

Stage 7 of the Triton NVFP4 expert rewrite, built on stage 6 (branch hyp/muse_prefill_layer/experts-triton-packed-dequant-turn4, whatever its score). Two edits in muse_prefill_layer/submission.py. (1) MARLIN_MAX_TOKENS from 1024 to 0 (the gate stays, so a control can restore it): every prefill shape and test (all T > 8) runs _experts_triton with stage 5's packed inline-PTX dequant; the Marlin bank and the e2e Triton-vs-Marlin check in prepare() stay as they are (Marlin becomes the check's reference only). (2) Replace the first EXPERT_CONFIGS entry (T <= 1024: turn_1 T=753 and the T=200/300/753 tests) with a bandwidth-shaped config: BLOCK_T=32 (turn_1 puts ~24 local rows on each expert, so one 32-wide token tile per expert and wgmma n32 instead of today's 64 with ~60% padding), BLOCK_W=128, BLOCK_K=128 (fixed by the nibble order), GROUP_SIZE_T=8, num_warps=4, num_stages=4 (~17 KB of codes+scales+activations per stage, ~68 KB in flight per CTA, so 3 CTAs per SM keep ~200 KB of weight bytes in flight per SM to stream HBM). At this T each weight element is dequantized once and the work is a weight stream: w13+w2 read 905 MB of NVFP4 codes and e4m3 scales (189 us at 4.8 TB/s). _check_expert_gemms already sizes its check to max(CHECK_ROWS, BLOCK_T) rows and compiles every entry, so the new config is checked in prepare(); the 4096 and first_turn entries are unchanged. If the prepare log shows spills or smem over budget, fall back to num_stages=3 and say so in the attempt file.

## Expected

- `*load=turn_1*` faster: replaced kernels at turn_1: Marlin x2 at 653.9 us (full) / 752.4 us (sliding), 45-67% of the shape, ~3.5-4x the 189 us weight-stream floor. Stage 2's Triton engine (old dequant, BLOCK_T 64) ran 0.80 ms here; with packed dequant, no token padding beyond 32 and 4 stages in flight, target w13+w2 at ~300-400 us (Milestone: under 500 us on both turn_1 shapes), 250-450 us off a 1.08-1.42 ms shape, 20-35% on the turn_1 load
- `*load=turn_4*` neutral: T=3012 keeps stage 6's BLOCK_T 128 config
- `*load=first_turn*` neutral: T=16384 keeps the BLOCK_T 256 config

## Risk

Risk: the dist=outliers test (T=200) can put far more than 32 rows on one expert; that only adds token tiles (moe_align pads per expert to BLOCK_T), so it is a correctness non-issue but not timed. At ~24 rows per expert the CTA count is 32 experts x 64 W tiles = 2048 for w13 and 1024 for w2, plenty to fill 132 SMs. If the turn_1 rows land between Marlin and the target, an ab-tune on num_stages (3 vs 5) or BLOCK_W 64 is the next step, not a revert.

## Milestones

Rewrite stage 7 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) at every prefill T`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
