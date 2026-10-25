---
problem: muse_moe_w4a16
slug: cuda-nvfp4-wgmma-rega-s2-fix-gate-from-8
created: 2026-10-25T011332Z
base: master
area: w13 marlin + silu-mul + w2 marlin
change: >
  Stage 2 of the wgmma register-A grouped NVFP4 rewrite with the gate widened to the small
  verify/decode shapes. Implement exactly the correctness work of
  cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix (read its spec: per-launch localization in the
  canary, zero smem ring/slab at CTA start and zero-filled padding rows, offline numpy check of
  the wgmma m64nNk16 register-A mapping, B smem descriptor LBO/SBO/swizzle, mbarrier expect_tx
  bytes and scale application), and additionally lower the wgmma path's own bound constant from
  num_tokens >= 128 to num_tokens >= 8 (bs=8 verify_1, bs=16, bs=32, and the bs=33 test), using
  the kernel's existing smallest N buckets (16 and 32 tokens per expert tile; under EP8 with
  skewed routing each local expert gets about 1-8 routes at bs 8-32). bs=1 and the bs=4 test stay
  on master Marlin. Extend the canary and the full self-check to M in {8, 16, 32, 33} as well as
  128/256/257, each with its own rel-L2 line, and enable the small-M range only if every small-M
  case passes (separate flag, separate PATH CHOSEN line for 8..127 and for 128..512). No tiling,
  pipeline or dequant change otherwise. Motivation: Marlin is furthest from the floor in the
  middle shapes (bs=8 114.0 us vs floor 37.3, bs=16 176.1 vs 52.5, bs=32 207.5 vs 70.8 in the best
  profile), and the persistent 132-CTA grid streams each touched expert's weight once regardless
  of rows, which is the property Marlin's m-block scheme lacks. First measurement is an okc check.
expected:
  - shapes: bs=8 *
    direction: faster
    why: Marlin pair 114.0 + 2.0 us (best profile) vs floor 37.3; M2 below Marlin, M3 <= 56
  - shapes: bs=16 *
    direction: faster
    why: 176.1 + 2.2 us vs floor 52.5; M3 <= 79
  - shapes: bs=32 *
    direction: faster
    why: 207.5 + 2.3 us vs floor 70.8; M3 <= 106. decode_32 is master weakest load (1.03x vs SGLang)
  - shapes: bs=128 *
    direction: faster
    why: "as in the finite-fix sibling: below 292.5 + 3.9 us"
  - shapes: bs=256 *
    direction: faster
    why: below 391.4 + 6.4 us
  - shapes: bs=1 *
    direction: flat
    why: "control: stays Marlin"
risk: high
depends_on: cuda-nvfp4-wgmma-rega-grouped-sm90a-rebuild-inproc
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 8 <= num_tokens <= 512
review: required
status: queued
---

# cuda-nvfp4-wgmma-rega-s2-fix-gate-from-8

## Change

Stage 2 of the wgmma register-A grouped NVFP4 rewrite with the gate widened to the small verify/decode shapes. Implement exactly the correctness work of cuda-nvfp4-wgmma-rega-grouped-s2-finite-fix (read its spec: per-launch localization in the canary, zero smem ring/slab at CTA start and zero-filled padding rows, offline numpy check of the wgmma m64nNk16 register-A mapping, B smem descriptor LBO/SBO/swizzle, mbarrier expect_tx bytes and scale application), and additionally lower the wgmma path's own bound constant from num_tokens >= 128 to num_tokens >= 8 (bs=8 verify_1, bs=16, bs=32, and the bs=33 test), using the kernel's existing smallest N buckets (16 and 32 tokens per expert tile; under EP8 with skewed routing each local expert gets about 1-8 routes at bs 8-32). bs=1 and the bs=4 test stay on master Marlin. Extend the canary and the full self-check to M in {8, 16, 32, 33} as well as 128/256/257, each with its own rel-L2 line, and enable the small-M range only if every small-M case passes (separate flag, separate PATH CHOSEN line for 8..127 and for 128..512). No tiling, pipeline or dequant change otherwise. Motivation: Marlin is furthest from the floor in the middle shapes (bs=8 114.0 us vs floor 37.3, bs=16 176.1 vs 52.5, bs=32 207.5 vs 70.8 in the best profile), and the persistent 132-CTA grid streams each touched expert's weight once regardless of rows, which is the property Marlin's m-block scheme lacks. First measurement is an okc check.

## Expected

- `bs=8 *` faster: Marlin pair 114.0 + 2.0 us (best profile) vs floor 37.3; M2 below Marlin, M3 <= 56
- `bs=16 *` faster: 176.1 + 2.2 us vs floor 52.5; M3 <= 79
- `bs=32 *` faster: 207.5 + 2.3 us vs floor 70.8; M3 <= 106. decode_32 is master weakest load (1.03x vs SGLang)
- `bs=128 *` faster: as in the finite-fix sibling: below 292.5 + 3.9 us
- `bs=256 *` faster: below 391.4 + 6.4 us
- `bs=1 *` flat: control: stays Marlin

## Risk

Shares the sibling's bug risk: if the finite fix is wrong both fail. Read per-shape rows: small-M and large-M ranges have separate flags, so one can be on while the other is off.

## Milestones

Rewrite stage 2 of `Marlin w13+w2 (moe_wna16_marlin_gemm x2) + silu_and_mul at 8 <= num_tokens <= 512`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
