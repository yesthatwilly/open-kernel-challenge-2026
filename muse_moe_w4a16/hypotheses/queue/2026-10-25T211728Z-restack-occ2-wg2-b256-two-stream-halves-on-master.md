---
problem: muse_moe_w4a16
slug: restack-occ2-wg2-b256-two-stream-halves-on-master
created: 2026-10-25T211728Z
base: master
area: expert GEMM path selection (wgmma wg2 at B256 + Marlin two-stream halves at T 8-32) restack
change: >
  COMBINATION RESTACK of two measured per-cell wins onto current master (bdbbc97b,
  combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2, 1.3866x), each behind its own prepare-time
  gate and its own bound constant, so the next wgmma-side hypotheses build on a base that contains
  wg2. Master already has F1 (w13 atomic at T=1) and F2 (align2 front end at B256); do NOT port
  either again, keep master's versions byte for byte. PIECE C2 (from branch
  hyp/muse_moe_w4a16/combo-wg-occ2-v64-hot-interleave-align2, job 31c27e4b8fc84bd2, 1.4021x board
  best): port the separate load_inline module WG2_EXT_NAME = muse_moe_w4a16_wgmma_occ2_v1
  (wg2_kernel<G, 64, 3, 2, ORD>: NMAX 64, 3 stages, 2 CTAs/SM via setmaxnreg 32/104 at 80 regs,
  grid 2 x SMs), _wg2 launch glue, _wg_gemm's variant argument (0 = master s5 wg_kernel, 1 = wg2
  V64; drop the print-only V64s variant 2), _wg_judge_ids, _wg_zipf_ids if master lacks it,
  _c2_build_phase, _c2_checks (canary per bucket, self-check, copy check over the judge's state
  copies) and _c2_select (cold-L2 graph A/B of the full _moe_wgmma path, V64@OCC2 vs master's s5,
  on _wg_judge_ids AND zipf routings, worse-of-two, require >= 1.02x). Live only at B256 (256 <= T
  <= 512, verify_32 and the bs=257 test); keep B128 on Marlin (the branch measured V64/Marlin
  1.067 there). Keep the COMBO_C2_*_DEADLINE_S guards so a slow build falls back to master. PIECE
  H (from branch hyp/muse_moe_w4a16/combo-smallm-two-stream-halves-x-w13-splitk, job 9a455a2a):
  port only mode H (two-stream expert halves: _align_small_halves_kernel/_align_small_halves,
  _experts_two_stream with w13_atomic=False, _sm_post_norm, _align_halves_self_check,
  _sm_cross_half, _sm_time_band) for 8 <= T <= 32 behind its own constant TWO_STREAM_MIN_T = 8 /
  TWO_STREAM_MAX_T = 32 (do not derive the gate from ALIGN_FUSED_MAX_TOKENS or any constant
  master's align/post-norm cells edit; LESSONS 137). Drop modes H+A and A (no H+A mode passed
  correctness there; master's F1 covers T=1). Per band (T=8, 16, 32) H goes live only if its
  cold-L2 graph A/B against MASTER's current path for that band (which now includes toprow's
  _align_parallel at 17..32 and hoisted post-norm at T=32) wins by >= 1.02x on both skewed and
  _wg_judge_ids routings; at T=32 H must also reuse master's hoisted post-norm or lose the cell.
  Print one PATH CHOSEN line per cell (C2: B256 variant; H: per band) and the A/B ratios. No other
  change: B128, T <= 7 and every kernel outside the two cells stay master. Scored submit (the C2
  piece already passed the judge as code)."
expected:
  - shapes: bs=256 *
    direction: faster
    why: wg2 V64@OCC2 gated 0.955 vs single-CTA on the branch; master verify_32 402.4 us vs board 358.1 us (wg2 w13 225.3 + w2 141.5 us). ~-44 us = ~+2.0% headline
  - shapes: bs=8 *
    direction: faster
    why: "two-stream halves 0.968-0.980 at T 8-32 on its branch, if its gate still wins against toprow master: verify_1 104.5 us -> ~101-102"
  - shapes: bs=16 *
    direction: faster
    why: halves 2-3% on decode_16 (150.9 us), gate-dependent
  - shapes: bs=32 *
    direction: flat
    why: halves 0.97-0.98 on its branch but master now has alignpar+hoisted post-norm at T=32; the gate decides, expect flat to -2%
  - shapes: bs=1 *
    direction: flat
    why: untouched
  - shapes: bs=128 *
    direction: flat
    why: B128 stays Marlin
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "restack-bucket-gate-on-master-align-atomic: that held restack carried an older bucket-gate branch; this one ports the board-best code itself (wg2 C2 module + selector, job 31c27e4b) plus the two-stream H cell from 9a455a2a, leaving out F1/F2 which master already has. Expected ~1.41-1.43 vs board 1.4021, so it clears the 1.24% margin only if C2 reproduces AND at least one H band goes live."
review: required
status: queued
---

# restack-occ2-wg2-b256-two-stream-halves-on-master

## Change

COMBINATION RESTACK of two measured per-cell wins onto current master (bdbbc97b, combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2, 1.3866x), each behind its own prepare-time gate and its own bound constant, so the next wgmma-side hypotheses build on a base that contains wg2. Master already has F1 (w13 atomic at T=1) and F2 (align2 front end at B256); do NOT port either again, keep master's versions byte for byte. PIECE C2 (from branch hyp/muse_moe_w4a16/combo-wg-occ2-v64-hot-interleave-align2, job 31c27e4b8fc84bd2, 1.4021x board best): port the separate load_inline module WG2_EXT_NAME = muse_moe_w4a16_wgmma_occ2_v1 (wg2_kernel<G, 64, 3, 2, ORD>: NMAX 64, 3 stages, 2 CTAs/SM via setmaxnreg 32/104 at 80 regs, grid 2 x SMs), _wg2 launch glue, _wg_gemm's variant argument (0 = master s5 wg_kernel, 1 = wg2 V64; drop the print-only V64s variant 2), _wg_judge_ids, _wg_zipf_ids if master lacks it, _c2_build_phase, _c2_checks (canary per bucket, self-check, copy check over the judge's state copies) and _c2_select (cold-L2 graph A/B of the full _moe_wgmma path, V64@OCC2 vs master's s5, on _wg_judge_ids AND zipf routings, worse-of-two, require >= 1.02x). Live only at B256 (256 <= T <= 512, verify_32 and the bs=257 test); keep B128 on Marlin (the branch measured V64/Marlin 1.067 there). Keep the COMBO_C2_*_DEADLINE_S guards so a slow build falls back to master. PIECE H (from branch hyp/muse_moe_w4a16/combo-smallm-two-stream-halves-x-w13-splitk, job 9a455a2a): port only mode H (two-stream expert halves: _align_small_halves_kernel/_align_small_halves, _experts_two_stream with w13_atomic=False, _sm_post_norm, _align_halves_self_check, _sm_cross_half, _sm_time_band) for 8 <= T <= 32 behind its own constant TWO_STREAM_MIN_T = 8 / TWO_STREAM_MAX_T = 32 (do not derive the gate from ALIGN_FUSED_MAX_TOKENS or any constant master's align/post-norm cells edit; LESSONS 137). Drop modes H+A and A (no H+A mode passed correctness there; master's F1 covers T=1). Per band (T=8, 16, 32) H goes live only if its cold-L2 graph A/B against MASTER's current path for that band (which now includes toprow's _align_parallel at 17..32 and hoisted post-norm at T=32) wins by >= 1.02x on both skewed and _wg_judge_ids routings; at T=32 H must also reuse master's hoisted post-norm or lose the cell. Print one PATH CHOSEN line per cell (C2: B256 variant; H: per band) and the A/B ratios. No other change: B128, T <= 7 and every kernel outside the two cells stay master. Scored submit (the C2 piece already passed the judge as code)."

## Expected

- `bs=256 *` faster: wg2 V64@OCC2 gated 0.955 vs single-CTA on the branch; master verify_32 402.4 us vs board 358.1 us (wg2 w13 225.3 + w2 141.5 us). ~-44 us = ~+2.0% headline
- `bs=8 *` faster: two-stream halves 0.968-0.980 at T 8-32 on its branch, if its gate still wins against toprow master: verify_1 104.5 us -> ~101-102
- `bs=16 *` faster: halves 2-3% on decode_16 (150.9 us), gate-dependent
- `bs=32 *` flat: halves 0.97-0.98 on its branch but master now has alignpar+hoisted post-norm at T=32; the gate decides, expect flat to -2%
- `bs=1 *` flat: untouched
- `bs=128 *` flat: B128 stays Marlin

## Risk

Merge risk, not kernel risk: DECISIONS 2026-10-25T190615Z/T192122Z recorded 6-7 conflicts when merging the occ2 branch; resolve by porting the C2 functions verbatim into master rather than git-merging. Confounders: master has PDL on the wg chain (wg-pdl-gather-w13-w2-chain); wg2 at OCC2 with PDL must be re-validated by _c2_checks (graph replay over both state copies, LESSONS 162). Read PATH CHOSEN lines before crediting rows (row noise +-3-4%). C2 already passed the judge first time; H passed on its branch.
