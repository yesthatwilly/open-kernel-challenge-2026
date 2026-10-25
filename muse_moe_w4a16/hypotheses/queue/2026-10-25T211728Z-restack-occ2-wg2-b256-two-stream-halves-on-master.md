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
  piece already passed the judge as code).
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
review: proceed
status: queued
review_at: 2026-10-25T212523Z
review_companions: null
---

# restack-occ2-wg2-b256-two-stream-halves-on-master

## Change

COMBINATION RESTACK of two measured per-cell wins onto current master (bdbbc97b, combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2, 1.3866x), each behind its own prepare-time gate and its own bound constant, so the next wgmma-side hypotheses build on a base that contains wg2. Master already has F1 (w13 atomic at T=1) and F2 (align2 front end at B256); do NOT port either again, keep master's versions byte for byte. PIECE C2 (from branch hyp/muse_moe_w4a16/combo-wg-occ2-v64-hot-interleave-align2, job 31c27e4b8fc84bd2, 1.4021x board best): port the separate load_inline module WG2_EXT_NAME = muse_moe_w4a16_wgmma_occ2_v1 (wg2_kernel<G, 64, 3, 2, ORD>: NMAX 64, 3 stages, 2 CTAs/SM via setmaxnreg 32/104 at 80 regs, grid 2 x SMs), _wg2 launch glue, _wg_gemm's variant argument (0 = master s5 wg_kernel, 1 = wg2 V64; drop the print-only V64s variant 2), _wg_judge_ids, _wg_zipf_ids if master lacks it, _c2_build_phase, _c2_checks (canary per bucket, self-check, copy check over the judge's state copies) and _c2_select (cold-L2 graph A/B of the full _moe_wgmma path, V64@OCC2 vs master's s5, on _wg_judge_ids AND zipf routings, worse-of-two, require >= 1.02x). Live only at B256 (256 <= T <= 512, verify_32 and the bs=257 test); keep B128 on Marlin (the branch measured V64/Marlin 1.067 there). Keep the COMBO_C2_*_DEADLINE_S guards so a slow build falls back to master. PIECE H (from branch hyp/muse_moe_w4a16/combo-smallm-two-stream-halves-x-w13-splitk, job 9a455a2a): port only mode H (two-stream expert halves: _align_small_halves_kernel/_align_small_halves, _experts_two_stream with w13_atomic=False, _sm_post_norm, _align_halves_self_check, _sm_cross_half, _sm_time_band) for 8 <= T <= 32 behind its own constant TWO_STREAM_MIN_T = 8 / TWO_STREAM_MAX_T = 32 (do not derive the gate from ALIGN_FUSED_MAX_TOKENS or any constant master's align/post-norm cells edit; LESSONS 137). Drop modes H+A and A (no H+A mode passed correctness there; master's F1 covers T=1). Per band (T=8, 16, 32) H goes live only if its cold-L2 graph A/B against MASTER's current path for that band (which now includes toprow's _align_parallel at 17..32 and hoisted post-norm at T=32) wins by >= 1.02x on both skewed and _wg_judge_ids routings; at T=32 H must also reuse master's hoisted post-norm or lose the cell. Print one PATH CHOSEN line per cell (C2: B256 variant; H: per band) and the A/B ratios. No other change: B128, T <= 7 and every kernel outside the two cells stay master. Scored submit (the C2 piece already passed the judge as code).

## Expected

- `bs=256 *` faster: wg2 V64@OCC2 gated 0.955 vs single-CTA on the branch; master verify_32 402.4 us vs board 358.1 us (wg2 w13 225.3 + w2 141.5 us). ~-44 us = ~+2.0% headline
- `bs=8 *` faster: two-stream halves 0.968-0.980 at T 8-32 on its branch, if its gate still wins against toprow master: verify_1 104.5 us -> ~101-102
- `bs=16 *` faster: halves 2-3% on decode_16 (150.9 us), gate-dependent
- `bs=32 *` flat: halves 0.97-0.98 on its branch but master now has alignpar+hoisted post-norm at T=32; the gate decides, expect flat to -2%
- `bs=1 *` flat: untouched
- `bs=128 *` flat: B128 stays Marlin

## Risk

Merge risk, not kernel risk: DECISIONS 2026-10-25T190615Z/T192122Z recorded 6-7 conflicts when merging the occ2 branch; resolve by porting the C2 functions verbatim into master rather than git-merging. Confounders: master has PDL on the wg chain (wg-pdl-gather-w13-w2-chain); wg2 at OCC2 with PDL must be re-validated by _c2_checks (graph replay over both state copies, LESSONS 162). Read PATH CHOSEN lines before crediting rows (row noise +-3-4%). C2 already passed the judge first time; H passed on its branch.

## Adversarial review

Reviewed 2026-10-25T212523Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is DIAGNOSIS action 1: it restacks two pieces of code that have already passed the judge onto master, which is the right next move. It is not a duplicate. The held restack-bucket-gate-on-master-align-atomic carried a different branch, and this one ports the board-best code itself. Each piece is per-cell gated and shows up in the profile under its own kernel name (wg2_kernel at bs=256, _align_small_halves_kernel at bs 8-32), so a single run can be read without companions. Two edits are needed. First, raise both gates to the new LESSONS rule (two interleaved rounds, both >= 1.03, IQR bound) instead of a single round at 1.02. Second, lower the headline expectation: the realistic outcome is about 1.40-1.42, which ties the board rather than clearing it by the 1.24% margin.

### Justification
A. C2 (wg2 V64 at 2 CTAs/SM, B256): the mechanism is plausible and was measured. On the occ2 branch it built cleanly under -Xptxas -Werror at 80 regs. It passed canary N16/N32/N64, self-check (rel-L2 6.5e-3) and the 60-replay copy check over states ABC, and it gated ON at a worse-routing ratio of 0.955. The judged profile shows wg2 w13 225.3 + w2 141.5 = 366.8 us, against s5 258.4 + 157.2 = 415.6 us in toprow (b395f5b6) and 271.7 + 150.2 = 421.9 us in combo-smallm (9a455a2a). Even after the C2 change, verify_32 is 3.3x its 108.9 us floor. The consumer runs at about 2.1-2.5 TB/s, as the occ2 log shows, so this is a TLP gain on a consumer-bound kernel and does not touch the floor gap. That matches the DIAGNOSIS framing, and action 3 is the real follow-up. WEAKEST LINK: the stated +2.0% headline (~-44 us) uses master's verify_32 row of 402.4 us as the baseline. That is the worst of the s5 measurements: combo-smallm judged the same s5 + align2 front end at 379.4 us. The prepare gate predicted 0.955, so the realistic verify_32 gain is 0.89-0.955, worth about +0.8-1.9% of headline. H (two-stream expert halves, T 8-32): this is the same mechanism as marlin-two-stream-expert-halves-small-m (neutral within CI, 1.3024), which DIAGNOSIS lists under 'local minimum'. It is re-proposed only because DIAGNOSIS action 1 names it explicitly. Its own gate ratios on 9a455a2a were 0.980 (T=8, exactly at threshold), 0.968 (T=16) and 0.971 (T=32), and those were against an older master without toprow's _align_parallel and hoisted post-norm at T=17..32. The judged rows cannot attribute H. verify_1 was 103.8 us with H, but 104.5 us on the occ2 run, which never had H, and 110.4 us on toprow, so row noise of +-5% swamps a 2-3% effect. Under the LESSONS rule added today (single-round 1.02x selectors mis-routed bf16 cells; require two interleaved rounds >= 1.03 with an IQR bound), T=8 would likely gate OFF and T=16/32 are borderline. The expected 'bs=8 faster' is therefore a coin flip at best and should read 'flat (gate-dependent)'. Expected directions are otherwise consistent with TASK.md, where every load is weighted equally in the geomean: bs=256 faster, bs=1 and bs=128 flat. Prior: the hypothesis's prior field is accurate. restack-bucket-gate-on-master-align-atomic was a different branch and is on DIAGNOSIS's close list. Nothing in the change touches a Do-not-try item: H is Marlin stream-level concurrency, not a Marlin knob.

### Confounders on the current kernel
- wg-pdl-gather-w13-w2-chain (PDL on the w13->w2 boundary): low risk. The occ2 branch already shipped PDL (WG_EXT_NAME v6_pdl, wg2 carries the same griddepcontrol sites, flags wg_pdl=True in 31c27e4b), so C2's 0.955 was measured with PDL on both arms. With 2 CTAs/SM and ~104 KB/CTA, early-launched w2 CTAs can sit on SM slots during the w13 tail. This was already included in the board measurement, but _c2_checks must re-run on the PDL module master builds now (LESSONS 162), and the build must not take master's non-PDL fallback silently.
- combo-toprow-w13t1-d32-alignpar-postnorm-wg-align2 (F2 align2 front end at B256, D32 alignpar plus hoisted post-norm at T=17..32): C2 inherits F2 unchanged, as on the branch, so there is no confound at B256. H displaces toprow's D32 cell at T=17..32: _align_small_halves replaces _align_parallel and _sm_post_norm competes with the hoisted kernel. Any decode_32 delta therefore mixes 'H vs toprow D32' and should be read from PATH CHOSEN plus kernel names, not the row. Toprow's own D32 cell is unproven (decode_32 was 205.3 us on toprow vs 199.1 on occ2, which lacked it).
- w2-marlin-atomic-add-small-m (w2 use_atomic_add at T<=32): H's two half-streams each run a w2 Marlin call with atomic add. The zeroing and accumulation of each half's output rows must stay disjoint across streams. This passed on 9a455a2a, whose base already had w2 atomic, but the port must keep that buffer split byte for byte.
- align-parallel-per-expert-verify / align-two-phase-port-from17: only through the D32 interaction above. At B256 the front end is align2 chunk 64 in both arms of the C2 selector.

### Interference and A/B needs
No companions. Both pieces are shape-disjoint (C2 at 256<=T<=512, H at 8<=T<=32), each sits behind its own prepare gate with printed A/B medians, and each leaves a distinct kernel name in the profile: wg2_kernel<..,64,3,2,..> at bs=256, and _align_small_halves_kernel plus two-stream Marlin at bs 8-32. A gain or a loss is therefore attributable per cell without a revert run. Reverting PDL is pointless because the board measurement already included it, and a C2-only control would just reproduce 31c27e4b. The one ambiguity companions cannot fix is the +-3-5% judged row noise on bs 8-32. The readout there is PATH CHOSEN plus kernel-only us, not the row, which is why the H gate must be strict (two rounds >= 1.03).

### Open questions
- Gate strength: apply the LESSONS 2026-10-25 rule to both the C2 and H selectors (two interleaved rounds, both >= 1.03, candidate IQR <= 1.5x master's). C2's 0.955 (1.047x) should survive it, and H at T=8 (1.020x) likely will not. Is a likely-OFF T=8 band worth porting at all, or should H's gate start at T=13?
- Promotion policy: the expected result is about 1.40-1.42, which ties the 1.4021 board and needs >= ~1.4195 to clear the 1.24% margin. Since the stated purpose is to put wg2 into master's lineage for actions 3 and 5, will churnsubs land it on master when it is neutral vs best but beats master's 1.3866? If not, the restack fails its purpose on a tie.
- Mode: the spec asks for a scored submit, while the orchestrator default is a check. A scored submit is defensible because both pieces passed the judge as code, but a check first costs nothing toward the submit quota (20 checks vs 10 submits) and verifies that the port did not silently fall back (PATH CHOSEN = master on deadline or build).
- _wg_judge_ids and its constants (WG_JUDGE_SEEDS, WG_POPULARITY, WG_JUDGE_TOUCHED) are absent from master (only _wg_zipf_ids exists at submission.py:2530) and must be ported with C2.
- At T=32, H must use master's hoisted post-norm (HOISTED_POST_NORM_TOKENS=32), not _sm_post_norm. The A/B there should compare H against toprow's full D32 path (alignpar plus hoisted), as the spec says, and print both.

