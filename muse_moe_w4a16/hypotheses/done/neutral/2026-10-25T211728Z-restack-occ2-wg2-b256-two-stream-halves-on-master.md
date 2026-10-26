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
status: done
review_at: 2026-10-25T212523Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T212549Z
base_sha: 2b505e1ba8846dfb42e476ada051a571345fbfbf
base_blob: 35b8a851e4ef5fa8fd66b12d09caac88b7ea7615
branch: hyp/muse_moe_w4a16/restack-occ2-wg2-b256-two-stream-halves-on-master
worktree: .worktrees/muse_moe_w4a16/restack-occ2-wg2-b256-two-stream-halves-on-master
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T010521Z
impl_commit: c680bbec3ebcf57b11d29786989250025009368a
submission_sha256: 2c0b1fc88f025cd8acbfddec84c7c7136053c1c706b379fce74343beda1bfe6f
norm_sha256: eb313baf583db0b5d5ae0c495057a1c73e1b501f8d3b55a7fa796145f340ed98
submission_blob: 28f02abd4f47ebe530bc077aa45168bb8027abc3
lint_warnings: 120
attempt_file: 2026-10-26T012541Z-check-restack-occ2-wg2-b256-two-stream-halves-on-master.md
prepared_at: 2026-10-26T012541Z
job: 6ee7b6e302a348c8
dispatched_at: 2026-10-26T012819Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.3996458875332363
speedup_ci: [1.3786511992202377, 1.4206405758462346]
harvested_at: 2026-10-26T023155Z
best_at_harvest: 1.4020871870995952
promotion: null
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


## Implementation notes

Edited only `.worktrees/muse_moe_w4a16/restack-occ2-wg2-b256-two-stream-halves-on-master/muse_moe_w4a16/submission.py` (4262 -> 5772 lines, +1562/-52 vs the claim base). Built on master's blob 35b8a851 (no git merge: functions ported verbatim from the two source branches, then adapted to master's `_combo_setup` framework). Not run (no GPU here): py_compile ok, okc_lint ok (0 errors, 120 prepare-only host-sync/dtype warnings), and the module imports cleanly with torch/triton/sglang stubbed (module-level asserts, WG2 smem/register budget, and `_wg_audit_cpp` of the wg2 source = no findings).

What changed
- **C2 (wg2 V64 at 2 CTAs/SM, B256 only).** Ported from `hyp/muse_moe_w4a16/combo-wg-occ2-v64-hot-interleave-align2`: `WG2_*` constants, `_WG2_CPP`, `_WG2_SRC_BODY` (wg2_kernel<G, 64, 3, 2, ORD>, setmaxnreg 32/104 at 80 regs, grid 2 x SMs, bounded mbarrier waits, `__CUDA_ARCH_FEAT_SM90_ALL` guard), `_wg2_src`, `_wg2_build` (sm_90a only, `-Xptxas -Werror` first, `-lineinfo`, arch flags printed, build time printed), `_c2_build_phase` (numRegs == 80, localSizeBytes == 0, CTAs/SM >= 2, smem fit, static/global audit; regs/spills printed per instantiation), `_wg_judge_ids` + `WG_POPULARITY` / `WG_JUDGE_SEEDS` / `WG_JUDGE_TOUCHED`, `_wg_route_stats`, `_wg2_chunk_stats`, `_c2_checks` (canary N16/N32/N64, `_wg_self_check`, 3-state `_wg_copy_check`, all with variant 1), `_c2_select`, `_c2_phase`. Master's wg plumbing gained the variant argument exactly as on the branch: `_wg_gemm(..., variant=0)` (0 = master's s5 module/grid, 1 = wg2 module with `state['wg2_grid']`, PDL only on w2 as master), `_moe_wgmma(..., variant=None)` reading `state['wg2_variant_256']` at T >= 256 (`wg2_variant_128` is always 0), `_wg_chunks(cut)`, and `variant` threaded through `_wg_eval_case`, `_wg_bucket_canary(n_buckets)`, `_wg_self_check`, `_wg_copy_check` (marker WG2_MAGIC, wg2_ext shared via the deepcopy memo). The print-only V64s variant 2 is dropped everywhere (its ORD=1 instantiations and the python rows are removed). B128 stays master's choice. Master's F1/F2 front end code is untouched (master's per-bucket `wg_align2` dict stays; the branch's `_wg_front` int variant is not ported, `_c2_front` mirrors master's three branches for the kernel-only prints).
- **H (two-stream expert halves, mode H only).** Ported from `hyp/muse_moe_w4a16/combo-smallm-two-stream-halves-x-w13-splitk`: `_align_small_halves_kernel`, `_align_small_halves`, `_experts_two_stream` (w13_atomic parameter and `sm_w13` dropped; w13 non-atomic, w2 mode from the cross-half wipe test, own side stream + second Marlin workspace made in the H phase after C2's state deepcopy), `_sm_post_norm`, `_align_halves_self_check`, `_sm_cross_half`, `_sm_time_band`, `_two_stream_phase` (trimmed `_combo_c3`). Own bound constants `TWO_STREAM_MIN_T = 8`, `TWO_STREAM_MAX_T = 32`, `TWO_STREAM_BANDS`, `TWO_STREAM_COUNT` (nothing derived from ALIGN_FUSED_MAX_TOKENS or any constant master's align/post-norm cells edit; LESSONS 137). Modes H+A and A are dropped. Renamed helpers that would have collided with master's `_combo_routings` (`_sm_routings`, `_sm_xw`). `kernel_fn` gains one branch: shape in 8..32 and `state['sm_mode'][band]` -> `_experts_two_stream`; everything else is master's call unchanged.
- **Selectors (review required fix 1).** New `_sel2_time` / `_sel2_gate`: two interleaved rounds of 41 cold-L2 (256 MB flush before every replay) graph replays, order rotated and flipped, empty-graph baseline subtracted per round; a candidate is live only if EVERY round on EVERY gating routing has master/candidate >= 1.03 AND candidate IQR <= 1.5x master's. Baseline is master's ACTUALLY ROUTED path: the full `kernel_fn` on a shallow state copy with the candidate flag off (so wg_on_256 / wg_align2 / d32 / hoisted post-norm / w13 mode as final), candidate = same `kernel_fn` with only its flag on. C2 gates on `_wg_judge_ids` and zipf a=0.3 at T=256; H gates per band (rep T 8 / 16 / 32) on skewed and `_wg_judge_ids`. Both print every round's medians and IQRs, kernel names, and a PATH CHOSEN line per cell (C2: B256 variant; H: per band) with a compact medians summary; those lines repeat as the last lines of prepare via `_combo_setup`'s `chosen` list.
- **Phasing / budgets.** New phases in `_combo_setup`: after F1, F2, C1a, C1b and master's copy check (so a slow C2 cannot starve it and revert landed flags): `C2-build` (start <= 170 s), `C2` (checks <= 215 s via `COMBO_C2_CHECK_DEADLINE_S`, selector <= 240 s via `COMBO_C2_SELECT_DEADLINE_S`, no timing past 255 s), `H` (start <= 250 s, no band timing past 262 s), then `copy2` (start <= 275 s) = `_combo_copy_check(only=("C2", "H"))`, a second graph copy check through kernel_fn on the state and a tensor-cloning copy (C2 at T=256/257, H at each band's lo/rep/hi); a miss resets only C2/H. Master's `F2-select-diag` (never changes a flag) now runs last. A skipped phase prints `skipped: deadline` and falls back to master. A build error, self-check miss, deadline, unsupported shape or raise leaves master's exact path; C2 failure restores master's `wg_on_256`.
- State flags added in `prepare()`: `wg2_variant_128 = 0`, `wg2_variant_256 = 0`, `sm_mode = {8: 0, 16: 0, 32: 0}`.

Review fixes applied
1. Both gates raised to the LESSONS 2026-10-25 rule (two interleaved rounds both >= 1.03x, IQR <= 1.5x master's) instead of one round at 1.02x.
2. Headline expectation lowered: realistic outcome ~1.40-1.42 (ties the 1.4021 board; needs ~1.4195 to clear the margin) and `bs=8 / 16 / 32` read "flat, gate-dependent". The front-matter `expected` block is left as minted (the review corrects the spec; the harvest should read PATH CHOSEN, not the row).
3. `_c2_checks` re-runs on the PDL module master builds now; the PDL state (`wg_ext_pdl`, `wg_pdl`) is printed at the start of the C2 phase so a non-PDL fallback is never silent.
4. w2 atomic/zero buffer split: H's halves keep the branch's shared cache3 with disjoint rows byte for byte, plus the cross-half wipe test (falls back to non-atomic w2, then H off).
5. `_wg_judge_ids` and its constants ported with C2 (and reused for H's gating routing).
6. At T=32 H reuses master's post-norm: `_sm_post_norm` calls master's hoisted Triton kernel iff master's own `state['d32_post_norm']` flag is on (else SGLang's fused kernel exactly as master); H is timed against master's full D32 path (alignpar + hoisted), printed in the PATH CHOSEN line (`master = [kernel names]`).

Open questions answered
- Gate strength: applied to both selectors. A likely-OFF T=8 band is still ported (the band costs one short A/B); the gate decides, no band starts at 13.
- Promotion policy: not a build matter; left to churnsubs. Mode: built for a check (as the brief says); nothing in the file depends on submit vs check.
- Deviations: (a) C2/H run after master's copy check with a separate copy2 pass (the branches ran theirs inside one combined copy check); (b) the C2 selector times `kernel_fn` itself on state copies instead of the branch's `_moe_wgmma` vs `_moe_marlin` pair, so the baseline is master's real routing including a Marlin B256 if master's own selector chose it; (c) the selector's third "random" routing and the B128 print-only row are dropped to save prepare time; (d) H warms the band's other T values only after the band wins (the Triton align variants are per T), then runs a per-T check against master.
- Biggest risks: prepare time (C2 build + checks + select, then H, then copy2 must finish inside 300 s: later phases skip by deadline and fall back to master, so a late start means master's path, not a failure); wg2 compile with `-Xptxas -Werror` (a setmaxnreg warning turns C2 off); the 2-CTA/SM PDL w13 -> w2 interaction (checked by the canary, self-check and the A/B/C copy check).

## Rules check

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints; no allocation or control flow on input values (shapes and `config` are fine). Checked against `git diff`: kernel_fn's only new code is a branch on `num_tokens`, `bank` constants and the plain int `state['sm_mode'][band]`; `_experts_two_stream` allocates by shape only, forks/joins streams with `wait_stream` (no sync, no record_stream); `_moe_wgmma` / `_wg_gemm` read the plain int `variant`. All `.item()`/`.tolist()`/synchronize lines are in prepare-only functions (okc_lint lists them as outside kernel_fn's call graph).
- [x] Nothing derived from `data` is cached across calls: every new tensor in the timed path is allocated per call from shapes; the new state entries (`wg2_ext`, `wg2_grid`, `wg2_variant_*`, `sm_mode`, `sm_side`, `sm_workspace_b`, `sm_w2`) are built from weights/constants/synthetic prepare inputs only. The tensor maps are encoded per call from the bsrc passed in that call (as master), no module-global device state (wg2 source passes `_wg_audit_cpp`).
- [x] Weights: repacked losslessly in `prepare()` only; NVFP4 stays NVFP4, BF16 stays BF16. n/a for new repacking: C2 reads master's existing retiled bank, H reads master's Marlin bank; no weight is stored at a different precision.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3. n/a for KV/queries (MoE problem); activations stay bf16 end to end in both ported paths, silu/accumulate in fp32 as on the branches.
- [x] Attention sees every key: n/a, MoE problem (no attention).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: grep of the file for subprocess / multiprocessing / os.fork / Popen / os.system finds nothing; the wg2 build is an in-process `load_inline` (arch list `9.0a` set around it and restored); no reference import.
- [x] All work happens inside the timed call; no output caching or replay: C2 and H run their kernels inside kernel_fn per call; prepare only builds, checks and selects.
- [x] Prepare-time selectors: two interleaved rounds, both clearing 1.03x, candidate IQR <= 1.5x master's; master's actually routed path (flag-off) timed; both rounds' medians/IQRs, kernel names and PATH CHOSEN printed per cell even when master is kept; compact PATH CHOSEN summary repeated as the last lines of prepare (via `_combo_setup`'s `chosen`).
- [x] New paths have their own per-state flags (`wg2_variant_256`, `sm_mode`); a miss falls back to master's exact path and never disables an unrelated landed flag (C2/H run after master's copy check; copy2 resets only C2/H; `reset_c2` restores master's `wg_on_256`).
- [x] Any build error, self-check miss, deadline or unsupported shape lands on master's path and never raises; bounded mbarrier waits in wg2_kernel (error word 1, `dead` flag); no host sync / allocation-on-values / data-derived caching / module-global device state / child processes in kernel_fn; wg2 kernel has a distinct symbol (`wg2_kernel`, `wg2_gemm`, `wg2_init`, module `muse_moe_w4a16_wgmma_occ2_v1`); build time and regs/spills printed.
- [x] Prepare budget (300 s) and 15-minute run limit respected by construction: every new phase has its own start deadline (`COMBO_DEADLINES_S`, `COMBO_C2_*_DEADLINE_S`, `COMBO_TS_TIMING_DEADLINE_S`) and prints `skipped: deadline`. Not measurable here (no GPU): the realized phase seconds are in the run's `[combo] phase ...` lines.

okc_lint output (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`): `OK with warnings`, 0 errors, 120 warnings, every one `host-sync` (prepare-only `.item()`/`.tolist()`/synchronize, "outside kernel_fn's call graph") or `dtype` (uint8 flush/pad buffers, prepare-only). Full output saved at /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/lint.txt. `python3 -I -m py_compile` ok; `__pycache__` deleted.

## Pre-flight audit

Verdict: dispatch (no fixes needed; submission.py untouched by the audit). Checked by reading and by AST diffs against the source branches.
- C2 vs hyp/muse_moe_w4a16/combo-wg-occ2-v64-hot-interleave-align2: _wg2_build, _wg_judge_ids, _wg_route_stats, _c2_select internals, WG2 constants identical; _WG2_SRC_BODY differs only by the intended V64s (variant 2) drop; _wg_gemm / _moe_wgmma / _wg_chunks / _wg_eval_case / _wg_bucket_canary / _wg_self_check / _wg_copy_check differ only by master's own F2 front end (kept byte for byte) and docstrings. PDL stays on the w2 launch only, as master.
- H vs hyp/muse_moe_w4a16/combo-smallm-two-stream-halves-x-w13-splitk: _align_small_halves(_kernel), _align_halves_self_check identical; _experts_two_stream differs only by the dropped w13-atomic branch; _sm_post_norm adds master's hoisted post-norm at T=32 iff master's d32_post_norm is on; renames only elsewhere.
- Graph safety: kernel_fn H path allocates by shape, forks with side.wait_stream(current), joins with current.wait_stream(side), no sync, no record_stream; side stream and workspace B are made in prepare after C2's deepcopy check. copy2 (_combo_copy_check only=("C2","H")) uses the tensor-cloning walk (stream shared, workspace cloned) and resets only C2 (wg_on_256 restored) / H bands; a skipped or raised copy2 also resets only C2/H.
- Deadlines: master's logged prepare is ~60 s, wg2 build 38 s, C2 phase 2.4 s, H branch phase 1.7 s, so the chain lands near 105-150 s against start deadlines 170/215/250/275; ample slack.
Residual risk: the new two-round 1.03x + IQR gate may leave H (esp. T=8) and possibly C2 OFF, giving a master-equal run; unmeasurable here without a GPU.

## Result

- job `6ee7b6e302a348c8` (check), status done, passed True, tests 5/5
- speedup 1.3996x, 95% CI [1.3787, 1.4206], score 9608.501778761987 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.4020871870995952, "ci": [1.3810558792931014, 1.423118494906089], "job": "31c27e4b8fc84bd2"}; base result: muse_moe_w4a16/hypotheses/done/neutral/combo-wg-occ2-v64-hot-interleave-align2.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/restack-occ2-wg2-b256-two-stream-halves-on-master.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/restack-occ2-wg2-b256-two-stream-halves-on-master.patch`; branch `hyp/muse_moe_w4a16/restack-occ2-wg2-b256-two-stream-halves-on-master`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| bs=1 load=decode_1 sharding=ep8 | 60 | 46.35 | 43.88 | 1.0212 | flat |
| bs=16 load=decode_16 sharding=ep8 | 60 | 150.9 | 147.25 | 0.9962 | flat |
| bs=32 load=decode_32 sharding=ep8 | 60 | 199.12 | 199.05 | 1.0134 | flat |
| bs=8 load=verify_1 sharding=ep8 | 60 | 104.51 | 113.31 | 0.977 | flat |
| bs=128 load=verify_16 sharding=ep8 | 60 | 323.16 | 318.93 | 0.9896 | flat |
| bs=256 load=verify_32 sharding=ep8 | 60 | 358.12 | 369.82 | 0.9927 | flat |

no shape moved outside noise. Expected hit: 3/6. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: restack-occ2-wg2-b256-two-stream-halves-on-master`) that adds a further change in the same area is the only way to learn more.

