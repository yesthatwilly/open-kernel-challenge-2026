---
problem: muse_decode_layer
slug: hothalf-a3-exactn-ring-wh
created: 2026-10-29T190020Z
base: master
mint_blob: "315119e26cb18e755adf23c3336af2c15c72aafc"
best_at_mint: 1.6904
area: moe hot half (wg_kernel w13/w2, dq_stage) at T=128/256
change: >
  Port the H100-validated 'wh' stack into decode's wgmma hot half (wg_kernel<true>/<false> and its
  dq_stage, muse_decode_layer/submission.py ~:15330; moe twin muse_moe_w4a16/submission.py:1128).
  Three parts, all bit-exact vs master: (1) A3 dequant = moe Engine A's dq2 form (paired-scale
  prmt+IMAD with mul_bf16x2_h<H> broadcast, plus ext_prop nibble extraction on a prepare-time
  word-permuted copy of the hot experts' code bank), exactly as landed in prefill
  (muse_prefill_layer/submission.py, the rega dq module of experts-rega-dq2-dequant-cost-diag,
  master 1b14174f) and as in takemyjob hopper/hh2-1029 e3bcfcd probes/hh2; (2) exact-N tiles
  ('nx'): each hot expert's N tile sized to its real route count rounded to the wgmma N granule
  instead of the current padded buckets (576 columns computed for 386 routes at T=256); (3) for
  tiles with N >= 48, a 4-deep register ring with wgmma_wait<2> ('w4b' arm limited to N>=48;
  neutral/slightly worse at N32, so N32 keeps master's ring). Keep the fork (W6-M), routing, cold
  half (ea4x), PDL and everything else unchanged. Route the new hot half by an in-prepare
  interleaved race per T class (T=128, T=256) vs master's hot half: on only if bit-exact
  (torch.equal) and forked pair ratio <= 0.98 every round; print [hh-wh] RATIO/ROUTE lines in a <=
  2 KB summary at the end of prepare. Prepare-time and size budget: decode is near the 1,000,000 B
  cap (check current size; trim diag text if needed) and the prepare deadline; build the new
  module in a background thread from the top of prepare if it is a separate load_inline module.
expected:
  - shapes: load=verify_32*
    direction: faster
    why: "H100 forked hot+cold stand-in: wh x0.89-0.92 on the T=256 hot half, ~-24.6 judge us per T=256 row"
  - shapes: load=verify_16*
    direction: faster
    why: wh forked x0.95 at T=128, ~-10.4 us per row
  - shapes: load=decode_*
    direction: same
    why: T<=32 rows do not use the hot half
  - shapes: load=verify_1*
    direction: same
    why: T=8
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "CLAIMS:98 (dq2 A3 in the wg hot half, this stack's parent lead) and CLAIMS:59 (wg2 big-T hot-half re-split, refuted); not S1a/CLAIMS:72"
target: check
value_gain: 1.31
value_p: 0.55
value_ev: 0.72
value_cost: 450k tokens, 1 judge run
evidence: >
  H100 gpunode06 hh2 probe (takemyjob hopper/hh2-1029 e3bcfcd; research/h100-hothalf2-1029.md), 20
  rounds/cell GPUs 1+3: forked pair a3 x0.95-0.96/x0.96-0.97 (+0.74%), nx x0.91-0.93/x0.94-0.96
  (+1.19%), wh x0.89-0.92/x0.95 (+1.31% decode headline via triage elasticities d lnH = 3.335e-4
  D256 + 4.652e-4 D128); bit-exact, 168 regs; A3 transfer proven on the judge in prefill (check
  835e1a44 low x0.854 vs node 0.84-0.89; landed 27f0b9af 1.5362); decode board 7cc034f9 anatomy
  hot half 134.5 us at T=256 vs ~39 us floor
user_requested: true
review: proceed
status: running
review_at: 2026-10-29T190531Z
review_companions: null
review_best: 1.6903955680487197
review_blob: 315119e26cb18e755adf23c3336af2c15c72aafc
stage: dispatched
claimed_at: 2026-10-29T190532Z
base_sha: 9a165fa3daa30898af37b9dcd47adf0056de57e3
base_blob: 315119e26cb18e755adf23c3336af2c15c72aafc
branch: hyp/muse_decode_layer/hothalf-a3-exactn-ring-wh
worktree: .worktrees/muse_decode_layer/hothalf-a3-exactn-ring-wh
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-29T194524Z
impl_commit: 2957d173e1e2123fd5d76a9539c144ec7c99fe26
submission_sha256: 354c8fbd10ed03f7edbbd5dda5babfb6ea1768c09b46d4110d01eebcd02f9d1e
norm_sha256: 4ef6127c045d3d98fab82ac17a0f377eb7b82683eac2ea1f83447cf5a1000ca8
submission_blob: 0e45203e9f2feab93221928726e1b59575c7e6d5
lint_warnings: 209
attempt_file: 2026-10-29T194525Z-check-hothalf-a3-exactn-ring-wh.md
prepared_at: 2026-10-29T194525Z
job: 44f303647100481b
dispatched_at: 2026-10-29T194534Z
---

# hothalf-a3-exactn-ring-wh

## Change

Port the H100-validated 'wh' stack into decode's wgmma hot half (wg_kernel<true>/<false> and its dq_stage, muse_decode_layer/submission.py ~:15330; moe twin muse_moe_w4a16/submission.py:1128). Three parts, all bit-exact vs master: (1) A3 dequant = moe Engine A's dq2 form (paired-scale prmt+IMAD with mul_bf16x2_h<H> broadcast, plus ext_prop nibble extraction on a prepare-time word-permuted copy of the hot experts' code bank), exactly as landed in prefill (muse_prefill_layer/submission.py, the rega dq module of experts-rega-dq2-dequant-cost-diag, master 1b14174f) and as in takemyjob hopper/hh2-1029 e3bcfcd probes/hh2; (2) exact-N tiles ('nx'): each hot expert's N tile sized to its real route count rounded to the wgmma N granule instead of the current padded buckets (576 columns computed for 386 routes at T=256); (3) for tiles with N >= 48, a 4-deep register ring with wgmma_wait<2> ('w4b' arm limited to N>=48; neutral/slightly worse at N32, so N32 keeps master's ring). Keep the fork (W6-M), routing, cold half (ea4x), PDL and everything else unchanged. Route the new hot half by an in-prepare interleaved race per T class (T=128, T=256) vs master's hot half: on only if bit-exact (torch.equal) and forked pair ratio <= 0.98 every round; print [hh-wh] RATIO/ROUTE lines in a <= 2 KB summary at the end of prepare. Prepare-time and size budget: decode is near the 1,000,000 B cap (check current size; trim diag text if needed) and the prepare deadline; build the new module in a background thread from the top of prepare if it is a separate load_inline module.

## Expected

- `load=verify_32*` faster: H100 forked hot+cold stand-in: wh x0.89-0.92 on the T=256 hot half, ~-24.6 judge us per T=256 row
- `load=verify_16*` faster: wh forked x0.95 at T=128, ~-10.4 us per row
- `load=decode_*` same: T<=32 rows do not use the hot half
- `load=verify_1*` same: T=8

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 gpunode06 hh2 probe (takemyjob hopper/hh2-1029 e3bcfcd; research/h100-hothalf2-1029.md), 20 rounds/cell GPUs 1+3: forked pair a3 x0.95-0.96/x0.96-0.97 (+0.74%), nx x0.91-0.93/x0.94-0.96 (+1.19%), wh x0.89-0.92/x0.95 (+1.31% decode headline via triage elasticities d lnH = 3.335e-4 D256 + 4.652e-4 D128); bit-exact, 168 regs; A3 transfer proven on the judge in prefill (check 835e1a44 low x0.854 vs node 0.84-0.89; landed 27f0b9af 1.5362); decode board 7cc034f9 anatomy hot half 134.5 us at T=256 vs ~39 us floor
- **Gain if it works:** 1.31% headline; **p** 0.55; **EV** 0.72% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.31% < 2.5%; EV 0.72% < 1.5%), minted because the user described it.
- **Cost:** 450k tokens, 1 judge run.

## Risk

Prior (campaign, not a churn slug): S1a / Engine S hot kernel was refuted at x0.89-0.95 of wg (CLAIMS:72); this keeps master's wg_kernel and its fork and changes only its dequant instruction form (A3, judge-proven on prefill), its N tiling and its ring depth for N>=48, all bit-exact. Kill/read: verify_32 sliding above ~416 us (the A3-only level) means the stack parts did not transfer. The fork arm on the node used a calibrated cold-half stand-in, not ea4x; only the judge measures the real fork. Registers: all variants 168 regs, 0 spill, no C75xx on cu130.

## Adversarial review

Reviewed 2026-10-29T190531Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a user-requested, below-bar judge check. Fresh (blob 315119e2 = mint blob, BEST 1.6904 unchanged); CLAIMS:98's >= 2.5% reopen is waived by the user steer. A +1.31% headline cannot be read in one run (board CI is about +-1.5%, the margin is 1.24%, rows drift 1.5-2.5%). The build must make its own readout: pinned or pair-gated routes, ROUTE lines, distinct kernel names, a DQ0 control instantiation, and an in-prepare master-vs-wh layer A/B on the real ea4x fork at T=128 and T=256. With those, no companion check is needed.

### Justification
A. The mechanism is plausible and measured. The 7cc034f9 profile has the wgmma hot half at 153.5 us (wg<true> 102.1 + wg<false> 51.4) on verify_32 sliding, 130.6 on verify_32 full and 69.6 on verify_16 sliding, against a ~39 us T=256 floor (LOG 1749Z; hothalf-1029 sec. 2: K-step x1.36-4 over the padded-N tensor floor, 576 token columns for 386 routes). All three parts attack measured gaps: (1) the serial dequant chain (A3, 287 -> 223 warp-inst/stage); (2) padding (nx, N80 1294 -> 1063 cyc/stage); (3) MMA exposure at N >= 48 (ring, N128 1296 -> ~1115-1170). These are per-SM cycle effects on a consumer-bound kernel, the class that transfers to the H200, and the prefill A3 transfer (node 0.84-0.89 vs judge 0.854/0.909) confirms that path. Expected shapes and directions match TASK's rows: only verify_16/32 (T=128/256) run the hybrid, and decode_* / verify_1 stay flat. WEAKEST LINK: the forked end-to-end ratio. The node used a spin stand-in for ea4x, with no ALU/L2 contention, calibrated to R=1.97 at T=128. The judge profile shows a much heavier cold side at verify_16 sliding: ea4x w13+w2 197 us vs hot 69.6 under concurrency. So the T=128 share of the estimate (D128 -10.4 us = ~0.48 of the 1.31 points) may hide behind the cold half, leaving ~+0.8% from T=256 alone. Second weakest: exact-N's 8 instantiations cost 2-6% at unchanged N (code size). On a judge draw with mostly N64/N128 chunks, nx alone could be neutral; the per-draw gain is very draw-dependent (verify_32 sliding vs full hot pair 153.5 vs 130.6 us on the board). Prior: the frontmatter says 'prior: none' but the body names S1a (CLAIMS:72). The true closest priors are CLAIMS:98 (A3 alone, this stack's parent lead), CLAIMS:59 (wg2 re-split, refuted) and the old wait<1>/ping-pong D4-varB, which ptxas serialized (C7512). The difference is real: the hh2 SASS shows DEPBAR.LE 0x1/0x2 per k16, not serialized, and the ring is limited to N >= 48, where it measured gains. This is a clerical defect, not grounds for hold. Fix the field to name CLAIMS:98/59. Nothing on FINDINGS' Do-not-try list matches.

### Confounders on the current kernel
- v4xd-stack (CLAIMS:95, the board, master d28b23d7): owns the priority-0 fork stream and the T=128 cold half v4cxs. The hot-half cut's end-to-end value is set by this fork's residency order: 2 x 113 KB ea4x CTAs/SM exclude a 187/204 KB wg CTA. A flat T=128 row could be the fork hiding a real kernel gain, not a non-transfer. Answered in-run by the forked layer A/B (see interference). No revert needed: the fork is only x1.013-1.026 over serial.
- pinned trim (CLAIMS:69) H-16 at T=128/256: fixes the hot set (> 16 routes, block 16) and so the N distribution exact-N sees. Retuning H would change both the hot set and nx's value. Keep it pinned and unchanged so the A/B stays clean.
- W6-G fast silu in wg (CLAIMS:68) and W6-B diet cells (CLAIMS:64): they share the wg epilogue and the T>=128 launch sequence. Exact-N changes the epilogue's NB loop bounds (NB/8 must stay even, true for 48/80/96/112). Outputs must stay bit-identical, so the build's torch.equal check covers the interaction.
- Engine A L3/L4 (CLAIMS:74/78): the same dq2 paired-scale form, already in decode for the cold half. Same numerics family; it does not confound timing, because the cold half's code is unchanged.
- v4x port (CLAIMS:93): the T=128 cold half is v4cxs. Its runtime sets how much of the T=128 hot cut can surface. See fork above.

### Interference and A/B needs
C. No companion check. Every ambiguity is cheaper to resolve inside the same run, and a revert or tune companion would not sharpen a +1.3% headline: one run's noise (~1.5% CI, 1.5-2.5% row drift) swamps it either way. What makes a single result interpretable, and what the build MUST include: (1) ROUTE certainty. CLAIMS:69 found that prepare-time layer margins do not track judged rows, and the T=128 layer cut is ~10 us of ~334 (x0.97, right at the proposed 0.98 bar), so a forked-layer race would flip. Route on the ISOLATED hot pair (w13 -> PDL w2 graph, read-flushed, interleaved), where wh reads x0.82-0.90. Require <= 0.97 vs master's dispatched launch AND vs an in-module DQ0 control, every round (prefill's REGA_DQ rule), plus bit-exactness. Otherwise pin wh, given exactness. Print [hh-wh] ROUTE T=128/T=256 with the per-round ratios. (2) PROOF IN THE PROFILE. Give the new kernels a distinct name or template argument (e.g. wg_kernel<G,1>), so the okc profile shows per row which hot half ran and its GPU time. The readable signal is the verify_32 sliding hot pair: 153.5 us on the board, ~126-131 predicted for wh, ~139-141 for A3 alone. That is a ~15-20% kernel-level move, far above drift, unlike the headline. (3) THE FORK ANSWER the node could not give. After routing, at T=128 and T=256 time a 2 x 2 layer-level arm set {master hot, wh hot} x {serial, fork} on the real ea4x cold half: interleaved, read-flushed, >= 2 inputs x 2 rounds as _sfk_gate does. Print it as [hh-wh] FORK lines. That alone decides whether a flat verify_16 row is 'the fork hid it' or 'no transfer'. (4) A DQ0 control instantiation in the new module: master's dq_stage, buckets and ring text compiled in the new TU. It separates the module/codegen effect from each part. Report fidelity DQ0 / master within 0.97-1.03, else mark INVALID and keep master. Optionally add a3-only and a3+nx pair arms (prepare-only timing, not routed) to attribute the 1.31% among the parts. Decision rule for the readout: a scored run only if the check headline beats 1.6904 by the 1.24% margin. If wh routed in both classes, the profile hot pair is <= x0.90 at verify_32 and the rows are flat or noisy, record [shelf] with the kernel/row numbers and carry wh as a bit-exact rider on the next decode candidate (as the forkguard rode). Kill if the profile hot pair at verify_32 sliding is > ~140 us (A3-only level), or if a route reads OFF. Use the row-level 416 us rule only as a secondary check: it sits inside verify_32 sliding's ~6-11 us run drift.

### Open questions
- Which judge us does the T=128 hot cut buy under the real ea4x fork? The profile's cold/hot at verify_16 sliding is ~2.8x, against the stand-in's calibrated 1.97x. If ~0, the expected headline is ~+0.8%, not +1.31%.
- Does nx's code-size tax (2-6% at unchanged N) show on the judge's actual T=256 draws? Board verify_32 sliding vs full differ by 23 us in the hot pair, so N mixes differ per row. A prepare-time a3 vs a3+nx pair arm would answer it.
- Should the moe twin get A3 alone in the same session? hh2 says moe gains ~+0.9% from A3 and nothing from nx or the ring. Keep this decode check decode-only, so its readout is not entangled.
- Should master's unpermuted wg bank be freed after routing (memory ~0.9 GB), or kept for a per-class split route (T=128 master, T=256 wh)? The build must keep it if the routes can differ per class.


## Implementation notes

**Mechanism:** the T=128/256 wgmma hot half (w13 + w2) runs as the new module's `wg_kernel<G, 7>` = A3 dequant on a prepare-time word-permuted copy of the whole wg bank, plus exact-N tiles (N = 16 nbk) and an R4 register ring with wait<2> for N >= 48, routed per T class on the isolated-pair gate (verify_16 / verify_32 rows).

Edited only `muse_decode_layer/submission.py` (+581/-5; 955,413 -> 984,021 B, under the 990 KB comment-strip line). Nothing compiled or ran on this workstation; built and prepare-validated on gpunode06 (below).

What changed
- **New module `muse_decode_wg_wh`**, derived by `_wh_src()` from master's `_WG_SRC_HEAD` / `_WG_SRC_BODY` with 22 asserted (old, new) edits, each old text occurring exactly once (else OFF). The kernel is templated `wg_kernel<G, V>`, with V bits 1 = A3, 2 = exact N, 4 = ring. Four arms x {w13, w2} = 8 kernels:
  - D0 = `<G,0>`: master's dq_stage, buckets and loop, compiled in this TU (the control).
  - a3 = `<G,1>` and a3nx = `<G,3>`: prepare-only attribution arms, never routed.
  - wh = `<G,7>`: the routed arm.
  - The profile tells master's `wg_kernel<true>` / `<false>` apart from `wg_kernel<true, 7>` / `<false, 7>`.
  - The kernel text equals hh2's `hh_wh.cuh` (takemyjob e3bcfcd), diffed. The helpers (hh_rt16 / hh_scale_pair / hh_mul_h / hh_ext_prop / hh_dq_step) and the ring are verbatim. The only deltas: master's dq_stage is kept, and the A3 form is named `dq_stage_a3`; exact N is a device-side `switch (m.nbk)` in `if constexpr (V & 2)`; the bucket arms index the same 8-map set at `(1 << bucket) - 1`, i.e. boxes of 16 / 32 / 64 / 128 rows and master's expect_tx.
  - Tensor maps are still encoded per call: 8 maps with box rows 16 (b + 1). `Wg<48/80/96/112>` comes from `_wg_wgmma_src(nbs)`, master's generator with a parameter whose default is unchanged.
- **Build:** `_wh_start` runs at the very top of `prepare()`, which is now a thin wrapper: `_wh_start`, then `_prepare_main` (master's former body), then a `finally` that joins. The build is a non-daemon thread at nice +10. TORCH_CUDA_ARCH_LIST is set to 9.0a by the main thread and restored after the final join (prefill's idiom).
  - It builds with master's WG_CUDA_CFLAGS plus `-Xptxas -Werror` first. On failure it greps the compiler text for C75xx. With no C75xx and less than 100 s elapsed it rebuilds without -Werror; any C75xx means OFF. The [hh-wh] BUILD line reports this, so a clean -Werror build proves no C75xx. No cuobjdump.
  - `wh_init` returns cudaFuncGetAttributes per kernel. ATTRS prints regs/spill for all 8 kernels. Any kernel that is not launchable, has occupancy < 1, is not an sm_90a image (bin 90), lacks the register pool, or has local bytes > 0 turns the whole phase OFF.
- **Dispatch:** `_hc_experts` calls `_wh_pair(state, state["whk"][class], ...)`: None runs master's two `_wg_gemm` launches unchanged, 7 runs the wh module. The branch is on the T class only (a shape), never on route counts. Master's gather is kept, as is its w2 PDL setting (`state["wg_pdl"]`, False on master, so neither arm sets the PDL attribute on w2; the griddepcontrol.wait in both kernels is a no-op then). The new module has its own error word `whk_err`.
- **Phase `_wh_phase`** runs after master's hot/cold, cells and anatomy, and just before `_sfk_gate`, so the [w6m-fork] gate sees the final route. Each step is behind a 240 s guard, and an exception sends both classes to master.
  1. Scale scan: every scale byte of the whole wg bank (w13 + w2) must lie in 0x08..0x7E, else OFF.
  2. `_wh_perm`: a copy of the WHOLE bank, every 32-bit code word bit-permuted (the hh2 `perm_word` / prefill LUT layout, checked here against both over 200k random words, with 0 ext_prop mismatches). Each chunk is checked lossless by the inverse; scale bytes are unchanged. It is derived from master's retiled bank (weights only), never from a routing.
  3. Exactness per class: 2 router draws (`_ead_cases`, the gates' inputs) + 4 synthetic routings (every local expert hot over the 4, with route counts giving nbk 1..8 at T=256, local_start 96). Master's eager pair is compared bitwise (torch.equal) to every arm, on both the w13 output `wg_hs[:n_padded]` and the junk-filled w2 output. Error words are checked as [0, magic].
  4. Timing: the ISOLATED pair (w13 -> w2, gather outside the graph) as master's dispatched launch, D0, wh, a3 and a3nx. Graphs are timed with `_rt_time` (read flush, interleaved / rotated, empty graph subtracted), 2 inputs x 2 rounds.
  5. Route ON iff D0 and wh are bit-exact, D0/m is in 0.97..1.03 in every round (else INVALID, master kept), wh/m <= 0.97 AND wh/D0 <= 0.97 in every round, and the replay check passes.
  6. Replay check: wh graphs (gather + w13 + w2) on two scratch sets (state's, and fresh xs/hs/err), refilled 8 times (router draws and all-N synthetic routings alternate), replayed in alternating order: 16 replays per class, each bitwise vs master's eager pair, plus the error words.
  7. FORK: for every exact class, `_ead_layer_gate` (as `_sfk_gate`: whole layer, real ea4x cold half, interleaved, read-flushed, 2 inputs x 2 rounds) over {master hot, wh hot} x {serial, fork}, where fork is `side_serial` False as in `_sfk_gate`. Prints [hh-wh] FORK lines with per-round wh/m under the fork and serially, plus master's fork/serial.
  8. Lines: the ROUTE line lists per-round wh/m, wh/D0, D0/m, a3/D0 and a3nx/D0, the pair us and exactness counts. `_EAD_INPUTS` is trimmed back afterwards (as `_sfk_gate`). If nothing routes, the permuted bank is freed.
- **Log:** the [hh-wh] summary (BUILD, ATTRS, SCALES, ROUTE x2, FORK x2, PATH) is printed last, after the `[w6m-fork] end` line, each line <= 300 chars. During the phase only ~10 short synced step markers print (`[hh-wh] > exact T=256`, ...).
- **Ring hazard** (review 6, unchanged from hh2):
  - The k16 step j writes ring slot j % 4 only after step j - 1's wait<2>, which retired every group <= j - 3. A slot is therefore rewritten 4 commits after the group that read it. An `asm volatile` clobber pins the next slot behind each wait.
  - The previous smem stage is released after this stage's second wait<2> (every group <= its last k16 retired). Items end on wait<0> and release their last stage.
  - The bounded mbar_wait / error-word path is master's.

Deviations / choices
- The pair is timed as master dispatches it: w2 without the PDL attribute, because master's `wg_pdl` is False. The review's "w13 -> PDL w2" wording describes the probe's pair. Routing on the launch that actually runs is the review's intent ("vs master's dispatched launch").
- "Fork" in the 2x2 is `_sfk_gate`'s `side_serial` toggle (the W6-M side-stream fork). CAMPAIGN_HCF is False on master, so the hot and cold halves share the main stream and overlap only through ea4x/wg PDL.
- The judge's prepare log is a ~16,000-char tail, so the summary is not echoed twice.
- The decode-only scope holds: no moe edits.
- Prior (review clerical fix): the frontmatter `prior` now names decode CLAIMS:98 (A3 lead, the parent) and CLAIMS:59 (wg2 re-split, refuted), not S1a / CLAIMS:72.

## Rules check

Each item was checked against `git -C .worktrees/muse_decode_layer/hothalf-a3-exactn-ring-wh diff`.

- [x] CUDA-graph capturable: the kernel_fn path adds only `_wh_pair`, which makes two `wh_gemm` launches or master's two `_wg_gemm` launches. It chooses by a dict lookup on the T class (a shape). No `.item()` / `.cpu()` / sync / tolist, no allocation and no branch on route counts: exact N is selected in-kernel from the chunk's nbk. Every sync / tolist in the diff is in `_wh_replay` / `_wh_phase` (prepare only).
- [x] Nothing derived from `data` is cached across calls. `state["whk"]` holds per-class route ints decided in prepare on synthetic / router-draw inputs. Tensor maps are encoded per call from the scratch passed in that call.
- [x] Weights: the permuted bank is a lossless prepare-time copy of master's retiled wg bank: a fixed bit permutation of each code word, inverse-checked per chunk, with scale bytes unchanged and checked. NVFP4 stays NVFP4 (the same 4-bit codes and E4M3 scales). Dequant is bit-exact to master (torch.equal gate on every routed class).
- [x] Activations BF16 as master (same gather, same bf16 hs / out). No KV / attention change.
- [x] Attention sees every key: n/a, attention untouched.
- [x] No subprocess / multiprocessing / os.fork / spawn in the diff. The build is torch `load_inline` (its ninja / nvcc, as master's modules) inside a `threading.Thread`. No timing patches, harness frames, network or reference import.
- [x] All work happens inside the timed call; no output caching or replay. The prepare-only graphs are freed.
- [x] Bounded waits: the new kernels keep master's bounded `mbar_wait` (error word, never a hang). sm_90a-only PTX stays under `__CUDA_ARCH_FEAT_SM90_ALL`, with error word 2 in the else-branch. TORCH_CUDA_ARCH_LIST=9.0a is set around the build and restored after the join.
- [x] Any miss (build, -Werror C75xx, attributes / spill, scale scan, permutation, exactness, fidelity, ratio, replay, deadline, exception) leaves master's hot half. Loud `[hh-wh] ROUTE` / `PATH CHOSEN` lines name the reason.
- [x] Decode-only: `git diff --stat` shows only `muse_decode_layer/submission.py`; no moe edit.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <wt>/muse_decode_layer/submission.py --problem muse_decode_layer`):
```
OK with warnings: .worktrees/muse_decode_layer/hothalf-a3-exactn-ring-wh/muse_decode_layer/submission.py
(warnings: host-sync / dtype lines, all "outside kernel_fn's call graph; fine in prepare()" or pre-existing master
 dtype lines; the new ones are L16345-16498, inside _wh_replay / _wh_phase)
```
py_compile ok; __pycache__ removed.

## Node validation (gpunode06 H100, GPU 1, 2026-10-29 19:37-19:44Z)

The run was one cold, isolated, prepare-only nodejudge process of the exact worktree bytes. Snapshot `hhwh-354c8fbd10ed.py` is 984,021 B; it is SGLang-shimmed, uses the pinned judge-like router and runs no tests or timed shapes. Config: takemyjob `probes/nodejudge/configs/hhwh-dec.json`. Output: `runs/20261029-123707-gpunode06/out/nodejudge-hhwh-dec.txt`. The nodejudge env was set up fresh for this run (863 s) and is left in place.

The [hh-wh] summary is the last thing printed: 1,381 B, including 207 B of step markers.
```
[hh-wh] BUILD 39s bg from 0.0s, join waited 0s; ptxas -Werror clean, C75xx none
[hh-wh] ATTRS regs/spill w13 D0 r168/l0 a3 r168/l0 a3nx r168/l0 wh r168/l0; w2 D0 r168/l0 a3 r168/l0 a3nx r168/l0 wh r168/l0
[hh-wh] SCALES all 100663296 in 0x08..0x7E; whole bank permuted, lossless (0.3s)
[hh-wh] ROUTE T=128 ON wg_kernel<G,7>: wh/m .884/.882/.874/.879 wh/D0 .879/.877/.875/.880 D0/m 1.005/1.005/.999/.999 a3/D0 .857/.855/.864/.867 a3nx/D0 .902/.899/.873/.874; us m 68.3 D0 68.4 wh 60.1; 6 routings exact; replay 16/16 exact
[hh-wh] ROUTE T=256 ON wg_kernel<G,7>: wh/m .809/.806/.805/.805 wh/D0 .805/.803/.802/.804 D0/m 1.005/1.004/1.004/1.002 a3/D0 .877/.875/.890/.892 a3nx/D0 .822/.820/.822/.821; us m 131.2 D0 131.7 wh 105.8; 6 routings exact; replay 16/16 exact
[hh-wh] FORK T=128 layer us m s/f 374.5/370.1 wh s/f 366.7/362.0; wh/m fork .978/.977/.977/.980 serial .976/.978/.980/.982; fork/serial m .988/.987/.990/.987
[hh-wh] FORK T=256 layer us m s/f 471.7/465.0 wh s/f 455.1/449.4; wh/m fork .967/.965/.966/.967 serial .964/.963/.967/.965; fork/serial m .985/.987/.985/.986
[hh-wh] PATH CHOSEN T=128 wh wg_kernel<G,7>, T=256 wh wg_kernel<G,7> (1.7s, prepare at 131.5s)
```

- Prepare took 131.9 s of 300, cold caches. Peak memory was 10.1 GiB allocated, which includes the permuted bank copy. passed=True. [hotcold], [w6g-hcgate] and [w6m-fork] routed as on the board (H=-16 both classes; fork x1.008-1.012).
- Isolated hot pair: wh x0.874-0.884 at T=128 and x0.805-0.809 at T=256. That matches hh2's pair numbers (x0.885-0.903 / x0.816-0.853). D0 fidelity is 0.999-1.005.
- Whole layer under the real ea4x cold half and the W6-M fork: wh/m x0.977-0.980 at T=128 (-8.1 us) and x0.965-0.967 at T=256 (-15.6 us), forked or serial alike.
- On these H100 layer cuts the headline arithmetic (3.335e-4 D256 + 4.652e-4 D128, before any H200 scaling) gives about +0.9%. That is below hh2's +1.31% stand-in estimate: the real cold half absorbs part of the cut, most at T=128, as the review predicted.
- Attribution at the pair: on these draws the ring+nx stack beats a3 alone by ~2-3 points at T=128 and by ~8 points at T=256. a3nx beats a3 at T=256 (.82 vs .88), but at T=128 it is worse than a3 (.87-.90 vs .86): nx's code-size tax shows where padding is small.
