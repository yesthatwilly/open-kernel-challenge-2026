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
prior: none
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
status: queued
review_at: 2026-10-29T190531Z
review_companions: null
review_best: 1.6903955680487197
review_blob: 315119e26cb18e755adf23c3336af2c15c72aafc
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

