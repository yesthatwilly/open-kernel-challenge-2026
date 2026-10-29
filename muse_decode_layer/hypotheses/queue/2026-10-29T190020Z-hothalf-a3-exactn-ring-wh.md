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
review: required
status: queued
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
