---
problem: muse_prefill_layer
slug: rega-s2-low-rs2-high-stack
created: 2026-10-30T002745Z
base: master
mint_blob: "7569df17e62ecfb6ecf85d12e2033c6611fcaade"
best_at_mint: 1.5721
area: "experts rega tile: S2 2-CTA at low + RS2 (wait<1> per group) at high, both on A5b+D+SC"
change: >
  Stack two shelved, bit-exact rega arms in master's per-class dq race (prefill master = A5b+D+SC,
  1.5721): (1) S2, the 2-CTA/SM low-class tile from experts-rega-low-2cta-a5b (branch
  hyp/muse_prefill_layer/experts-rega-low-2cta-a5b; judge 074a0e02 routed it at low: S2/A5b
  .931-.932, guard .968), restacked from its patch with all its gates; (2) RS2 at the HIGH class
  only (takemyjob hopper/rgx3 2b6d50e; research/h100-rega3-1029.md: wait<1> every group so the
  next stage's dequant runs under the previous stage's second wgmma group; node pair vs A5b+D+SC
  at high 0.959-0.963, w2 0.955; low/ft no gain; check ptxas C7510-C7514 and route only if clean),
  raced at high with the same <= 0.97 vs A5b every round + exact rule. ft keeps A5b+D+SC. Keep the
  build gate: count instantiations (S2 2 + RS2 2 for high) and probe only the new ones so the 233
  s need holds; print the [rega-dq] summary with both arms.
expected:
  - shapes: load=turn_1*
    direction: faster
    why: "S2 at low: judge 074a0e02 S2/A5b .931-.932"
  - shapes: load=turn_4*
    direction: faster
    why: "RS2 at high: node pair 0.959-0.963 vs A5b+D+SC"
  - shapes: load=first_turn*
    direction: same
    why: ft keeps A5b+D+SC
risk: medium
depends_on: experts-rega-low-2cta-a5b
priority: 1
kind: primary
controls_for: null
prior: "experts-rega-low-2cta-a5b: S2 alone read flat on the headline (1.5678 vs 1.5721) while routed; this adds RS2 at high so the stacked gain clears one run's noise"
target: check
value_gain: 1.8
value_p: 0.5
value_ev: 0.9
value_cost: 350k tokens, 1 judge run
evidence: >
  judge 074a0e02f4e94efd: S2 routed at low, S2/A5b 0.932 pop warm, 0.931 cold, 0.885 random, guard
  1348 routes 0.968, vs master 0.743-0.772 (turn_1 rega ~317 us of 643 us); H100 rgx3
  (research/h100-rega3-1029.md): RS2 pair vs A5b+D+SC at high L2 0.963, HBM 0.959, w2 0.955
  (turn_4 rega ~1076 us)
user_requested: true
review: required
status: queued
---

# rega-s2-low-rs2-high-stack

## Change

Stack two shelved, bit-exact rega arms in master's per-class dq race (prefill master = A5b+D+SC, 1.5721): (1) S2, the 2-CTA/SM low-class tile from experts-rega-low-2cta-a5b (branch hyp/muse_prefill_layer/experts-rega-low-2cta-a5b; judge 074a0e02 routed it at low: S2/A5b .931-.932, guard .968), restacked from its patch with all its gates; (2) RS2 at the HIGH class only (takemyjob hopper/rgx3 2b6d50e; research/h100-rega3-1029.md: wait<1> every group so the next stage's dequant runs under the previous stage's second wgmma group; node pair vs A5b+D+SC at high 0.959-0.963, w2 0.955; low/ft no gain; check ptxas C7510-C7514 and route only if clean), raced at high with the same <= 0.97 vs A5b every round + exact rule. ft keeps A5b+D+SC. Keep the build gate: count instantiations (S2 2 + RS2 2 for high) and probe only the new ones so the 233 s need holds; print the [rega-dq] summary with both arms.

## Expected

- `load=turn_1*` faster: S2 at low: judge 074a0e02 S2/A5b .931-.932
- `load=turn_4*` faster: RS2 at high: node pair 0.959-0.963 vs A5b+D+SC
- `load=first_turn*` same: ft keeps A5b+D+SC

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** judge 074a0e02f4e94efd: S2 routed at low, S2/A5b 0.932 pop warm, 0.931 cold, 0.885 random, guard 1348 routes 0.968, vs master 0.743-0.772 (turn_1 rega ~317 us of 643 us); H100 rgx3 (research/h100-rega3-1029.md): RS2 pair vs A5b+D+SC at high L2 0.963, HBM 0.959, w2 0.955 (turn_4 rega ~1076 us)
- **Gain if it works:** 1.8% headline; **p** 0.5; **EV** 0.9% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.8% < 2.5%; EV 0.9% < 1.5%), minted because the user described it.
- **Cost:** 350k tokens, 1 judge run.

## Risk

Each part alone is ~+1% headline (below one run); stacked ~+1.5-2%. S2's judge read is direct; RS2's is node-only (A5b-family node->judge transfer held within 0.01-0.02).
