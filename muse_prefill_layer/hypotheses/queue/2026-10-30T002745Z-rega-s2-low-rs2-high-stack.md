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
review: proceed
status: queued
review_at: 2026-10-30T003004Z
review_companions: null
review_best: 1.5721154841318719
review_blob: 7569df17e62ecfb6ecf85d12e2033c6611fcaade
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

## Adversarial review

Reviewed 2026-10-30T003004Z by an adversarial subagent (model opus).

**Verdict: proceed.** E: fresh (board 1.5721, blob 7569df17 unchanged); CLAIMS.md:79 S2 [shelf] reopen ('ride in the next prefill candidate') is met in form, but the stack sums to ~1.8% < the 2.5% stack bar. It proceeds only because the user asked for it. Fix the build-gate arithmetic before building: with RS2 the need is ~259 s, not 233 s. Read the result from the per-class [rega-dq] pair ratios, not from the headline.

### Justification
A: Both arms are plausible, bit-exact per-SM schedule changes on the A5b+D+SC tile. S2 has a direct judge read: check 074a0e02 routed it at low with pair S2/A5b .932 warm, .931 cold, .885 random and guard 1348r .945-.959 (gate g.968 <= 1.00). The probe shows 80 regs, 0 spill and C7510/C7512 0. RS2 is node-only (campaigns/2026-10-26-expert-bandwidth/research/h100-rega3-1029.md). wait<1> per group removes the per-stage wgmma-wait barrier at high: w13 cycles 968k -> 940k and tensor 78.7 -> 81.1%. The high pair is 0.963 at L2 and 0.959 at HBM, w2 0.955. At low it is neutral or worse (low hbm 1.045), and at ft it is noise (0.990 / 1.016, +-10% round spread), so high-only routing is right. RS2-2CTA hit C7512, so RS2 must be instantiated only on the 1-CTA, 168-reg high tile, never combined with S2's 80-reg template. WEAKEST LINK: whether RS2 routes at all on the judge. The gate is <= 0.97 vs A5b on every round plus exact. RS2's interval is [0.950-0.976] at L2 and [0.950-0.976] at HBM, and the earlier 2-round val runs read 0.970-0.974, which would FAIL that gate. Treat p(RS2 routes) as ~0.4-0.5. If it does not route, the run is S2 alone, already measured flat (1.5678; its single run also 'lost on full first_turn', a row it does not touch, which shows the ~1-1.5% run noise). SECOND: the build gate claim is wrong. On the S2 branch: est = 40 x (INST + PROBE_INST)/4 and need = used + 1.3 x est + 25. Today 14 + 2 gives need 233. Adding RS2's 2 instantiations (16) plus probing only RS2 (2) gives est 180 and need = used + 259. That passes the 270 limit only while used <= 11 s. The judge started the build at 0.1 s and the real build took 40.5 s of a 160 s estimate, so it passes, but with 11 s of slack, not 37. Probing both S2 and RS2 (16 + 4) gives 285 and the whole dq module goes OFF, losing the landed A5b routing too. Required: point PROBE_INST at RS2's 2 instantiations only, pin S2 by its judge-proven source (no re-probe), log need, and assert need <= 270 in a local dry count. Expected directions match TASK weights: turn_1 rega ~317 of 643 us x ~7% gives ~+1% headline, turn_4 rega ~1076 us x ~4% gives ~+0.7%, ft flat. PRIOR: S2's own run is cited correctly. RS2 has no earlier prefill attempt. It is the deferred wgmma_wait<1> sibling named in CLAIMS.md:71's second line. It is not the refuted anti-phase or K-pair (CLAIMS.md:57-58), because RS2 changes neither warp count nor issue order across WGs. Not a do-not-retry duplicate.

### Confounders on the current kernel
- experts-rega-a5b-desc-splitcommit (CLAIMS.md:76, landed): the base of both arms. RS2 is a modification of its split commit (rolling instead of drained per stage), so any gain is measured relative to it by construction. The race gates vs A5b (b) and master (m), so this is attribution, not confounding.
- experts-rega-w13-fast-silu-epilogue (landed) and the w2st arm a pinned at high/ft (CLAIMS.md:56): the rgx3 node tile is rgx2's generated tile and may not carry master's fast-silu epilogue or the w2 early gs/sid loads. RS2's 0.955-0.963 is vs a node A5b+D+SC that may differ from master's high-class launch. The in-prepare race against master's dispatched launch (m) covers the routing decision. A node figure that does not transfer shows up as RS2 not routing, not as a wrong route.
- rega_lo / REGA_LO_GUARD_ROUTES (landed low-class gate): S2's route depends on the guard routing (~1350 routes) at <= 1.00. That passed at .945-.959 on 074a0e02 and should hold unchanged, since the S2 arm is a byte-identical restack.

### Interference and A/B needs
No companions. The two arms touch disjoint classes (S2 low only, RS2 high only, ft untouched), and the [rega-dq] summary prints per-class pair ratios and the ROUTE decision for each arm inside the same run. Attribution is therefore in the log: a turn_1 move belongs to S2 and a turn_4 move to RS2. A revert companion of A5b is not implementable, since both arms are built on its encoded bank. A control of S2 alone already exists (074a0e02, 1.5678). The headline alone stays ambiguous at ~1.8% expected vs a ~+-1.5% single-run CI. Judge the run on the route lines, the pair ratios, and the turn_1/turn_4 rega us in the profile vs 074a0e02 and da80fdb9. Do not judge it on the board.

### Open questions
- Does master's RS2 port keep the gate's 0.97 every-round rule at high? Given node 0.959-0.976 and val 0.970-0.974, would the user accept a per-class RS2 threshold, or is it better to know it will likely not route? Do not loosen the threshold in this run.
- Is RS2 a BIG=true high-class instantiation pair (w13, w2) only, with no DQ-mode cross product? Any extra template axis puts need at >= 285 s and turns the dq module OFF.
- Should REGA_DQ_EST_4INST_S be recalibrated (the judge built 14 inst in 40.5 s vs est 160 s)? That would restore build slack. It is a separate change and should not ride silently in this stack.

