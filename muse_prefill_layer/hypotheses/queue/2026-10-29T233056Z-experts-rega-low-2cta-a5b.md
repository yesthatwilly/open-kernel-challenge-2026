---
problem: muse_prefill_layer
slug: experts-rega-low-2cta-a5b
created: 2026-10-29T233056Z
base: master
mint_blob: "7569df17e62ecfb6ecf85d12e2033c6611fcaade"
best_at_mint: 1.5721
area: "experts rega tile at the LOW class (turn_1): occupancy / tile shape"
change: >
  Add a low-class-only arm to master's rega dq module: the H100-validated SC-2CTA tile from
  takemyjob hopper/rgx4 290f097 (research/h100-rega4-1029.md): master's A5b+D+SC k loop in a
  2-CTAs-per-SM tile (grid 264), launch registers 80 with setmaxnreg 32 producer / 104 consumer,
  3-stage ring, 64-row slab, chunks capped at N64 (at low, 28 of 32 chunks are N16/N32). Bit-exact
  vs master. Route it ONLY at the low class through master's existing per-class rega race: pick it
  if bit-exact on every draw (torch.equal, replay) and <= 0.97 vs the routed A5b+D+SC path in
  every round (both L2 modes, pop and random routings); high and ft keep A5b+D+SC. Build it as
  extra instantiations in the existing background dq build, keeping the 233 s build gate (retire
  nothing else; count instantiations). Print it in the [rega-dq] summary (RATIOS/PAIR/ROUTE for
  the new arm at low) inside the 16 KB tail.
expected:
  - shapes: load=turn_1*
    direction: faster
    why: "node pair vs A5b+D+SC at low: L2 0.896, HBM 0.938, random 0.887; ncu eligible warps 0.73 -> 1.41, SM cycles -13%"
  - shapes: load=turn_4*
    direction: same
    why: high keeps A5b+D+SC (2-CTA loses x1.13 there)
  - shapes: load=first_turn*
    direction: same
    why: ft keeps A5b+D+SC (x1.23)
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "experts-rega-a5b-desc-splitcommit: same k loop (A5b+D+SC, landed); this changes only the low-class tile occupancy (2 CTAs/SM, smaller slab/ring) to double eligible warps, which the earlier K-pair four-consumer attempt could not at 232 regs"
target: check
value_gain: 1.6
value_p: 0.55
value_ev: 0.88
value_cost: 350k tokens, 1 judge run
evidence: >
  H100 rgx4 (takemyjob hopper/rgx4 290f097; research/h100-rega4-1029.md, GPU 5): SC-2CTA pair vs
  A5b+D+SC at low L2 0.896, HBM 0.938, random 0.887 (high 1.129, ft 1.23); bit-exact, no spill, 2
  CTAs/SM at 80 regs; ncu low w13 eligible warps/scheduler 0.73 -> 1.41, issue active 52 -> 63%,
  SM cycles -13%; prefill board 1.5721 (9f934846)
user_requested: true
review: required
status: queued
---

# experts-rega-low-2cta-a5b

## Change

Add a low-class-only arm to master's rega dq module: the H100-validated SC-2CTA tile from takemyjob hopper/rgx4 290f097 (research/h100-rega4-1029.md): master's A5b+D+SC k loop in a 2-CTAs-per-SM tile (grid 264), launch registers 80 with setmaxnreg 32 producer / 104 consumer, 3-stage ring, 64-row slab, chunks capped at N64 (at low, 28 of 32 chunks are N16/N32). Bit-exact vs master. Route it ONLY at the low class through master's existing per-class rega race: pick it if bit-exact on every draw (torch.equal, replay) and <= 0.97 vs the routed A5b+D+SC path in every round (both L2 modes, pop and random routings); high and ft keep A5b+D+SC. Build it as extra instantiations in the existing background dq build, keeping the 233 s build gate (retire nothing else; count instantiations). Print it in the [rega-dq] summary (RATIOS/PAIR/ROUTE for the new arm at low) inside the 16 KB tail.

## Expected

- `load=turn_1*` faster: node pair vs A5b+D+SC at low: L2 0.896, HBM 0.938, random 0.887; ncu eligible warps 0.73 -> 1.41, SM cycles -13%
- `load=turn_4*` same: high keeps A5b+D+SC (2-CTA loses x1.13 there)
- `load=first_turn*` same: ft keeps A5b+D+SC (x1.23)

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 rgx4 (takemyjob hopper/rgx4 290f097; research/h100-rega4-1029.md, GPU 5): SC-2CTA pair vs A5b+D+SC at low L2 0.896, HBM 0.938, random 0.887 (high 1.129, ft 1.23); bit-exact, no spill, 2 CTAs/SM at 80 regs; ncu low w13 eligible warps/scheduler 0.73 -> 1.41, issue active 52 -> 63%, SM cycles -13%; prefill board 1.5721 (9f934846)
- **Gain if it works:** 1.6% headline; **p** 0.55; **EV** 0.88% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.6% < 2.5%; EV 0.88% < 1.5%), minted because the user described it.
- **Cost:** 350k tokens, 1 judge run.

## Risk

Node->judge transfer for this tile family held within 0.01-0.02 (A3: node low 0.84-0.89 vs judge 0.854; A5b: node HBM-like low .961-.972 vs judge .943). Low class = turn_1 rows (sliding share 21.7%, full 11.7%), rega 48.6% of turn_1 sliding.
