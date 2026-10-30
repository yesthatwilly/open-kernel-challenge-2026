---
problem: muse_moe_w4a16
slug: t128-hybrid-eacombine-judge-race
created: 2026-10-30T000244Z
base: master
mint_blob: "d80df5ecc71315466253dae86b28e4f6dfc41642"
best_at_mint: 2.0201
area: "moe T=128 expert block: forked hot/cold hybrid (wh5ds hot + E5 cold) with ea_combine tail vs plain Engine A E5"
change: >
  Route moe T=128 to the forked H-16 hybrid (wh5ds hot half + E5-encoded cold half) with master's
  ea_combine as its tail (the 'gon' arm of takemyjob hopper/t128g-1029 c778f99,
  research/h100-t128g-1029.md), decided by an in-prepare race through kernel_fn on the JUDGE's own
  routing distribution: draw K >= 12 judge-like T=128 routings (the reference's generator with
  fresh seeds, as master's W6-M race does), time hybrid vs plain E5 interleaved per draw (>= 6
  rounds), route the hybrid only if the MEAN over draws is <= 0.97 and no draw's median is > 1.03,
  and every draw is exact vs master. Otherwise T=128 stays plain E5. Print the per-draw ratios,
  mean, worst and PATH CHOSEN. Keep everything else unchanged (T=256 routes, E races, fork setup).
expected:
  - shapes: load=verify_16*
    direction: faster
    why: "node: gon mean 0.934 over 12 judge-shaped draws, 11/12 win; s2 p10/p50 lose x1.04-1.05"
  - shapes: load=decode_*
    direction: same
    why: T<=32 untouched
risk: high
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: none
target: check
value_gain: 1.0
value_p: 0.35
value_ev: 0.35
value_cost: 300k tokens, 1 judge run
evidence: >
  H100 t128h/t128g (research/h100-t128h-1029.md, h100-t128g-1029.md; takemyjob hopper/t128g-1029
  c778f99): master plain E5 257-320 us per T=128 draw vs forked H-16 hybrid 242-296 us; hybrid +
  ea_combine tail mean 0.934 over 12 judge-shaped draws (11 win; worst 1.05x), all exact; moe
  board 2.0201 judged T=128 as plain E5 (702f1004 E5/m 0.955)
user_requested: true
review: required
status: queued
---

# t128-hybrid-eacombine-judge-race

## Change

Route moe T=128 to the forked H-16 hybrid (wh5ds hot half + E5-encoded cold half) with master's ea_combine as its tail (the 'gon' arm of takemyjob hopper/t128g-1029 c778f99, research/h100-t128g-1029.md), decided by an in-prepare race through kernel_fn on the JUDGE's own routing distribution: draw K >= 12 judge-like T=128 routings (the reference's generator with fresh seeds, as master's W6-M race does), time hybrid vs plain E5 interleaved per draw (>= 6 rounds), route the hybrid only if the MEAN over draws is <= 0.97 and no draw's median is > 1.03, and every draw is exact vs master. Otherwise T=128 stays plain E5. Print the per-draw ratios, mean, worst and PATH CHOSEN. Keep everything else unchanged (T=256 routes, E races, fork setup).

## Expected

- `load=verify_16*` faster: node: gon mean 0.934 over 12 judge-shaped draws, 11/12 win; s2 p10/p50 lose x1.04-1.05
- `load=decode_*` same: T<=32 untouched

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** H100 t128h/t128g (research/h100-t128h-1029.md, h100-t128g-1029.md; takemyjob hopper/t128g-1029 c778f99): master plain E5 257-320 us per T=128 draw vs forked H-16 hybrid 242-296 us; hybrid + ea_combine tail mean 0.934 over 12 judge-shaped draws (11 win; worst 1.05x), all exact; moe board 2.0201 judged T=128 as plain E5 (702f1004 E5/m 0.955)
- **Gain if it works:** 1.0% headline; **p** 0.35; **EV** 0.35% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 1.0% < 2.5%; EV 0.35% < 1.5%), minted because the user described it.
- **Cost:** 300k tokens, 1 judge run.

## Risk

The T=128 hybrid flipped sign node vs judge before (CLAIMS:63, H-16f x0.922-0.942 then refuted on the judge); the race on the judge's own draws is the safeguard. A worst draw > 1.03 vetoes.
