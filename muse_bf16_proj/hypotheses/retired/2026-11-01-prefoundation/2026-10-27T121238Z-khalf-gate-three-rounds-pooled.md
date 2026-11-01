---
problem: muse_bf16_proj
slug: khalf-gate-three-rounds-pooled
created: 2026-10-27T121238Z
base: master
area: pooler m=128 khalf (b2) cell gate (_khalf prepare gate only)
change: >
  Built on master (8f233e88 lineage; khalf b2 landed, scored bc147c5b 1.2735). The khalf (b2)
  pooler cell at m=128 is a real, judged win, but its gate is flaky: KHALF_ROUNDS = 2 read-flush
  _cell_gate rounds that must BOTH clear KHALF_MIN_RATIO = 1.03. Check a6424247 routed it (r1
  1.036, r2 1.048); check c65ffcb6 did not (r1 1.025, r2 1.054), and that run's headline fell to
  1.2507 from the board's 1.2735 (bf16 lost ~1.8% to a gate miss, not to its own change). CHANGE,
  in the khalf gate only (the constants block at ~3700 and the function that runs the rounds and
  decides the flag): run KHALF_ROUNDS = 3 rounds; route when EVERY round has master/cand >=
  KHALF_FLOOR_RATIO = 1.01 (never a measured loss or tie) AND the median of the three rounds'
  ratios is >= 1.03 (the old bar, now on the median), with the existing IQR rule unchanged per
  round. Print all three rounds' net us and ratios, the median, and which rule decided, in the
  PATH CHOSEN K line (keep it in the end-of-prepare re-print). Nothing else changes: same
  candidate (b2: bmm fp32 partials + _add2_cast_kernel), same canary, same A/B copy check, same
  deadlines (one extra round costs ~0.1 s).
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "routes khalf in runs where one of the two rounds dips just under 1.03 (c65ffcb6 r1 1.025): the judged pooler m=128 row keeps its ~-1 us; headline-wise this removes a ~-1.8% downside draw rather than adding upside"
  - shapes: "*"
    direction: flat
    why: gate-only change; every other cell and kernel identical
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "pooler-khalf-m128-fp32-partials-triton-castadd: that change (landed as khalf b2 in 8f233e88) built the cell and its two-round all-must-pass gate; this one changes only the gate's decision rule (three rounds, per-round floor 1.01, median >= 1.03) after c65ffcb6 showed the two-round rule dropping a real win"
review: hold
status: hold
review_at: 2026-10-27T121627Z
review_companions: [khalf-gate-three-rounds-pooled-control-gate-stats-pooled-iqr]
hold_reason: "adversarial review: The premise is contradicted by the archived logs: the board run bc147c5b (1.2735) itself did not route khalf, and failed on the IQR rule, not the ratio rule; c6"
---

# khalf-gate-three-rounds-pooled

## Change

Built on master (8f233e88 lineage; khalf b2 landed, scored bc147c5b 1.2735). The khalf (b2) pooler cell at m=128 is a real, judged win, but its gate is flaky: KHALF_ROUNDS = 2 read-flush _cell_gate rounds that must BOTH clear KHALF_MIN_RATIO = 1.03. Check a6424247 routed it (r1 1.036, r2 1.048); check c65ffcb6 did not (r1 1.025, r2 1.054), and that run's headline fell to 1.2507 from the board's 1.2735 (bf16 lost ~1.8% to a gate miss, not to its own change). CHANGE, in the khalf gate only (the constants block at ~3700 and the function that runs the rounds and decides the flag): run KHALF_ROUNDS = 3 rounds; route when EVERY round has master/cand >= KHALF_FLOOR_RATIO = 1.01 (never a measured loss or tie) AND the median of the three rounds' ratios is >= 1.03 (the old bar, now on the median), with the existing IQR rule unchanged per round. Print all three rounds' net us and ratios, the median, and which rule decided, in the PATH CHOSEN K line (keep it in the end-of-prepare re-print). Nothing else changes: same candidate (b2: bmm fp32 partials + _add2_cast_kernel), same canary, same A/B copy check, same deadlines (one extra round costs ~0.1 s).

## Expected

- `layer=pooler load=verify_16 m=128` faster: routes khalf in runs where one of the two rounds dips just under 1.03 (c65ffcb6 r1 1.025): the judged pooler m=128 row keeps its ~-1 us; headline-wise this removes a ~-1.8% downside draw rather than adding upside
- `*` flat: gate-only change; every other cell and kernel identical

## Risk

Not a speedup by itself: it makes an existing win route reliably. The judge's headline from one run will usually be flat vs the board; evidence is the PATH CHOSEN K line (routed, with the three ratios). Risk: a looser gate could route khalf on a run where it is genuinely slower; the 1.01 per-round floor plus the 1.03 median bar keeps that unlikely.

## Adversarial review

Reviewed 2026-10-27T121627Z by an adversarial subagent (model opus).

**Verdict: hold.** The premise is contradicted by the archived logs: the board run bc147c5b (1.2735) itself did not route khalf, and failed on the IQR rule, not the ratio rule; c65ffcb6 also failed the IQR rule in r1. The proposal keeps the per-round IQR rule and adds a third draw of it, so it fixes neither observed miss and probably lowers the routing rate. A judged check cannot measure a ~0.3% headline gate change. Rewrite so the IQR test runs on pooled samples, and test the rule with a multi-round gate diagnostic.

### Justification
b2 is a real win: across 8 recorded rounds master/cand is 1.025-1.054 (mean ~1.041, SD ~0.009); judged pooler verify_16 21.87 us with khalf vs 23.08 us without. Median-of-3 at 1.03 with a 1.01 floor cuts ratio misses from ~21% to ~4% without loosening the bar. But KHALF_MAX_IQR (cand IQR <= 1.5x max(master IQR, 0.5 us) per round on raw 41-rep samples) is the binding constraint: bc147c5b r2 cand IQR 1.12 vs bar 0.915 (routed master); c65ffcb6 r1 cand IQR 1.15 vs bar 1.11. A 41-sample IQR has ~25-30% relative SE, so a 3-round all-must-pass IQR test lowers the pass rate (~0.56 -> ~0.42); combined routing ~0.44 -> ~0.40. The '~1.8% lost to a gate miss' premise is wrong: the 1.2735 board run is itself khalf-off, and c65ffcb6's decode_1/verify_1/decode_16 loads also fell 1.7-3.6% (drift or its own change). The pooler row is ~0.27% of the headline. The slug says 'pooled' but nothing is pooled.

### Confounders on the current kernel
- 8f233e88 (khalf b2 + read-flush gates, hand-landed): the inherited per-round IQR clause, not the median rule, decides the outcome.
- pooler-nn-layout-widen-m256 / restack-three-cell-wins: the gate's master side is NN torch.mm (splitK nvjet + splitKreduce), whose variance feeds the ratio and IQR; background term.
- Earlier read-flush gates in prepare share flush size, reps and clock state; a reason to measure the IQR distribution in situ.

### Interference and A/B needs
A single primary run is uninterpretable (one Bernoulli draw of the routing decision; headline effect ~0.27%). The right instrument is one prepare-time diagnostic drawing many gate rounds and evaluating old, parent and pooled rules on the same data: given as a control companion that should replace the parent.

### Companions queued
- `khalf-gate-three-rounds-pooled-control-gate-stats-pooled-iqr`: Shows from one check which clause makes the khalf gate flaky (per-round IQR fail fraction vs ratio < 1.03 fraction) and gives each candidate rule's miss rate on identical data (OLD vs PARENT vs POOLED, 3 triples each). If the IQR fail fraction is ~0.25 per round as the 8 logged rounds suggest, PARENT is refuted and POOLED is the rule to land.

### Open questions
- What does the IQR clause guard against? A pooled IQR or p90/p10 on pooled samples guards bimodality without tripping on 41-rep noise.
- Accept on the gate diagnostic plus a routed PATH CHOSEN K line; a flat headline is not neutral evidence.
- Run 474490c8 (BEST.json's check_job) shows master splitK kernels for pooler verify_16 and no khalf line: is BEST's check_job pre-khalf?
- Re-attribute c65ffcb6's drop (1.2735 -> 1.2507): every load fell 1.7-3.6% (drift or qkv16-s5), not khalf; this matters for how qkv-swapab16-five-stage-ring-m8-m16 was judged.

