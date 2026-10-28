---
problem: muse_prefill_layer
slug: experts-rega-mc2-bm256-stack-high-ft
created: 2026-10-25T211825Z
base: master
area: "w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft): 256-row wide tile x 2-CTA slab multicast, stacked"
change: >
  DIAGNOSIS action 2 stacking clause. Run only if BOTH experts-rega-mc2-slab-multicast-all-t and
  experts-rega-s4-bm256-wide-tile-high-ft gate ON (or measure faster) in at least one of high/ft;
  otherwise retire. Start from the mc2 branch and merge the bm256 branch's rega_wide_kernel (both
  append new kernels and leave master's rega_kernel untouched, so the merge is additive). New
  kernel rega_wide_mc_kernel<G, BIG>: bm256's CTA tile (two adjacent master tiles j0=2J, j1=2J+1,
  two register-A fragment sets and two accumulator sets, in-warpgroup dequant/MMA overlap, STAGES
  3, 51200 B per stage) with mc2's cluster pairing applied to WIDE tiles: the 2 CTAs of a (2,1,1)
  cluster take wide tiles 2W and 2W+1 of the same chunk (w13 32 wide tiles -> 16 pairs, w2 16 ->
  8), each producer issues one slab half with .multicast::cluster mask 0b11, empty barrier counts
  both CTAs' consumer warps (cluster-scope remote arrive via mapa), cluster barrier before exit.
  Per stage each CTA then issues 18432 B of weight + 16384 B of slab for 2048 clk of MMA (~17
  B/clk) against master's 41984 B per 1024 clk (~41 B/clk): a ~2.4x cut in issued L2 bytes per
  FLOP. Grid from cudaOccupancyMaxActiveClusters. Gate per class (high T=3012, ft T=16384) in
  _rega_setup exactly as mc2's: exact-compare vs master on random/skewed/one-expert routings with
  every bucket forced, two-state replay check, cold-L2 interleaved graph timing; ON only at <=
  0.97x the FASTEST of {master, mc2, bm256} for that class, so the gate attributes the stack's own
  increment. Low class untouched. Print regs/spill (bm256 already sits near setmaxnreg 232 with
  two accumulator sets: if NB=128 spills, ship the stack for w2 only and say so), grid, per-kernel
  us and issued TB/s for all four candidates, PATH CHOSEN.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: bm256 targets w13 <= 2900 us and w2 <= 1450 us at T=16384; mc2 targets w13+w2 ~4000-4400 us; the stack removes the ingest cap in both ways and should reach MMA/dequant-bound ~3800-4000 us for w13+w2 vs master 5504 us, i.e. another 3-6% of the first_turn load beyond the better single winner. If the increment over the better single is < ~3% of rega it will not clear the 1.24% headline margin alone and must ride with that winner.
  - shapes: "*load=turn_4*"
    direction: faster
    why: high class rega 1098 us -> <= 820 us (bm256 target 880, mc2 target 880).
  - shapes: "*load=turn_1*"
    direction: flat
    why: low class untouched.
risk: high
depends_on: experts-rega-mc2-slab-multicast-all-t
priority: 3
kind: primary
controls_for: null
prior: "experts-rega-s4-bm256-wide-tile-high-ft: queued and unmeasured; it halves activation bytes per FLOP inside one CTA; this adds mc2's cross-CTA slab multicast on top, so issued slab bytes per FLOP drop 4x, and the gate compares against the better single winner"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): STALE: mc2 and bm256 both slower; DISTILL retire"
---

# experts-rega-mc2-bm256-stack-high-ft

## Change

DIAGNOSIS action 2 stacking clause. Run only if BOTH experts-rega-mc2-slab-multicast-all-t and experts-rega-s4-bm256-wide-tile-high-ft gate ON (or measure faster) in at least one of high/ft; otherwise retire. Start from the mc2 branch and merge the bm256 branch's rega_wide_kernel (both append new kernels and leave master's rega_kernel untouched, so the merge is additive). New kernel rega_wide_mc_kernel<G, BIG>: bm256's CTA tile (two adjacent master tiles j0=2J, j1=2J+1, two register-A fragment sets and two accumulator sets, in-warpgroup dequant/MMA overlap, STAGES 3, 51200 B per stage) with mc2's cluster pairing applied to WIDE tiles: the 2 CTAs of a (2,1,1) cluster take wide tiles 2W and 2W+1 of the same chunk (w13 32 wide tiles -> 16 pairs, w2 16 -> 8), each producer issues one slab half with .multicast::cluster mask 0b11, empty barrier counts both CTAs' consumer warps (cluster-scope remote arrive via mapa), cluster barrier before exit. Per stage each CTA then issues 18432 B of weight + 16384 B of slab for 2048 clk of MMA (~17 B/clk) against master's 41984 B per 1024 clk (~41 B/clk): a ~2.4x cut in issued L2 bytes per FLOP. Grid from cudaOccupancyMaxActiveClusters. Gate per class (high T=3012, ft T=16384) in _rega_setup exactly as mc2's: exact-compare vs master on random/skewed/one-expert routings with every bucket forced, two-state replay check, cold-L2 interleaved graph timing; ON only at <= 0.97x the FASTEST of {master, mc2, bm256} for that class, so the gate attributes the stack's own increment. Low class untouched. Print regs/spill (bm256 already sits near setmaxnreg 232 with two accumulator sets: if NB=128 spills, ship the stack for w2 only and say so), grid, per-kernel us and issued TB/s for all four candidates, PATH CHOSEN.

## Expected

- `*load=first_turn*` faster: bm256 targets w13 <= 2900 us and w2 <= 1450 us at T=16384; mc2 targets w13+w2 ~4000-4400 us; the stack removes the ingest cap in both ways and should reach MMA/dequant-bound ~3800-4000 us for w13+w2 vs master 5504 us, i.e. another 3-6% of the first_turn load beyond the better single winner. If the increment over the better single is < ~3% of rega it will not clear the 1.24% headline margin alone and must ride with that winner.
- `*load=turn_4*` faster: high class rega 1098 us -> <= 820 us (bm256 target 880, mc2 target 880).
- `*load=turn_1*` flat: low class untouched.

## Risk

Cluster launch + two register sets near the 232-reg ceiling: hang/fault and spill risk; first measurement is an okc check. Conditional: do not build unless both parents gate ON somewhere.
