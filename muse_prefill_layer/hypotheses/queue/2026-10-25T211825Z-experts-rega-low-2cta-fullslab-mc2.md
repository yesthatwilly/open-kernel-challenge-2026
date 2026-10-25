---
problem: muse_prefill_layer
slug: experts-rega-low-2cta-fullslab-mc2
created: 2026-10-25T211825Z
base: master
area: "w13 + w2 expert GEMMs at 200 < T <= 1024 (rega low class, turn_1): 2 CTAs/SM with the full 128-route slab, slab multicast across a 2-CTA cluster"
change: >
  DIAGNOSIS action 3. Built on the mc2 branch whatever its score (needs its cluster pairing,
  multicast producer and remote empty arrive). Only the low class (REGA_MIN_TOKENS < T <=
  REGA_LOW_MAX). New instantiation rega_mc_lo_kernel<G>: combo B's two-CTAs-per-SM register split
  (setmaxnreg 32 producer / 104 consumers, __launch_bounds__(384, 2), the V64 consume_item<NMAX,
  STG> ring-geometry variant from muse_prefill_rega_lo, take it from the combo-b patch since mc2
  deletes that module) but with NMAX = 128 (full 128-route slab, the same chunking as master: one
  chunk per expert at the judge's ~1356 local routes, max ~228 per expert) and STG = 2 stages per
  CTA, plus mc2's (2,1,1) cluster slab multicast. SMEM: per CTA 1024 + 2 x 41984 + epilogue (w13
  18432, w2 34816) + static tables; two CTAs must fit in 228 KB (233472 B incl. 1 KB reserve per
  CTA). Print Smem bytes and static size per G at build. If w2 does not fit, stage w2's 256-column
  epilogue in two 128-column passes through an 18432 B buffer with named_bar between passes; if
  w13 still does not fit, report and gate OFF (no smaller slab: the narrow slab is the
  Do-not-try). Grid = 2 x cudaOccupancyMaxActiveClusters at 2 CTAs/SM; print it (expect ~264).
  GATE (combo B's protocol, keep it exactly): candidates master, mc2-low (1 CTA/SM) and this
  kernel, cold-L2 interleaved graph timing at T=753 on BOTH master's random gate routing and the
  1.86x-skew guard routing; score = the WORSE of the two ratios; ON only when that is <= 0.95x the
  best other candidate; exact-compare and two-state replay check first, every bucket forced. Print
  both routings' us for each candidate and PATH CHOSEN.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: turn_1 is the weakest load (1.403). rega w13+w2 ~418 us at T=753 (278+140 sliding). V64 at 2 CTAs/SM won 12% on random routing (0.878) but lost under skew (1.035) because the 64-route slab split experts into 4 chunks; the full slab keeps 1 chunk per expert, and multicast keeps 2 CTAs/SM under the ~6.4 TB/s ingest V64 already hit. Target 418 -> ~370 us, 5-7% of the turn_1 load, ~+2% headline; clears the margin alone only at the top of that range, otherwise stack with mc2.
  - shapes: "*load=turn_4*"
    direction: flat
    why: high class untouched
  - shapes: "*load=first_turn*"
    direction: flat
    why: ft class untouched
risk: high
depends_on: experts-rega-mc2-slab-multicast-all-t
priority: 4
kind: primary
controls_for: null
prior: "combo-b-normgather-two-cta-low-regate: its V64/V32 narrow slabs split popular experts under the 1.86x-skew guard (V64 1.035, V32 1.321) and gated OFF; this keeps the full 128-route slab (one chunk per expert) with 2 stages per CTA and multicast to stay under the ingest cap"
review: required
status: queued
---

# experts-rega-low-2cta-fullslab-mc2

## Change

DIAGNOSIS action 3. Built on the mc2 branch whatever its score (needs its cluster pairing, multicast producer and remote empty arrive). Only the low class (REGA_MIN_TOKENS < T <= REGA_LOW_MAX). New instantiation rega_mc_lo_kernel<G>: combo B's two-CTAs-per-SM register split (setmaxnreg 32 producer / 104 consumers, __launch_bounds__(384, 2), the V64 consume_item<NMAX, STG> ring-geometry variant from muse_prefill_rega_lo, take it from the combo-b patch since mc2 deletes that module) but with NMAX = 128 (full 128-route slab, the same chunking as master: one chunk per expert at the judge's ~1356 local routes, max ~228 per expert) and STG = 2 stages per CTA, plus mc2's (2,1,1) cluster slab multicast. SMEM: per CTA 1024 + 2 x 41984 + epilogue (w13 18432, w2 34816) + static tables; two CTAs must fit in 228 KB (233472 B incl. 1 KB reserve per CTA). Print Smem bytes and static size per G at build. If w2 does not fit, stage w2's 256-column epilogue in two 128-column passes through an 18432 B buffer with named_bar between passes; if w13 still does not fit, report and gate OFF (no smaller slab: the narrow slab is the Do-not-try). Grid = 2 x cudaOccupancyMaxActiveClusters at 2 CTAs/SM; print it (expect ~264). GATE (combo B's protocol, keep it exactly): candidates master, mc2-low (1 CTA/SM) and this kernel, cold-L2 interleaved graph timing at T=753 on BOTH master's random gate routing and the 1.86x-skew guard routing; score = the WORSE of the two ratios; ON only when that is <= 0.95x the best other candidate; exact-compare and two-state replay check first, every bucket forced. Print both routings' us for each candidate and PATH CHOSEN.

## Expected

- `*load=turn_1*` faster: turn_1 is the weakest load (1.403). rega w13+w2 ~418 us at T=753 (278+140 sliding). V64 at 2 CTAs/SM won 12% on random routing (0.878) but lost under skew (1.035) because the 64-route slab split experts into 4 chunks; the full slab keeps 1 chunk per expert, and multicast keeps 2 CTAs/SM under the ~6.4 TB/s ingest V64 already hit. Target 418 -> ~370 us, 5-7% of the turn_1 load, ~+2% headline; clears the margin alone only at the top of that range, otherwise stack with mc2.
- `*load=turn_4*` flat: high class untouched
- `*load=first_turn*` flat: ft class untouched

## Risk

Cluster + 2 CTAs/SM + remote mbarrier arrive: hang/fault risk, and smem may not fit both CTAs; first measurement is an okc check. Keep the worse-of-two-routings gate: a random-only gate is how V64 nearly shipped a skew loss.
