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
review: proceed
status: hold
review_at: 2026-10-25T212524Z
review_companions: null
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): STALE: mc2 never landed (1.31-1.81x slower); DISTILL 3.1 retire; REGA_LO_SKIP on master"
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

## Adversarial review

Reviewed 2026-10-25T212524Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a real follow-up to the combo-B and two-CTA lesson, not a retry: it keeps the full 128-route chunking so popular experts are not split, and its own three-way gate (master, mc2-low, this kernel) reports the result per candidate and per routing, so no companion run is needed. Two things should be fixed in the brief before the build: the stated reason the shared memory fits is wrong, and the NB=128 register budget has not been checked.

### Justification
A. The mechanism is plausible, but part of it is overstated. Evidence for it: FINDINGS and LESSONS show turn_1 is the weakest load (1.403). The low-class rega pair costs 278 + 140 us on sliding turn_1 and 293 us for w13 on full turn_1. Two CTAs per SM gave about 10% from thread-level parallelism alone (V64 0.887x vs the occ1 diagnostic at 0.984x). V64 lost only under the 1.86x-skew guard (1.035) because its 64-route slab cut popular experts into 4-8 chunks. NMAX=128 restores master's chunking (one chunk per expert up to 128 routes, the same item decomposition as master), so the skew-guard penalty should go away, and with the same accumulation order bit-identical output is plausible. The prior field is accurate. The combo-b and two-ctas-per-sm results files exist, and DIAGNOSIS lists only the narrow-slab V64/V32 version as do-not-try. So this is a different mechanism, and it is DIAGNOSIS action 3 as written.

Weak points:
(1) Multicast does not save any shared memory. Each CTA still receives the full slab in its own shared memory, so the claim that multicast makes the smem fit (in DIAGNOSIS and implied by the change) is wrong. The fit comes from STG=2. I checked the arithmetic against combo-b's SmemLo: 1024 + 2 x 41984 + 18432 = 103424 B, and 2 x (103424 + STATIC_LO 12288 + 1024) = 233472 B. That is exactly the budget, with zero slack. Any extra static barrier or cluster scratch that mc2 adds beyond STATIC_LO breaks it, and w2 only fits through the two-pass epilogue.
(2) Register risk at NB=128, which I rank as the weakest link. Under setmaxnreg 104 and __launch_bounds__(384,2), the consumers of an m64 x N=128 swap-AB tile need 64 fp32 accumulator registers per thread, double V64's 32. That leaves about 40 for register-A fragments, FP4/E4M3 dequant temporaries and addressing. V64 never instantiated NB=128 at OCC 2. Popular experts under skew run exactly this NB=128 path, and that is the case this hypothesis is meant to win. A spill there would bring back the skew loss through a different cause.
(3) The two goals pull against each other. The thread-level gain came from two independent CTAs interleaving on one SM. mc2 pairs CTAs across a cluster, with a remote empty-barrier arrive and lockstep stage counters, so a stall in one CTA stalls its peer's producer. Also, at turn_1 most experts are in buckets 32/64, so the slab is only 4-8 KB per stage and multicast saves little ingest in the random case. The low class is described as partly latency-bound (V32 at 4.4 TB/s was no faster).
(4) Two stages per CTA, versus V64's 3, may expose latency in that latency-bound regime.

The expected directions are inconsistent: the hypothesis is built on the mc2 branch, so turn_4 and first_turn will move by mc2's own effect. They are flat only relative to the mc2 parent run, not relative to master or BEST. The target (418 -> ~370 us, about 5-7% of turn_1, about +2% headline) is plausible if the random-routing V64 win carries over. On its own it is near the 1.24% margin at best, as the hypothesis itself admits.

### Confounders on the current kernel
- experts-rega-mc2-slab-multicast-all-t (the depends_on parent, not landed): this run's headline and its turn_4 and first_turn rows carry mc2's effect, so read them against mc2's own check, not master or BEST. Only the low-class PATH CHOSEN line and turn_1 rows belong to this hypothesis.
- combo-b-normgather-two-cta-low-regate (BEST/master code): it supplies the rega_lo template (consume_item_lo, SmemLo, the 32/104 setmaxnreg split) and the worse-of-two-routings gate. mc2 deletes that module, so this change must restore the template from the combo-b patch exactly. If the restore drifts, the baseline candidate is no longer what V64 measured.
- align-two-phase-all-t-rega: it determines chunk and item order. NMAX=128 keeps master's chunking, so the interaction is neutral, but the claim of bit-identical output depends on the align's item order staying unchanged.
- combo-a-master-normgather-naive-stack (fused norm + gather): it writes the bf16 activation rows that the multicast TMA reads. That is the same layout for every candidate, so there is no confounding, but the gate must use master's real gather output, not a synthetic slab.

### Interference and A/B needs
No companion is warranted. The gate already runs the A/B inside this one run: master, mc2-low (1 CTA/SM, the parent's kernel) and this kernel are timed interleaved under cold L2 on both the random and the skew-guard routing, and every candidate's time is printed. A gain or a loss can therefore be attributed from the log without another okc check. The only thing that log cannot separate is whether a win comes from the full slab at 2 CTAs/SM or from multicast. That is cheap to answer in the same prepare: add rega_lo<G,128,2,2> without a cluster as a fourth diagnostic candidate. It is timed but never selectable, and it needs the same two-pass w2 epilogue. combo-b's template instantiates it directly, at the cost of one more nvcc instantiation, so a separate check is not worth spending. Under the new LESSONS rule, the 0.95x gate should also be confirmed by two interleaved rounds, both at or below 0.95 (that is, at least 1.05x, which exceeds the 1.03 floor), with the candidate's IQR at most 1.5x the best other candidate's. 'Keep combo B's protocol exactly' must not leave this kernel on a single round.

### Open questions
- With 2 CTAs/SM and (2,1,1) clusters, does the hardware place both CTAs of a cluster on the same SM? If it does, and the ingest cap is the per-SM port rather than total L2 bandwidth, multicast saves nothing. Print each CTA's %smid in a diagnostic launch.
- Does ptxas fit the NB=128 consume path for 2 CTAs/SM under setmaxnreg 104 with 0 spill? Print regs and spill per NB, and gate OFF if NB=128 spills.
- Does mc2 add static shared memory (cluster barriers, peer-arrive scratch) beyond combo-b's STATIC_LO of 12288? At 2 x (103424 + 12288 + 1024) = 233472 B there is no slack; static_assert the OCC-2 fit with mc2's real static size.
- If mc2 itself gates OFF in the low class, or fails M1, this hypothesis's comparison against mc2-low is moot. Should it still run against master alone, or wait?
- How much does the pairing slow things down? Print the time the producer waits on the remote empty barrier, so a loss can be pinned on pair lockstep rather than on 2 stages.

