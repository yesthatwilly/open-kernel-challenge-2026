---
problem: muse_prefill_layer
slug: experts-rega-mc4-cluster-high-ft
created: 2026-10-25T211857Z
base: master
area: "w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft): activation slab shared across a 4-CTA cluster by TMA multicast"
change: >
  Rewrite stage 2 of the slab-multicast line, built on experts-rega-mc2-slab-multicast-all-t's
  branch whatever its score. Template the cluster size: rega_mc_kernel<G, BIG, CL> with CL in {2,
  4}; CL=2 stays stage 1's code path. At CL=4 the four CTAs of a (4,1,1) cluster take weight tiles
  4Q..4Q+3 of the same chunk (n_tiles 64 / 32 are multiples of 4). Each producer issues a QUARTER
  of the slab by TOKEN rows, not by k: rank r issues both 64-column k-halves for token rows [32r,
  32r+32) of the bucket (box rows 32, same SWIZZLE_128B box width, smem destination offset r*32
  rows*128 B inside each half; the 128B swizzle repeats every 8 rows so the landed layout is
  byte-identical to master's), multicast mask 0b1111; for buckets 16 and 32 (fewer rows than 4 x
  32) only ranks whose row range is non-empty issue, and expect_tx stays the bucket's full slab
  bytes + own weight. Empty barrier init count = 8 consumer warps x 4 CTAs; each consumer warp
  arrives on all four CTAs' empty barriers (mapa per rank). Cluster barrier at start (after
  fence.mbarrier_init) and before exit. Grid = 4 x cudaOccupancyMaxActiveClusters for (4,1,1);
  print it: on 132 SMs a 4-cluster may leave SMs idle (a GPC with an SM count not divisible by 4),
  which costs up to ~3% and is the main reason this can lose. Low class stays at stage 1's choice.
  GATE per class (high T=3012, ft T=16384): exact-compare, every bucket forced, two-state replay,
  cold-L2 interleaved graph timing of master vs CL=2 vs CL=4; CL=4 ON only at <= 0.97x the better
  of the other two. Print per candidate: grid, w13/w2 us, issued L2 bytes per stage and TB/s.
  MILESTONES: M1 CL=4 correct with grid >= 128 CTAs. M2 ft w13+w2 <= 0.95x stage 1's CL=2 time. M3
  ft w13+w2 <= 3900 us (vs master 5504 us, floor ~3100 us). FALLBACK: the CL=2 path (or master)
  per class.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "Per stage per CTA, issued L2 reads go 41 KB (master) -> 25.6 KB (CL=2) -> 17.2 KB (CL=4: 9.2 KB weight + 8 KB slab) for 1024 clk of MMA. If CL=2 lands near the ingest/MMA balance point (~25 KB at ~6 TB/s is ~1030 clk), CL=4 gives ingest headroom so dequant + MMA set the pace: another ~3-6% of ft rega (5504 -> ~4000 us at CL=2, -150-250 us more), ~1-2% of the first_turn load. Alone that is under the 1.24% headline margin: it must stack with stage 1 or land in the same submit."
  - shapes: "*load=turn_4*"
    direction: faster
    why: high class rega 1098 us (master); ~-3-5% beyond CL=2.
  - shapes: "*load=turn_1*"
    direction: flat
    why: low class untouched
risk: high
depends_on: experts-rega-mc2-slab-multicast-all-t
priority: 5
kind: rewrite
controls_for: null
prior: "experts-rega-mc2-slab-multicast-all-t: stage 1 shares the slab across 2 CTAs by k-halves; this shares it across 4 by 32-token row quarters, so issued slab reads per CTA halve again, at the cost of possible idle SMs from cluster-4 occupancy"
rewrite_stage: 2
replaces: rega_kernel / rega_mc_kernel (w13, w2) in the rega high (T=3012) and ft (T=16384) classes
review: hold
status: hold
review_at: 2026-10-26T215518Z
review_companions: null
hold_reason: "adversarial review: Hold: this hypothesis's own build condition has failed. Stage 1's 2-CTA slab multicast cut issued L2 bytes by 24-39% yet rega ran 1.31-1.38x slower at high/ft a"
---

# experts-rega-mc4-cluster-high-ft

## Change

Rewrite stage 2 of the slab-multicast line, built on experts-rega-mc2-slab-multicast-all-t's branch whatever its score. Template the cluster size: rega_mc_kernel<G, BIG, CL> with CL in {2, 4}; CL=2 stays stage 1's code path. At CL=4 the four CTAs of a (4,1,1) cluster take weight tiles 4Q..4Q+3 of the same chunk (n_tiles 64 / 32 are multiples of 4). Each producer issues a QUARTER of the slab by TOKEN rows, not by k: rank r issues both 64-column k-halves for token rows [32r, 32r+32) of the bucket (box rows 32, same SWIZZLE_128B box width, smem destination offset r*32 rows*128 B inside each half; the 128B swizzle repeats every 8 rows so the landed layout is byte-identical to master's), multicast mask 0b1111; for buckets 16 and 32 (fewer rows than 4 x 32) only ranks whose row range is non-empty issue, and expect_tx stays the bucket's full slab bytes + own weight. Empty barrier init count = 8 consumer warps x 4 CTAs; each consumer warp arrives on all four CTAs' empty barriers (mapa per rank). Cluster barrier at start (after fence.mbarrier_init) and before exit. Grid = 4 x cudaOccupancyMaxActiveClusters for (4,1,1); print it: on 132 SMs a 4-cluster may leave SMs idle (a GPC with an SM count not divisible by 4), which costs up to ~3% and is the main reason this can lose. Low class stays at stage 1's choice. GATE per class (high T=3012, ft T=16384): exact-compare, every bucket forced, two-state replay, cold-L2 interleaved graph timing of master vs CL=2 vs CL=4; CL=4 ON only at <= 0.97x the better of the other two. Print per candidate: grid, w13/w2 us, issued L2 bytes per stage and TB/s. MILESTONES: M1 CL=4 correct with grid >= 128 CTAs. M2 ft w13+w2 <= 0.95x stage 1's CL=2 time. M3 ft w13+w2 <= 3900 us (vs master 5504 us, floor ~3100 us). FALLBACK: the CL=2 path (or master) per class.

## Expected

- `*load=first_turn*` faster: Per stage per CTA, issued L2 reads go 41 KB (master) -> 25.6 KB (CL=2) -> 17.2 KB (CL=4: 9.2 KB weight + 8 KB slab) for 1024 clk of MMA. If CL=2 lands near the ingest/MMA balance point (~25 KB at ~6 TB/s is ~1030 clk), CL=4 gives ingest headroom so dequant + MMA set the pace: another ~3-6% of ft rega (5504 -> ~4000 us at CL=2, -150-250 us more), ~1-2% of the first_turn load. Alone that is under the 1.24% headline margin: it must stack with stage 1 or land in the same submit.
- `*load=turn_4*` faster: high class rega 1098 us (master); ~-3-5% beyond CL=2.
- `*load=turn_1*` flat: low class untouched

## Risk

4-CTA cluster, quarter-box TMA multicast and 32 remote arrivals per stage: hang/fault risk; first measurement is an okc check. Only worth building if stage 1 printed an ingest drop (issued TB/s) that moved time; if stage 1 was flat because the cap is per-SM receive, retire this.

## Milestones

Rewrite stage 2 of `rega_kernel / rega_mc_kernel (w13, w2) in the rega high (T=3012) and ft (T=16384) classes`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-26T215518Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold: this hypothesis's own build condition has failed. Stage 1's 2-CTA slab multicast cut issued L2 bytes by 24-39% yet rega ran 1.31-1.38x slower at high/ft and 1.80x at low (OFF in every class, job 0ca60bb781db48e4), and the in-CTA bm256 byte cut also lost 1.16-1.21x (job a6f6063051f14750), so slab ingest is not the limiter and CL=4 widens exactly the cluster handshake that cost CL=2 ~0.3-0.5 us per stage. Spend the check on diagnosing that cost (or master's real limiter), not on CL=4.

### Justification
Its build condition (an issued-TB/s drop that moved time) failed in stage 1 (job 0ca60bb781db48e4): issued bytes fell 24-39% while rega_mc_kernel got 1.32-1.90x slower; bm256 (job a6f6063051f14750) cut the same ~38% bytes and also lost 1.16-1.20x. The ingest-cap premise behind DIAGNOSIS actions 1-2 is falsified; CL=4 widens the per-stage cluster handshake (4-way lockstep, 32-arrival empty barriers, 3 remote arrives per consumer warp per stage). Spec defects noted: tail-bucket boxes vs expect_tx, rank skipping at bucket 64, quad pairing in the decode.

### Confounders on the current kernel
- experts-turn-rega-s4-port-t4096 / experts-turn-rega-s4-first-turn-n128: the incumbent's serial consumer, not ingest, sets the pace
- align-two-phase-all-t-rega: fixes chunking k and the tail-bucket mix
- combo-a-master-normgather-naive-stack: the gate must time the norm-gather-on routed block
- combo-b-normgather-two-cta-low-regate: prepare-only difference

### Interference and A/B needs
No companion warranted; the prepare gate already A/Bs master vs CL=2 vs CL=4. The open question is upstream: why CL=2 added ~0.3-0.5 us per stage.

### Open questions
- Re-scope: print-only variants (a) master rega as (2,1,1) clusters, (b) plus 16-arrival cross-CTA empty barrier, (c) stage-1 mc2, (d) mc2 with remote arrives spread across lanes at .release.cta
- Re-diagnose master's limiter (clock64 stamps of the consumer's waits / dq_stage / wgmma_wait) before minting new rega stages
- Siblings experts-rega-mc2-bm256-stack-high-ft and experts-rega-low-2cta-fullslab-mc2 share the falsified premise
- The -cf confirm (13cfaf1e25c341af) re-measures master: its parent's gate kept master in high and ft

