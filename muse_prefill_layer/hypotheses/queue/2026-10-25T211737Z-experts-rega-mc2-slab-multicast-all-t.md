---
problem: muse_prefill_layer
slug: experts-rega-mc2-slab-multicast-all-t
created: 2026-10-25T211737Z
base: master
area: "w13 + w2 expert GEMMs, every rega class: activation slab loaded once per 2-CTA cluster and shared by TMA multicast"
change: >
  DIAGNOSIS action 1. Built on master (combo-b-normgather-two-cta-low-regate code, blob f4751fc0).
  New kernel rega_mc_kernel<G, BIG> appended to the rega module source (_REGA_BODY; master's
  rega_kernel strings untouched), launched with cluster dims (2,1,1) via cudaLaunchKernelEx
  (cudaLaunchAttributeClusterDimension) from a new host entry rega_gemm_mc with rega_gemm's exact
  signature. PAIRING: the two CTAs of cluster c take adjacent weight tiles j=2J and 2J+1 of the
  same chunk; the persistent loop walks pair index p = clusterId; p < total/2; p += numClusters,
  item it = 2p + cta_rank (n_tiles is even: 64 for w13, 32 for w2; decode_item unchanged).
  PRODUCER (tid 0 of warp 0 per CTA, still setmaxnreg.dec 40): per k-stage it bulk-copies its OWN
  tile's 9216 B weight record exactly as master, and issues ONE slab half (rank 0: k 0-63 into
  st+WST, rank 1: k 64-127 into st+WST+SLAB_HALF) with
  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster,
  ctaMask 0b11, so the half lands at the same smem offset in both CTAs and completes tx on both
  CTAs' full barrier at that offset. full barrier expect_tx = WST + 2 * (16<<bucket) * 128
  (unchanged total: own weight + both halves). Before reusing a stage the producer must know BOTH
  CTAs' consumers released it: the empty barrier init count becomes 2x master's (16 arrivals: 8
  consumer warps x 2 CTAs), and each consumer warp's lane 0 arrives on its own CTA's empty barrier
  and on the peer's via mapa.shared::cluster + mbarrier.arrive.release.cluster.shared::cluster.
  Both CTAs of a pair always have the same bucket and KS (same chunk), so stage counters stay in
  lockstep. Barrier init is followed by fence.mbarrier_init.release.cluster and
  barrier.cluster.arrive/wait before any TMA; the kernel ends with a cluster barrier so no CTA
  exits while its peer can still multicast into it. CONSUMERS: consume_item, the dequant, the
  wgmma sequence, the bank, align, gather and epilogue are master's, so outputs are bit-identical.
  GRID: numClusters from cudaOccupancyMaxActiveClusters for (2,1,1) at Smem<G>::BYTES, grid = 2 *
  numClusters (do not assume 132); print it. GATE in _rega_setup per class (low T=753, high
  T=3012, ft T=16384): exact-compare outputs of rega_gemm_mc vs rega_gemm on random, skewed and
  one-expert routings (every bucket forced, LESSONS 156), the two-state replay check (LESSONS
  162), then master's cold-L2 interleaved graph timing; ON only at <= 0.97x master's rega pair;
  state['rega_mc_ok'][cls] is a plain bool per state; _experts_rega calls rega_gemm_mc when it is
  set. Print per class: build regs/spill, grid, w13 and w2 us for both kernels, and achieved
  L2->SM TB/s computed as (issued weight + issued slab bytes)/time for both, plus 'PATH CHOSEN'.
  PREPARE BUDGET: in the same change delete the dead ssdq build/canary (_ssdq_setup and its phase
  calls; ssdq_ok is never ON on a scored shape) and the OFF two-CTA low gate (_rega_lo_phase,
  muse_prefill_rega_lo build): ~33 s freed. MILESTONES: M1 builds sm_90a, 0 spill, exact-compare
  passes in all classes (correctness only). M2 sliding first_turn w13+w2 5504 -> <= 4800 us,
  turn_4 1098 -> <= 980 us. M3 ft <= 4400 us, turn_4 <= 880 us, turn_1 418 -> <= 390 us. FALLBACK:
  any gate miss leaves master's rega_kernel for that class; the module still builds master's
  kernel.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: rega w13 3652-3848 us + w2 1851-1891 us at T=16384 run at ~5.8-6.4 TB/s L2->SM with 41 KB/stage/CTA (9.2 KB weight + 32 KB slab) every ~1530 clk vs 1024 clk of MMA; multicast halves issued slab reads to 25.6 KB/stage. Upper bound w13+w2 5504 -> 4000-4400 us, ~-10% of the first_turn load (sliding weight 138).
  - shapes: "*load=turn_4*"
    direction: faster
    why: rega w13 741-763 us + w2 357-371 us at T=3012 (high class); 1098 -> ~800-880 us, ~-8-11% of the turn_4 load.
  - shapes: "*load=turn_1*"
    direction: faster
    why: rega 278+140 us sliding, 293 us w13 full at T=753 (low class) is partly latency bound (V32 at 4.4 TB/s was no faster than V64), so ~418 -> 350-390 us, ~-3-5%. Headline upper bound ~x1.08-1.10; even a third of that clears the 1.24% margin.
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
prior: "experts-rega-pingpong-consumers-all-t: changed only the consumer schedule and LOST ingest (5.80 -> 4.76 TB/s, 1.18-1.22x slower); this leaves the consumer byte-identical and changes only the producer's slab TMA (multicast) and the empty-barrier count, cutting issued L2 slab reads in half"
rewrite_stage: 1
replaces: "rega_kernel<true,BIG> + rega_kernel<false,BIG> (w13, w2) in all three rega classes: low T<=1024, high 1024<T<=4096, ft 4096<T<=16384"
review: required
status: queued
---

# experts-rega-mc2-slab-multicast-all-t

## Change

DIAGNOSIS action 1. Built on master (combo-b-normgather-two-cta-low-regate code, blob f4751fc0). New kernel rega_mc_kernel<G, BIG> appended to the rega module source (_REGA_BODY; master's rega_kernel strings untouched), launched with cluster dims (2,1,1) via cudaLaunchKernelEx (cudaLaunchAttributeClusterDimension) from a new host entry rega_gemm_mc with rega_gemm's exact signature. PAIRING: the two CTAs of cluster c take adjacent weight tiles j=2J and 2J+1 of the same chunk; the persistent loop walks pair index p = clusterId; p < total/2; p += numClusters, item it = 2p + cta_rank (n_tiles is even: 64 for w13, 32 for w2; decode_item unchanged). PRODUCER (tid 0 of warp 0 per CTA, still setmaxnreg.dec 40): per k-stage it bulk-copies its OWN tile's 9216 B weight record exactly as master, and issues ONE slab half (rank 0: k 0-63 into st+WST, rank 1: k 64-127 into st+WST+SLAB_HALF) with cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster, ctaMask 0b11, so the half lands at the same smem offset in both CTAs and completes tx on both CTAs' full barrier at that offset. full barrier expect_tx = WST + 2 * (16<<bucket) * 128 (unchanged total: own weight + both halves). Before reusing a stage the producer must know BOTH CTAs' consumers released it: the empty barrier init count becomes 2x master's (16 arrivals: 8 consumer warps x 2 CTAs), and each consumer warp's lane 0 arrives on its own CTA's empty barrier and on the peer's via mapa.shared::cluster + mbarrier.arrive.release.cluster.shared::cluster. Both CTAs of a pair always have the same bucket and KS (same chunk), so stage counters stay in lockstep. Barrier init is followed by fence.mbarrier_init.release.cluster and barrier.cluster.arrive/wait before any TMA; the kernel ends with a cluster barrier so no CTA exits while its peer can still multicast into it. CONSUMERS: consume_item, the dequant, the wgmma sequence, the bank, align, gather and epilogue are master's, so outputs are bit-identical. GRID: numClusters from cudaOccupancyMaxActiveClusters for (2,1,1) at Smem<G>::BYTES, grid = 2 * numClusters (do not assume 132); print it. GATE in _rega_setup per class (low T=753, high T=3012, ft T=16384): exact-compare outputs of rega_gemm_mc vs rega_gemm on random, skewed and one-expert routings (every bucket forced, LESSONS 156), the two-state replay check (LESSONS 162), then master's cold-L2 interleaved graph timing; ON only at <= 0.97x master's rega pair; state['rega_mc_ok'][cls] is a plain bool per state; _experts_rega calls rega_gemm_mc when it is set. Print per class: build regs/spill, grid, w13 and w2 us for both kernels, and achieved L2->SM TB/s computed as (issued weight + issued slab bytes)/time for both, plus 'PATH CHOSEN'. PREPARE BUDGET: in the same change delete the dead ssdq build/canary (_ssdq_setup and its phase calls; ssdq_ok is never ON on a scored shape) and the OFF two-CTA low gate (_rega_lo_phase, muse_prefill_rega_lo build): ~33 s freed. MILESTONES: M1 builds sm_90a, 0 spill, exact-compare passes in all classes (correctness only). M2 sliding first_turn w13+w2 5504 -> <= 4800 us, turn_4 1098 -> <= 980 us. M3 ft <= 4400 us, turn_4 <= 880 us, turn_1 418 -> <= 390 us. FALLBACK: any gate miss leaves master's rega_kernel for that class; the module still builds master's kernel.

## Expected

- `*load=first_turn*` faster: rega w13 3652-3848 us + w2 1851-1891 us at T=16384 run at ~5.8-6.4 TB/s L2->SM with 41 KB/stage/CTA (9.2 KB weight + 32 KB slab) every ~1530 clk vs 1024 clk of MMA; multicast halves issued slab reads to 25.6 KB/stage. Upper bound w13+w2 5504 -> 4000-4400 us, ~-10% of the first_turn load (sliding weight 138).
- `*load=turn_4*` faster: rega w13 741-763 us + w2 357-371 us at T=3012 (high class); 1098 -> ~800-880 us, ~-8-11% of the turn_4 load.
- `*load=turn_1*` faster: rega 278+140 us sliding, 293 us w13 full at T=753 (low class) is partly latency bound (V32 at 4.4 TB/s was no faster than V64), so ~418 -> 350-390 us, ~-3-5%. Headline upper bound ~x1.08-1.10; even a third of that clears the 1.24% margin.

## Risk

Cluster launch, cross-CTA mbarrier arrives and multicast TMA can hang or fault (a missing peer arrive deadlocks the ring; a CTA exiting early while the peer multicasts into it corrupts smem): first measurement is an okc check. Keep master's mbar_wait timeout -> error word so a hang becomes a gate OFF, not a 15-min timeout. Caveat for the reviewer: multicast cuts L2 READS, not bytes landed in each SM's smem (still 41 KB/stage). If the cap is per-SM receive rather than L2 output, mc gives ~0 and the printed TB/s will show it; experts-rega-s4-bm256-wide-tile-high-ft (halves slab bytes per FLOP inside the CTA) is the hedge and runs in the same batch. Precedent: muse_bf16_proj bf16-verify-swapab-wgmma-cluster-dsmem-rewrite ran multicast::cluster on the judge.

## Milestones

Rewrite stage 1 of `rega_kernel<true,BIG> + rega_kernel<false,BIG> (w13, w2) in all three rega classes: low T<=1024, high 1024<T<=4096, ft 4096<T<=16384`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
