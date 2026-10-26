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
review: proceed
status: running
review_at: 2026-10-25T212524Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T212550Z
base_sha: 98cd3251d87b4318340450cec6f52e37f574895c
base_blob: f4751fc054335a94d1b401af272e98a2d4760363
branch: hyp/muse_prefill_layer/experts-rega-mc2-slab-multicast-all-t
worktree: .worktrees/muse_prefill_layer/experts-rega-mc2-slab-multicast-all-t
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T004959Z
impl_commit: c6815034ebe12c1593e9e10d06b41ab1d245902e
submission_sha256: 6acb348c520b8f6e4d9575c4406c961607a57df50925e8143d0868ac857dff6b
norm_sha256: 40c1ebf8b9037d1428d7e00f4ef6ff74e0d891c00df03c9c1d5f7775c570ad59
submission_blob: 0689e0198fcf8c7a711b4f69ce2c1972c9970cb9
lint_warnings: 78
attempt_file: 2026-10-26T012540Z-check-experts-rega-mc2-slab-multicast-all-t.md
prepared_at: 2026-10-26T012540Z
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

## Adversarial review

Reviewed 2026-10-25T212524Z by an adversarial subagent (model opus).

**Verdict: proceed.** The multicast-slab rewrite targets the limiter that the logs actually name: rega is L2->SM ingest-capped at 5.8-6.4 TB/s, and most of that ingest is the bf16 slab. Its producer and barrier design matches the standard CUTLASS multicast pattern. There is one blocking spec defect to fix before building: master's decode_item is chunk-minor, so 'it = 2p + cta_rank with decode_item unchanged' pairs DIFFERENT chunks of the SAME weight tile whenever an expert has more than one chunk. That is k=4 at first_turn, so the exact-compare would fail and the multicast kernel would be gated OFF exactly where most of the gain is.

### Justification
A. The mechanism is plausible. At bucket 128 a stage is 128x128x128 bf16 MMA, about 1024 clk at ~4096 FLOP/clk/SM. It moves 41 KB/stage/CTA (9216 B weight + 2x16 KB slab), and 132 SMs at ~27 B/clk is about the 6 TB/s measured, which is near H100's aggregate L2 read ceiling. So cutting issued L2 reads (slab 32 KB -> 16 KB per CTA, 25.6 KB total per stage) can plausibly move the kernel off that cap. FINDINGS supports the target. rega w13+w2 is 3652+1851 us (sliding ft, weight 138), 741+357 us (sliding turn_4) and 278+140 us (sliding turn_1), 32-36% + 16-18% of those shapes, and DIAGNOSIS puts rega at 43-49% of each load's weighted time and about 1.75x its floor. The expected directions (all faster, ft largest, turn_1 smallest) agree with TASK's equal-weight geomean over loads.

The prior is real and different. experts-rega-pingpong-consumers-all-t changed the consumer schedule and cut ingest (5.80 -> 4.76 TB/s). This change leaves consume_item byte-identical and changes only the producer, the empty-barrier count and the launch. DIAGNOSIS's Do-not-try list bans consumer-side overlap, not this, and DIAGNOSIS explicitly names multicast as the missing fundamental (action 1). No earlier prefill attempt used multicast or clusters on rega, so this is not a duplicate. experts-rega-low-class-two-ctas-per-sm and combo-b changed TLP, not slab ingest.

BLOCKING SPEC DEFECT (must fix in the build). decode_item maps local = it - f*T to chunk c = f + local % k and tile j = local / k, so consecutive items walk CHUNKS of one tile first. With N_EXPERTS=256, 32 local and top-8, first_turn has ~512 local routes per expert, i.e. k=4 chunks of 128. Turn_4 is mostly k=1 but k=2 for popular experts, and turn_1 under skew (max 228 per expert) can also reach k=2. With 'decode_item unchanged', pair (2p, 2p+1) is (chunk f, j) and (chunk f+1, j): same weight, different slab rows. The multicast slab would then be wrong for one CTA of every such pair, and the bucket and tx can differ too (the last chunk is shorter). The exact-compare gate would turn ft OFF (and high/low on the skewed and one-expert cases), so the run measures master. The fix: add an mc-specific decode. Pair index pl within the expert group (k*T/2 pairs, and T is even, so groups never straddle pairs) maps to c = f + pl % k, J = pl / k, j = 2J + cta_rank. This keeps master's chunk-minor order, and with it the cross-cluster L2 reuse of weight tiles. Both CTAs then share c, so bucket, tx and KS match and the stage counters stay in lockstep.

WEAKEST LINK: the size of the gain, not its sign. Inside each consumer warpgroup, dequant and wgmma are still serial (dq_stage, then MMAs, then wgmma_wait<0>), and there is no ping-pong. Once ingest drops to ~940 clk/stage, the cap becomes ~1024 clk MMA plus the exposed dequant, plausibly 1250-1350 clk, against 1530 today. That is ~12-18% on rega, not the ~25-30% behind the 4000-4400 us ft bound. I expect ft w13+w2 at ~4600-4900 us: M2 (<=4800) is reachable, M3 (<=4400) probably needs a later stage (bm256 or retried overlap). The secondary risk is the hypothesis's own caveat: if the per-SM receive path rather than aggregate L2 output is the cap, multicast gives ~0. The printed TB/s settles that. A tertiary cost: max active 2-CTA clusters may be fewer than 66 if GPCs have odd usable SM counts, which loses a few percent of SMs. The grid print shows it.

### Confounders on the current kernel
- experts-turn-rega-s4-port-t4096 / experts-turn-rega-s4-first-turn-n128: these are the incumbent kernel itself. rega_mc_kernel reuses consume_item, the bank and the tensor maps, so any change in those (e.g. a concurrent bm256 landing first) changes the baseline the 0.97x gate compares against. Build and gate against master's exact rega_kernel only.
- align-two-phase-all-t-rega: it sets the chunk structure (8 blocks of 16 per chunk, the chunk count per expert k, the bucket of the tail chunk), which decides how many items pair cleanly and the bucket mix. That is the k>1 issue above. Under a different align chunking (e.g. 64-route chunks) the mc pairing and its benefit would differ.
- combo-a-master-normgather-naive-stack (norm fused into the gather): it produces the xs slab that is now multicast, and the norm_gather_ok flags pick the rega entry. Output equality holds whether prenorm is on or off, but the gate must time rega_gemm_mc inside the same prenorm path master runs per class, or the ratio mixes two front ends.
- combo-b-normgather-two-cta-low-regate: its low-class module is gated OFF on master (variant 0), so deleting _rega_lo_phase has no timed-path effect. It only frees prepare time, but it also removes the code DIAGNOSIS action 3 (full slab, 2 CTAs/SM, plus multicast) would build on.
- experts-turn4-ssdq-s1b-dataptr-symbol-fix: ssdq is never ON on a scored shape (rega owns 200 < T <= 16384), so deleting its build and canary is timing-neutral. Confirm in the log that the ssdq PATH CHOSEN lines are gone and that no fallback path referenced ssdq_ok.

### Interference and A/B needs
No companion is warranted. This is kind: rewrite, judged on the replaced kernels' own us per class (profile top_kernels) and on the gate's printed rega vs rega_mc medians and TB/s. The per-class prepare gate (exact-compare, then replay check, then cold-L2 interleaved timing, ON at <= 0.97x) already acts as the in-run A/B: if PATH CHOSEN says OFF, the headline is master re-measured (the row noise is +-3-4% per LESSONS); if ON, the kernel row is read directly. The landed pieces that touch this area are the kernel's own substrate, not a removable stack, so an ab-revert would measure nothing useful. The two ablations one might want, weight-only vs slab multicast and cluster launch without multicast, would only matter if the result were flat. The printed achieved L2->SM TB/s already separates 'ingest cap was per-SM receive' (TB/s unchanged, time unchanged) from 'cap moved to MMA+dequant' (issued TB/s down, time down). The in-batch hedge experts-rega-s4-bm256-wide-tile-high-ft is the natural comparison. Follow the new LESSONS rule: the gate should use two interleaved rounds and require <= 0.97x in both (the brief's 0.97 already exceeds the 1.03 margin), with an IQR bound printed.

### Open questions
- Fix the pairing: will the builder add an mc decode (c = f + pl % k, j = 2*(pl / k) + rank) instead of 'decode_item unchanged'? As written, ft (k~4) fails the exact-compare and gates OFF.
- Does the exact-compare force k>=2 experts in the high and low classes (skewed and one-expert routings) so that the pairing across multi-chunk experts is tested, not only buckets? LESSONS 161 covers buckets; this is a chunk-count edge.
- What does cudaOccupancyMaxActiveClusters return for (2,1,1) at Smem<G,BIG>::BYTES (~170 KB) on the judge's H100? If it is below 66 clusters, how much does losing those SMs eat into the gain at turn_1, where items per CTA are only ~16-19?
- After the change, the dequant inside each consumer warpgroup is exposed serially before its MMAs. If the printed issued TB/s drops but the time drops less than ~15%, retry DIAGNOSIS's banned consumer-overlap items (pingpong, scale-expand) on top. DIAGNOSIS says they are dead only until the ingest cap moves.
- Is the deleted _rega_lo_phase code needed by DIAGNOSIS action 3 (full slab, 2 CTAs/SM, plus multicast)? If so, keep it in git history or the archive, or move the deletion to a separate cleanup.


## Implementation notes

Edited only `<worktree>/muse_prefill_layer/submission.py` (cannot run: no GPU or nvcc on this node; CUDA reviewed by eye only).

What changed:
- New CUDA block `_REGA_MC_BODY` (+ `_REGA_CPP_MC`, `_rega_mc_src`) appended by `_rega_src(mc=True)`: `rega_mc_kernel<G,BIG>`, host `rega_mc_info` (regs/spill, max active 2-CTA clusters via cudaOccupancyMaxActiveClusters at Smem<G>::BYTES) and `rega_gemm_mc` (rega_gemm's exact signature; cudaLaunchKernelEx with cudaLaunchAttributeClusterDimension (2,1,1); grid must be even, n_tiles even). Master's `_REGA_HEAD` / `_REGA_BODY` strings are byte-untouched (v1 fallback hash intact); `consume_item_mc` is master's consume_item text copied at assembly time with three asserted edits (name, `peer` parameter, stage release = own empty barrier + peer's via mapa + mbarrier.arrive.release.cluster).
- Producer: own 9216 B weight record as master, plus ONE slab half by `cp.async.bulk.tensor.2d...multicast::cluster` (rank 0 k 0-63 at st+WST, rank 1 k 64-127 at st+WST+SLAB_HALF, ctaMask 0b11); expect_tx unchanged (WST + 2*(16<<bucket)*128); empty barrier init count 16; fence.mbarrier_init + cluster barrier before any TMA; cluster barrier at the end. All waits remain master's bounded `mbar_wait`.
- Python: `_rega_mc_phase` (after norm-gather's setup, so it gates the routed block with its fused-gather flags): occupancy/regs print, exact compare, two-state replay check, per-kernel us + achieved L2->SM TB/s (issued weight + slab bytes), two interleaved rounds of the whole routed `_experts(pre_norm=True)` flag-off vs flag-on. `state['rega_mc_ok'][cls]` (plain bool) and `state['rega_mc_grid']` (plain ints) are in REGA_STATE_KEYS; `_experts_rega` reads them by shape (class from T, capacity class from the block-table length).
- Build safety: the mc module is built under its own name `muse_prefill_rega_mc_v3`; if it fails, master's exact v2 module is built (so a bug in the new kernel never costs the ft class), then master's v1 fallback.
- Prepare budget: deleted `_ssdq_setup`, `_ssdq_build`, `_ssdq_gate`, `_ssdq_replay_check`, `_ssdq_reference_y`, `_ssdq_source_audit` and the ssdq call, and the whole combo-b low-class module (`_rega_lo_phase` and everything it used: muse_prefill_rega_lo source/build/gate, pop-count helpers; `_rega_pop_case` kept, `_rega_case` uses it). Dead ssdq CUDA source strings and `_experts_ssdq` were left (never built; `state.get('ssdq_ok')` is simply absent).

Review fixes applied (review wins over the spec):
1. BLOCKING pairing defect: `decode_item_mc(pi, T, rank)`: group `cg = 2pi/T`, `pl = pi - f*T/2`, `c = f + pl % k`, `j = 2*(pl/k) + rank`, so both CTAs of a cluster share chunk, bucket, tx and KS and take adjacent tiles; the loop walks pair index `pi = cluster; pi < total/2; pi += clusters`.
2. Exact-compare forces k >= 2 experts in EVERY class (cases: low 753 skewed/one/local, 1024 local; high 3012 skewed/one, 4096 local; ft 16384 skewed/one/local, 4500 local), prints max k per case and turns the class OFF unless >= 3 cases contain a k >= 2 expert.
3. Two interleaved rounds, ON only if BOTH <= 0.97 and candidate IQR <= 1.5x master's; master timed as its actually routed path (flag off, same norm-gather flags); both rounds' medians/IQRs, kernel names and a compact PATH CHOSEN summary are the last prepare() lines.
4. Open questions: grid is the occupancy query's cluster count x 2, printed per instantiation; `_rega_lo_phase` deleted per the spec (it stays in git history at master's blob f4751fc0, DIAGNOSIS action 3 can restore it); post-change TB/s/time are printed so the consumer-overlap retry decision can be read from the log.

Deviations: (a) separate module name for the mc build instead of the same module, for fallback safety; (b) the gate times the whole routed block (not only the rega pair) plus a per-kernel print; (c) compile-time risks I could not check: ptxas register fit of the producer under setmaxnreg 40, cudaLaunchKernelEx template argument coercion of the `__grid_constant__ Maps` parameter.

## Rules check

- [x] CUDA-graph capturable: kernel_fn path adds only a per-state bool/int lookup and the cudaLaunchKernelEx host launch (capturable); every sync/.cpu() is in prepare() only (lint: host-sync warnings all "outside kernel_fn's call graph").
- [x] Nothing derived from `data` is cached across calls: flags and grids are prepare-time plain data of the state; nothing new at module/file scope (the C++ audit finds no file-scope variable).
- [x] Weights: unchanged retiled bank, NVFP4 stays NVFP4; mc kernel reads the same bank.
- [x] Activations BF16, KV FP8, queries unchanged: the slab is the same bf16 xs, consumers are master's code.
- [x] Attention untouched: n/a, no attention code changed.
- [x] No timing patches, harness access, module edits, subprocesses or network (grep for subprocess/Popen/os.system/fork: none).
- [x] All work inside the timed call; no output caching or replay: the mc kernels run inside _experts_rega each call.
- [x] Prepare-time selector, flags and fallback: per-class `rega_mc_ok`, a miss/exception/deadline lands on master's rega_kernel and never disables rega_ok or other flags; 'skipped: deadline' printed.

okc_lint: `OK with warnings` (only host-sync warnings in prepare() and pre-existing dtype warnings; zero errors). `python3 -I -m py_compile` passed; `__pycache__` removed.

## Pre-flight audit

Verdict: dispatch-with-fixes (audited by reading; no GPU/nvcc here).
- ssdq / rega_lo deletion: an AST undefined-name scan finds no dangling names; no code reads rega_lo_ext / rega_low_variant / rega_low_grid; ssdq_ok is only read through state.get (absent = off); unreachable `_experts_ssdq` and dead ssdq source strings remain, harmless.
- Pairing: total = nc * n_tiles is always even (n_tiles 64 / 32), so there is never an odd last pair; both CTAs run pi = cid; pi < npairs; pi += ncl, share chunk / bucket / KS, so the 16-arrival empty barrier, the peer arrives and the stage counters stay in lockstep. (c, j) is a bijection per expert group. Final cluster_sync_all is reached by every thread of both CTAs.
- Fix 1: `_experts_rega` fell to a KeyError on `state["rega_mc_grid"][...]` if a capacity-class instantiation had failed its prepare check but the class was ON; it now falls back to master's rega_gemm.
- Fix 2: the gate's IQR veto (candidate IQR <= 1.5x master's over 11 samples) could reject a real win on a near-zero master IQR; added 1% of master's time as slack. The ratio <= 0.97 in both rounds is unchanged and times master's routed `_experts(pre_norm=True)` with the flag off.
Residual risk: nvcc/ptxas compile of the cluster/multicast PTX and the producer register fit at setmaxnreg 40 are unverified; any build failure lands on master's v2 module.
