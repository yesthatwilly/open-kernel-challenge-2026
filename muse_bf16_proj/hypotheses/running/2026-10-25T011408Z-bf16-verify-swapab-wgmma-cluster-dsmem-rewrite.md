---
problem: muse_bf16_proj
slug: bf16-verify-swapab-wgmma-cluster-dsmem-rewrite
created: 2026-10-25T011408Z
base: master
area: bf16 GEMM dispatch (m=65..256, swap-AB wgmma + cluster DSMEM split-K)
change: >
  Built on master (blob a8bb42a). WHY: the m=128/256 rows are the last ones on cuBLAS and the
  furthest from their floors in absolute us (master profile 1602a4e947e64090): qkv_gate m=128
  16.81 row (nvjet splitK 13.0 + splitKreduce 3.0, floor 8.4), m=256 19.62 (nvjet 19.4 at 498
  TFLOP/s, floor 9.8); pooler m=128 25.55 (nvjet 25.7 at ~2.6 TB/s, floor 14.6), m=256 30.53 (29.5
  at 583 TFLOP/s, floor 17.4); unpooler m=128 19.79 (20.4, floor 14.6). The three failed attempts
  all put TOKENS on the MMA M side with 128x128 output tiles: REWRITE_READ section 1 found the
  Triton mainloop stuck at ~2.4 TB/s because every CTA re-reads its x K-slice from L2 (~2x the
  weight bytes), and the CUTLASS stream-K run (job f5b1361aaa0e48b5) lost on all m=256 rows (qkv
  28.5 us: 36 tiles spread over 132 units with deterministic serialized fixup). This rewrite
  changes the decomposition, not the load mechanism: SWAP-AB (weight rows on wgmma M, tokens on
  wgmma N, the orientation that won at m<=32), each weight byte read from DRAM exactly once, deep
  contiguous K per CTA, x TMA-MULTICAST across a 2-CTA cluster pair, and split-K reduced through
  distributed shared memory inside the cluster (no workspace, no memset node, no fixup
  serialization). STAGE 1: one CUDA kernel template in torch.utils.cpp_extension.load_inline
  (reuse the gemv/lossless/stream-K scaffold: compile-seconds print, per-route in-process
  self-check, print-and-fall-back to master). Per LESSONS 2026-10-25: set
  os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' around the build and restore it, pass
  -gencode=arch=compute_90a,code=sm_90a and -lineinfo, guard the wgmma/setmaxnreg code with #if
  defined(__CUDA_ARCH_FEAT_SM90_ALL) and put an #error in the #else for any __CUDA_ARCH__ pass so
  a stray sm_90 pass fails the BUILD (clean fallback) instead of trapping; print the arch flags.
  Kernel: CTA = 3 warpgroups (384 threads): WG0 producer (setmaxnreg.dec 40; one elected thread
  issues TMA cp.async.bulk.tensor 2D loads with mbarrier complete_tx), WG1-2 consumers
  (setmaxnreg.inc 232), each issuing wgmma.mma_async m64nTk16 bf16->fp32 with A = 64 weight rows
  from smem and B = the T-token x tile from smem (both K-major, 128B swizzle), so the CTA tile is
  128 weight rows x T tokens; T = 128 for 65 <= m <= 128 and T = 256 for 129 <= m <= 256 (two
  template instances; TMA zero-fills token rows >= m). BLOCK_K = 64; stage = 16 KB weight + T*128
  B of x (16 or 32 KB); 4 stages (<= 192 KB). Grid = (n/128 weight tiles) x S K-splits, launched
  with cudaLaunchKernelEx and a cluster of (2 along weight tiles, S along K): pooler n=4096 -> 32
  tiles, S=4, K-slice 2048, 128 CTAs, cluster 2x4=8; unpooler (m<=128 only) n=8192 -> 64 tiles,
  S=2, K-slice 2048, 128 CTAs, cluster 2x2; qkv_gate n=2304 -> 18 tiles, S=7, K-slices of
  19/19/18x5 k-blocks, 126 CTAs, cluster 2x7=14 (set
  cudaFuncAttributeNonPortableClusterSizeAllowed=1; if cudaOccupancyMaxActiveClusters returns 0,
  rebuild the launch with cluster 1x7, no multicast, and print it). The x tile for a K-block is
  fetched once per cluster pair with .multicast::cluster (ctaMask of the two CTAs sharing a
  K-split), halving x L2->SM traffic; weight tiles are not multicast (each is unique). Epilogue:
  after the mainloop, each consumer writes its fp32 accumulator (128 x T x 4 B = 64/128 KB) into
  its CTA's smem (reusing the stage ring), barrier.cluster.arrive/wait, then cluster rank r along
  K sums rows [r*128/S, (r+1)*128/S) of the tile from all S peers through mapa +
  ld.shared::cluster in fixed order 0..S-1, converts to bf16 and stores out[token, n] (out is [m,
  n] row-major, so the transposed tile is written with tokens as rows; coalesce along n), then a
  final cluster barrier before exit. Route in kernel_fn, just before master's cuBLAS fallback:
  layer in {qkv_gate, pooler} and 65 <= cfg m <= 256, or layer == unpooler and 65 <= cfg m <= 128;
  cfg n = the full width; x contiguous bf16 and x.data_ptr() % 16 == 0; state ok flag set. o, the
  router, m <= 32, unpooler at m >= 129 and the m=512/1000 tests keep master. TMA descriptors for
  the weights are built once in prepare; the x descriptor is built per call on the host from the
  pointer and m (shape-only, graph-safe; pass it as a __grid_constant__ CUtensorMap kernel
  parameter). Prepare self-check per routed layer at m in {65, 128, 130, 200, 256} (unpooler: 65,
  128): rel L2 <= 3e-3 vs x.double() @ W.double().T, a one-hot power-of-two probe row within 1e-4,
  finite, and run 2 bit-identical to run 1; sync and print after EACH launch so a fault is
  attributed; on failure print '[swapab-cl] <layer> DISABLED: <reason>'. Diagnostics in prepare,
  print only: regs/spills via cudaFuncGetAttributes, the cluster shape actually launched, and a
  cold-L2 cudaEvent time per layer at m=128/256 (flush a 256 MB buffer before each event pair)
  next to master's cuBLAS on the same x. Run as an okc CHECK. MILESTONES (kernel us from
  top_kernels, rows second): M1 builds, all self-checks pass, PATH CHOSEN swapab-cl on all routed
  layers; M2 at or below the incumbent: qkv m=128 <= 16.0 (both cuBLAS nodes), qkv m=256 <= 19.4,
  pooler m=128 <= 25.7, pooler m=256 <= 29.5, unpooler m=128 <= 20.4; M3 qkv m=256 <= 14, pooler
  m=256 <= 24, pooler m=128 <= 19, qkv m=128 <= 11. Stage 2 levers: cluster orientation and S per
  layer, T=192 at 129..192, o at m=256, persistent tile order.
expected:
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: one node replaces nvjet splitK 13.0 + splitKreduce 3.0 us (row 16.81); 37.7 MB read once, floor 8.4; M3 <= 11
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: nvjet 19.4 us at 498 TFLOP/s on 36-72 tiles; 126 CTAs with deep K at ~700 TFLOP/s ~14 us; floor 9.8
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "memory-bound at m=128: nvjet 25.7 at ~2.6 TB/s; weight once + multicast x at ~3.5 TB/s ~19-20 us; floor 14.6"
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: nvjet 29.5 at 583 TFLOP/s (64 tiles); 128 CTAs at ~700 TFLOP/s ~24 us; floor 17.4
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: nvjet 20.4 us vs 14.6 floor; 128 CTAs, 512 KB of weight each
  - shapes: layer=unpooler load=verify_32 m=256
    direction: flat
    why: m >= 129 unpooler keeps nvjet (~700 TFLOP/s already)
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
  - shapes: layer=* load=decode_*
    direction: flat
    why: m <= 32 untouched
  - shapes: layer=* load=verify_1 m=8
    direction: flat
    why: m <= 32 untouched
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: cuBLAS nvjet at m=128/256 for qkv_gate (13.0+3.0 splitKreduce / 19.4 us), pooler (25.7 / 29.5 us) and unpooler m=128 (20.4 us)
review: proceed
status: running
review_at: 2026-10-25T012031Z
review_companions: null
hold_reason: null
stage: claimed
claimed_at: 2026-10-25T042030Z
base_sha: 2de39fe31bc772463640a6e1897c6c89e11c919a
base_blob: 970b415b746fa03cb92e419d270be92588273178
branch: hyp/muse_bf16_proj/bf16-verify-swapab-wgmma-cluster-dsmem-rewrite
worktree: .worktrees/muse_bf16_proj/bf16-verify-swapab-wgmma-cluster-dsmem-rewrite
mode: null
restack_of: null
stale_base: null
---

# bf16-verify-swapab-wgmma-cluster-dsmem-rewrite

## Change

Built on master (blob a8bb42a). WHY: the m=128/256 rows are the last ones on cuBLAS and the furthest from their floors in absolute us (master profile 1602a4e947e64090): qkv_gate m=128 16.81 row (nvjet splitK 13.0 + splitKreduce 3.0, floor 8.4), m=256 19.62 (nvjet 19.4 at 498 TFLOP/s, floor 9.8); pooler m=128 25.55 (nvjet 25.7 at ~2.6 TB/s, floor 14.6), m=256 30.53 (29.5 at 583 TFLOP/s, floor 17.4); unpooler m=128 19.79 (20.4, floor 14.6). The three failed attempts all put TOKENS on the MMA M side with 128x128 output tiles: REWRITE_READ section 1 found the Triton mainloop stuck at ~2.4 TB/s because every CTA re-reads its x K-slice from L2 (~2x the weight bytes), and the CUTLASS stream-K run (job f5b1361aaa0e48b5) lost on all m=256 rows (qkv 28.5 us: 36 tiles spread over 132 units with deterministic serialized fixup). This rewrite changes the decomposition, not the load mechanism: SWAP-AB (weight rows on wgmma M, tokens on wgmma N, the orientation that won at m<=32), each weight byte read from DRAM exactly once, deep contiguous K per CTA, x TMA-MULTICAST across a 2-CTA cluster pair, and split-K reduced through distributed shared memory inside the cluster (no workspace, no memset node, no fixup serialization). STAGE 1: one CUDA kernel template in torch.utils.cpp_extension.load_inline (reuse the gemv/lossless/stream-K scaffold: compile-seconds print, per-route in-process self-check, print-and-fall-back to master). Per LESSONS 2026-10-25: set os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' around the build and restore it, pass -gencode=arch=compute_90a,code=sm_90a and -lineinfo, guard the wgmma/setmaxnreg code with #if defined(__CUDA_ARCH_FEAT_SM90_ALL) and put an #error in the #else for any __CUDA_ARCH__ pass so a stray sm_90 pass fails the BUILD (clean fallback) instead of trapping; print the arch flags. Kernel: CTA = 3 warpgroups (384 threads): WG0 producer (setmaxnreg.dec 40; one elected thread issues TMA cp.async.bulk.tensor 2D loads with mbarrier complete_tx), WG1-2 consumers (setmaxnreg.inc 232), each issuing wgmma.mma_async m64nTk16 bf16->fp32 with A = 64 weight rows from smem and B = the T-token x tile from smem (both K-major, 128B swizzle), so the CTA tile is 128 weight rows x T tokens; T = 128 for 65 <= m <= 128 and T = 256 for 129 <= m <= 256 (two template instances; TMA zero-fills token rows >= m). BLOCK_K = 64; stage = 16 KB weight + T*128 B of x (16 or 32 KB); 4 stages (<= 192 KB). Grid = (n/128 weight tiles) x S K-splits, launched with cudaLaunchKernelEx and a cluster of (2 along weight tiles, S along K): pooler n=4096 -> 32 tiles, S=4, K-slice 2048, 128 CTAs, cluster 2x4=8; unpooler (m<=128 only) n=8192 -> 64 tiles, S=2, K-slice 2048, 128 CTAs, cluster 2x2; qkv_gate n=2304 -> 18 tiles, S=7, K-slices of 19/19/18x5 k-blocks, 126 CTAs, cluster 2x7=14 (set cudaFuncAttributeNonPortableClusterSizeAllowed=1; if cudaOccupancyMaxActiveClusters returns 0, rebuild the launch with cluster 1x7, no multicast, and print it). The x tile for a K-block is fetched once per cluster pair with .multicast::cluster (ctaMask of the two CTAs sharing a K-split), halving x L2->SM traffic; weight tiles are not multicast (each is unique). Epilogue: after the mainloop, each consumer writes its fp32 accumulator (128 x T x 4 B = 64/128 KB) into its CTA's smem (reusing the stage ring), barrier.cluster.arrive/wait, then cluster rank r along K sums rows [r*128/S, (r+1)*128/S) of the tile from all S peers through mapa + ld.shared::cluster in fixed order 0..S-1, converts to bf16 and stores out[token, n] (out is [m, n] row-major, so the transposed tile is written with tokens as rows; coalesce along n), then a final cluster barrier before exit. Route in kernel_fn, just before master's cuBLAS fallback: layer in {qkv_gate, pooler} and 65 <= cfg m <= 256, or layer == unpooler and 65 <= cfg m <= 128; cfg n = the full width; x contiguous bf16 and x.data_ptr() % 16 == 0; state ok flag set. o, the router, m <= 32, unpooler at m >= 129 and the m=512/1000 tests keep master. TMA descriptors for the weights are built once in prepare; the x descriptor is built per call on the host from the pointer and m (shape-only, graph-safe; pass it as a __grid_constant__ CUtensorMap kernel parameter). Prepare self-check per routed layer at m in {65, 128, 130, 200, 256} (unpooler: 65, 128): rel L2 <= 3e-3 vs x.double() @ W.double().T, a one-hot power-of-two probe row within 1e-4, finite, and run 2 bit-identical to run 1; sync and print after EACH launch so a fault is attributed; on failure print '[swapab-cl] <layer> DISABLED: <reason>'. Diagnostics in prepare, print only: regs/spills via cudaFuncGetAttributes, the cluster shape actually launched, and a cold-L2 cudaEvent time per layer at m=128/256 (flush a 256 MB buffer before each event pair) next to master's cuBLAS on the same x. Run as an okc CHECK. MILESTONES (kernel us from top_kernels, rows second): M1 builds, all self-checks pass, PATH CHOSEN swapab-cl on all routed layers; M2 at or below the incumbent: qkv m=128 <= 16.0 (both cuBLAS nodes), qkv m=256 <= 19.4, pooler m=128 <= 25.7, pooler m=256 <= 29.5, unpooler m=128 <= 20.4; M3 qkv m=256 <= 14, pooler m=256 <= 24, pooler m=128 <= 19, qkv m=128 <= 11. Stage 2 levers: cluster orientation and S per layer, T=192 at 129..192, o at m=256, persistent tile order.

## Expected

- `layer=qkv_gate load=verify_16 m=128` faster: one node replaces nvjet splitK 13.0 + splitKreduce 3.0 us (row 16.81); 37.7 MB read once, floor 8.4; M3 <= 11
- `layer=qkv_gate load=verify_32 m=256` faster: nvjet 19.4 us at 498 TFLOP/s on 36-72 tiles; 126 CTAs with deep K at ~700 TFLOP/s ~14 us; floor 9.8
- `layer=pooler load=verify_16 m=128` faster: memory-bound at m=128: nvjet 25.7 at ~2.6 TB/s; weight once + multicast x at ~3.5 TB/s ~19-20 us; floor 14.6
- `layer=pooler load=verify_32 m=256` faster: nvjet 29.5 at 583 TFLOP/s (64 tiles); 128 CTAs at ~700 TFLOP/s ~24 us; floor 17.4
- `layer=unpooler load=verify_16 m=128` faster: nvjet 20.4 us vs 14.6 floor; 128 CTAs, 512 KB of weight each
- `layer=unpooler load=verify_32 m=256` flat: m >= 129 unpooler keeps nvjet (~700 TFLOP/s already)
- `layer=o *` flat: untouched
- `layer=router *` flat: untouched
- `layer=* load=decode_*` flat: m <= 32 untouched
- `layer=* load=verify_1 m=8` flat: m <= 32 untouched

## Risk

High. (1) Compute ceiling: 128 CTAs on 132 SMs at m=256 is ~97% fill; a consumer pair of m64n256 wgmma per k16 is the densest Hopper issue pattern, but if it lands near the stream-K CUTLASS rate (343-608 TFLOP/s) it loses; the prepare cold-L2 diag shows that before the rows do. (2) The DSMEM epilogue moves 128 x T x 4 B per CTA across the cluster (128 KB at T=256): budget ~1-1.5 us; if it dominates, stage 2 reduces in bf16x2 pairs or shrinks S. (3) Non-portable cluster 2x7 for qkv may not be schedulable; the 1x7 fallback is built in. (4) A sticky fault in the self-check fails the whole run (LESSONS): build guards above, -lineinfo, sync+print per launch, plain TMA forms the stream-K build already ran on the judge. (5) Correctness is fp32 accumulation over K with a fixed-order sum, same precision class as cuBLAS. The running router/o-unpooler DSMEM rewrites test the reduce pattern on small streams; this does not depend on them, but their harvest notes on mapa/barrier.cluster apply.

## Milestones

Rewrite stage 1 of `cuBLAS nvjet at m=128/256 for qkv_gate (13.0+3.0 splitKreduce / 19.4 us), pooler (25.7 / 29.5 us) and unpooler m=128 (20.4 us)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T012031Z by an adversarial subagent (model opus).

**Verdict: proceed.** The design can physically reach M2, most plausibly on qkv m=128, which replaces two cuBLAS nodes (16.0 us). It has two concrete defects to fix before building. First, the qkv 2x7 (and 1x7 fallback) cluster cannot be co-resident in one wave on 132 SMs at 1 CTA/SM, and the brief's 'max active clusters == 0' test does not catch that. Second, the DSMEM epilogue moves 6-14 MB per launch, several times the 1-1.5 us budget. Dispatch as an okc check once the two in-flight cluster/DSMEM checks have at least reported build and self-check status.

### Justification
Incumbents (master 1602a4e947e64090, untouched by the restack at m>=128): qkv m=128 row 16.81 (nvjet splitK 13.0 + splitKreduce 3.0), qkv m=256 19.62/19.4, pooler m=128 25.55/25.7, pooler m=256 30.53/29.5, unpooler m=128 19.79/20.4. The hypothesis quotes all of these correctly, its expected directions match the TASK floors, and its untouched shapes are correctly marked flat. Mechanism: swap-AB with T=256 gives a 128x256 CTA tile. That is 1.33x the arithmetic intensity of the 128x128 tiles that failed. Multicasting x across the pair halves the x L2->SM re-read that REWRITE_READ section 1 named as the ceiling. Real WS plus setmaxnreg is what Triton 3.8 could not deliver. So the change is more than a re-skin. Two caveats. (i) At m=128 the T=128 tile is the same 128x128 product tile as the failed attempts, only transposed, so the m=128 gain must come from multicast plus the single-node reduce, not from the tile. (ii) The best prior for a hand-written WS wgmma kernel is CUTLASS's own (job f5b1361aaa0e48b5: 128x128x64, cluster 2x1 multicast, cooperative WS). With NO split-K (unpooler m=256, Heuristic/DataParallel) it ran 26.8 us at 608 TFLOP/s, against nvjet's 24.6. Pooler m=256 reached 495 TFLOP/s. The '~700 TFLOP/s' behind the qkv m=256 <= 14 and pooler m=256 <= 24 targets is therefore above anything measured here. Per-shape view: qkv m=128 needs only ~500 TFLOP/s plus a <=2 us epilogue to beat 16.0, so it is the likeliest M2 win. Pooler m=128 is mixed-bound (needs ~3.5 TB/s and ~450 TFLOP/s together), which is plausible. Both m=256 rows are long shots in stage 1. WEAKEST LINK, in two parts. (a) Cluster residency. A 2x7=14-CTA cluster at 1 CTA/SM (192 KB smem) fits at most once per GPC. H100/H200 GPCs hold roughly 14-18 SMs, so at most ~8 clusters (112 CTAs) are co-resident, against the 9 clusters (126 CTAs) the grid needs. The 1x7 fallback fits 2 per GPC, so at most ~16 clusters against 18 needed. Either way a second wave roughly doubles qkv time. cudaOccupancyMaxActiveClusters will return a positive number, so the stated fallback test (== 0) never fires. Pooler's 2x4=8 cluster needs 16 co-resident clusters, which fails if any GPC has fewer than 16 usable SMs. (b) Epilogue bytes. Total DSMEM traffic is about CTAs x tile_fp32 x (S-1)/S: qkv m=256 ~126 x 128 KB x 6/7 ~ 14 MB, pooler m=256 ~12.6 MB, pooler m=128 ~6.3 MB. Published Hopper microbenchmarks put aggregate SM-to-SM throughput around ~3.3 TB/s for the whole GPU, lower for larger clusters. If that holds, the epilogue is 2-4+ us, not 1-1.5 us, and it alone erases the m=256 margin. Fixes to put in the implement brief: (1) in prepare, compute needed_clusters = grid_CTAs / cluster_size, print it next to cudaOccupancyMaxActiveClusters, and treat max_active < needed as 'does not fit in one wave'. For qkv, choose among single-wave candidates (e.g. 36 token-half x weight tiles at T=128 with S=3 and cluster 2x3; or 18 tiles with S=4 and cluster 2x4, at the cost of fill) and print the one chosen. (2) Build a print-only no-reduce variant (mainloop + local bf16 store of the partial) and print full vs no_reduce cold-L2 times per layer. The m128-wspec read split mainloop from tail the same way, and that split is what makes stage 2 decidable. (3) S=7 does not divide 128 rows: specify an integer row partition, e.g. ceil-split 19/19/18/18/18/18/18. (4) The multicast pair needs the remote empty-barrier protocol: each consumer must arrive on BOTH CTAs' empty mbarrier (mapa + mbarrier.arrive.shared::cluster) before the issuing producer reuses that slot. Missing it is a silent, intermittent race that a 2-run bit-identity check may not catch.

### Confounders on the current kernel
- sglang-gemm-dispatch-parity: defines the cuBLAS fallback that is the incumbent on every routed shape. The new route sits in front of it, so there is no interaction on kernel time. cuBLAS m>=128 rows swing up to ~7% between runs (pooler m=128 nvjet 23.9 vs 25.7), so judge against the in-prepare side-by-side cold-L2 cuBLAS time and the 1602 profile, not against one row.
- router-triton-hilo-fused-control-splitk-all-m: owns the router at m=128/256 in the same loads, a separate graph per shape. No shared kernel or memory; it only moves the load-level headline.
- bf16-m32-swapab-wgmma-triton / bf16-shape-gated-triton-winners-combo: m<=32 only; disjoint shapes. They matter only if the routing gate is written as an elif chain that changes their m bounds, so keep separate bound constants (LESSONS 2026-10-25 stack lesson).

### Interference and A/B needs
No companion. This is stage 1 of a rewrite, judged on the replaced kernels' own time in top_kernels plus the prepare diagnostics, and no landed hypothesis shares its shapes or code path. What decides stage 2 is internal: mainloop vs epilogue (the no-reduce print) and wave count (cluster residency print). Both belong in this run as print-only diagnostics, not as extra checks. On the in-flight checks: router-cuda-cluster-dsmem-splitk-rewrite (dc5d03c4d9604b0e) and o-unpooler-slice-cuda-cluster-dsmem-rewrite (3f3330f70b6e4cdc) are the first judge runs of the shared machinery: load_inline + cudaLaunchKernelEx cluster attribute, mapa/ld.shared::cluster, and barrier.cluster arrive.release/wait.acquire. A fault there poisons the context (LESSONS), and the same bug would kill this run. So WAIT for their logs (build OK, cluster launch accepted, max-active-clusters print, self-check pass or fault) before dispatching, and fold any fix into this kernel. Do NOT wait for their performance verdicts. They reduce tiny [16,NT] partials over portable clusters of 8 with mma.sync and no TMA, so they say nothing about the epilogue bandwidth (64-128 KB per CTA), non-portable cluster sizes, TMA multicast, wgmma or setmaxnreg, which are this run's real risks.

### Open questions
- What does cudaOccupancyMaxActiveClusters return on the judge for 14, 8, 7, 6 and 4 CTA clusters at ~192 KB smem? This decides the qkv and pooler decompositions. Print all of them in prepare regardless of which is launched.
- Is the DSMEM epilogue bandwidth-bound? Compare full vs no_reduce per layer, and if it exceeds ~1.5 us, consider in stage 2 a smaller S, or 64-row weight tiles with T=256 and no split.
- Is the T=256 consumer (m64n256 per warpgroup, 128 fp32 accumulator regs/thread at setmaxnreg 232) spill-free? Print the spill line and do not trust a run that spills (LESSONS 2026-10-25: 'a wgmma build that runs is not a fast one').


## Implementation notes

Edited only `muse_bf16_proj/submission.py` in the worktree (base = master 2de39fe, the landed restack 1.2215; master's paths untouched). New section (8) before `prepare`, +792 lines, plus `import os, time`, a prepare call `_swcl_prepare(state)` after `_swapab16_prepare`, and a route in `kernel_fn` just before master's cuBLAS fallback.

- Kernel (load_inline at import, `TORCH_CUDA_ARCH_LIST=9.0a` set and restored, `-gencode=arch=compute_90a,code=sm_90a -O3 -lineinfo -std=c++20`, arch flags printed). Hand-written PTX, no cute/cuda:: headers. CTA = 384 threads: WG0 producer (setmaxnreg.dec 40, one thread issues TMA with mbarrier complete_tx), WG1-2 consumers (setmaxnreg.inc 232) issuing wgmma m64nTk16 SS (A = 64 weight rows, B = T x tile, 128B swizzle, descriptor form copied from the sm_90a moe build that ran), T = 128 (65..128, 6 stages) or 256 (129..256, 4 stages), BLOCK_K 64. All wgmma/setmaxnreg bodies are under `__CUDA_ARCH_FEAT_SM90_ALL`; the `#else` sets device error word 2 which the self-check reads (the brief's form; the spec's `#error` was not used).
- Cluster (CX, S): grid (n/128, S). CX = 2 multicasts the x tile (each CTA loads half the rows, `.multicast::cluster`, mask 3 << 2*cy); CX = 1 is the same kernel without multicast. Epilogue: fp32 accumulator to the CTA's smem (aliasing the ring, token stride 132 floats, conflict-free), cluster barrier, rank cy sums its share of 4-row groups over the S peers in fixed order via mapa + `ld.shared::cluster.v4`, bf16 8 B stores, final cluster barrier. Bounded mbarrier waits (clock64, error word 1). Cluster-rank check (error word 3). TMA descriptors via the driver entry point (as the moe build); weight maps built once in prepare, x map per call from pointer and m (host only, graph-safe).
- Review fixes applied (the review wins over the spec): (1) every candidate prints max-active vs needed clusters (tiles/CX) and is dropped if it does not fit one wave; qkv candidates (2,7),(1,7),(2,6),(1,6),(2,4),(1,4), pooler (2,4),(1,4),(2,2), unpooler (2,2),(1,2); survivors are timed cold-L2 and the fastest kept (so the spec's single 2x7 qkv launch and its "== 0" fallback test are replaced). (2) print-only no-reduce variant (mode 0: own-tile store, no peer reads; the two cluster barriers stay) timed beside the full kernel. (3) integer partitions: K-blocks floor(r*nkb/S), row groups of 4 (32 groups, floor(r*32/S)) instead of a 19/19/18 row split, so the v4 loads stay 16 B aligned. (4) remote empty-barrier protocol: each consumer warp arrives on the EMPTY barrier of both CTAs of its pair (8 warps x CX arrivals), FULL expects 16 KB + T*128 B.
- Task requirements: gate in prepare per (layer, T class) on cold-L2 CUDA-graph time (L2 flushed before every replay) at m = 128 and m = 256 vs master's cuBLAS on the same x, routed only if <= 0.98 x master, else master for that class; PATH CHOSEN printed per range and per representative m. Self-check m in {65,128} (T=128) and {130,200,256} (T=256): rel L2 <= 3e-3 vs fp64, one-hot probe row <= 1e-4, finite, run 2 bit-identical, flushed print before and sync + print after each launch, error word read. Route: qkv_gate/pooler 65..256, unpooler 65..128, cfg n/k = full widths, x contiguous (stride k), 16 B aligned; everything else (o, router, m < 65, m > 256, unpooler m >= 129, tests m=130 unpooler/m=512/1000) stays on master. Own constants (SWCL_*), no shared bound with other paths.
- No process/subprocess code anywhere. Not compiled or run here (no GPU, nvcc or torch on this node).

Biggest risks: (1) a blind first compile of ~400 lines of PTX-level CUDA (wgmma operand lists, TMA multicast, mapa forms): a build error just falls back to master and prints the nvcc tail; (2) a first-launch fault in the self-check poisons the context (every launch is bracketed by prints); (3) `regs * 384 >= 64512` must hold or setmaxnreg.inc would block, it is gated on cudaFuncGetAttributes.

## Rules check

- [x] CUDA-graph capturable: kernel_fn only reads shapes/cfg, builds the x TMA map on the host from the pointer and m, one cudaLaunchKernelEx on the current stream, torch.empty output; no sync, no value-dependent control (diff: `_swcl_route`, `_swcl_forward`). Syncs/.item() exist only in prepare-time self-check/timing.
- [x] Nothing derived from `data` is cached across calls: state holds only weight TMA maps, the error word and the per-class choice from synthetic data.
- [x] Weights: bf16 weights untouched (only a TMA descriptor over the originals); no repack, no precision change.
- [x] Activations BF16: x bf16 in, bf16 out, fp32 accumulation (cuBLAS class); FP32 router untouched.
- [x] n/a: attention rule (this is a GEMM, every output computed over all of K).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (grep for subprocess/multiprocessing/fork/spawn clean; lint clean).
- [x] All work inside the timed call; no output caching or replay; the prepare timing/self-check is diagnostic and selector only.

okc_lint output:
```
warn  host-sync  torch.cuda.synchronize() / err.item()  (outside kernel_fn's call graph; fine in prepare())  x5
warn  dtype      flush = torch.empty(SWCL_FLUSH_BYTES, dtype=torch.uint8, ...)  (L2 flush buffer in prepare, not an activation)
OK with warnings
```
