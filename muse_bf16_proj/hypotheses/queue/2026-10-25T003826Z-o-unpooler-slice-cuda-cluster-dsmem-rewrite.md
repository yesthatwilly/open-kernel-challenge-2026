---
problem: muse_bf16_proj
slug: o-unpooler-slice-cuda-cluster-dsmem-rewrite
created: 2026-10-25T003826Z
base: master
area: o GEMM + unpooler GEMM (latency-bound small streams)
change: >
  Built on master (8b9ad43, blob a8bb42a); restack not landed and not needed. WHY: o (16.8 MB,
  floor 3.5 us) and the unpooler slice (8.4 MB, floor 1.75 us) are the shapes furthest from their
  floors in ratio (1.7x and 3.4x on rows). REWRITE_READ_2026-10-25 section 2 found why the gemv
  rewrite lost on exactly these: they are latency-bound (each CTA streams only 32-64 KB, a ring
  never fills) and the cost grows with m in the per-row partial store and the serialized global
  ticket-reduce tail. This stage removes both: every CTA issues ALL its weight loads at once (no
  ring), and split-K partials are reduced through thread-block-cluster distributed shared memory
  instead of a global workspace + ticket. STAGE 1: one CUDA kernel template compiled with
  torch.utils.cpp_extension.load_inline (reuse the gemv/lossless scaffold: compile-seconds print,
  per-route in-process self-check, print-and-fall-back to master; TORCH_CUDA_ARCH_LIST=9.0a around
  the build, -lineinfo; mma.sync m16n8k16 swap-AB with weight rows on M and tokens on N, NT 8 or
  16 template instances, cp.async.cg 16 B only, cluster launch via cudaLaunchKernelEx with cluster
  dim 8 along K, mapa + ld.shared::cluster, barrier.cluster; all baseline sm_90). Decompositions:
  o (n=8192, k=1024): 32 clusters x 8 CTAs = 256 CTAs (2 per SM), cluster owns 256 output rows,
  rank r owns K [128r, 128r+128): 256 x 128 x 2 B = 64 KB weight per CTA, issued in 2 commit
  groups. Unpooler slice (n=1024 of the full [8192, 4096] weight, k=4096): 16 clusters x 8 CTAs =
  128 CTAs, cluster owns 64 rows, rank r owns K [512r, 512r+512): 64 KB per CTA, 4 commit groups.
  Each CTA: 4 or 8 warps split its K range, reduce warp partials in smem, write its [rows, NT]
  fp32 partial to smem, cluster barrier (release/acquire), then rank r reduces its 1/8 of the
  cluster rows over ranks 0..7 in fixed order via DSMEM and stores bf16 out[m, n] (transposed
  store, tokens < m only); a second cluster barrier before exit. torch.empty output, no memset, no
  atomics, one node. ROUTING in kernel_fn: o at 1 <= m <= 16 and unpooler with cfg n == 1024 at 1
  <= m <= 8, when state cluster_ok[layer] (a prepare-time constant); everything else keeps master
  byte for byte. Tests through it: m=5 unpooler. SELF-CHECK in prepare per route at m in {1, 5, 8,
  16 (o only)}, twice: rel L2 <= 3e-3 vs fp64, one-hot probe rows exact within 1e-4, finite, run 2
  bit-identical; synchronize and print after each launch. DIAGNOSTICS: cudaFuncGetAttributes
  regs/spills/smem, cudaOccupancyMaxActiveClusters, warm-L2 and cold-L2 (flush > L2 first) event
  times vs master on the same synthetic x, and the m=1 vs m=8/16 kernel time (the gemv rewrite
  cost grew 1.6-1.7 us with m; this design should keep it flat). Stage 2 (later): extend to o m
  17..32 (NT 32) and qkv_gate m<=8 if M2 is met. Run as a check.
expected:
  - shapes: layer=o load=decode_1 m=1
    direction: faster
    why: "incumbent skinny kernel 4.1 us, row 5.9; floor 3.5. All 64 KB per CTA in flight, DSMEM reduce: M2 row <= 5.9, M3 row <= 5.0"
  - shapes: layer=o load=verify_1 m=8
    direction: faster
    why: incumbent 4.2 us, row 6.1; floor 3.5; M3 <= 5.0
  - shapes: layer=o load=decode_16 m=16
    direction: faster
    why: incumbent 4.4 us, row 6.2; floor 3.6; M3 <= 5.2
  - shapes: layer=unpooler load=decode_1 m=1
    direction: faster
    why: "incumbent split-K-8 ticket kernel 4.3 us, row 6.2; floor 1.7: M2 row <= 5.9, M3 row <= 4.0"
  - shapes: layer=unpooler load=verify_1 m=8
    direction: faster
    why: incumbent 4.4 us, row 6.0; floor 1.8; M3 <= 4.2 (no m-proportional tail)
  - shapes: layer=o load=decode_32 m=32
    direction: flat
    why: m > 16 keeps master swap-AB
  - shapes: layer=unpooler load=decode_16 m=16
    direction: flat
    why: full weight, master
  - shapes: layer=router *
    direction: flat
    why: untouched
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: o at m<=16 (_pooler_skinny_kernel, 4.1-4.4 us kernel, rows 5.9-6.3) and the unpooler slice at m<=8 (_bf16_splitk_kernel/ticket split-K 8, 4.3-4.4 us kernel, rows 5.9-6.1)
review: proceed
status: queued
review_at: 2026-10-25T004153Z
review_companions: null
---

# o-unpooler-slice-cuda-cluster-dsmem-rewrite

## Change

Built on master (8b9ad43, blob a8bb42a); restack not landed and not needed. WHY: o (16.8 MB, floor 3.5 us) and the unpooler slice (8.4 MB, floor 1.75 us) are the shapes furthest from their floors in ratio (1.7x and 3.4x on rows). REWRITE_READ_2026-10-25 section 2 found why the gemv rewrite lost on exactly these: they are latency-bound (each CTA streams only 32-64 KB, a ring never fills) and the cost grows with m in the per-row partial store and the serialized global ticket-reduce tail. This stage removes both: every CTA issues ALL its weight loads at once (no ring), and split-K partials are reduced through thread-block-cluster distributed shared memory instead of a global workspace + ticket. STAGE 1: one CUDA kernel template compiled with torch.utils.cpp_extension.load_inline (reuse the gemv/lossless scaffold: compile-seconds print, per-route in-process self-check, print-and-fall-back to master; TORCH_CUDA_ARCH_LIST=9.0a around the build, -lineinfo; mma.sync m16n8k16 swap-AB with weight rows on M and tokens on N, NT 8 or 16 template instances, cp.async.cg 16 B only, cluster launch via cudaLaunchKernelEx with cluster dim 8 along K, mapa + ld.shared::cluster, barrier.cluster; all baseline sm_90). Decompositions: o (n=8192, k=1024): 32 clusters x 8 CTAs = 256 CTAs (2 per SM), cluster owns 256 output rows, rank r owns K [128r, 128r+128): 256 x 128 x 2 B = 64 KB weight per CTA, issued in 2 commit groups. Unpooler slice (n=1024 of the full [8192, 4096] weight, k=4096): 16 clusters x 8 CTAs = 128 CTAs, cluster owns 64 rows, rank r owns K [512r, 512r+512): 64 KB per CTA, 4 commit groups. Each CTA: 4 or 8 warps split its K range, reduce warp partials in smem, write its [rows, NT] fp32 partial to smem, cluster barrier (release/acquire), then rank r reduces its 1/8 of the cluster rows over ranks 0..7 in fixed order via DSMEM and stores bf16 out[m, n] (transposed store, tokens < m only); a second cluster barrier before exit. torch.empty output, no memset, no atomics, one node. ROUTING in kernel_fn: o at 1 <= m <= 16 and unpooler with cfg n == 1024 at 1 <= m <= 8, when state cluster_ok[layer] (a prepare-time constant); everything else keeps master byte for byte. Tests through it: m=5 unpooler. SELF-CHECK in prepare per route at m in {1, 5, 8, 16 (o only)}, twice: rel L2 <= 3e-3 vs fp64, one-hot probe rows exact within 1e-4, finite, run 2 bit-identical; synchronize and print after each launch. DIAGNOSTICS: cudaFuncGetAttributes regs/spills/smem, cudaOccupancyMaxActiveClusters, warm-L2 and cold-L2 (flush > L2 first) event times vs master on the same synthetic x, and the m=1 vs m=8/16 kernel time (the gemv rewrite cost grew 1.6-1.7 us with m; this design should keep it flat). Stage 2 (later): extend to o m 17..32 (NT 32) and qkv_gate m<=8 if M2 is met. Run as a check.

## Expected

- `layer=o load=decode_1 m=1` faster: incumbent skinny kernel 4.1 us, row 5.9; floor 3.5. All 64 KB per CTA in flight, DSMEM reduce: M2 row <= 5.9, M3 row <= 5.0
- `layer=o load=verify_1 m=8` faster: incumbent 4.2 us, row 6.1; floor 3.5; M3 <= 5.0
- `layer=o load=decode_16 m=16` faster: incumbent 4.4 us, row 6.2; floor 3.6; M3 <= 5.2
- `layer=unpooler load=decode_1 m=1` faster: incumbent split-K-8 ticket kernel 4.3 us, row 6.2; floor 1.7: M2 row <= 5.9, M3 row <= 4.0
- `layer=unpooler load=verify_1 m=8` faster: incumbent 4.4 us, row 6.0; floor 1.8; M3 <= 4.2 (no m-proportional tail)
- `layer=o load=decode_32 m=32` flat: m > 16 keeps master swap-AB
- `layer=unpooler load=decode_16 m=16` flat: full weight, master
- `layer=router *` flat: untouched
- `layer=qkv_gate *` flat: untouched
- `layer=pooler *` flat: untouched

## Risk

Counter-evidence to weigh: bf16-small-m-cuda-gemv-rewrite lost on exactly these rows (o m=8 6.32 -> 9.52, slice m=8 6.13 -> 7.55); this stage differs in the two mechanisms that read named as the cause (no ring on 32-64 KB streams; no global partial store + ticket tail). Risks: a first-launch fault poisons the context (LESSONS 2026-10-25), so prints around each self-check launch and check mode only; cluster co-residency (print cudaOccupancyMaxActiveClusters; fall back if below the grid). Milestones: M1 9/9 tests with m=5 unpooler through the kernel; M2 every routed row at or below master; M3 o rows <= 5.0-5.2, slice rows <= 4.2. Headline at M3: ~1 us o + ~2 us slice on decode_1 and verify_1, ~1 us o on decode_16: about +4-6% on decode_1/verify_1, +1.5% decode_16.

## Milestones

Rewrite stage 1 of `o at m<=16 (_pooler_skinny_kernel, 4.1-4.4 us kernel, rows 5.9-6.3) and the unpooler slice at m<=8 (_bf16_splitk_kernel/ticket split-K 8, 4.3-4.4 us kernel, rows 5.9-6.1)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T004153Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound stage-1 rewrite aimed at the two shapes furthest from their floors; it fixes both causes REWRITE_READ section 2 found for the gemv loss (a ring that never fills; a global partial store plus a ticket tail that grows with m). Proceed as a check. Before building, correct the incumbent baselines (it quotes the restack's kernels, not master's), and add an smem swizzle and a cluster-residency gate.

### Justification
o (16.8 MB) and the unpooler slice (8.4 MB) are latency-bound 32-64 KB/CTA streams; issuing each CTA's whole slice at once puts the full weight in flight in one wave, and the DSMEM fixed-order reduce removes the gpu-scope ticket and fp32 global workspace (the m-proportional costs the read measured). Uses only baseline sm_90 (mma.sync, cp.async.cg, cluster launch, mapa, ld.shared::cluster, barrier.cluster); no wgmma. Weak points: (1) M3 needs the slice row at 4.0-4.2 us, but slice rows sit ~1.5 us above kernel time (master 4.5->5.94, restack 4.3->6.2), needing a ~2.5 us cold kernel for 8.4 MB plus cluster fixed cost; M2 credible, M3 on the slice a stretch, o <=5.0 plausible. (2) Baselines: master 8b9ad43 routes o m<=16 to _pooler_skinny_kernel and the slice to _bf16_splitk_kernel split-K 8; run 1602a4e947e64090 has o kernels 4.6/5.2/4.8 rows 6.32/6.32, slice kernels 4.5/4.6 rows 5.94/6.13. The quoted figures come from the restack run 60b20fd0bf924267. Judge M2 against master and also report vs the restack. (3) Bank conflicts: 256 B / 1 KB power-of-two row pitches make unswizzled ldmatrix ~8-way conflicted (~2 us); XOR-swizzle or +16 B pad the cp.async destination and ldmatrix addresses. Cluster residency: o needs 32 co-resident 8-CTA clusters at 2 CTAs/SM, an exact fit on 132 SMs; fall back to cluster dim 4 (64 x 4, K 256/rank) when cudaOccupancyMaxActiveClusters < 32, not straight to master; the slice (16 clusters at 1 CTA/SM) has the same risk. Smem ~80 KB, 2 CTAs/SM fits; alias the fp32 partial onto consumed weight buffer. Judge milestones on rows (warm-L2 caveat).

### Confounders on the current kernel
- bf16-shape-gated-triton-winners-combo (landed): supplies the replaced kernels; the true incumbent; compare to run 1602a4e947e64090.
- bf16-restack-m16-swapab-router-swapab-m32 (not landed, pending user): re-routes the same shapes; report deltas vs both.
- bf16-m32-swapab-wgmma-triton (landed): owns o at 17..32; rewrite stops at m=16, so o decode_32 flat is valid.
- router-hilo / sglang-gemm-dispatch-parity / router-hilo-at-m8: other layers; no interference.

### Interference and A/B needs
No companion. Judge on replaced kernels' time per shape vs master (1602a4e947e64090) and restack (60b20fd0bf924267). The queued router-cuda-cluster-dsmem-splitk-rewrite uses the same design: give each its own load_inline extension name and its own route-ok constant (LESSONS 2026-10-25).

### Open questions
- Fix the expected/replaces text to master's numbers before the build.
- cudaOccupancyMaxActiveClusters on the judge for cluster 8 at 2 CTAs/SM (o needs 32) and 1 CTA/SM (slice needs 16)? Fall back to cluster dim 4.
- Is the smem tile swizzled or padded for ldmatrix?
- Slice at 256 CTAs (32 KB, 2/SM) instead of 128 x 64 KB? Worth a prepare diagnostic.
- Use the prepare cold-L2 event time to decide whether slice M3 is reachable before stage 2.

