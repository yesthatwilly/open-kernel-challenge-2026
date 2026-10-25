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
review: required
status: queued
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
