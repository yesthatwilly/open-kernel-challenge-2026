---
problem: muse_bf16_proj
slug: router-cluster-dsmem-s2-flat-in-m-occupancy
created: 2026-10-25T042040Z
base: master
area: router GEMM (m<=32, cluster DSMEM kernel)
change: >
  Build on branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (stage 1, dc5d03c4).
  Stage 1 won rows at m=1/8 only by deleting the memset node: its kernel was SLOWER than master's
  (3.6/4.3/4.6 us at m=1/8/16 vs hilo 3.2/3.2/3.5), i.e. 8.4 MB at ~2.3 TB/s with 4 warps and 1
  CTA/SM, and it grew with m. Stage 2 changes only the kernel's decomposition, not the routing:
  (1) cluster dim 16 along K (set cudaFuncAttributeNonPortableClusterSizeAllowed=1; fall back to 8
  if cudaOccupancyMaxActiveClusters returns 0 for 16, and print which): 16 clusters x 16 CTAs =
  256 CTAs (2 per SM), each CTA owning K [512r, 512r+512): 32 KB of hi+lo weight; (2) 8 warps per
  CTA each taking a 64-K slice, so 16 warps/SM are in flight instead of 4; (3) load x for the
  CTA's K slice once into smem with tokens padded to NT and reuse the A (hi/lo) fragments across
  all NT/8 B fragments (decode-once, as stage 1 of the lossless line learned); (4) the DSMEM
  reduce stores each rank's slice of columns once over 16 ranks in fixed order. Keep NT 8/16/32,
  the same self-check bounds and run-twice bit-identical check, torch.empty output, one node.
  Routing stays as stage 1 (cluster at 1 <= m <= 32, master above). Print per-m kernel time (cold
  L2) and regs/spills/smem.
expected:
  - shapes: layer=router load=decode_1 m=1
    direction: faster
    why: stage-1 row 5.24 (kernel 3.6); floor 1.8; target kernel <= 2.8, row <= 4.6
  - shapes: layer=router load=verify_1 m=8
    direction: faster
    why: stage-1 row 5.08 (kernel 4.3); target flat-in-m kernel <= 2.9
  - shapes: layer=router load=decode_16 m=16
    direction: faster
    why: stage-1 row 5.87 (4.6) lost to restack swapab 5.47; target <= 5.0
  - shapes: layer=router load=decode_32 m=32
    direction: faster
    why: stage-1 fell back to hilo 6.85; now routed, target <= 5.5 (restack swapab 6.2)
  - shapes: layer=router load=verify_* m=12*
    direction: flat
    why: m=128 keeps master
  - shapes: layer=router load=verify_32 m=256
    direction: flat
    why: keeps master
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
    direction: flat
    why: untouched
risk: medium
depends_on: router-cuda-cluster-dsmem-splitk-rewrite
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _router_swapab_kernel + memset at router m<=32 (restack) / _router_hilo_kernel + memset (master)
review: required
status: queued
---

# router-cluster-dsmem-s2-flat-in-m-occupancy

## Change

Build on branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (stage 1, dc5d03c4). Stage 1 won rows at m=1/8 only by deleting the memset node: its kernel was SLOWER than master's (3.6/4.3/4.6 us at m=1/8/16 vs hilo 3.2/3.2/3.5), i.e. 8.4 MB at ~2.3 TB/s with 4 warps and 1 CTA/SM, and it grew with m. Stage 2 changes only the kernel's decomposition, not the routing: (1) cluster dim 16 along K (set cudaFuncAttributeNonPortableClusterSizeAllowed=1; fall back to 8 if cudaOccupancyMaxActiveClusters returns 0 for 16, and print which): 16 clusters x 16 CTAs = 256 CTAs (2 per SM), each CTA owning K [512r, 512r+512): 32 KB of hi+lo weight; (2) 8 warps per CTA each taking a 64-K slice, so 16 warps/SM are in flight instead of 4; (3) load x for the CTA's K slice once into smem with tokens padded to NT and reuse the A (hi/lo) fragments across all NT/8 B fragments (decode-once, as stage 1 of the lossless line learned); (4) the DSMEM reduce stores each rank's slice of columns once over 16 ranks in fixed order. Keep NT 8/16/32, the same self-check bounds and run-twice bit-identical check, torch.empty output, one node. Routing stays as stage 1 (cluster at 1 <= m <= 32, master above). Print per-m kernel time (cold L2) and regs/spills/smem.

## Expected

- `layer=router load=decode_1 m=1` faster: stage-1 row 5.24 (kernel 3.6); floor 1.8; target kernel <= 2.8, row <= 4.6
- `layer=router load=verify_1 m=8` faster: stage-1 row 5.08 (kernel 4.3); target flat-in-m kernel <= 2.9
- `layer=router load=decode_16 m=16` faster: stage-1 row 5.87 (4.6) lost to restack swapab 5.47; target <= 5.0
- `layer=router load=decode_32 m=32` faster: stage-1 fell back to hilo 6.85; now routed, target <= 5.5 (restack swapab 6.2)
- `layer=router load=verify_* m=12*` flat: m=128 keeps master
- `layer=router load=verify_32 m=256` flat: keeps master
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched

## Risk

Judge against stage 1's rows (dc5d03c4) and the restack's router rows. Milestone M2: kernel time within 0.3 us from m=1 to m=32 and <= 3.0 us. Kill: m=16 and m=32 rows not below the restack's 5.47/6.2. Built on master (stage 1's base); a win restacks onto the restack afterwards.

## Milestones

Rewrite stage 2 of `_router_swapab_kernel + memset at router m<=32 (restack) / _router_hilo_kernel + memset (master)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
