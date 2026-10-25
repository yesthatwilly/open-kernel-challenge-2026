---
problem: muse_bf16_proj
slug: router-m128-m256-cluster-dsmem-tiled-rewrite
created: 2026-10-25T042040Z
base: master
area: router GEMM (m=65..256)
change: >
  Built on master (the router at m > 32 is identical on master and the restack). Router rows at
  m=128/256 are 7.3/9.6 us against floors of 2.2/2.7 (3.3-3.6x, the worst ratio left at m>=128):
  _router_hilo_kernel 5.0/6.9 us plus a 0.7-0.8 us vectorized_elementwise memset node for the
  atomically-added split-K output. New CUDA kernel, one node, no memset, no atomics, reusing the
  router-cuda-cluster-dsmem-splitk-rewrite scaffold (load_inline, TORCH_CUDA_ARCH_LIST=9.0a,
  -std=c++20 replacing any -std, -lineinfo, per-route self-check, print-and-fall-back). Tiles the
  [m, 256] fp32 output: m=128 -> 64 token x 32 expert tiles (2 x 8 = 16 tiles); m=256 -> 64 x 64
  tiles (4 x 4 = 16 tiles); each tile is one cluster of 8 CTAs splitting K=8192 into 1024-wide
  slices = 128 CTAs. Per CTA a 3-stage cp.async.cg ring of 128-K chunks of x [64, 128] and
  router_hi/router_lo [BN, 128]; mma.sync m16n8k16 with hi and lo as two MMAs into one fp32
  accumulator (the same hi+lo fp32-equivalent math as master); 4 warps per CTA (2x2 warp tiles).
  Split-K reduce: each CTA writes its fp32 tile partial to smem, barrier.cluster release/acquire,
  rank r reduces rows [8r, 8r+8) of the tile over ranks 0..7 in fixed order via
  mapa/ld.shared::cluster and stores fp32 with plain stores (deterministic); second cluster
  barrier before exit. torch.empty output. Rows m in 33..64 and m > 256 keep master; m 65..256
  that is not a multiple of the token tile is handled by zero-fill on load and a guarded store
  (covers the m=130/200 tests). Self-check at m in {65, 128, 130, 200, 256} twice: rel L2 <= 2e-5
  vs fp64, one-hot power-of-two probe rows within 1e-6 of the hi+lo reconstruction, run 2
  bit-identical, sync+print after each launch. Diagnostics: regs/spills/smem,
  cudaOccupancyMaxActiveClusters, cold-L2 event time vs master _router_hilo_forward.
expected:
  - shapes: layer=router load=verify_16 m=128
    direction: faster
    why: incumbent kernel 5.0 + memset 0.8, row 7.3, floor 2.2; 12.6 MB of DRAM + ~25 MB L2 re-reads; M2 row <= 6.5, M3 <= 5.5
  - shapes: layer=router load=verify_32 m=256
    direction: faster
    why: incumbent 6.9 + 0.7, row 9.6, floor 2.7; M2 <= 8.5, M3 <= 7.0
  - shapes: layer=router load=decode_*
    direction: flat
    why: m<=32 untouched
  - shapes: layer=router load=verify_1 m=8
    direction: flat
    why: untouched
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
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: _router_hilo_kernel + torch.zeros memset at router m=65..256
review: required
status: queued
---

# router-m128-m256-cluster-dsmem-tiled-rewrite

## Change

Built on master (the router at m > 32 is identical on master and the restack). Router rows at m=128/256 are 7.3/9.6 us against floors of 2.2/2.7 (3.3-3.6x, the worst ratio left at m>=128): _router_hilo_kernel 5.0/6.9 us plus a 0.7-0.8 us vectorized_elementwise memset node for the atomically-added split-K output. New CUDA kernel, one node, no memset, no atomics, reusing the router-cuda-cluster-dsmem-splitk-rewrite scaffold (load_inline, TORCH_CUDA_ARCH_LIST=9.0a, -std=c++20 replacing any -std, -lineinfo, per-route self-check, print-and-fall-back). Tiles the [m, 256] fp32 output: m=128 -> 64 token x 32 expert tiles (2 x 8 = 16 tiles); m=256 -> 64 x 64 tiles (4 x 4 = 16 tiles); each tile is one cluster of 8 CTAs splitting K=8192 into 1024-wide slices = 128 CTAs. Per CTA a 3-stage cp.async.cg ring of 128-K chunks of x [64, 128] and router_hi/router_lo [BN, 128]; mma.sync m16n8k16 with hi and lo as two MMAs into one fp32 accumulator (the same hi+lo fp32-equivalent math as master); 4 warps per CTA (2x2 warp tiles). Split-K reduce: each CTA writes its fp32 tile partial to smem, barrier.cluster release/acquire, rank r reduces rows [8r, 8r+8) of the tile over ranks 0..7 in fixed order via mapa/ld.shared::cluster and stores fp32 with plain stores (deterministic); second cluster barrier before exit. torch.empty output. Rows m in 33..64 and m > 256 keep master; m 65..256 that is not a multiple of the token tile is handled by zero-fill on load and a guarded store (covers the m=130/200 tests). Self-check at m in {65, 128, 130, 200, 256} twice: rel L2 <= 2e-5 vs fp64, one-hot power-of-two probe rows within 1e-6 of the hi+lo reconstruction, run 2 bit-identical, sync+print after each launch. Diagnostics: regs/spills/smem, cudaOccupancyMaxActiveClusters, cold-L2 event time vs master _router_hilo_forward.

## Expected

- `layer=router load=verify_16 m=128` faster: incumbent kernel 5.0 + memset 0.8, row 7.3, floor 2.2; 12.6 MB of DRAM + ~25 MB L2 re-reads; M2 row <= 6.5, M3 <= 5.5
- `layer=router load=verify_32 m=256` faster: incumbent 6.9 + 0.7, row 9.6, floor 2.7; M2 <= 8.5, M3 <= 7.0
- `layer=router load=decode_*` flat: m<=32 untouched
- `layer=router load=verify_1 m=8` flat: untouched
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched

## Risk

Kill: neither target row below its incumbent with the new kernel in the profile, or a spill in the build line. Risk: 2x hi/lo MMA count makes m=256 ~2.1 GFLOP; mma.sync at ~400 TFLOP/s is ~5.4 us, so M3 at m=256 may need stage 2 (wgmma). Stage-1 cluster kernel ran clean on the judge (dc5d03c4), so the cluster/DSMEM instruction set is proven.

## Milestones

Rewrite stage 1 of `_router_hilo_kernel + torch.zeros memset at router m=65..256`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
