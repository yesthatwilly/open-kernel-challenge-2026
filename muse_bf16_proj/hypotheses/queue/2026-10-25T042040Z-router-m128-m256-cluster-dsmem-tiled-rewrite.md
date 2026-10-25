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
review: proceed
status: queued
review_at: 2026-10-25T042454Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T042454Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a legitimate stage-1 rewrite with no rules problem, and router m=128/256 really are the worst ratios left at m>=128. As specified, though, it probably will not beat the incumbent. The m=256 instance's 144 KB smem breaks the 16-cluster single wave (the stage-1 log shows cs=8 at 133 KB gives max active clusters 15). Its m=256 tiling is identical to the incumbent's, and its m=128 tiling moves about 30% MORE L2->SM bytes. It also downgrades the wgmma the incumbent already uses to 4-warp mma.sync. Build it with the smem, tiling and diagnostic fixes below.

### Justification
Profile (restack run 1e90292c7b954d96 = current master b99b1dd, router m>32 unchanged by the restack): m=128 row 7.3 us = _router_hilo_kernel 5.0 + memset 0.8; m=256 row 9.6 = 6.9 + 0.7; floors 2.2 / 2.7 (DRAM; hi+lo doubles the math to 2.15 GFLOP at m=256, a 2.2 us wgmma math floor, so m=256 is math- and L2-bound, not DRAM-bound). The memset node is real, but it is only 0.7-0.8 us. The bigger cost is L2->SM operand traffic: bytes = x(m*K*2)*(N/BN) + hi+lo(8.4 MB)*(m/BM). Incumbent m=128 (64x64, split 16) = 2*4+8.4*2 = 24.8 MB. Incumbent m=256 (64x64, split 8) = 4*4+8.4*4 = 49.6 MB, which at ~7 TB/s L2 is ~7 us and matches the 6.9 us kernel. The proposed m=256 tiling (64x64 tiles, 8-way K split, 128 CTAs) is IDENTICAL to the incumbent's (64,64,8) grid, so its mainloop has the same traffic and the only possible saving is the memset. The proposed m=128 tiling (64x32, split 8) moves 2*8+8.4*2 = 32.8 MB, 32% more than the incumbent. Stage-1 evidence from dc5d03c4 (same scaffold): the cluster kernel was 3.6 us vs the Triton swap-AB kernel's 3.1 at m=1 (kernel vs kernel). Its DSMEM reduce tail cost 1.1-1.2 us in the cold-L2 eager diagnostic (10.08 vs 9.02 no-reduce at m=1; 10.82 vs 9.60 at m=16), which is more than the 0.7-0.8 us memset it removes. Smem: the m=256 instance's 3-stage ring is 3 x (x 16 KB + hi/lo 32 KB) = 144 KB, so 1 CTA/SM. The stage-1 log shows cs=8 at 133 KB gives max active clusters = 15 < 16, so the 16th cluster runs as a second wave (~2x). The m=128 instance at 96 KB is fine (cs=8 at 82-99 KB gave 30). Tensor-core path: the incumbent's BLOCK_M=64, 4-warp tl.dot lowers to wgmma at m>=64; the rewrite uses mma.sync with 4 warps per SM, which the hypothesis itself prices at ~5.4 us of pure math at m=256. WEAKEST LINK: the mechanism (memset + atomics removal) is worth <=0.8 us, which the measured DSMEM-reduce tail (~1.1 us) and the mma.sync downgrade can each erase. The kernel's real cost is L2->SM re-reads, which this tiling leaves unchanged (m=256) or worsens (m=128). Expected directions are consistent with TASK shapes; other layers and m<=64 are correctly gated off. One factual slip: no router test runs at m=130/200 (the router tests are m=8 and m=9 dist outliers; m=130 is unpooler), so m=65..256 is checked only by the timed calls' per-call correctness and the in-prepare self-check.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32 (b99b1dd, current master): gates the swap-AB router to m<=32 and leaves m>32 on _router_hilo_forward byte for byte, so it does not confound m=128/256. Use its run 1e90292c7b954d96 as the incumbent (kernel 5.0/6.9, rows 7.28/9.60).
- router-triton-hilo-fused-control-splitk-all-m: this is the incumbent being replaced (_router_hilo_kernel (64,64,16)/(64,64,8) + torch.zeros memset), not a confounder. Judge against its top_kernels on the two target rows.
- bf16-shape-gated-triton-winners-combo, bf16-m32-swapab-wgmma-triton, sglang-gemm-dispatch-parity, router-hilo-at-m8: other layers or m<=8 only. No shared launch or numerics at m>=65. Adding a ~40 s load_inline to prepare is within budget (master has no load_inline today).
- Cross-run headline noise: in dc5d03c4 the untouched qkv/o decode rows moved -2 to -8%, so the headline is not attributable. Read only the router m=128/256 top_kernels.

### Interference and A/B needs
No companions. This is a rewrite judged on the replaced kernels' own time. The gate (65<=m<=256, router only) is clean, and the profile names the new kernel on exactly the two target rows. The in-prepare cold-L2 eager timing vs _router_hilo_forward on the same synthetic x, plus a no-DSMEM-reduce diagnostic mode, already give a same-run A/B that splits mainloop from reduce tail. A revert or tune of any landed item would not disambiguate anything, because nothing landed touches router m>32. The next stage (wgmma and/or multicast tiling) is the follow-up.

### Open questions
- BUILDER (must): cap every instance's dynamic smem at ~110 KB so 2 CTAs/SM fit and cudaOccupancyMaxActiveClusters(cs=8) >= 16. For the m=256 BN=64 instance, use BLOCK_K=64 with 3-4 stages (24 KB/stage) or 2 stages at 128-K instead of 3x48 KB = 144 KB. Alias the fp32 partial tile onto the ring. Print regs/spills/smem/max-active-clusters per instance and fall back to master if < 16.
- BUILDER (should): print the L2->SM operand bytes per instance and do not ship a tiling that exceeds the incumbent's (24.8 MB at m=128, 49.6 MB at m=256). For m=128, prefer 64x64 tiles x split 16 (K-slice 512) with a non-portable cluster of 16 (cudaFuncAttributeNonPortableClusterSizeAllowed; gate on max active clusters >= 8), or compile both 64x32/cs8 and 64x64/cs16 and pick per m bucket by in-prepare cold-L2 timing. For m=256, the specified 64x64/split-8 equals the incumbent's traffic. Optionally add a 128x64 x split-16 cs16 instance (32.8 MB) as the stage's real traffic win.
- BUILDER (should): keep the dc5d03c4 diagnostics. Report cold-L2 eager medians for the new kernel, the same kernel in no-DSMEM-reduce mode, and master _router_hilo_forward (which includes the memset) at m=65/128/200/256, so harvest can tell mainloop from reduce tail. Stage 1 measured that tail at ~1.1 us, more than the 0.8 us memset it removes; shrink it (8 rows x BN x 8 ranks via 16-B ld.shared::cluster vector loads, all 128 threads).
- BUILDER (could): consider 8 warps (2 per SM sub-partition, 4x2 warp tiles) to give mma.sync latency hiding; 4 warps/SM is the likely mainloop limiter at m=256. Print the mma.sync count.
- Self-check: add m=64 to the self-check to confirm it stays on master, alongside 65/128/129/130/200/256. Note that the judge has no router test at m=65..256, so a wrong-but-finite tile edge only shows up as a failed timed call.
- Harvest: before reading numbers, confirm the build line, the occupancy prints, PATH CHOSEN for m=65..256, and no vectorized_elementwise memset node on the two target rows. M2 means the kernel is below 5.8/7.6 us (incumbent kernel + memset), rows below 7.28/9.60. If M2 fails at m=256 with the traffic-equal tiling, stage 2 should be wgmma (sm_90a) + TMA multicast of x across N-tiles in the cluster, not more mma.sync tuning.
- Coordinate with the queued bf16-verify-swapab-wgmma-cluster-dsmem-rewrite (m=65..256, bf16 layers). If its wgmma+cluster scaffold lands first, router stage 2 should reuse it.

