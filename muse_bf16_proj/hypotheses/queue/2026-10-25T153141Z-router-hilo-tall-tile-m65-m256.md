---
problem: muse_bf16_proj
slug: router-hilo-tall-tile-m65-m256
created: 2026-10-25T153141Z
base: master
area: router GEMM (m=65..256, hilo tile config)
change: >
  Built on current master (blob 353e3b19). Only the m > 32 router path changes:
  _router_hilo_forward and _router_hilo_config. WHY: the hilo kernel is L2->SM bound, not DRAM
  bound. Master's config is (BLOCK_M 64, BLOCK_N 64, SPLIT_K 16) at m<=128 and (64, 64, 8) at
  m=256, 128 CTAs both times. Each tile re-reads hi+lo (8.4 MB) once per M-tile and x once per
  N-tile, which is 25.2 MB at m=128 and 50.3 MB at m=256 (router_tiled printed these bytes, job
  b7b2b55d04544660). The kernel time follows those bytes, not the 8.4 MB of DRAM: hilo is 5.0 us
  at m=128 and 7.0 us at m=256 with the same CTA count, so the extra 25 MB cost 2 us (about 12.5
  TB/s from L2). CHANGE: give _router_hilo_config taller token tiles and deeper split-K at the
  same CTA count: m in 65..128 -> (BLOCK_M 128, BLOCK_N 64, SPLIT_K 32, BLOCK_K 64), grid 1x4x32 =
  128 CTAs, 16.8 MB L2->SM (-33%). m in 129..256 -> candidate A (128, 64, 16, BLOCK_K 128), grid
  2x4x16 = 128 CTAs, 33.6 MB (-33%); candidate B (256, 64, 32, BLOCK_K 64, num_warps 8), grid
  1x4x32 = 128 CTAs, 25.2 MB (-50%). Add a per-config BLOCK_K and num_warps to the launch;
  _router_hilo_kernel's body is unchanged (two tl.dot per K-step into one fp32 acc, atomic_add
  into the torch.zeros output). Print n_regs, n_spills and wgmma count per compiled variant; smem
  per stage is 64 KB at (128, 64, BK 128), so use 3 stages at BK 128 and 4 at BK 64. Precision is
  unchanged (same hi+lo fp32 accumulate, only the split order differs). The router at m <= 32
  (cluster kernel and swap-AB) and every other layer stay byte-identical. Selector (copy
  restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture
  each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB
  L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both
  medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH
  CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert
  the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Risk:
  twice as many fp32 atomics into L2 (1.05M vs 0.52M at m=128 and m=256, 2.1M for candidate B);
  the selector catches that.
expected:
  - shapes: layer=router load=verify_16 m=128
    direction: faster
    why: "hilo 5.0 us + memset 0.8 / row 7.38, floor 2.21; L2->SM bytes 25.2 -> 16.8 MB: kernel ~4.0, row ~6.4"
  - shapes: layer=router load=verify_32 m=256
    direction: faster
    why: "hilo 7.0 us + memset 0.7 / row 9.31, floor 2.68; 50.3 -> 25-34 MB: kernel ~4.5-5.3, row ~7.0-7.6"
  - shapes: layer=router load=decode_* m=*
    direction: unchanged
    why: m <= 32 router paths untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# router-hilo-tall-tile-m65-m256

## Change

Built on current master (blob 353e3b19). Only the m > 32 router path changes: _router_hilo_forward and _router_hilo_config. WHY: the hilo kernel is L2->SM bound, not DRAM bound. Master's config is (BLOCK_M 64, BLOCK_N 64, SPLIT_K 16) at m<=128 and (64, 64, 8) at m=256, 128 CTAs both times. Each tile re-reads hi+lo (8.4 MB) once per M-tile and x once per N-tile, which is 25.2 MB at m=128 and 50.3 MB at m=256 (router_tiled printed these bytes, job b7b2b55d04544660). The kernel time follows those bytes, not the 8.4 MB of DRAM: hilo is 5.0 us at m=128 and 7.0 us at m=256 with the same CTA count, so the extra 25 MB cost 2 us (about 12.5 TB/s from L2). CHANGE: give _router_hilo_config taller token tiles and deeper split-K at the same CTA count: m in 65..128 -> (BLOCK_M 128, BLOCK_N 64, SPLIT_K 32, BLOCK_K 64), grid 1x4x32 = 128 CTAs, 16.8 MB L2->SM (-33%). m in 129..256 -> candidate A (128, 64, 16, BLOCK_K 128), grid 2x4x16 = 128 CTAs, 33.6 MB (-33%); candidate B (256, 64, 32, BLOCK_K 64, num_warps 8), grid 1x4x32 = 128 CTAs, 25.2 MB (-50%). Add a per-config BLOCK_K and num_warps to the launch; _router_hilo_kernel's body is unchanged (two tl.dot per K-step into one fp32 acc, atomic_add into the torch.zeros output). Print n_regs, n_spills and wgmma count per compiled variant; smem per stage is 64 KB at (128, 64, BK 128), so use 3 stages at BK 128 and 4 at BK 64. Precision is unchanged (same hi+lo fp32 accumulate, only the split order differs). The router at m <= 32 (cluster kernel and swap-AB) and every other layer stay byte-identical. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Risk: twice as many fp32 atomics into L2 (1.05M vs 0.52M at m=128 and m=256, 2.1M for candidate B); the selector catches that.

## Expected

- `layer=router load=verify_16 m=128` faster: hilo 5.0 us + memset 0.8 / row 7.38, floor 2.21; L2->SM bytes 25.2 -> 16.8 MB: kernel ~4.0, row ~6.4
- `layer=router load=verify_32 m=256` faster: hilo 7.0 us + memset 0.7 / row 9.31, floor 2.68; 50.3 -> 25-34 MB: kernel ~4.5-5.3, row ~7.0-7.6
- `layer=router load=decode_* m=*` unchanged: m <= 32 router paths untouched

## Risk

Distinct from router-m128-m256-cluster-dsmem-tiled-rewrite (CUDA mma.sync with the same 64x64 tiles and the same 50.3 MB; the gate kept master) and from router-swapab-wgmma-all-m (swap-AB, lost m=128). Neither changed the L2 byte count. Atomic contention could cancel the byte saving; the selector would then print master and the run reads flat.
