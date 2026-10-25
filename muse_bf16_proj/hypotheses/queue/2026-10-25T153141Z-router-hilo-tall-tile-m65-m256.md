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
review: proceed
status: queued
review_at: 2026-10-25T203138Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T203138Z by an adversarial subagent (model opus).

**Verdict: proceed.** Cheap, self-gated retile of master's _router_hilo_kernel at m=128/256; distinct from earlier router attempts, no rules issue. Mechanism weaker than stated (m=256 mixes math and L2 bytes; doubled fp32 atomics may cancel savings): expect a small win at m=128 and likely PATH CHOSEN master at m=256. The in-run selector medians and profile make the single run readable without companions.

### Justification
Per-M-tile hi+lo re-reads are real (25.2/50.3 MB at m=128/256, matches job b7b2b55d). _router_hilo_kernel 5.0/6.9 us + 0.7-0.8 us memset; rows 7.6/9.7 us vs floors 2.2/2.7. Weakest link: the L2-bound claim from 5.0 -> 7.0 us also doubles math (2.15 GFLOP ~2.2 us wgmma floor at m=256) and K-steps; implied 12.5 TB/s marginal L2 is above H100/H200 L2. SPLIT_K 32 doubles red.global (0.52M -> 1.05M); BK 64 gives 4 K-steps, all prologue/epilogue. Prior check: cluster-dsmem-tiled-rewrite kept 64x64 tiling (never tested bytes); swapab-wgmma-all-m changed operand roles. Base drift (353e3b19 -> ad8d7254) does not touch the router code. Spec gap: master's bucket m<=128 covers 33..128; keep 33..64 on (64,64,16) explicitly.

### Confounders on the current kernel
- router-triton-hilo-fused-control-splitk-all-m: the incumbent kernel; body/memset/epilogue unchanged.
- bf16-restack-m16-swapab-router-swapab-m32: ROUTER_SWAPAB_MAX_M=32 puts 33..64 in the hilo m<=128 bucket; keep the boundary explicit.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (master): prepare's _ab_selfcheck rebuilds state B via _ab_flags; the new selector flag must be inherited like NEW_CELL_FLAGS.
- pooler/unpooler/qkv/o and router m<=32 hypotheses: no shared traffic or numerics with router m=128/256.

### Interference and A/B needs
No companion: the change is ~0.3-0.5% headline at most, inside the CI; the run is read from the prepare() selector's same-run cold-L2 medians (master vs A vs B), the profiler kernel-name assertion + PATH CHOSEN, and the judge rows vs 7.6/9.7 us.

### Open questions
- Keep m 33..64 at (64,64,16) and print the m=64 path.
- _router_shape_ok / _router_hilo_forward must take per-config BLOCK_K, stages and warps.
- Selector flag must pass through _ab_flags into state B.
- 4-stage pipeline at 4 K-steps: try 3 stages; print n_regs/spills.
- Overlap with queued router-m128-m256-wgmma-swapab-tall-cluster-stage2 (re-baseline if this lands).

