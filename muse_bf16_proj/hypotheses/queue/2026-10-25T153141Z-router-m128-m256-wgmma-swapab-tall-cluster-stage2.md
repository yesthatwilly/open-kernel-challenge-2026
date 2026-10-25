---
problem: muse_bf16_proj
slug: router-m128-m256-wgmma-swapab-tall-cluster-stage2
created: 2026-10-25T153141Z
base: master
area: router GEMM (m=65..256, CUDA wgmma swap-AB cluster kernel)
change: >
  Stage 2 of router-m128-m256-cluster-dsmem-tiled-rewrite (job b7b2b55d04544660, neutral). Port
  its load_inline scaffold, self-check, cold-L2 timing table and PATH CHOSEN gate onto CURRENT
  master (blob 353e3b19), not onto its stale branch. Stage 1 compiled and self-checked, and its
  gate correctly kept master. It lost on the kernel: mma.sync over the incumbent's same 64x64
  tiles, printing the same 25.2 / 50.3 MB L2->SM operand bytes. Cold-L2 eager it ran 17.3 / 24.0
  us against master's 12.5 / 14.7 at m=128 / 256. Its no-DSMEM-reduce variant was 15.2-22.2, so
  the mainloop, not the reduce, was slow. STAGE 2 CHANGE, three parts. (1) An sm90a wgmma swap-AB
  mainloop. The A operand is weight rows from smem: 64 hi rows and the 64 lo rows of the same
  output columns, TMA-loaded. The B operand is the x tile [BLOCK_T tokens x 64 K], TMA-loaded. One
  consumer warpgroup issues wgmma m64nBLOCK_Tk16 twice per K-step (hi, then lo) into one fp32
  accumulator, so each x tile in smem feeds both halves. There is one producer warp, a 4-6 stage
  mbarrier ring, and setmaxnreg. (2) Tall token tiles to cut L2 bytes. At m=256, BLOCK_T 128 x 4
  column groups x cluster 16 split-K = 128 CTAs, with 33.6 MB L2->SM (-33%). At m=128, two
  candidates: BLOCK_T 128 x 4 x cluster 16 = 64 CTAs (16.8 MB), or BLOCK_T 64 x 2 token tiles x 4
  x 16 = 128 CTAs (25.2 MB). (3) Keep stage 1's in-cluster DSMEM split-K reduce, so each output
  tile is owned by one cluster and stored once. That deletes the torch.zeros memset node (LESSONS:
  on 5-10 us rows a graph node matters). The output is torch.empty. Build per LESSONS:
  TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20, the #error guard, -lineinfo, per-launch
  'launching'/'ok' prints in the self-check, and cudaFuncAttributeNonPortableClusterSizeAllowed
  for cluster 16 (stage 1 measured max active clusters 14 at cs=16). Keep TMA descriptors per
  state, never module-global. Self-check at m 65, 128, 129, 200 and 256 with outlier rows and
  power-of-two single-nonzero rows: rel L2 <= 1e-5 against fp64 (bound 1e-4), exact-column rows <=
  1e-3; plus a two-state alternating graph replay. Selector (copy
  restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture
  each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB
  L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both
  medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH
  CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert
  the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The
  selector runs per m-band (65..128, 129..256) against whatever master's m > 32 router is at
  dispatch time; if router-hilo-tall-tile-m65-m256 has landed, it is the incumbent to beat. Router
  m <= 32 and every bf16 layer stay byte-identical. Run as an okc check.
expected:
  - shapes: layer=router load=verify_32 m=256
    direction: faster
    why: "incumbent hilo 7.0 + memset 0.7 / row 9.31, floor 2.68: M2 kernel <= 6.0 with no memset node, M3 <= 4.5 (row ~6.5)"
  - shapes: layer=router load=verify_16 m=128
    direction: faster
    why: "incumbent hilo 5.0 + memset 0.8 / row 7.38, floor 2.21: M2 kernel <= 4.5 with no memset, M3 <= 3.5 (row ~5.5)"
  - shapes: layer=router load=decode_* m=*
    direction: unchanged
    why: router m <= 32 untouched
risk: high
depends_on: router-m128-m256-cluster-dsmem-tiled-rewrite
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _router_hilo_kernel + torch.zeros memset at router m=65..256 (hilo 5.0 + 0.8 us at m=128, 7.0 + 0.7 us at m=256)
review: required
status: queued
---

# router-m128-m256-wgmma-swapab-tall-cluster-stage2

## Change

Stage 2 of router-m128-m256-cluster-dsmem-tiled-rewrite (job b7b2b55d04544660, neutral). Port its load_inline scaffold, self-check, cold-L2 timing table and PATH CHOSEN gate onto CURRENT master (blob 353e3b19), not onto its stale branch. Stage 1 compiled and self-checked, and its gate correctly kept master. It lost on the kernel: mma.sync over the incumbent's same 64x64 tiles, printing the same 25.2 / 50.3 MB L2->SM operand bytes. Cold-L2 eager it ran 17.3 / 24.0 us against master's 12.5 / 14.7 at m=128 / 256. Its no-DSMEM-reduce variant was 15.2-22.2, so the mainloop, not the reduce, was slow. STAGE 2 CHANGE, three parts. (1) An sm90a wgmma swap-AB mainloop. The A operand is weight rows from smem: 64 hi rows and the 64 lo rows of the same output columns, TMA-loaded. The B operand is the x tile [BLOCK_T tokens x 64 K], TMA-loaded. One consumer warpgroup issues wgmma m64nBLOCK_Tk16 twice per K-step (hi, then lo) into one fp32 accumulator, so each x tile in smem feeds both halves. There is one producer warp, a 4-6 stage mbarrier ring, and setmaxnreg. (2) Tall token tiles to cut L2 bytes. At m=256, BLOCK_T 128 x 4 column groups x cluster 16 split-K = 128 CTAs, with 33.6 MB L2->SM (-33%). At m=128, two candidates: BLOCK_T 128 x 4 x cluster 16 = 64 CTAs (16.8 MB), or BLOCK_T 64 x 2 token tiles x 4 x 16 = 128 CTAs (25.2 MB). (3) Keep stage 1's in-cluster DSMEM split-K reduce, so each output tile is owned by one cluster and stored once. That deletes the torch.zeros memset node (LESSONS: on 5-10 us rows a graph node matters). The output is torch.empty. Build per LESSONS: TORCH_CUDA_ARCH_LIST=9.0a only, -std=c++20, the #error guard, -lineinfo, per-launch 'launching'/'ok' prints in the self-check, and cudaFuncAttributeNonPortableClusterSizeAllowed for cluster 16 (stage 1 measured max active clusters 14 at cs=16). Keep TMA descriptors per state, never module-global. Self-check at m 65, 128, 129, 200 and 256 with outlier rows and power-of-two single-nonzero rows: rel L2 <= 1e-5 against fp64 (bound 1e-4), exact-column rows <= 1e-3; plus a two-state alternating graph replay. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The selector runs per m-band (65..128, 129..256) against whatever master's m > 32 router is at dispatch time; if router-hilo-tall-tile-m65-m256 has landed, it is the incumbent to beat. Router m <= 32 and every bf16 layer stay byte-identical. Run as an okc check.

## Expected

- `layer=router load=verify_32 m=256` faster: incumbent hilo 7.0 + memset 0.7 / row 9.31, floor 2.68: M2 kernel <= 6.0 with no memset node, M3 <= 4.5 (row ~6.5)
- `layer=router load=verify_16 m=128` faster: incumbent hilo 5.0 + memset 0.8 / row 7.38, floor 2.21: M2 kernel <= 4.5 with no memset, M3 <= 3.5 (row ~5.5)
- `layer=router load=decode_* m=*` unchanged: router m <= 32 untouched

## Risk

Milestones: M1 build OK, 9/9 tests, the router kernel name on both rows (or a printed gate decision keeping master, plus the cold-L2 table); M2 as above; M3 router rows within ~2.5x of floor, about +4% on the verify_32 load and +3% on verify_16. Kill: if the wgmma mainloop without the reduce is not under master's cold-L2 eager time at m=256 (14.7 us in stage 1's table), the CUDA router line is dead; leave the router on Triton. Overlaps router-hilo-tall-tile-m65-m256 (a Triton config change on the same rows). Run the primary first: if it lands, this stage's gate measures against it. Fallback: master through the gate. Risk: two prior CUDA router kernels (mma.sync) lost; this stage changes the MMA path and the L2 byte count, the two things stage 1 did not.

## Milestones

Rewrite stage 2 of `_router_hilo_kernel + torch.zeros memset at router m=65..256 (hilo 5.0 + 0.8 us at m=128, 7.0 + 0.7 us at m=256)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
