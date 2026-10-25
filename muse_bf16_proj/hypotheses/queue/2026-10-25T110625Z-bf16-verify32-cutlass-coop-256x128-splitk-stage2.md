---
problem: muse_bf16_proj
slug: bf16-verify32-cutlass-coop-256x128-splitk-stage2
created: 2026-10-25T110625Z
base: master
area: bf16 GEMM dispatch (m=129..256, CUTLASS sm90 cooperative split-K)
change: >
  Stage 2 of bf16-verify-streamk-cutlass-wgmma-rewrite (job f5b1361aaa0e48b5), but ported onto
  CURRENT master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128,
  1.2366x), not built on the stage-1 branch, whose small-m paths are two masters old. WHY:
  verify_32 is the weakest load (1.0745x vs 1.25/1.20/1.19/1.58/1.17 for the other five) and every
  m=256 GEMM row sits at SGLang parity on nvjet: qkv 19.7, o 9.6, pooler 31.3, unpooler 24.5 us
  against compute floors 9.8/4.5/17.4/17.4 (pooler nvjet ~580 TFLOP/s). cuBLAS knobs are spent
  there (NN layout flat at m=256, cublasLt algo pick lost qkv, DeepGEMM never ran, Triton m128
  mainloops stuck at ~26 us). Stage 1 proved the scaffold (CUTLASS 4.5 headers found in
  flashinfer, load_inline OK in 49 s with TORCH_CUDA_ARCH_LIST=9.0a, bit-identical self-checks,
  the cutlass device_kernel on all four m=256 rows) but lost on the kernel shape: TileShape
  128x128x64, ClusterShape 2x1x1, StreamK over 132 units for qkv/pooler (25.3 / 31.0 us + a 0.9 us
  Memset node) and Heuristic for o/unpooler (9.2 / 26.8 us). What stage 1 got wrong: with 128-row
  tiles the m=256 problem is two M-tiles, so W (the 67 MB operand) is fetched by two CTAs
  (multicast only within a 2-CTA cluster) and stream-K fixups serialize partial tiles. STAGE 2
  CHANGE: copy the extension code (_STREAMK_CUDA, _build_streamk, header discovery, workspace,
  self-check, prints) from branch hyp/muse_bf16_proj/bf16-verify-streamk-cutlass-wgmma-rewrite
  (patch done/regression/bf16-verify-streamk-cutlass-wgmma-rewrite.patch) onto master and change
  the kernel: (1) TileShape 256x128x64 with KernelTmaWarpSpecializedCooperative (2 consumer
  warpgroups of 128x128 each, 1 producer, setmaxnreg via the collective), so one M-tile covers all
  256 tokens and every W byte is fetched from DRAM by exactly one CTA. (2) ClusterShape 1x2x1: the
  two N-neighbour CTAs of a cluster multicast the x tile (256 x 64 per stage, the reused operand)
  instead of W. (3) DecompositionMode::SplitK with static splits that give one wave of <= 132
  CTAs: pooler (n 4096 -> 32 N-tiles) x 4 = 128 CTAs (K 2048 each); unpooler (n 8192 -> 64 tiles,
  k 4096) x 2 = 128; qkv_gate (2304 -> 18 tiles) x 7 = 126 (uneven K slices as CUTLASS computes
  them); o (64 tiles, k 1024) x 2 = 128. Print, per layer and m, can_implement, grid, splits,
  smem, workspace bytes and the decomposition actually applied (stage 1's print line), and assert
  the grid is <= 132 CTAs. (4) Stages via StageCountAutoCarveout (expect 4 x 48 KB); epilogue as
  stage 1 (bf16 out, fp32 accumulate). (5) Keep stage 1's one prepare-allocated workspace and its
  initialize_workspace (the Memset node) for the split-K barrier; record its cost separately in
  the profile. Routing: for qkv_gate, o, pooler, unpooler at 129 <= cfg m <= 256 only (scored row:
  verify_32 m=256; tests m=130/200 also take it), gate on cfg layer/m/n and shapes only; every m
  <= 128 path (swap-AB, cluster kernels, NN layout at m=65..128, cuBLAS) stays byte-identical to
  master; m > 256 (the m=512 / m=1000 tests) stays on master. Per-layer correctness self-check at
  m 130/200/256 as stage 1 (rel L2, finite, run2 bit-identical); a failed build or self-check
  prints DISABLED and that layer keeps master. In prepare print, per layer at m=256, the cold-L2
  CUDA-graph replay median of the new kernel and of master's dispatch on the same synthetic x
  (print-only, never routing). Run as a check.
expected:
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: "incumbent nvjet 128x64 29.7 us kernel / 31.3 row, floor 17.4; one 256-row M-tile reads W once, 128x128 per-WG wgmma tiles instead of N=64: M2 <= 29.0 kernel, M3 <= 25.0"
  - shapes: layer=unpooler load=verify_32 m=256
    direction: faster
    why: "incumbent nvjet 128x128 24.8 us / 24.5 row, floor 17.4; split 2 fills 128 SMs with 256x128 tiles: M2 <= 24.0, M3 <= 20.5"
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: "incumbent nvjet 96x64 19.4 us / 19.7 row, floor 9.8; 18 tiles x 7 splits = 126 CTAs vs stage 1 stream-K 25.3: M2 <= 19.0, M3 <= 15.5"
  - shapes: layer=o load=verify_32 m=256
    direction: faster
    why: "incumbent nvjet 8.1 us / 9.6 row, floor 4.5; risky: k=1024 gives each split only 8 K-steps and the memset node costs ~1 us; M2 <= 8.0, M3 <= 7.0"
  - shapes: layer=* load=decode_* m=*
    direction: flat
    why: m <= 32 paths byte-identical to master
  - shapes: layer=* load=verify_1 m=8
    direction: flat
    why: untouched
  - shapes: layer=* load=verify_16 m=128
    direction: flat
    why: m=128 keeps master (NN layout / nvjet / hilo)
  - shapes: layer=router load=verify_32 m=256
    direction: flat
    why: router untouched
risk: high
depends_on: bf16-verify-streamk-cutlass-wgmma-rewrite
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: cuBLAS nvjet at m=129..256 (verify_32) for qkv_gate (nvjet 96x64 2x4 TNN 19.4 us), o (nvjet 128x128 TNT 8.1 us), pooler (nvjet 128x64 TNT 29.7 us) and unpooler (nvjet 128x128 TNT 24.8 us)
review: required
status: queued
---

# bf16-verify32-cutlass-coop-256x128-splitk-stage2

## Change

Stage 2 of bf16-verify-streamk-cutlass-wgmma-rewrite (job f5b1361aaa0e48b5), but ported onto CURRENT master (blob 353e3b19, restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128, 1.2366x), not built on the stage-1 branch, whose small-m paths are two masters old. WHY: verify_32 is the weakest load (1.0745x vs 1.25/1.20/1.19/1.58/1.17 for the other five) and every m=256 GEMM row sits at SGLang parity on nvjet: qkv 19.7, o 9.6, pooler 31.3, unpooler 24.5 us against compute floors 9.8/4.5/17.4/17.4 (pooler nvjet ~580 TFLOP/s). cuBLAS knobs are spent there (NN layout flat at m=256, cublasLt algo pick lost qkv, DeepGEMM never ran, Triton m128 mainloops stuck at ~26 us). Stage 1 proved the scaffold (CUTLASS 4.5 headers found in flashinfer, load_inline OK in 49 s with TORCH_CUDA_ARCH_LIST=9.0a, bit-identical self-checks, the cutlass device_kernel on all four m=256 rows) but lost on the kernel shape: TileShape 128x128x64, ClusterShape 2x1x1, StreamK over 132 units for qkv/pooler (25.3 / 31.0 us + a 0.9 us Memset node) and Heuristic for o/unpooler (9.2 / 26.8 us). What stage 1 got wrong: with 128-row tiles the m=256 problem is two M-tiles, so W (the 67 MB operand) is fetched by two CTAs (multicast only within a 2-CTA cluster) and stream-K fixups serialize partial tiles. STAGE 2 CHANGE: copy the extension code (_STREAMK_CUDA, _build_streamk, header discovery, workspace, self-check, prints) from branch hyp/muse_bf16_proj/bf16-verify-streamk-cutlass-wgmma-rewrite (patch done/regression/bf16-verify-streamk-cutlass-wgmma-rewrite.patch) onto master and change the kernel: (1) TileShape 256x128x64 with KernelTmaWarpSpecializedCooperative (2 consumer warpgroups of 128x128 each, 1 producer, setmaxnreg via the collective), so one M-tile covers all 256 tokens and every W byte is fetched from DRAM by exactly one CTA. (2) ClusterShape 1x2x1: the two N-neighbour CTAs of a cluster multicast the x tile (256 x 64 per stage, the reused operand) instead of W. (3) DecompositionMode::SplitK with static splits that give one wave of <= 132 CTAs: pooler (n 4096 -> 32 N-tiles) x 4 = 128 CTAs (K 2048 each); unpooler (n 8192 -> 64 tiles, k 4096) x 2 = 128; qkv_gate (2304 -> 18 tiles) x 7 = 126 (uneven K slices as CUTLASS computes them); o (64 tiles, k 1024) x 2 = 128. Print, per layer and m, can_implement, grid, splits, smem, workspace bytes and the decomposition actually applied (stage 1's print line), and assert the grid is <= 132 CTAs. (4) Stages via StageCountAutoCarveout (expect 4 x 48 KB); epilogue as stage 1 (bf16 out, fp32 accumulate). (5) Keep stage 1's one prepare-allocated workspace and its initialize_workspace (the Memset node) for the split-K barrier; record its cost separately in the profile. Routing: for qkv_gate, o, pooler, unpooler at 129 <= cfg m <= 256 only (scored row: verify_32 m=256; tests m=130/200 also take it), gate on cfg layer/m/n and shapes only; every m <= 128 path (swap-AB, cluster kernels, NN layout at m=65..128, cuBLAS) stays byte-identical to master; m > 256 (the m=512 / m=1000 tests) stays on master. Per-layer correctness self-check at m 130/200/256 as stage 1 (rel L2, finite, run2 bit-identical); a failed build or self-check prints DISABLED and that layer keeps master. In prepare print, per layer at m=256, the cold-L2 CUDA-graph replay median of the new kernel and of master's dispatch on the same synthetic x (print-only, never routing). Run as a check.

## Expected

- `layer=pooler load=verify_32 m=256` faster: incumbent nvjet 128x64 29.7 us kernel / 31.3 row, floor 17.4; one 256-row M-tile reads W once, 128x128 per-WG wgmma tiles instead of N=64: M2 <= 29.0 kernel, M3 <= 25.0
- `layer=unpooler load=verify_32 m=256` faster: incumbent nvjet 128x128 24.8 us / 24.5 row, floor 17.4; split 2 fills 128 SMs with 256x128 tiles: M2 <= 24.0, M3 <= 20.5
- `layer=qkv_gate load=verify_32 m=256` faster: incumbent nvjet 96x64 19.4 us / 19.7 row, floor 9.8; 18 tiles x 7 splits = 126 CTAs vs stage 1 stream-K 25.3: M2 <= 19.0, M3 <= 15.5
- `layer=o load=verify_32 m=256` faster: incumbent nvjet 8.1 us / 9.6 row, floor 4.5; risky: k=1024 gives each split only 8 K-steps and the memset node costs ~1 us; M2 <= 8.0, M3 <= 7.0
- `layer=* load=decode_* m=*` flat: m <= 32 paths byte-identical to master
- `layer=* load=verify_1 m=8` flat: untouched
- `layer=* load=verify_16 m=128` flat: m=128 keeps master (NN layout / nvjet / hilo)
- `layer=router load=verify_32 m=256` flat: router untouched

## Risk

Milestones (judged on the replaced kernels' own time in top_kernels plus the Memset node, and on the m=256 rows): M1 build OK, 9/9 tests, the profile shows a cutlass device_kernel on all four m=256 rows and no fallback (grep [streamk] build OK / DISABLED / PATH CHOSEN before reading any number); M2 pooler and unpooler at or below nvjet (29.7 / 24.8 us) with qkv not worse than stage 1's 25.3; M3 pooler <= 25, unpooler <= 20.5, qkv <= 15.5, o <= 7.0, i.e. verify_32 load ~94 -> ~82 us, +13% on that load and about +2% headline. Risks: (a) 256x128 cooperative accumulators are 128 fp32 regs per consumer thread; with setmaxnreg 240 that is the standard CUTLASS config, but print regs/spills. (b) SplitK fixup still serializes on the workspace barrier; if the tail dominates, stage 3 is a DSMEM in-cluster reduce (cluster 1x1xS) instead of the global workspace. (c) The build bug in LESSONS (torch adds a plain compute_90 gencode): keep stage 1's TORCH_CUDA_ARCH_LIST=9.0a and the #error guard on !__CUDA_ARCH_FEAT_SM90_ALL. (d) cuBLAS-incumbent m=256 rows were stable across runs (qkv 19.57-19.69, o 9.01-9.56, pooler 30.53-31.12, unpooler 23.89-24.74), so a per-row gain over ~3% is readable from one check. Fallback: any layer that misses M2 stays on master in a later restack; the scaffold is reused. Not blocked by the queued lossless stage 3 (held, line DEAD per REWRITE_READ) or by any pending hypothesis.

## Milestones

Rewrite stage 2 of `cuBLAS nvjet at m=129..256 (verify_32) for qkv_gate (nvjet 96x64 2x4 TNN 19.4 us), o (nvjet 128x128 TNT 8.1 us), pooler (nvjet 128x64 TNT 29.7 us) and unpooler (nvjet 128x128 TNT 24.8 us)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
