---
problem: muse_bf16_proj
slug: bf16-verify16-cutlass-m128-onetile-splitk-rewrite
created: 2026-10-25T153141Z
base: master
area: bf16 GEMM dispatch (m=65..128, CUTLASS sm90 one-M-tile split-K)
change: >
  Built on current master (blob 353e3b19). This is a new line for the m=128 band, which the queued
  bf16-verify32-cutlass-coop-256x128-splitk-stage2 (m=129..256 only) leaves to cuBLAS. It takes
  its scaffold from bf16-verify-streamk-cutlass-wgmma-rewrite stage 1 (patch
  done/regression/bf16-verify-streamk-cutlass-wgmma-rewrite.patch): CUTLASS 4.5 header discovery
  from flashinfer, load_inline with TORCH_CUDA_ARCH_LIST=9.0a and -std=c++20, the #error guard on
  !__CUDA_ARCH_FEAT_SM90_ALL, the one prepare-allocated workspace per state, the self-check and
  the prints. WHY: verify_16 is 73.2 us against a 43.8 us floor, and its four GEMM rows are all
  cuBLAS at 1.3-2.0x their floors (qkv 16.58/8.42, o 7.35/3.99, pooler 22.94/14.64, unpooler
  18.95/14.64). cuBLAS knobs are spent there: NN layout (landed for pooler/unpooler), cublasLt
  algo pick (did not carry over), and DeepGEMM, which never ran. Two Triton m=128 mainloops
  stalled at ~26 us, about 2.4 TB/s; REWRITE_READ named the x K-slice re-read per CTA (about 2x
  the weight bytes) and one CTA per SM. DESIGN, stage 1: (1) TileShape 128x128x64 with M = tokens,
  so m <= 128 is ONE M-tile and every W byte is fetched from DRAM by exactly one CTA. Use
  KernelTmaWarpSpecializedCooperative (2 consumer warpgroups of 64x128, 1 TMA producer,
  setmaxnreg) and TN operands: x [m, k] K-major as A, W [n, k] K-major as B, the layout master
  already holds, so no repack. (2) ClusterShape 1x2x1 so the two N-neighbour CTAs multicast the x
  tile (the reused operand) by TMA, halving x L2->SM bytes. (3) DecompositionMode::SplitK with
  static splits for one wave of <= 132 CTAs: pooler 32 N-tiles x 4 = 128 (K 2048), unpooler 64 x 2
  = 128 (K 2048), qkv 18 x 7 = 126, o 64 x 2 = 128 (K 512). Print can_implement, grid, splits,
  smem, workspace bytes, regs/spills and the decomposition actually applied, and assert the grid
  is <= 132. (4) StageCountAutoCarveout; bf16 out, fp32 accumulate. (5) Rows >= m are handled by
  TMA OOB zero-fill and a predicated store, never by padding x, so there is no allocation or copy
  in kernel_fn. Routing: qkv_gate, o, pooler and unpooler (full width, n == 8192) at 65 <= cfg m
  <= 128 only, gated on cfg layer/m/n and shapes. Every m <= 64 path and every m > 128 path stays
  byte-identical to master. Self-check per layer at m 65, 100 and 128: rel L2 against fp32 torch,
  finite, and run2 bit-identical; plus capture once and replay with refilled inputs while
  alternating two prepared states (LESSONS: the judge keeps 2 state copies; no module-global
  workspace or TMA descriptors). Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model
  selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same
  synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41
  reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new
  path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of
  the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS:
  PATH CHOSEN proves the selector, not the kernel). The selector runs per layer at m=128; a failed
  build or self-check prints DISABLED and that layer keeps master. Run as an okc check.
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "incumbent NN nvjet splitK 20.2 + splitKreduce 3.0 / row 22.94, floor 14.64; W read once, x multicast, in-kernel split-K fixup: M2 row <= 21.5, M3 row <= 18.5"
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: "incumbent nvjet 12.7 + splitKreduce 3.1 / row 16.58, floor 8.42; 126 CTAs and no separate reduce node: M2 <= 14.5, M3 <= 12.0"
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: "incumbent NN nvjet 20.4 kernel / row 18.95, floor 14.64: M2 <= 18.5, M3 <= 17.0"
  - shapes: layer=o load=verify_16 m=128
    direction: faster
    why: "incumbent nvjet 128x64 5.5 kernel / row 7.35, floor 3.99; only 8 K-steps per split, so the risky one: M2 <= 7.2, M3 <= 6.0"
  - shapes: layer=* load=decode_* m=*
    direction: unchanged
    why: m <= 32 untouched
  - shapes: layer=* load=verify_32 m=256
    direction: unchanged
    why: m > 128 untouched (owned by the cutlass stage-2 line)
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: "cuBLAS at m=65..128 (verify_16): qkv_gate nvjet 96x64 splitK 12.7 + cublasLt splitKreduce 3.1 us, o nvjet 128x64 TNT 5.5 us, pooler NN nvjet 128x64 splitK 20.2 + splitKreduce 3.0 us, unpooler NN nvjet 128x64 NNT 20.4 us"
review: required
status: queued
---

# bf16-verify16-cutlass-m128-onetile-splitk-rewrite

## Change

Built on current master (blob 353e3b19). This is a new line for the m=128 band, which the queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 (m=129..256 only) leaves to cuBLAS. It takes its scaffold from bf16-verify-streamk-cutlass-wgmma-rewrite stage 1 (patch done/regression/bf16-verify-streamk-cutlass-wgmma-rewrite.patch): CUTLASS 4.5 header discovery from flashinfer, load_inline with TORCH_CUDA_ARCH_LIST=9.0a and -std=c++20, the #error guard on !__CUDA_ARCH_FEAT_SM90_ALL, the one prepare-allocated workspace per state, the self-check and the prints. WHY: verify_16 is 73.2 us against a 43.8 us floor, and its four GEMM rows are all cuBLAS at 1.3-2.0x their floors (qkv 16.58/8.42, o 7.35/3.99, pooler 22.94/14.64, unpooler 18.95/14.64). cuBLAS knobs are spent there: NN layout (landed for pooler/unpooler), cublasLt algo pick (did not carry over), and DeepGEMM, which never ran. Two Triton m=128 mainloops stalled at ~26 us, about 2.4 TB/s; REWRITE_READ named the x K-slice re-read per CTA (about 2x the weight bytes) and one CTA per SM. DESIGN, stage 1: (1) TileShape 128x128x64 with M = tokens, so m <= 128 is ONE M-tile and every W byte is fetched from DRAM by exactly one CTA. Use KernelTmaWarpSpecializedCooperative (2 consumer warpgroups of 64x128, 1 TMA producer, setmaxnreg) and TN operands: x [m, k] K-major as A, W [n, k] K-major as B, the layout master already holds, so no repack. (2) ClusterShape 1x2x1 so the two N-neighbour CTAs multicast the x tile (the reused operand) by TMA, halving x L2->SM bytes. (3) DecompositionMode::SplitK with static splits for one wave of <= 132 CTAs: pooler 32 N-tiles x 4 = 128 (K 2048), unpooler 64 x 2 = 128 (K 2048), qkv 18 x 7 = 126, o 64 x 2 = 128 (K 512). Print can_implement, grid, splits, smem, workspace bytes, regs/spills and the decomposition actually applied, and assert the grid is <= 132. (4) StageCountAutoCarveout; bf16 out, fp32 accumulate. (5) Rows >= m are handled by TMA OOB zero-fill and a predicated store, never by padding x, so there is no allocation or copy in kernel_fn. Routing: qkv_gate, o, pooler and unpooler (full width, n == 8192) at 65 <= cfg m <= 128 only, gated on cfg layer/m/n and shapes. Every m <= 64 path and every m > 128 path stays byte-identical to master. Self-check per layer at m 65, 100 and 128: rel L2 against fp32 torch, finite, and run2 bit-identical; plus capture once and replay with refilled inputs while alternating two prepared states (LESSONS: the judge keeps 2 state copies; no module-global workspace or TMA descriptors). Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The selector runs per layer at m=128; a failed build or self-check prints DISABLED and that layer keeps master. Run as an okc check.

## Expected

- `layer=pooler load=verify_16 m=128` faster: incumbent NN nvjet splitK 20.2 + splitKreduce 3.0 / row 22.94, floor 14.64; W read once, x multicast, in-kernel split-K fixup: M2 row <= 21.5, M3 row <= 18.5
- `layer=qkv_gate load=verify_16 m=128` faster: incumbent nvjet 12.7 + splitKreduce 3.1 / row 16.58, floor 8.42; 126 CTAs and no separate reduce node: M2 <= 14.5, M3 <= 12.0
- `layer=unpooler load=verify_16 m=128` faster: incumbent NN nvjet 20.4 kernel / row 18.95, floor 14.64: M2 <= 18.5, M3 <= 17.0
- `layer=o load=verify_16 m=128` faster: incumbent nvjet 128x64 5.5 kernel / row 7.35, floor 3.99; only 8 K-steps per split, so the risky one: M2 <= 7.2, M3 <= 6.0
- `layer=* load=decode_* m=*` unchanged: m <= 32 untouched
- `layer=* load=verify_32 m=256` unchanged: m > 128 untouched (owned by the cutlass stage-2 line)

## Risk

Milestones (judged on the replaced kernels' own time in top_kernels, including any Memset node, and on the m=128 rows). M1: build OK, 9/9 tests, a cutlass device_kernel on the m=128 rows the selector routed, no silent fallback (grep the build-OK / DISABLED / PATH CHOSEN lines first). M2: pooler and qkv rows below the incumbent, which proves the reduce-node removal. M3: verify_16 load 73.2 -> ~62 us (+18% on the load, about +3% headline). Stage 2 if M2 is met: replace the global-workspace split-K fixup and its Memset node with an in-cluster DSMEM reduce (cluster 1x2xS), and merge this extension with the m=256 line into one load_inline module. Kill: if the pooler kernel is not under 20.2 us at M1, the CUTLASS tile is no better than nvjet at m=128; stop the line. Fallback: master through the prepare-time gate. Risks: the compile adds about 50 s, and the m=256 line builds a second module; keep both under the 10-minute run budget by compiling only the instantiations routed here. o's 8 K-steps per split may leave it slower: let the gate drop it.

## Milestones

Rewrite stage 1 of `cuBLAS at m=65..128 (verify_16): qkv_gate nvjet 96x64 splitK 12.7 + cublasLt splitKreduce 3.1 us, o nvjet 128x64 TNT 5.5 us, pooler NN nvjet 128x64 splitK 20.2 + splitKreduce 3.0 us, unpooler NN nvjet 128x64 NNT 20.4 us`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
