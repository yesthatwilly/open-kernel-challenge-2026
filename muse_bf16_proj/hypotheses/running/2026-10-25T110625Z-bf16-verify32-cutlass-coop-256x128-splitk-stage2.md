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
review: proceed
status: running
review_at: 2026-10-25T212522Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T212548Z
base_sha: e4182610b1552724381a1f2451f6bcd56b7dc1ed
base_blob: bcf8bff0b622eecf0d9dc9abecf9e4c7067df872
branch: hyp/muse_bf16_proj/bf16-verify32-cutlass-coop-256x128-splitk-stage2
worktree: .worktrees/muse_bf16_proj/bf16-verify32-cutlass-coop-256x128-splitk-stage2
mode: check
restack_of: null
stale_base: true
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

## Adversarial review

Reviewed 2026-10-25T212522Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound stage 2 of the CUTLASS line that DIAGNOSIS action 3 explicitly owns. The one-M-tile 256x128 cooperative split-K design directly fixes stage 1's diagnosed tile-shape error and can physically reach M2 on pooler/unpooler/qkv. The builder must port it onto current master 36c26199, not 353e3b19, and re-baseline the pooler incumbent to master's NN-layout kernel. No companion run is needed.

### Justification
Mechanism: at m=256 the four rows sit near SGLang parity on nvjet (master 741072dc: qkv row 19.5/kernel 20.0, o 9.4/8.3, pooler 28.0/28.8 NNT, unpooler 24.2/24.8). Their compute floors are 9.8/4.5/17.4/17.4 us, so verify_32 holds about 25-40 us of headroom, which DIAGNOSIS ranks as the right place to spend. A 256x128x64 cooperative tile (2 consumer WGs x 128x128, 128 fp32 acc regs/thread, 4 x 48 KB stages fits under 227 KB) is the standard Hopper config. With static split-K giving 126-128 CTAs in one wave, the per-CTA compute floor is about 17.9 us for pooler (17.4 x 132/128). At a realistic 75-80% wgmma efficiency plus a 3-partial fixup and the ~0.9 us Memset node, that lands around 23-25 us. M2 (<= nvjet) is reachable and M3 is borderline but not impossible. qkv (18 x 7 = 126 CTAs, uneven 18-19 k-steps each, 6 partials per tile) lands around 13-16 us, also borderline at M3 15.5. Weakest link: the causal story that stage 1 lost because 'W was fetched twice by two M-tiles'. pooler at m=256 is near compute/DRAM balance (17.4 us compute vs about 14 us DRAM), and nvjet already uses 128-row tiles at 28.8 us, so the second M-tile's W read is probably largely an L2 hit. The real limiter is more likely mainloop efficiency plus the serialized split-K fixup plus the per-row Memset. Either way the profile will attribute it, because it is a rewrite judged on per-row kernel time. o is the weakest shape: k=1024 x 2 splits gives 8 k-steps per CTA against a 4-stage pipeline, so prologue/epilogue and the memset dominate, and it will likely miss M2 (8.3 us kernel). Stale numbers: the hypothesis quotes pooler as TNT 29.7 kernel / 31.3 row and base blob 353e3b19. Master has since landed pooler-nn-layout-widen-m256 (pooler m=256 is now nvjet 128x64 NNT 28.8 kernel / 28.0 row) and combo-rowowner (36c26199). The M2 bar for pooler must be 28.8, not 29.7. Prior: depends_on bf16-verify-streamk-cutlass-wgmma-rewrite (stage 1, regression 1.1635x, f5b1361a). The difference is real, not the same mechanism again: a 256-row single M-tile with N-multicast of x, and static SplitK to exactly one wave instead of 128x128 StreamK/Heuristic. It is not on any Do-not-try list. DIAGNOSIS's Do-not-try explicitly hands the m>=128 band to 'the CUTLASS line (action 3)'. Expected directions are consistent with TASK: only verify_32 is scored at m in 129..256; m=130/200 tests also take the path; all other loads should be flat.

### Confounders on the current kernel
- pooler-nn-layout-widen-m256: this redefines the pooler m=129..256 incumbent as NN-layout nvjet 128x64 NNT (28.8 kernel / 28.0 row, flag nn_p256_ok). The CUTLASS route takes precedence over that path, so a pooler 'gain' must be judged against 28.8/28.0, not the hypothesis's TNT 29.7/31.3. With the stale bar, a 29.0 us CUTLASS kernel would read as an M2 pass while actually losing to master. state['pooler_t'] stays allocated either way, so this is a measurement confounder, not a functional one.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (current master 36c26199): this does not touch m=256, but the hypothesis says it is built on blob 353e3b19. If the builder ports onto the stale blob, every m<=128 row (bulk row-owner m=1, the qkv m=128 Lt B-cell) differs from master. Headline and non-target rows would then move for reasons unrelated to this change. It must be built on 36c26199 so the m<=256-excluded paths are byte-identical.
- sglang-gemm-dispatch-parity: this is the fallback for qkv_gate/o/unpooler at m=256 and the DISABLED path. If any layer self-check fails, that row silently measures this incumbent. Grep [streamk] build OK / DISABLED / PATH CHOSEN before reading any per-row number (LESSONS: load_inline wgmma builds silently fall back without TORCH_CUDA_ARCH_LIST=9.0a and -std=c++20).
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128: the NN gate is 65..128 (NN_GATED_MAX_M=128), so it does not overlap 129..256 for unpooler. There is no interaction as long as the new gate starts at m >= 129. Verify that the test at m=130 does not double-route.

### Interference and A/B needs
No companion is warranted. This is a rewrite judged on the replaced kernels' own per-row time (profile top_kernels) against the incumbent and floor, and each of the four m=256 rows is attributable directly from one profile because the gate is 129 <= m <= 256 only. Every other path is byte-identical to master when built on 36c26199. The only landed interaction (the pooler NN layout) is a re-baselining issue, fixed by quoting the right incumbent, not by an ab-revert. Reverting it would only make the bar easier and the result less meaningful. The one sub-change that is ambiguous is split-K fixup + memset cost vs mainloop/tile-shape cost. That is best answered for free inside this run by the prepare-time print-only cold-L2 timings (add a splits=1 variant per layer, especially o and unpooler, alongside the chosen split). A separate check would spend a run on it. In-flight pooler-k-halves-fp32-partials-restack-m128-m256 and the DIAGNOSIS action-2 per-cell cuBLAS selector would move the pooler/qkv/o/unpooler m=256 incumbents if they land first. Whichever lands second must be re-measured against the new master, and a restack decision should be per cell.

### Open questions
- Does the CUTLASS sm90 stream-K/SplitK tile scheduler accept ClusterShape 1x2x1 with static splits=7 for qkv (18 N-tiles, uneven K of 128 k-steps / 7) and still give exactly 126 CTAs? Print can_implement, the grid and the applied decomposition, and assert the grid in prepare.
- For o (k=1024, 8 k-steps per split) and possibly unpooler, would splits=1 (64 CTAs, no workspace, no Memset node) beat splits=2? Print both in prepare's cold-L2 replay so stage 3 can choose per layer without another run.
- Can the per-row Memset node (~0.9 us x 4 rows ≈ 3.6 us on a ~91 us verify_32 load, about 4%) be removed, for example by self-resetting barrier counters in the epilogue? Per LESSONS, deleting a graph node can matter as much as kernel speed. Record its cost separately as planned.
- Should the pooler M2 bar be restated as <= 28.8 kernel / 28.0 row (master NN layout), and the qkv bar as master's 20.0 kernel / 19.5 row rather than stage 1's 25.3? 'qkv not worse than stage 1' is too weak a bar for promotion.
- Ordering vs DIAGNOSIS actions 1-2: if the pooler K-halves restack (26.75 row) or the per-cell cuBLAS selector lands first, the pooler/qkv incumbents drop. Promotion should then be per cell against that newer master, and should follow the new LESSONS rule (two interleaved rounds >= 1.03 with an IQR bound) for any prepare-time selector that gates the CUTLASS path.

