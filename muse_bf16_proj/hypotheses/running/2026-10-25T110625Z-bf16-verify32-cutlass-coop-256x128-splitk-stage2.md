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
stage: implemented
claimed_at: 2026-10-25T212548Z
base_sha: e4182610b1552724381a1f2451f6bcd56b7dc1ed
base_blob: bcf8bff0b622eecf0d9dc9abecf9e4c7067df872
branch: hyp/muse_bf16_proj/bf16-verify32-cutlass-coop-256x128-splitk-stage2
worktree: .worktrees/muse_bf16_proj/bf16-verify32-cutlass-coop-256x128-splitk-stage2
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-26T011122Z
impl_commit: 47ba5bff021ae82d631a382c31109f1453b02f83
submission_sha256: 96f9312ccf31ac76419c263bd8a546c3fe0c1a67688eae377eea0938ae8fcc7b
norm_sha256: 702c7b386fab0da985c904c0ba1866079b13f1f3782f61571cbba62f49a491f9
submission_blob: 6e8cbd9208410b2fc62ee8b2e90ee064981e7a31
lint_warnings: 38
attempt_file: 2026-10-26T012540Z-check-bf16-verify32-cutlass-coop-256x128-splitk-stage2.md
prepared_at: 2026-10-26T012540Z
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


## Implementation notes

Built 2026-10-26 (overnight) in the worktree, `muse_bf16_proj/submission.py` only. Nothing could be compiled or run on this node (no GPU, torch or CUTLASS headers); `py_compile` and `okc_lint` pass.

**Base.** The worktree was claimed on the parent's commit (stale_base). As the review says, `git checkout master -- muse_bf16_proj/submission.py` brought in master 36c26199 (blob ad8d7254, byte-identical to master before my edits), then the stage-1 scaffold was ported onto it. Diff vs master: +761 / -4 lines; the 4 removed lines are the edited `NEW_CELL_FLAGS`, `_ab_component` and `comp = ...` lines and the one `kernel_fn` hunk is a pure insertion.

**What changed (all new code is section "(10)" before `_prepare_state`).**
- CUTLASS extension `sp2_gemm / sp2_query / sp2_set_attr` (namespace `sp2`, ext name `muse_bf16_sp2_cutlass_coop256x128_splitk_v1_c<ver>_cn<N>`; kernel symbols are distinct from every other module): TileShape 256x128x64, `KernelTmaWarpSpecializedCooperative` + `TmaWarpSpecializedCooperative` epilogue (bf16 out, fp32 accumulate, no C), ClusterShape 1x2x1 (the two N-neighbour CTAs multicast the x tile), `StageCountAutoCarveout`, `StreamKScheduler` run in `DecompositionMode::SplitK` with a static `splits` (pooler 4, unpooler 2, qkv_gate 7, o 2), `ReductionMode::Deterministic`. splits=1 selects DataParallel (used only by the print-only diagnostic). Stage 1's one-workspace + `initialize_workspace` (the Memset node) + host-side `to_underlying_arguments` + `Gemm::run` split is kept; `cudaFuncSetAttribute` happens once in prepare via `sp2_set_attr`, which also returns regs/thread, spill bytes and static smem (printed).
- Header discovery (`_find_cutlass_includes`) and the build (`_build_sp2`) copied from stage 1 (CUTLASS 4.5.0 from flashinfer built in 49 s there). Build uses `TORCH_CUDA_ARCH_LIST=9.0a` around `load_inline` (restored in `finally`), `-gencode=arch=compute_90a,code=sm_90a`, `-std=c++20`, `-lineinfo`, prints the nvcc arch flags and build seconds, and the source has `#error` when `__CUDA_ARCH__` is defined without `__CUDA_ARCH_FEAT_SM90_ALL` (a plain compute_90 pass fails the build loudly, master kept).
- Routing: `_sp2_route_ok` / `_sp2_forward`, inserted in `kernel_fn` after every m <= 128 path and before the `layer != "router"` block (so it precedes the pooler NN 129..256 path). Gate: per-layer state flag `sp2_<layer>_ok`, 129 <= cfg m <= 256, cfg n/k == the full weight, x contiguous bf16 [m, k] 16 B aligned. m <= 128 and m > 256 (the m=512 / m=1000 tests) never reach it; router never reaches it. m=130 unpooler (test) takes it; `_ouns`, bulk, swapab and `_pooler/_qkv/_unpool/_o` skinny gates are all m <= 32 so there is no double routing.
- Per-state flags: `SP2_FLAGS` appended to `NEW_CELL_FLAGS`, so `_ab_flags` hands them to state B, the A/B self-check's fail path turns them off, and FINAL FLAGS prints them. State B (`_sp2_prepare(state, flags=...)`) inherits the flags and allocates its OWN workspace and SM count; no module-global device state (only the host int `_SP2_CLUSTER_N` records which cluster build loaded). `AB_CASES` gained unpooler / qkv_gate / o at m=256 (pooler m=256 already existed); `_ab_component(layer, m, state)` names the sp2 flag for those cells only when it is on, otherwise the old component (pooler: `nn_p256_ok`).
- `_sp2_prepare` (state A), per layer: (1) host queries for every m 129..256 (can_implement, grid, scheduler-applied bits, workspace), printing can_implement / grid / CTAs / tiles x splits / smem / workspace / sk_tiles / sk_units / splits_ / CUTLASS version at m = 129, 200, 256 and ASSERTING 1 <= CTAs <= min(sms, 132) (a miss prints `DISABLED`); (2) one workspace per state sized to the max; (3) canary + correctness via master's `_cell_check` at m 130/200/256 (flushed LAUNCH before and synced DONE after every launch, rel L2 <= 3e-3 vs fp64, finite, run 2 bit-identical, power-of-two probe row); (4) print-only `_sp2_diag` at m=256, cold-L2 graph replay: new kernel at its splits, the same kernel at splits=1 (answers the review's o / unpooler question), master's dispatch (flag off), and an upper bound for the memset node (a memset of the whole workspace; the real node zeroes only the barrier region); (5) `_sp2_gate`: two interleaved rounds (A,B,A,B; 41 reps each, 256 MB flush, empty-graph baseline subtracted) of kernel_fn with the flag ON versus OFF (master's ACTUALLY routed path: for pooler the NN-layout path while `nn_p256_ok`, else cuBLAS), both rounds must show master/cand >= 1.03 AND candidate IQR <= 1.5x master's; (6) `torch.profiler` kernel-name assertion that one routed call launches a `device_kernel`; (7) one `[sp2] PATH CHOSEN` line per layer. Every miss, raise, or unsupported shape lands on master's exact path and never disables another layer's flag or any landed flag.
- Deadlines: no build if the import has already run 170 s (`[sp2] build skipped: deadline`), no cluster-1x1x1 retry after 200 s, no sp2 prepare phase after 240 s, no further layer after 270 s (`skipped: deadline`), all measured from `_T0` (import start). Expected sp2 cost: ~50 s build, a few seconds per layer in prepare.
- Last lines of prepare: after `FINAL FLAGS`, the four `[sp2] PATH CHOSEN ... (gate rounds ...)` lines and `[sp2] LAST-LINES flags now: ...` are re-printed.

**Review fixes applied.**
1. Built on master 36c26199, not 353e3b19 (see Base).
2. Pooler incumbent re-baselined: the gate compares against master's real routed path (NN-layout `nn_p256_ok` nvjet 128x64 NNT, 28.8 kernel / 28.0 row), never the stale TNT 29.7 / 31.3; the milestone bar in the Expected section is superseded by the review's 28.8 / 28.0. Likewise qkv's bar is master's own cuBLAS (20.0 / 19.5), not stage 1's 25.3: the gate itself enforces it.
3. Gate starts at m >= 129 (NN_GATED_MAX_M = 128 is untouched); m=130 does not double-route.
4. Prints can_implement, grid and the applied decomposition, asserts the grid in prepare (open question 1).
5. splits=1 variant per layer in prepare's cold-L2 print (open question 2); the memset node cost is recorded separately as an upper bound (open question 3).
6. Selector follows the new LESSONS rule (two interleaved rounds >= 1.03, IQR <= 1.5x master's, master = the actually routed path, flag-off), per cell, with PATH CHOSEN per cell and a compact repeat as the last lines.
7. Greppable `[sp2] build OK / DISABLED / PATH CHOSEN` before any number is read.

**Open questions answered in code.** (Q1) Whether CUTLASS accepts cluster 1x2x1 with SplitK 7 and gives exactly 126 CTAs is only answered at run time; the build tries cluster 1x2x1 first and, if the compile fails (e.g. a static_assert), once more at 1x1x1. (Q3) Not attempted: removing the memset node by self-resetting barriers would mean forking the CUTLASS scheduler; recorded only. (Q4) Bars restated above. (Q5) The selector is per cell, against whatever master the file is built on.

**Deviations from the spec / brief.**
- Bounded mbarrier waits: the mainloop, epilogue and split-K barrier waits are CUTLASS-internal and unbounded, and I did not fork them. Mitigation: build `-lineinfo`, flushed print before and synced print after every self-check launch (names the kernel on a fault), the first launch is the canary in prepare.
- No self-check at m=129 or m=255 beyond the three m of stage 1 (130 / 200 / 256); the host query covers every m 129..256.
- The cluster 1x1x1 retry build is an addition (not in the spec) to survive a CUTLASS-side static_assert on the split-K + cluster combination.
- Stage 1's second-CUTLASS-version retry was dropped (the newest tree, 4.5.0 from flashinfer, is the one stage 1 proved).

**Biggest risks.** (a) First compile of the 256x128 cooperative StreamK scheduler with cluster 1x2x1 and `splits` on CUTLASS 4.5 (the one thing stage 1 did not prove); a failed build costs 50-100 s of prepare and lands on master. (b) A hang in a CUTLASS-internal barrier would only end at the run timeout. (c) o (8 k-steps per split) and qkv may lose the 1.03x gate and stay on master, which is the designed outcome.

## Rules check

Checked against `git -C .worktrees/muse_bf16_proj/bf16-verify32-cutlass-coop-256x128-splitk-stage2 diff master -- muse_bf16_proj/submission.py` (+761 / -4).

- [x] CUDA-graph capturable: `kernel_fn` only gains `_sp2_route_ok` (cfg / shape / state flags) and `_sp2_forward` (`torch.empty` + one `sp2_gemm`: a memset node, host-side TMA descriptors from pointers, one cluster launch on the current stream); no `.item()` / `.cpu()` / `synchronize()` / prints; no branching on values. Every sync and print is in prepare-only functions (`_sp2_*`, which the gate captures in graphs itself).
- [x] Nothing derived from `data` is cached across calls: the workspace is zeroed by `initialize_workspace` each call and holds only partial sums; `state["sp2_ws"]`, `sp2_sms`, flags and summary strings come from weights / the device only. The extension has no globals.
- [x] Weights: untouched; the CUTLASS kernel reads `state[layer]` (the original contiguous bf16 weight) in place, no repack, no precision change.
- [x] Activations BF16 where SGLang keeps BF16: x bf16, out bf16, fp32 accumulate, same as nvjet.
- [x] n/a: attention rules do not apply to this GEMM problem.
- [x] No timing patches, harness frames / I/O / secrets, module edits, reference import, subprocesses or network: grep for subprocess / multiprocessing / fork / Popen / os.system is empty; the build is `torch.utils.cpp_extension.load_inline` like the existing `bulk` / `ouns` modules; header discovery only globs local package directories.
- [x] All work happens inside the timed call (one CUTLASS GEMM per call, outputs recomputed from the call's x); no output caching or replay.
- [x] Per-state flags and fallback: each layer has its own `sp2_<layer>_ok`, passed to state B through `NEW_CELL_FLAGS` / `_ab_flags`; any build error, self-check miss, deadline or unsupported shape lands on master's exact path; no landed flag is touched.
- [x] Prepare budget: deadlines at 170 / 200 / 240 / 270 s from import, explicit `skipped: deadline` prints, build time printed, regs / spills printed.
- [x] Does not edit any file but `muse_bf16_proj/submission.py` in the worktree (plus this md); no commit, no staging, no okc.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`): `OK with warnings` (0 errors; 36 warnings, all `host-sync` in prepare-only code or `dtype` for the uint8 L2-flush buffers, the same pattern as master's gates; none in `kernel_fn`'s call graph). `python3 -I -m py_compile`: OK; `__pycache__` deleted.

## Pre-flight audit

Verdict: dispatch-with-fixes. Audited by reading (no GPU, torch or CUTLASS headers on this node; none found by `find /`).

- Scheduler members (concern 1): judge log f5b1361a (stage 1, CUTLASS 4.5.0 from flashinfer) proves `decomposition_mode` settable (mode_set=True), `sk_tiles_`, `sk_units_`, `get_grid_shape(params)` and `params.scheduler` compile. Untested there: `splits` / `splits_` / `reduction_mode`. These are detected by void_t SFINAE, so an absent member never breaks the build: a missing `splits` leaves applied bits != 3 and `_sp2_query_layer` prints `DISABLED` and keeps master per layer. A member that exists with an unexpected enum (`DM::SplitK`, `RM::Deterministic`) is a hard nvcc error, caught by `except Exception` around load_inline, then `_SP2_EXT=None` and master everywhere with `[sp2] build FAILED` / `PATH CHOSEN ... master`.
- Retry (concern 2): the 1x1x1 retry is a separate module name (`..._cn1` vs `..._cn2`, so a separate torch_extensions dir and version entry), torch releases its build lock in a finally, and `TORCH_CUDA_ARCH_LIST` is restored in a finally. No half-built state is reused. Deadlines are from `_T0`, which starts AFTER `import torch/triton/sglang`; master's ouns + router-cluster builds (~37 s each) run first, so the sp2 build starts near 80-100 s; two failed builds fit under the 200 s retry cutoff and prepare starts before 240 s.
- Grid and splits (concern 3): emulation `/tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/audits26/emu_sp2_tiles_bf16_verify32.py` (output `.out` beside it): for every m 129..256 there is exactly one 256-row M-tile; units = pooler 128, unpooler 128, qkv 126 (7 splits of 19,19,..,14 k-tiles), o 128, all <= 132, all even, N-tile counts even (cluster 1x2 divides), 63-64 clusters <= 66 co-resident; 4 stages x 48 KB = 192 KB leaves 35 KB for epilogue. Note the `<= min(sms,132)` assert is nearly tautological (persistent grid = min(units, SMs)); the real protection is the can_implement / applied-bits / self-check / gate chain.
- Shadowing (concern 4): the sp2 route sits after every m <= 32 path, the m == 1 bulk, m == 128 Lt and ouns routes, and before the NN-layout block. In 129..256 the only landed cell it can shadow is the pooler NN layout (`nn_p256_ok`); the per-layer gate compares against exactly that routed path (flag off) at m=256 and needs >= 1.03 in both rounds. Gate runs at m=256 only, so m=129..255 (test m=130 unpooler n=8192 only) rides the m=256 decision; unscored, correctness checked at 130/200/256. unpooler slice (n=1024) and m=128/512/1000 never reach it.
- Fixes: (1) IQR veto floor was `max(im, 0.05)` us, so a timer-resolution IQR on the incumbent could veto a real win; raised to `max(im, 0.5)`. (2) `_sp2_diag` (print-only) was inside the layer's try, so any exception in it (e.g. the splits=1 DataParallel query/launch) would disable the layer; wrapped in its own try. okc_lint OK with warnings, py_compile OK, __pycache__ removed.
- Residual risk: first compile and launch of the 256x128 cooperative StreamK SplitK instantiation with `splits` on CUTLASS 4.5.0 and a CUTLASS-internal unbounded barrier hang or fault in the canary would poison the context rather than fall back.
