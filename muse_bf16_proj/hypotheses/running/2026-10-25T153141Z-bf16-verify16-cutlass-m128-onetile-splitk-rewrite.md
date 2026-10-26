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
review: proceed
status: running
review_at: 2026-10-25T212614Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T212614Z
base_sha: f8d14416f350af92938d3e41f3256af4aa532299
base_blob: ad8d7254a64a52db2a8f56ec0ad262748153a2e8
branch: hyp/muse_bf16_proj/bf16-verify16-cutlass-m128-onetile-splitk-rewrite
worktree: .worktrees/muse_bf16_proj/bf16-verify16-cutlass-m128-onetile-splitk-rewrite
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-26T010753Z
impl_commit: f1adab325b6c9225785a9d72c237374e50c0366e
submission_sha256: 6f4f815ba8afe271d06d8512e6c879a259027f19ff8c799288664833232fa5fb
norm_sha256: 89033c648111f6a5bba3a295216709f0c559d26cc8451e5ddc9032d34adbfe3d
submission_blob: 18bc92b5a7d3cfa10ac6120d8ab524a4fd8cb25a
lint_warnings: 35
attempt_file: 2026-10-26T012539Z-check-bf16-verify16-cutlass-m128-onetile-splitk-rewrite.md
prepared_at: 2026-10-26T012539Z
job: d4edd38aeed44334
dispatched_at: 2026-10-26T012632Z
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

## Adversarial review

Reviewed 2026-10-25T212614Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a sound rewrite stage. The CUTLASS 4.5 scaffold is proven on the judge (f5b1361a: built in 49.0 s from flashinfer's tree, every self-check passed, cutlass::device_kernel rows in the profile), and a one-M-tile split-K design at m=128 can physically reach M2 on pooler, unpooler and qkv. Before the build, the stale base and incumbents must be fixed. The selector must reuse master's _cell_gate flag on/off machinery so it races master's actual NN and Lt routes, and it should adopt today's two-round >=1.03 lesson.

### Justification
Mechanism. At m<=128 a 128x128x64 tile makes the token dimension a single M-tile, so each W byte is fetched from DRAM by exactly one CTA. Split-K then fills 126-128 CTAs in one wave. The cooperative TMA/wgmma mainloop with setmaxnreg is the one mainloop family not yet tried at m=128: both Triton mainloops are on Do-not-try at 26-27.6 us, and the swap-AB cluster design is dead. DIAGNOSIS action 3 explicitly endorses this line.

Numbers from the prior run. The same template (128x128x64 coop, 384 threads, 214 KB smem) ran at m=256 at roughly 0.42 us per 64-wide k-step per CTA. Unpooler data-parallel took 26.8 us for 64 k-steps. At m=128 the per-CTA math per k-step is identical, but there are only 32 k-steps per split for pooler and unpooler (K 2048). That gives about 13.4 us of mainloop if math-limited, or about 15.5-16 us if limited by DRAM at roughly 4.3 TB/s for 67 MB, plus the fixup and memset. Expected rows are about 17-18.5 us for unpooler against 18.95, and about 18-19.5 us for pooler against 22.94 (20.8 + 3.0 splitKreduce). For qkv (126 CTAs, about 18-19 k-steps, 7-way fixup) expect about 11-12 us. All of these are consistent with the M2 and M3 targets. o (8 k-steps per split, fill and epilogue dominated) is marginal; letting the gate drop it is correct.

Weakest link 1 (empirical). The only measurement of this exact template on this hardware lost to nvjet on all four layers at m=256: qkv 25.3 vs 19.4, pooler 31.0 vs 29.5, o 9.2 vs 8.3, unpooler 26.8 vs 24.6. That run ran at 495-608 TFLOP/s warm, and with cluster 2x1x1 W was already multicast there. DIAGNOSIS's 'fetched W twice' explanation of that loss is therefore partly wrong, and the m=128 win rests on the bytes-per-k-step argument holding when DRAM, not the tensor cores, sets the pace. The kill criterion covers this. Judge pooler on the cutlass kernel plus its Memset node (incumbent 23.8 us kernel time), not the bare kernel against 20.2.

Weakest link 2 (stale facts):
- The md says 'Built on current master (blob 353e3b19)'. Master is now 36c26199, blob ad8d7254.
- On that master, qkv_gate m=128 already runs the B-cell cuBLASLt single-node plan: nvjet tst_40x64, 15.5 us row, no splitKreduce. The hypothesis still lists the qkv incumbent as 12.7 + 3.1 = 16.58, so the expected entry and M2 must use 15.5. The rationale that M2 on qkv 'proves the reduce-node removal' is void, because master already removed it.
- Pooler (22.9, NN splitK + reduce) and unpooler (18.9, NN) match the md.

Prior and Do-not-try. There is no prior field, but the body names the scaffold source, bf16-verify-streamk-cutlass-wgmma-rewrite (regression at m=256, different band, tile quantization and stream-K issue). The m=128 band was explicitly left to it as 'stage 2 scope'. The CUTLASS line is not a re-run of the dead Triton m=128 designs. It is not a duplicate.

Shapes and weights. All verify_16 rows carry weight 62. A load drop from 73.2 to about 62 us is roughly +2.5-3% headline, in line with M3. The decode, verify_1 and verify_32 'unchanged' entries are right if the gating is cfg-m based as stated.

### Confounders on the current kernel
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (latest landed, BEST): its B cell routes qkv_gate m=128 to a prepare-fixed cuBLASLt single-node plan (15.5 us; lt_qkv128_ok, _lt_route_ok sits before the cuBLAS fallback in kernel_fn). The CUTLASS qkv candidate must be raced against that path, not against _bf16_gemm_dispatch_impl. If the selector is copied from _ouns_gate and races plain dispatch (16.6-17.4 us), it can route CUTLASS over a faster incumbent and record a false 'win'. The md's qkv incumbent numbers predate this landing.
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128: nn_ok sends pooler and unpooler at 65..128 to NN-layout torch.mm on pooler_t / unpooler NN copies. That NN path is the real incumbent for both rows. The md's numbers already reflect it, but the gate must keep nn_ok's route as 'master', and the new branch must preempt it only when its own state flag is set.
- pooler-nn-layout-widen-m256: touches only m=129..256. No interaction, provided the new branch's upper gate is cfg m <= 128.
- Not landed, but in flight on the same cells: ooler-k-halves-fp32-partials-restack-m128-m256 (pooler m=128, row about 21.4) and f16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole. DIAGNOSIS action 3 says to dispatch this one after those and race whatever they routed. If either lands before tonight's build, rebuild on it so the selector's incumbent is current. Otherwise a later stack could double-route the cell.

### Interference and A/B needs
No companion is warranted. This is a rewrite stage judged on the replaced kernels' own time in top_kernels. The run carries its own in-prepare A/B: the interleaved graph-replay gate per (layer, m=128) against master's flag-off path, with both medians printed and the launched kernel names asserted by the profiler. Both a gain and a loss are therefore attributable from one run, without a revert of any landed hypothesis. The landed hypotheses that touch these cells (B-cell Lt for qkv, nn_ok for pooler/unpooler) are exactly the incumbents the gate must race. Reverting them would only measure the CUTLASS kernel against a weaker baseline nobody would ship. If one control were ever needed, it would be the in-prepare print of the CUTLASS kernel with cluster 1x1x1 vs 1x2x1 (x multicast on/off). That belongs in stage 2 as a second instantiation, not in a separate check.

### Open questions
- Rebase onto master 36c26199 (blob ad8d7254). Update the qkv expected and M2 line to the 15.5 us Lt single-node incumbent (and the pooler kill line to kernel + memset < 23.8 us), not 12.7 + 3.1.
- Selector: reuse master's _cell_gate / _cell_kernels with a new per-layer state flag (cand = flag on, master = flag off, same kernel_fn), not a copy of _ouns_gate racing plain dispatch. Per today's lesson, require two interleaved rounds, each >= 1.03 (not one round at 1.02), with an IQR bound printed.
- Does CUTLASS 4.5's sm90 stream-K scheduler in DecompositionMode::SplitK accept ClusterShape 1x2x1 with uneven K splits? qkv uses 7 splits over 128 k-tiles (18.3 per split). Print can_implement, the splits actually applied and sk_units per layer. If can_implement fails for qkv, try 8 splits (18 x 8 = 144 CTAs > 132) or fall back to 4 rather than disabling.
- Is the deterministic split-K fixup of 4-7 peers serialized inside the single wave (the last split waits on its peers' workspace flags)? If pooler or qkv misses M2 by about the fixup time, stage 2's in-cluster DSMEM reduce is the right next step, as the md already plans.
- Confirm that kernel-name assertion and harvest grep pick up a cutlass::device_kernel<...GemmUniversal...> row on each routed m=128 row before any number is read. The m=256 run proved the selector prints, not the routed kernel.
- If the per-cell cuBLASLt selector (action 2) or the pooler K-halves restack lands first, which incumbent will tonight's build race? The build should rebase on whatever master holds at build time.


## Implementation notes

Edited only `.worktrees/muse_bf16_proj/bf16-verify16-cutlass-m128-onetile-splitk-rewrite/muse_bf16_proj/submission.py` (+714/-4 vs the worktree's HEAD, master blob ad8d7254, which already is the review's rebase base). Everything outside the new code is unchanged except the small hooks listed below.

What changed:
- New load_inline module `muse_bf16_cut128_v1_c<ver>` (`_CUT_CPP`/`_CUT_CUDA`, `_cut_build`), built at import after master's own modules. Scaffold = stage-1 streamk patch: CUTLASS header discovery (flashinfer / deep_gemm / sgl_kernel / tilelang / cutlass_library, newest 3.x+ tree), `TORCH_CUDA_ARCH_LIST=9.0a` set around the build and restored, `-gencode=arch=compute_90a,code=sm_90a`, `-std=c++20`, `-lineinfo`, arch flags and build seconds printed, an `#error` if a non-sm_90a device pass is ever compiled (build then fails loudly and master runs). No subprocess/fork anywhere.
- Kernel: TileShape 128x128x64, ClusterShape 1x2x1, KernelTmaWarpSpecializedCooperative + TmaWarpSpecializedCooperative epilogue, StreamKScheduler with DecompositionMode::SplitK, runtime `splits`, ReductionMode::Deterministic, StageCountAutoCarveout, TN operands (x [m,k] A, W [n,k] B, no repack), bf16 out / fp32 accumulate, rows >= m by TMA OOB zero-fill and clipped TMA store (no padding, no copy). Distinct symbols (`c128::GemmKernel`, `cut128_*`).
- Per-layer state flags `cut_qkv_ok / cut_o_ok / cut_pooler_ok / cut_unpooler_ok` (CUT_LAYERS), added to NEW_CELL_FLAGS so the A/B state B inherits them; a miss in one cell touches no other flag. Splits: qkv 7 (fallback 8, 4), o 2 (4), pooler 4 (2), unpooler 2 (4); first candidate with can_implement, split mode applied and grid <= SM count is used (grid, splits, smem, workspace, sk_tiles/sk_units, regs/spills, decomposition applied are printed).
- Routing in kernel_fn: one new branch after the swapab routes and BEFORE the nn_ok / Lt / cuBLAS block, gated on layer flag, 65 <= cfg m <= 128, cfg n/k == full weight, x contiguous bf16 [m,k] 16-B aligned. Every m <= 64 and m > 128 path is byte-identical to master; with a flag off the call falls through to master's exact route (nn_ok, Lt, cuBLAS).
- prepare(): `_cut_prepare` after `_lt_prepare` (state A). Per layer: self-check at m 65/100/128 through `_cell_check` (announced launches, sync after each, rel L2 vs fp64 <= 3e-3, finite, run2 bit-identical, one-hot probe row), then `_cut_gate` at m=128, then a profiler kernel-name assertion (exactly one cutlass device_kernel<GemmUniversal> plus memset). One workspace per state (state B allocates its own); no module-global workspace or descriptors. A/B/A/B graph check extended with an o m=128 case and `_ab_component` now takes state and blames the cut flag when it serves the cell.
- Last lines of prepare() repeat every `PATH CHOSEN CUT <layer>` line and a `[cut128] FINAL` flags line.

Review fixes applied:
- Rebase: worktree already at blob ad8d7254 (checked with git rev-parse). The qkv incumbent is the B-cell cuBLASLt single-node plan (15.5 us), not 12.7+3.1: the gate races kernel_fn flag-off, which routes through `_lt_route_ok` first, so the qkv cell must beat 15.5 us (M2 target for qkv = below 15.5). Pooler judged as cutlass kernel + Memset vs 23.8 us.
- Selector reuses master's flag-on / flag-off kernel_fn capture (`_cut_gate`, modelled on `_cell_gate`) instead of a copy of `_ouns_gate` racing plain dispatch; nn_ok keeps being master's route for pooler/unpooler.
- Two interleaved rounds, both master/cand >= 1.03x (net of empty-graph baseline), candidate IQR <= 1.5x master's; both rounds' medians and IQRs, both sides' profiled kernel names and PATH CHOSEN are printed even when master is kept.
- Open questions answered in code: can_implement / splits-applied / sk_units / grid printed per layer, qkv falls back 7 -> 8 -> 4 splits rather than disabling; fixup serialisation is not directly observable, the printed per-layer gate times decide whether stage 2 (DSMEM reduce) is needed; the kernel-name assertion requires a cutlass device_kernel<GemmUniversal> row per routed cell; rebase on whatever master holds at build time (done, no unlanded companion present in this tree).

Deviations:
- CUTLASS owns the mainloop, so its mbarrier waits are not bounded by us (cannot patch them); mitigation: the canary runs every launch announced and synced before anything is routed, and the module is never run on an unchecked shape. A hang would still run to the timeout.
- The brief's `#if __CUDA_ARCH_FEAT_SM90_ALL ... #else error flag` is realised as the scaffold's `#error` (loud build failure -> master), since CUTLASS itself is only usable on sm_90a.
- IQR rule has a 0.5 us absolute floor (event timer resolution), else a master IQR of 0 makes the rule unsatisfiable.
- The gate runs at m=128 only (the scored cell); the 65..127 band is routed by the same flag on the strength of the m 65/100/128 self-checks, untimed.
- Deadlines: build skipped if import took > 180 s; phase skipped (prints "skipped: deadline") if state A's build is > 150 s in; each layer skipped if > 200 s in.
- Nothing could be compiled or run on this node (no nvcc/GPU/CUTLASS); the CUTLASS scheduler member names (`splits`, `decomposition_mode`) are detected by SFINAE, and if absent the split mode reports not applied and the layers are DISABLED (master).

## Rules check

- [x] CUDA-graph capturable: kernel_fn adds one branch calling `cut128_gemm` (host-side TMA descriptor build, stream memset, cluster launch), a fixed-shape torch.empty, no .item()/.cpu()/sync; branches on cfg, shapes and state flags only. Same structure as stage 1, which was graph-captured by the judge.
- [x] Nothing derived from `data` is cached: workspace and splits are weight/shape constants made in prepare(); descriptors are rebuilt per call from the live pointers.
- [x] Weights: untouched (read in place as [N, K] bf16, no repack, no dtype change).
- [x] Activations BF16, output bf16 as SGLang's; fp32 accumulation.
- [x] n/a: attention rules (no attention in this problem).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (load_inline compile only, no child process of our own, no subprocess import).
- [x] All work inside the timed call; no output caching or replay (checked against `git -C .worktrees/muse_bf16_proj/bf16-verify16-cutlass-m128-onetile-splitk-rewrite diff`: new branch returns the fresh CUTLASS output).
- [x] Per-layer flags, own workspace per state, a miss falls back to master's exact path and never disables an unrelated flag (state B inherits via _ab_flags / flags["_cut_*"]).

okc_lint (warnings only, all in prepare()-time code; tail of output):
```
warn  host-sync    L1452: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1498: flush = torch.empty(OUNS_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L1507: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1513: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1517: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1546: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1567: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1613: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2013: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2647: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2650: code = int(err.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2679: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2683: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2700: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2709: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2729: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2738: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2741: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2807: code = int(state["bulk_err"].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2838: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2851: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2876: state["lt_ws_qkv128"] = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  dtype        L2887: ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2892: algo_id, tile, stg, splitk, red, ws_need, waves, st = (int(v) for v in info[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2898: key = tuple(int(v) for v in algos[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2903: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2950: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3385: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L3394: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3427: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3604: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3791: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3795: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3808: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3847: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_bf16_proj/bf16-verify16-cutlass-m128-onetile-splitk-rewrite/muse_bf16_proj/submission.py
```
py_compile: OK; __pycache__ removed.

## Pre-flight audit

Verdict: dispatch (no fixes). Read-only audit; no CUDA run possible here.
- Scheduler API: no CUTLASS gemm/ headers on disk (only a partial 4.8.1 cute/cutlass tree without gemm/). The judge image's flashinfer tree is CUTLASS 4.5.0 (log f5b1361a, built in 49.0 s, same scaffold, `scheduler` arg, initialize_workspace / to_underlying_arguments / Gemm::run all compiled there). `splits`, `decomposition_mode`, `reduction_mode` are StreamK scheduler Arguments members in 3.x/4.x; each is SFINAE-detected, so a missing member gives applied=0 and the layer prints DISABLED. Only `DM::SplitK` is a hard reference, and a miss is a compile error caught by load_inline's except, so _CUT_EXT=None and master everywhere. Nothing raises past the guards.
- Split fallback: qkv 7 -> 8 -> 4 is only reached if can_implement or split mode fails; 8 gives 144 units, and the grid<=132 check rejects it if CUTLASS does not cap it, else the correctness check and 1.03x gate decide. Waves: /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/audits26/b16_waves.py (N-tile counts all even, so cluster 1x2x1 fits; qkv 7 splits is uneven over 128 k-tiles, covered by the self-check at m 65/100/128).
- Shadowing: the cut route sits after every earlier landed route, all m<=32 or m==1, and requires 65<=cfg m<=128 plus per-layer flag, so nothing outside 65..128 reaches it and it only preempts nn_ok / Lt / cuBLAS in-band; the gate races exactly those.
- Determinism: reduction_mode=Deterministic set (fp32 workspace accumulate, fixed order); the self-check asserts run2 bit-identical.
- Deadlines: _MODULE_T0 is set after import torch/triton/sglang, and the cut build runs last at module level so earlier builds count against its 180 s cap; master prepare is about 4 s (logs), build about 49 s, caps 180/150/200 s against a 600 s budget. IQR veto has a 0.5 us floor.
- Lint OK, py_compile OK, __pycache__ removed.
Residual risk: a device fault or deadlock in the deterministic split-K fixup with cluster 1x2x1 would crash prepare rather than fall back.
