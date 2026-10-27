---
problem: muse_bf16_proj
slug: bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush
created: 2026-10-27T132819Z
base: master
area: bf16 GEMM dispatch (m=65..256, DeepGEMM sm90 BF16 kernel)
change: >
  Stage 2 (job 566a79e4) measured NOTHING: its generator filled DeepGEMM's sm90_bf16_gemm_impl
  template by parameter NAME and stopped at 'unknown template parameter BLOCK_K_' (log: '[dg]
  signature mismatch ... BLOCK_K_'), so every cell ran master. The header on the judge (deep_gemm
  0.2.1.post1+muse) declares 21 template params in this order: kMajorA, kMajorB, SHAPE_M, SHAPE_N,
  SHAPE_K, kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K_, kSwizzleAMode, kSwizzleBMode, kSwizzleDMode,
  kNumStages_, kNumTMAThreads, kNumMathThreads, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs,
  kGemmType, kWithAccumulation, cd_dtype_t; and 7 kernel params: grouped_layout, shape_m, shape_n,
  shape_k, tensor_map_a, tensor_map_b, tensor_map_cd. CHANGE: port stage 2's [dg] section
  (done/neutral/bf16-verify-deepgemm-sm90-explicit-instantiation-stage2.patch: _dg_parse_kernel,
  _dg_gen_source, _dg_make_variants, the tensor-map encode, canary, gate, FINAL PATH CHOSEN lines)
  onto CURRENT master as a new last section, and fix the name map: in _dg_gen_source look a name
  up as-is, then with one trailing underscore stripped (BLOCK_K_ -> BLOCK_K, kNumStages_ ->
  kNumStages); kGemmType = GemmType::Normal, kNumGroups 1, kWithAccumulation false, cd_dtype_t
  cutlass::bfloat16_t, kMajorA/B K. Print the generated template-argument list once. Replace stage
  2's bytes/flops cost model (it chose BLOCK_M 64 at m=128 with 'model 26.2 us' for pooler, worse
  than the incumbent before running) with a small measured pre-screen: per cell up to 3 variants
  from {BLOCK_M 128 with 256 math threads, BLOCK_M 64 with 128}, BLOCK_N in {128, 256 when it
  divides n and 112/144/160 for qkv's 2304 when allowed by the header's static_asserts},
  kNumTMAMulticast 2 on A when the tile count is even else 1, stages filling 227 KB; all variants
  in ONE load_inline TU built for sm_90a only with -std=c++20 (LESSONS 2026-10-25), build-time
  budget printed, a build over 150 s or a failure keeps master everywhere. TMA descriptors per
  state, never module-global (LESSONS cuda-nvfp4 s3: the judge prepares two state copies). Cells
  and incumbents (each its own flag dg_<layer>_<m>_ok, kernel_fn branch placed BEFORE the route it
  replaces): qkv_gate m=128 (incumbent cell B Lt plan), qkv_gate m=256 (dispatch), pooler m=256
  (NN torch.mm, nn_p256_ok), unpooler m=128 (NN torch.mm, nn_ok) and m=256 (dispatch); pooler
  m=128 stays on khalf/NN. Per variant: canary (synced launch, error-free), _cell_check (rel L2 <=
  3e-3 vs fp64, probe row exact, bitwise repeat), read-flush 15-rep pre-screen; the best variant
  per cell then takes two read-flush _cell_gate rounds against the actually routed incumbent
  (master/cand >= 1.03 both, cand IQR <= 1.5x master's), and _cell_kernels must show exactly one
  launch named sm90_bf16_gemm_impl (LESSONS: PATH CHOSEN proves the selector, not the kernel).
  AB_CASES / copy check across both states. Print per cell, compactly at the end of prepare: each
  variant's net us, TFLOP/s and TB/s against the incumbent, and PATH CHOSEN DG.
expected:
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: incumbent nvjet 96x64 2x4 fills only ~96-108 CTAs at 493 TFLOP/s (floor 9.77 us, compute); a persistent 128xN multicast wgmma kernel at ~650 TFLOP/s is ~15 us vs 19.97 row
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: "nvjet 128x64 NNT 29.0 us (593 TFLOP/s) vs the same-FLOPs unpooler at 24.7 us on 128x128: a persistent kernel that keeps 128x128-class tiles busy on 132 SMs could reach ~25-26 us"
  - shapes: layer=unpooler load=verify_32 m=256
    direction: flat
    why: incumbent already ~700 TFLOP/s; routes only at >= 3%
  - shapes: layer=unpooler load=verify_16 m=128
    direction: faster
    why: NN 128x64 19.1 us at ~3.5 TB/s vs floor 14.64; one M tile per CTA reads W once
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: flat
    why: cell B Lt plan is the incumbent; routes only at >= 3%
  - shapes: layer=* load=decode_* m=*
    direction: flat
    why: m <= 32 untouched
risk: high
depends_on: bf16-verify-deepgemm-sm90-explicit-instantiation-stage2
priority: 2
kind: rewrite
controls_for: null
prior: "bf16-verify-deepgemm-sm90-explicit-instantiation-stage2: it was VOID (no kernel ever built, all cells master) because the name-keyed generator did not know BLOCK_K_ / kNumStages_; this stage adds the underscore alias, replaces its bytes/flops tile model (which picked BLOCK_M 64 configs modelled slower than the incumbents) with a measured read-flush pre-screen of 2-3 variants per cell, and gates on the read flush in two rounds on current master"
rewrite_stage: 3
replaces: cuBLAS nvjet at m=128/256 for qkv_gate (Lt 40x64 15.7 us at m=128; 96x64 2x4 TNN 19.6 us at m=256), pooler m=256 (NN 128x64 29.0 us) and unpooler (NN 128x64 19.1 us at m=128; 128x128 TNT 24.7 us at m=256)
review: proceed
status: done
review_at: 2026-10-27T133822Z
review_companions: null
stage: harvested
claimed_at: 2026-10-27T133822Z
base_sha: 0dc50d0a5d027659c02f0e6a45778fa2662139e4
base_blob: ff4b91b7ccf2b0111c4ed8a900c2f48cfed8d73b
branch: hyp/muse_bf16_proj/bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush
worktree: .worktrees/muse_bf16_proj/bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-27T140050Z
impl_commit: a88d68ed70f2425b659e5218d5c621ee90d39a1b
submission_sha256: ef7265da3cefafe9804489586cb45aa8636ff5175dc102b3bc2b714e699dbe7b
norm_sha256: 66d556a6d10ef4f52706b5cd65aee5f750935e39b28afb4fd38c297e5ea11e27
submission_blob: 73b304b6996a68f9ae633659e65f1be5e31247ac
lint_warnings: 50
attempt_file: 2026-10-27T140051Z-check-bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush.md
prepared_at: 2026-10-27T140051Z
job: 6407497bec88470e
dispatched_at: 2026-10-27T140119Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.274087889871094
speedup_ci: [1.2549765715230274, 1.2931992082191601]
harvested_at: 2026-10-27T140553Z
best_at_harvest: 1.273478298751914
promotion: null
---

# bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush

## Change

Stage 2 (job 566a79e4) measured NOTHING: its generator filled DeepGEMM's sm90_bf16_gemm_impl template by parameter NAME and stopped at 'unknown template parameter BLOCK_K_' (log: '[dg] signature mismatch ... BLOCK_K_'), so every cell ran master. The header on the judge (deep_gemm 0.2.1.post1+muse) declares 21 template params in this order: kMajorA, kMajorB, SHAPE_M, SHAPE_N, SHAPE_K, kNumGroups, BLOCK_M, BLOCK_N, BLOCK_K_, kSwizzleAMode, kSwizzleBMode, kSwizzleDMode, kNumStages_, kNumTMAThreads, kNumMathThreads, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, kGemmType, kWithAccumulation, cd_dtype_t; and 7 kernel params: grouped_layout, shape_m, shape_n, shape_k, tensor_map_a, tensor_map_b, tensor_map_cd. CHANGE: port stage 2's [dg] section (done/neutral/bf16-verify-deepgemm-sm90-explicit-instantiation-stage2.patch: _dg_parse_kernel, _dg_gen_source, _dg_make_variants, the tensor-map encode, canary, gate, FINAL PATH CHOSEN lines) onto CURRENT master as a new last section, and fix the name map: in _dg_gen_source look a name up as-is, then with one trailing underscore stripped (BLOCK_K_ -> BLOCK_K, kNumStages_ -> kNumStages); kGemmType = GemmType::Normal, kNumGroups 1, kWithAccumulation false, cd_dtype_t cutlass::bfloat16_t, kMajorA/B K. Print the generated template-argument list once. Replace stage 2's bytes/flops cost model (it chose BLOCK_M 64 at m=128 with 'model 26.2 us' for pooler, worse than the incumbent before running) with a small measured pre-screen: per cell up to 3 variants from {BLOCK_M 128 with 256 math threads, BLOCK_M 64 with 128}, BLOCK_N in {128, 256 when it divides n and 112/144/160 for qkv's 2304 when allowed by the header's static_asserts}, kNumTMAMulticast 2 on A when the tile count is even else 1, stages filling 227 KB; all variants in ONE load_inline TU built for sm_90a only with -std=c++20 (LESSONS 2026-10-25), build-time budget printed, a build over 150 s or a failure keeps master everywhere. TMA descriptors per state, never module-global (LESSONS cuda-nvfp4 s3: the judge prepares two state copies). Cells and incumbents (each its own flag dg_<layer>_<m>_ok, kernel_fn branch placed BEFORE the route it replaces): qkv_gate m=128 (incumbent cell B Lt plan), qkv_gate m=256 (dispatch), pooler m=256 (NN torch.mm, nn_p256_ok), unpooler m=128 (NN torch.mm, nn_ok) and m=256 (dispatch); pooler m=128 stays on khalf/NN. Per variant: canary (synced launch, error-free), _cell_check (rel L2 <= 3e-3 vs fp64, probe row exact, bitwise repeat), read-flush 15-rep pre-screen; the best variant per cell then takes two read-flush _cell_gate rounds against the actually routed incumbent (master/cand >= 1.03 both, cand IQR <= 1.5x master's), and _cell_kernels must show exactly one launch named sm90_bf16_gemm_impl (LESSONS: PATH CHOSEN proves the selector, not the kernel). AB_CASES / copy check across both states. Print per cell, compactly at the end of prepare: each variant's net us, TFLOP/s and TB/s against the incumbent, and PATH CHOSEN DG.

## Expected

- `layer=qkv_gate load=verify_32 m=256` faster: incumbent nvjet 96x64 2x4 fills only ~96-108 CTAs at 493 TFLOP/s (floor 9.77 us, compute); a persistent 128xN multicast wgmma kernel at ~650 TFLOP/s is ~15 us vs 19.97 row
- `layer=pooler load=verify_32 m=256` faster: nvjet 128x64 NNT 29.0 us (593 TFLOP/s) vs the same-FLOPs unpooler at 24.7 us on 128x128: a persistent kernel that keeps 128x128-class tiles busy on 132 SMs could reach ~25-26 us
- `layer=unpooler load=verify_32 m=256` flat: incumbent already ~700 TFLOP/s; routes only at >= 3%
- `layer=unpooler load=verify_16 m=128` faster: NN 128x64 19.1 us at ~3.5 TB/s vs floor 14.64; one M tile per CTA reads W once
- `layer=qkv_gate load=verify_16 m=128` flat: cell B Lt plan is the incumbent; routes only at >= 3%
- `layer=* load=decode_* m=*` flat: m <= 32 untouched

## Risk

Milestones (judged on the replaced kernels, not the headline): M1 the TU compiles on the judge for every variant, each canary and _cell_check passes, and the pre-screen prints net us per cell (a run that prints no DG net us is VOID again, not a measurement); M2 qkv m=256 or pooler m=256 within 1.05x of its incumbent in the read-flush pre-screen; M3 one cell routes (>= 1.03 in both rounds) and the profile row shows sm90_bf16_gemm_impl. Fallback at every step: master. Ceiling if M3 on qkv m=256 and pooler m=256: verify_32 -5..-8 us (0.184 %/us, ~+1-1.5% headline); decode-portable at T=128/256 (decode's T>32 dense path is SGLang's kernels verbatim, wave6-panel2 L1). Risks: further header surprises (a static_assert on BLOCK_N for n=2304; a required kNumLastStages-like param), compile time inside the 10-minute run (keep variants <= 8, one TU), register spills (print the spill line, LESSONS prefill wgmma), wrong-copy TMA maps. Run as an okc check only. The five earlier m>=128 custom kernels (CUTLASS 0.50-0.78x, Triton, swap-AB cluster) lost in their mainloops; DeepGEMM is the one production-tuned persistent sm90 kernel on the image that has never been timed here.

## Milestones

Rewrite stage 3 of `cuBLAS nvjet at m=128/256 for qkv_gate (Lt 40x64 15.7 us at m=128; 96x64 2x4 TNN 19.6 us at m=256), pooler m=256 (NN 128x64 29.0 us) and unpooler (NN 128x64 19.1 us at m=128; 128x128 TNT 24.7 us at m=256)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-27T133822Z by an adversarial subagent (model opus).

**Verdict: proceed.** A real stage 3: stage 2's log (566a79e4 line 19) shows the run was VOID at 'unknown template parameter BLOCK_K_', so the alias plus a measured pre-screen is the right next step, and the rules are clean. Two spec errors: its variant set cannot reach the qkv m=256 and unpooler m=128 targets it names, and it reintroduces TMA multicast, which the stage-1 audit removed as an unbounded-hang source. Fix both in the build; no companion.

### Justification
No DG kernel has ever been built or timed. This kernel has no split-K, so tile count decides everything. pooler m=256 is the strongest cell: 64x128 gives 128 tiles in one wave (~21-25 us realistic vs incumbent NN ~29.0 us; needs <= 28.1). 128x128 gives 64 tiles and can only lose. qkv m=256 (n=2304): every 128-row variant gives 32-42 tiles (>= 31 us vs 19.6 us); a win needs BLOCK_M 64 with BLOCK_N 64/80/96/112 (64x112 is marginal). unpooler m=128: only 128x64 (128 tiles, W read once, ~16-17.5 us) matches the stated mechanism; BLOCK_N 64 is not in the set. unpooler m=256 flat; qkv m=128 dead on arrival (18-36 tiles vs Lt 15.7 us). Ceiling ~+1-1.5% headline, at the readability line; judge on pre-screen net-us lines, the two gate rounds and profile rows naming sm90_bf16_gemm_impl. 15 instantiations contradicts the risk section's <= 8: use SHAPE_M=0 (runtime M supported) to share m=128/256 builds. Multicast: a box/expect-tx mismatch is an unbounded mbarrier hang in DeepGEMM's header that would cost the whole run; the 'tile count even' legality rule is also wrong (it needs ceil(n/BLOCK_N) % 2 == 0, an even grid and a cluster launch attribute). Make this stage mc=1 only.

### Confounders on the current kernel
- pooler-nn-layout-widen-m256 (nn_p256_ok) is the pooler m=256 incumbent: the DG branch must sit above that block (and above _khalf_route_ok), else the gate reads ~1.0; print which incumbent the gate actually ran.
- restack-three-cell-wins (nn_ok, pooler/unpooler 65..128): _ab_component maps unpooler 65..128 to nn_ok; map dg_unpooler_128_ok first (as KHALF_FLAG is), else a DG miss turns nn_ok off.
- combo-rowowner-reduce-free (Lt cell B) is the qkv m=128 incumbent; it makes that cell hopeless.
- sglang-gemm-dispatch-parity: the fallback is the qkv/unpooler m=256 incumbent; a direct DG vs nvjet comparison.
- master 8f233e88: KHALF_START_DEADLINE_S=170 s; the DG section must run AFTER _khalf_prepare, or a long nvcc build skips khalf and regresses pooler m=128.
- m<=32 / router items: no shared traffic; do not credit or blame untouched rows.

### Interference and A/B needs
No companion: each target cell attributes itself (pre-screen net us per variant, the two-round gate, profile rows naming sm90_bf16_gemm_impl). Drop mc=2 rather than spend a control on it. The real risks are non-results (a multicast hang, no viable tile count, DG placed below the NN/khalf/Lt returns, a build overrun): fix them in the build. Queued lt-config-sweep, pooler-khalf-lt-batched-plan and lt-nn-layout-cells compete for the same cells; a later restack picks per cell by the printed gate numbers.

### Open questions
- Replace the fixed grid with a tile-count pre-filter (>= ~0.7 x 132 tiles): pooler m=256 64x128; unpooler m=128 128x64 and 64x128; qkv m=256 64x{64,80,96,112} if static_asserts allow; drop qkv m=128?
- SHAPE_M=0 to share m=128/256 builds (<= 8 instantiations); split the TU for parallel ninja; a pre-build deadline (start only if prepare < ~120 s); print [build-time].
- Self-check: also print the incumbent's own rel L2 and require DG <= 1.5x of it.
- Keep pre-screen and gate lines compact; repeat the per-cell summary as the last prepare lines.
- Reject any variant with local-memory spill > 0 or regs > 168 before the canary.


## Implementation notes

Implemented 2026-10-27 in the worktree's `muse_bf16_proj/submission.py` (not committed or staged). The worktree is +934 / -2 lines against master 8f233e88's file.

**Stale base (step 0).** 0dc50d0a's file is master 36c26199's file (blob ad8d7254) plus stage 2's DG section: `git diff 36c26199 0dc50d0a` has only DG hunks. Its commit parent e4008591 is stage 1 on an older base, so neither `git merge master` (merge-base e5548a78 holds a 1079-line file) nor re-applying `0dc50d0a~1..0dc50d0a` would give a clean 3-way. Instead I ran `git merge-file -p master.py base(36c26199).py stage2(0dc50d0a).py`, which left 6 conflicts. I resolved each by hand:
- module docstring: kept both paragraphs;
- `_prepare_state` state A: `_khalf_prepare` first, then `_dg_prepare`;
- `_ab_component`: kept master's `st` signature, khalf and b12 checks, with the dg flag check placed first;
- `_ab_selfcheck`: kept both `fl["_b12_inst"]` and `fl["_dg_route"]`;
- prepare() tail: master's DECISIONS reprint, then `_dg_final_summary`;
- kernel_fn: the DG branch sits ABOVE `_khalf_route_ok`.

After the merge, every master line was intact. Only the docstring, master's `NEW_CELL_FLAGS` literal, the state-B tick label and the `_ab_component` docstring differed. Stage 3 then restored the `NEW_CELL_FLAGS` literal and appends `DG_FLAGS` at the end of the section, the way section (11) appends `KHALF_FLAG`.

**What changed.** The stage-2 DG section was replaced by section (12), placed after section (11) and before `_prepare_state`. All its constants live there now.
- `_dg_targs`: looks each template parameter name up as-is, then with one trailing underscore stripped (BLOCK_K_ -> BLOCK_K, kNumStages_ -> kNumStages). It sets kGemmType `deep_gemm::GemmType::Normal`, kNumGroups 1, kWithAccumulation false, cd_dtype_t `cutlass::bfloat16_t`, kMajorA/B `cute::UMMA::Major::K`, kNumTMAMulticast 1, kIsTMAMulticastOnA false, SHAPE_M `0u` and kNumSMs 132. The generated `name=value` list of v0 is printed once (`[dg] template args ...`).
- Variants: a fixed set chosen by tile count (review open question 1). The cost model and `_dg_find_jit_template` are dropped, and the per-cell candidates are in `DG_CELL_TILES`. Stages fill 227 KB but are capped at 9, so the header's kDoMergeStages path (>= 10 stages, 128 math threads) never runs and the TMA box stays BLOCK_K_ = 64. Multicast is 1 everywhere, with no cluster launch attribute.
- Build: three `load_inline` modules (g0/g1/g2), started at the same time with `threading.Thread`.
  - load_inline joins all `cuda_sources` into one cuda.cu, so splitting into 2-3 cuda_sources would NOT let ninja compile them in parallel. Separate modules do.
  - Each module has a torch-free `.cu` (the DeepGEMM header plus static launchers and setup functions) and a host `.cpp`: descriptor encode, checks, bindings, `dg_run` and `dg_weight_desc`.
  - A module that fails drops only its variants. The two 32-byte-swizzle-D variants (BLOCK_N 80 / 112) are isolated in g2.
  - Flags: sm_90a only (TORCH_CUDA_ARCH_LIST=9.0a plus an explicit gencode), `-std=c++20`, DeepGEMM's `--diag-suppress` list.
  - The build starts only if state build < 120 s and module clock < 260 s. Over 150 s of wall time keeps master everywhere. `[build-time]` is printed per module and in total.
- Per variant, before the canary: `cudaFuncGetAttributes`. Reject on sass != sm_90, local bytes > 0, or regs > 168 at 256 math threads. Then the per-state weight descriptor, and a synced canary at m=128 and 256 under the host watchdog (10 s), checked for finite output.
- Per cell (`_dg_cell`):
  - incumbent kernels profiled with `_cell_kernels` (flag off);
  - incumbent rel L2 vs fp64 on a seeded x;
  - per variant: `_cell_check` (rel L2 <= 3e-3, probe row, bitwise repeat), then DG rel L2 <= 1.5x the incumbent's on the same x;
  - `_dg_prescreen`: 15 read-flush (`DG_GATE_FLUSH = B12_GATE_FLUSH`) graph replays per variant, interleaved with the incumbent in rotated order, empty-graph baseline subtracted;
  - the best variant takes two `_cell_gate(flush_mode=B12_GATE_FLUSH, stats=...)` rounds against kernel_fn with the flag off. It routes only if master/cand >= 1.03 in both rounds and cand IQR <= 1.5x max(master IQR, 0.5 us);
  - `_cell_kernels` with the flag on must show exactly one launch containing `sm90_bf16_gemm_impl`.
- Cells and flags: `dg_pooler_256_ok`, `dg_qkv_gate_256_ok`, `dg_unpooler_128_ok`, `dg_unpooler_256_ok`. They are mapped in `_ab_component` before khalf / nn_ok / nn_p256_ok / Lt, and checked by the A/B copy check (AB_CASES has all four).
- State B re-encodes the weight descriptors over its own weights. A route entry is (module, local index, variant id); there are no module globals. x and out descriptors are encoded on every `dg_run` call from pointer and shape, with no memo.
- Log tail: one compact `PRE` line per cell, master's one-line `_cell_gate` print per round, and one `CELL` line per cell: variant, stages, regs/spill, pre-screen net us / TFLOP/s / TB/s for every variant and the incumbent, gate ratios, PATH CHOSEN with the profiled DG kernel, and the incumbent's profiled kernels. After FINAL FLAGS and master's DECISIONS, `_dg_final_summary` re-prints the build, variants and CELL lines and one `[dg] FINAL` flag line (post-A/B) as the last lines of prepare.

**Variant / instantiation list (8, SHAPE_M = 0, mc = 1).** Tiles are at m=128 / m=256.

| id | layer | BLOCK_M x BLOCK_N | math thr | stages | swizzle-D | smem | tiles | module |
|---|---|---|---|---|---|---|---|---|
| v0 | pooler | 64x128 | 128 | 8 | 128 | 217088 | 64 / 128 | g0 |
| v1 | unpooler | 128x64 | 256 | 8 | 128 | 217088 | 128 / 256 | g0 |
| v2 | unpooler | 64x128 | 128 | 8 | 128 | 217088 | 128 / 256 | g0 |
| v3 | unpooler | 128x128 | 256 | 5 | 128 | 200704 | 64 / 128 | g1 |
| v4 | qkv_gate | 64x64 | 128 | 9 | 128 | 159744 | 72 / 144 | g1 |
| v5 | qkv_gate | 64x96 | 128 | 9 | 64 | 200704 | 48 / 96 | g1 |
| v6 | qkv_gate | 64x80 | 128 | 9 | 32 | 180224 | 58 / 116 | g2 |
| v7 | qkv_gate | 64x112 | 128 | 9 | 32 | 221184 | 42 / 84 | g2 |

Cell candidates:
- pooler m=256: v0
- unpooler m=128: v1, v2
- unpooler m=256: v3, v1, v2
- qkv_gate m=256: v4, v6, v5, v7

**Deviations from the md's `change`, all per the review or the caller's binding notes:**
- no multicast;
- tile-count variant set instead of the {BM 128/64} x {BN} grid;
- qkv_gate m=128 and pooler m=128 are not cells;
- m=130/200 are not self-checked, because nothing routes there (exact (layer, m) cells only);
- three modules instead of ONE TU, so the builds run in parallel and a failure is isolated;
- stages capped at 9 (my addition, see Risks).

## Rules check

Each item below was checked against `git -C .worktrees/muse_bf16_proj/bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush diff` and against the worktree file compared with master.

- [x] CUDA-graph capturable. The new kernel_fn path is `_dg_route_ok` (it reads cfg m/n, x shape, dtype, contiguity, and state flags and route) followed by `_dg_forward`. `_dg_forward` does torch.empty sized by x.shape[0], then `dg_run`, which encodes two TMA descriptors on the host and calls cudaLaunchKernelEx on the current stream. It has no `.item()`, `.cpu()` or synchronize, no tensor prints, and no branch on input values; its TORCH_CHECKs read shapes and dtypes only. Every synchronize, `ev.synchronize`, `.isfinite().all()` and watchdog poll is in prepare-only functions (okc_lint confirms they are outside kernel_fn's call graph).
- [x] Nothing derived from `data` is cached across calls. The x and out descriptors are rebuilt inside every `dg_run` call from data_ptr and shape and never stored. The one C++ static is the cuTensorMapEncodeTiled driver entry point. The weight descriptors are per-state CPU tensors built in prepare from the state's own weights.
- [x] Weights: BF16 weights are used as they are (state[layer], contiguous [n, k]). DG only builds TMA descriptors over them, and no copy or precision change is made. The router is untouched.
- [x] Activations BF16 where SGLang keeps BF16: x is bf16, out is bf16 (cd_dtype_t cutlass::bfloat16_t), and accumulation is fp32 in wgmma. KV cache and queries: n/a (dense projections, no attention).
- [x] Attention sees every key: n/a (no attention in muse_bf16_proj).
- [x] No timing patches, harness frames, harness I/O or secrets, module edits, reference import, subprocesses or network.
  - The build is torch `load_inline`, the same mechanism master already uses five times.
  - Its three builds run concurrently from `threading.Thread` workers inside prepare(). These are threads, not processes, and they only wait on load_inline.
  - The only file reads are DeepGEMM's own installed header and a bounded `os.walk` for cutlass includes in site-packages (a fallback). There are no writes beyond load_inline's own build directory.
- [x] All work happens inside the timed call. Each routed call launches exactly one sm90_bf16_gemm_impl (asserted with the profiler before routing), and there is no output caching or replay.
- [x] Shape gating only: DG routes exactly (pooler, 256), (qkv_gate, 256), (unpooler, 128) and (unpooler, 256), each under its own flag. The m = 130 / 512 / 1000 tests, probe rows, m <= 32, o and the router keep master's paths.
- [x] Two state copies: each state has its own `dg_wdesc` over its own weights, and state B re-encodes them from the inherited routes. There are no module-global descriptors (LESSONS cuda-nvfp4 s3).

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`) exits 0: **OK with warnings**. It gives 51 warnings, and master's file gives 46. The 5 new ones are prepare-only:
- the read-flush buffer `torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8)` in `_dg_prescreen`, a copy of master's `_cell_gate`;
- `ev[1].synchronize()` and `torch.cuda.synchronize()` in `_dg_prescreen`;
- `torch.cuda.synchronize()` at the end of `_dg_prepare_a`;
- the watchdog's final `torch.cuda.synchronize()` in `_dg_sync`.

Full output:

```
warn  host-sync    L1456: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1502: flush = torch.empty(OUNS_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L1511: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1517: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1521: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1550: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1571: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1616: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2016: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2637: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2640: code = int(err.item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2669: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2673: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2698: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2712: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2737: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2746: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2749: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2815: code = int(state["bulk_err"].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2846: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2859: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2884: state["lt_ws_qkv128"] = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  dtype        L2895: ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2900: algo_id, tile, stg, splitk, red, ws_need, waves, st = (int(v) for v in info[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L2906: key = tuple(int(v) for v in algos[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L2911: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L2958: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L3440: lf = frag(lo_enc).to(torch.uint8).reshape(ctas, nch, ns, mt, 256)  (allowed only where SGLang does the same)
warn  dtype        L3442: cwd = (cf[..., 0:4] | (cf[..., 4:8] << 4)).to(torch.uint8).reshape(ctas, nch, ns, mt, 128)  (allowed only where SGLang does the same)
warn  dtype        L3447: blocks = torch.zeros((ctas, B12_ESC_BYTES), dtype=torch.uint8)  (allowed only where SGLang does the same)
warn  host-sync    L3448: per_cta = torch.bincount((idx[:, 0] // r).cpu(), minlength=ctas) if n_esc else torch.zeros(ctas, dtype=torch.int64)  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3453: rr, kk = idx[:, 0].cpu(), idx[:, 1].cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3454: vals = bits[idx[:, 0], idx[:, 1]].cpu()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3457: for c in torch.nonzero(per_cta).flatten().tolist():  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3498: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3610: code = int(state["b12_err"].item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3665: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3804: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3808: nbad = int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3841: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3893: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L3993: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L4516: flush = torch.empty(CELL_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L4529: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4545: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L4773: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5007: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5011: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5024: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L5064: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_bf16_proj/bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush/muse_bf16_proj/submission.py
```

Other checks:
- `python3 -I -m py_compile`: OK. No `__pycache__` is left in the worktree.
- The CUDA syntax tool (`.worktrees/b12p-fix/rewrite/bin/cuda_syntax.py`) scans only module-level strings that contain `__global__` or `__device__`.
  - It reports `_BULK_CUDA` ok and `_B12_CUDA` ok.
  - It reports `_OUNS_CUDA` and `_RC_CUDA` as FAIL. Both are master's, unchanged, and run on the judge; the tool's stub lacks `cudaOccupancyMaxActiveClusters`.
  - The DG sources are generated at runtime, so the tool does not see them.
- Instead, I extracted section (12) and exec'd it with stub globals. I parsed a stand-in header that carries the judge's printed 21-parameter template and 7-parameter kernel signature, then generated all three modules.
  - g++ `-std=c++20 -fsyntax-only -Wall`, using the same `cuda_stub` / `torch_stub` plus a stub `deep_gemm` namespace, passes on all 6 sources (g0/g1/g2, both .cpp and .cu).
  - That includes the stub's `cudaLaunchKernelEx` static_asserts on argument count and convertibility.
  - The parse gave the expected 21 tparams (BLOCK_K_ and kNumStages_ resolved through the alias), 7 fparams, runtime_m True and merge-stages True.

### Implementation uncertainties (none can be run here: no GPU or nvcc)
- **Concurrent load_inline.** Three load_inline calls run from threads, with distinct names and build dirs and TORCH_CUDA_ARCH_LIST set once before the threads start. torch's JIT does not document thread-safety. If a thread errors, only that module's variants are lost and its cells run master. The judge's static check has passed `load_inline` before but has never seen `threading`.
- **Host-side TMA assumptions.**
  - A box (64, BLOCK_M) and B box (64, BLOCK_N), both with 128-byte swizzle. These match the kernel's expect-tx only when there is no multicast and no stage merge, which this stage ensures.
  - D box (swizzleD/2 or BLOCK_N, BLOCK_M), following DeepGEMM's host code.
  - A wrong D box gives wrong output, which `_cell_check` catches. A wrong A/B box can hang: the watchdog names it, but the run is lost.
- **Template values not seen in a successful build.** cd_dtype_t `cutlass::bfloat16_t` and kGemmType `deep_gemm::GemmType::Normal` match DeepGEMM's to_string. A compile error here costs only that module.
- **Stage cap.** Capping stages at 9 keeps the 64-row variants off DeepGEMM's merge-stages path, which may cost a few % on 64xN tiles. Stage 4 can lift the cap once a variant is shown to run.
- **Below the tile-count rule.** qkv 64x64 at m=256 (144 tiles, 2 waves, 0.55 fill) and 64x112 (84 tiles, 0.64) are under the ~0.7 rule. They are included because the caller listed them explicitly, and the pre-screen decides.
- **Local-bytes rejection.** `localSizeBytes > 0` rejects a stack frame as well as a spill. That is stricter than "spill > 0", as the review asked.
- **Build time.** Unknown: 8 instantiations, about 3 per module, the modules in parallel. A failed or rejected build is printed and keeps master; it is not VOID-silent.

## Result

- job `6407497bec88470e` (check), status done, passed True, tests 9/9
- speedup 1.2741x, 95% CI [1.2550, 1.2932], score 3791.5751640090994 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.273478298751914, "ci": [1.2543761242706353, 1.2925804732331925], "job": "bc147c5b87bc465d"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-verify-deepgemm-sm90-explicit-instantiation-stage2.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush.patch`; branch `hyp/muse_bf16_proj/bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 10.94 | 10.59 | 1.0233 | flat |
| layer=o load=decode_1 m=1 | 62 | 6.19 | 5.95 | 1.0092 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 17.47 | 14.62 | 1.176 | gained |
| layer=unpooler load=decode_1 m=1 | 62 | 5.03 | 5.05 | 0.951 | lost |
| layer=router load=decode_1 m=1 | 62 | 4.86 | 5.1 | 0.9494 | lost |
| layer=qkv_gate load=decode_16 m=16 | 62 | 12.81 | 13.14 | 0.9983 | flat |
| layer=o load=decode_16 m=16 | 62 | 5.78 | 6.11 | 0.9646 | lost |
| layer=pooler load=decode_16 m=16 | 62 | 19.11 | 19.46 | 0.9674 | lost |
| layer=unpooler load=decode_16 m=16 | 62 | 18.27 | 18.54 | 0.9783 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.56 | 5.68 | 0.9758 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.82 | 14.12 | 0.9898 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.61 | 6.6 | 0.9599 | lost |
| layer=pooler load=decode_32 m=32 | 62 | 19.6 | 19.9 | 0.9726 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.2 | 17.98 | 0.9825 | flat |
| layer=router load=decode_32 m=32 | 62 | 5.79 | 6.1 | 0.9797 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.79 | 12.58 | 0.9996 | flat |
| layer=o load=verify_1 m=8 | 62 | 6.07 | 5.89 | 1.0253 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 18.75 | 19.05 | 0.9719 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 5.26 | 5.29 | 1.0089 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.11 | 5.08 | 1.0139 | flat |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.27 | 15.49 | 1.07 | gained |
| layer=o load=verify_16 m=128 | 62 | 7.32 | 7.33 | 1.0301 | gained |
| layer=pooler load=verify_16 m=128 | 62 | 22.8 | 23.06 | 1.0336 | gained |
| layer=unpooler load=verify_16 m=128 | 62 | 18.77 | 18.85 | 0.9877 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.68 | 7.39 | 1.0499 | gained |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.68 | 19.3 | 1.0236 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.25 | 9.11 | 1.0239 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 30.03 | 28.8 | 1.0073 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 23.99 | 23.77 | 1.0187 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.41 | 8.99 | 1.0396 | gained |

gained on pooler decode_1, verify_16; qkv_gate verify_16; o verify_16; router verify_16, verify_32; lost on unpooler decode_1; router decode_1; o decode_16, decode_32; pooler decode_16. Expected hit: 1/6. Noise ±1.5%.

## Follow-ups

- rewrite stage 3 of `cuBLAS nvjet at m=128/256 for qkv_gate (Lt 40x64 15.7 us at m=128; 96x64 2x4 TNN 19.6 us at m=256), pooler m=256 (NN 128x64 29.0 us) and unpooler (NN 128x64 19.1 us at m=128; 128x128 TNT 24.7 us at m=256)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_bf16_proj "<what to fix or tune>" --kind rewrite --depends-on bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush --stage 4`; a slow or failed first cut is the expected start, not a do-not-retry.
- unproven-vs-best: faster than the best but inside the drift margin; churn queues one confirm rerun when the margin allows (CONFIRM_MIN), and pools the two runs.

