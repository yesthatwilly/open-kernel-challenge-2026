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
status: running
review_at: 2026-10-27T133822Z
review_companions: null
stage: claimed
claimed_at: 2026-10-27T133822Z
base_sha: 0dc50d0a5d027659c02f0e6a45778fa2662139e4
base_blob: ff4b91b7ccf2b0111c4ed8a900c2f48cfed8d73b
branch: hyp/muse_bf16_proj/bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush
worktree: .worktrees/muse_bf16_proj/bf16-verify-deepgemm-sm90-s3-tparam-alias-readflush
mode: null
restack_of: null
stale_base: true
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

