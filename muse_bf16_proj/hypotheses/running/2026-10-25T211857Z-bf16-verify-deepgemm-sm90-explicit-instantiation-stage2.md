---
problem: muse_bf16_proj
slug: bf16-verify-deepgemm-sm90-explicit-instantiation-stage2
created: 2026-10-25T211857Z
base: master
area: bf16 GEMM dispatch (m=65..256, DeepGEMM sm90 BF16 kernel)
change: >
  DIAGNOSIS Action 5. Built on master 36c26199, porting stage 1's [dg] scaffold (patch
  done/neutral/bf16-verify-deepgemm-sm90-kernel-asserted-rewrite.patch: TORCH_CUDA_ARCH_LIST=9.0a
  around load_inline, -std=c++20 -gencode=arch=compute_90a,code=sm_90a, -I deep_gemm/include
  (bundled CUTLASS/cute), the #error guard on !__CUDA_ARCH_FEAT_SM90_ALL, cuTensorMapEncodeTiled
  via cudaGetDriverEntryPoint, weight descriptors once in prepare, x/out descriptors per call from
  data_ptr and shape, cudaLaunchKernelEx with the cluster attribute, the self-check and [dg] PATH
  CHOSEN / ROUTE TAKEN prints). STAGE 2 CHANGE: (1) Discovery first: print the impls listing, then
  for sm90_bf16_gemm.cuh print every line matching __global__, __launch_bounds__, CUTLASS_GLOBAL
  or 'template <' with 8 lines of context, and grep deep_gemm's own JIT sources in the package
  (csrc/jit_kernels/impls/*sm90*bf16*, any *.hpp/*.py that builds the instantiation string) and
  print the generated-code template if found. Print these BEFORE any build so the log tail keeps
  them. (2) Replace the regex parse with an explicit instantiation: locate the kernel symbol
  (sm90_bf16_gemm_impl or the name found in 1), and if DeepGEMM's JIT generator string is found,
  reuse its exact template-argument order and fill it with our per-(layer, m) config; otherwise
  instantiate by the printed template parameter names, refusing (print 'signature mismatch',
  master runs) any name it does not know rather than guessing. (3) Config: BLOCK_K 64, BLOCK_N
  restricted to multiples of 32 (stage 1 flagged BLOCK_N=48 for qkv as untested), BLOCK_M 64/128,
  multicast 1 or 2; per layer two instantiations ({128} and {130,200,256}), routed only at the
  exact verified (layer, m) pairs; o stays on cuBLAS. (4) Gate per cell against kernel_fn with the
  flag off, i.e. whatever master (and any landed pooler K-halves or Lt cells) routes. Gate
  (hardened from today's evidence): copy master's _cell_gate model selector (graph replay of
  kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved reps,
  empty-graph baseline subtracted) but route only if BOTH of two independent interleaved rounds
  give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two picks
  that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted 1.041
  (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The pick
  that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the
  chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both
  medians, both IQRs and the kernels. The profiler assert must see the DeepGEMM symbol and no
  nvjet/cublas kernel. Milestones: M0 the TU builds in < 150 s and prints regs/spills/SASS 90a; M1
  every (layer, m) self-check passes incl. unpooler m=130; M2 kernel-only cold-L2 time beats the
  incumbent on >= 2 of the 6 cells (targets: pooler m=256 <= 24 us vs 28.8, qkv m=256 <= 16 vs
  20.0, unpooler m=256 <= 22 vs 24.8, pooler m=128 <= 18 vs 20.8+3.0, unpooler m=128 <= 17.5 vs
  19.3, qkv m=128 <= 13.5 vs 15.5); M3 those rows faster in the judge. Fallback at every step: the
  per-pair flag stays off and master runs.
expected:
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: tuned persistent TMA/wgmma kernel at ~700 TFLOP/s vs nvjet 28.8 us (580 TFLOP/s)
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: nvjet 96x64 at 497 TFLOP/s, floor 9.8
  - shapes: layer=unpooler load=verify_* m=*
    direction: faster
    why: one M-tile per CTA fetches W once; incumbent 2.9-3.5 TB/s
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: vs cell B nvjet 40x64 15.5 us, floor 8.4
  - shapes: layer=o load=* m=*
    direction: flat
    why: o stays on cuBLAS
risk: high
depends_on: bf16-verify-deepgemm-sm90-kernel-asserted-rewrite
priority: 1
kind: rewrite
controls_for: null
prior: "bf16-verify-deepgemm-sm90-kernel-asserted-rewrite: stage 1 never compiled (header parse: 'no __global__ template kernel found', VOID RUN, 4c224beb); stage 2 replaces the regex signature parse with a printed-then-explicit instantiation of the header's kernel and makes the first lines of the log the kernel declaration, so even a failed build hands stage 3 the signature"
rewrite_stage: 2
replaces: cuBLAS nvjet (+splitKreduce) for qkv_gate, pooler, unpooler at m=128/256
review: proceed
status: running
review_at: 2026-10-25T212521Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T212548Z
base_sha: e4008591fb186026deeb850980bc31aed9fbe89a
base_blob: 489174846c51a034e3c119f0f67cec9c213404f5
branch: hyp/muse_bf16_proj/bf16-verify-deepgemm-sm90-explicit-instantiation-stage2
worktree: .worktrees/muse_bf16_proj/bf16-verify-deepgemm-sm90-explicit-instantiation-stage2
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-26T011203Z
impl_commit: 0dc50d0a5d027659c02f0e6a45778fa2662139e4
submission_sha256: 1ed49c2ec768aba15650df07f73e25aa54f700875a1bcdd816bb55da5f584b2c
norm_sha256: adf5c5221ecf31df1fc106cc0d05df26528b73a50f155216a710ff0ed4577f93
submission_blob: ff4b91b7ccf2b0111c4ed8a900c2f48cfed8d73b
lint_warnings: 37
attempt_file: 2026-10-26T012540Z-check-bf16-verify-deepgemm-sm90-explicit-instantiation-stage2.md
prepared_at: 2026-10-26T012540Z
---

# bf16-verify-deepgemm-sm90-explicit-instantiation-stage2

## Change

DIAGNOSIS Action 5. Built on master 36c26199, porting stage 1's [dg] scaffold (patch done/neutral/bf16-verify-deepgemm-sm90-kernel-asserted-rewrite.patch: TORCH_CUDA_ARCH_LIST=9.0a around load_inline, -std=c++20 -gencode=arch=compute_90a,code=sm_90a, -I deep_gemm/include (bundled CUTLASS/cute), the #error guard on !__CUDA_ARCH_FEAT_SM90_ALL, cuTensorMapEncodeTiled via cudaGetDriverEntryPoint, weight descriptors once in prepare, x/out descriptors per call from data_ptr and shape, cudaLaunchKernelEx with the cluster attribute, the self-check and [dg] PATH CHOSEN / ROUTE TAKEN prints). STAGE 2 CHANGE: (1) Discovery first: print the impls listing, then for sm90_bf16_gemm.cuh print every line matching __global__, __launch_bounds__, CUTLASS_GLOBAL or 'template <' with 8 lines of context, and grep deep_gemm's own JIT sources in the package (csrc/jit_kernels/impls/*sm90*bf16*, any *.hpp/*.py that builds the instantiation string) and print the generated-code template if found. Print these BEFORE any build so the log tail keeps them. (2) Replace the regex parse with an explicit instantiation: locate the kernel symbol (sm90_bf16_gemm_impl or the name found in 1), and if DeepGEMM's JIT generator string is found, reuse its exact template-argument order and fill it with our per-(layer, m) config; otherwise instantiate by the printed template parameter names, refusing (print 'signature mismatch', master runs) any name it does not know rather than guessing. (3) Config: BLOCK_K 64, BLOCK_N restricted to multiples of 32 (stage 1 flagged BLOCK_N=48 for qkv as untested), BLOCK_M 64/128, multicast 1 or 2; per layer two instantiations ({128} and {130,200,256}), routed only at the exact verified (layer, m) pairs; o stays on cuBLAS. (4) Gate per cell against kernel_fn with the flag off, i.e. whatever master (and any landed pooler K-halves or Lt cells) routes. Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both medians, both IQRs and the kernels. The profiler assert must see the DeepGEMM symbol and no nvjet/cublas kernel. Milestones: M0 the TU builds in < 150 s and prints regs/spills/SASS 90a; M1 every (layer, m) self-check passes incl. unpooler m=130; M2 kernel-only cold-L2 time beats the incumbent on >= 2 of the 6 cells (targets: pooler m=256 <= 24 us vs 28.8, qkv m=256 <= 16 vs 20.0, unpooler m=256 <= 22 vs 24.8, pooler m=128 <= 18 vs 20.8+3.0, unpooler m=128 <= 17.5 vs 19.3, qkv m=128 <= 13.5 vs 15.5); M3 those rows faster in the judge. Fallback at every step: the per-pair flag stays off and master runs.

## Expected

- `layer=pooler load=verify_32 m=256` faster: tuned persistent TMA/wgmma kernel at ~700 TFLOP/s vs nvjet 28.8 us (580 TFLOP/s)
- `layer=qkv_gate load=verify_32 m=256` faster: nvjet 96x64 at 497 TFLOP/s, floor 9.8
- `layer=unpooler load=verify_* m=*` faster: one M-tile per CTA fetches W once; incumbent 2.9-3.5 TB/s
- `layer=qkv_gate load=verify_16 m=128` faster: vs cell B nvjet 40x64 15.5 us, floor 8.4
- `layer=o load=* m=*` flat: o stays on cuBLAS

## Risk

Ceiling: verify_32 90.8 -> ~78 us and verify_16 72.6 -> ~62 us if M2 holds on most cells, +2-3% headline. Judged on the replaced kernels' own us against the milestones, not the headline. The waits live in DeepGEMM's shipped header (cannot be bounded): fault/hang risk, so the first measurement is an okc check; flushed per-launch prints before/after every canary launch. Compile budget: skip the build if prepare passes 110 s.

## Milestones

Rewrite stage 2 of `cuBLAS nvjet (+splitKreduce) for qkv_gate, pooler, unpooler at m=128/256`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T212521Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a real stage 2 and not a duplicate. Stage 1 (4c224beb) was VOID at the header parse and measured nothing, so fixing kernel discovery is the right next step and the targets on pooler/unpooler are physically reachable. Before building, fix three implementation traps: where the DG branch sits in kernel_fn on the current master, the TMA multicast that the stage-1 audit removed because it hangs, and discovery lines printed first in a log that is kept tail-only.

### Justification
A. The mechanism is plausible for pooler and unpooler, weaker for qkv_gate, and FINDINGS supports where the time is. On master 36c26199 the target rows are pooler m=256 nvjet NNT 28.8 us (17.2 GFLOP, about 597 TFLOP/s, floor 17.4), pooler m=128 nvjet splitK 20.8 + splitKreduce 3.0 (floor 14.6), unpooler m=128 19.3 (67 MB at about 3.5 TB/s, floor 14.6), unpooler m=256 24.8 (about 693 TFLOP/s already, floor 17.4), qkv m=256 20.0 (floor 9.8), and qkv m=128 Lt single-node nvjet 40x64 15.5 (floor 8.4). Tile arithmetic for a DeepGEMM sm90 BF16 kernel with no split-K on 132 SMs: (1) pooler m=256 at 128x64 or 64x128 gives 128 tiles in one wave at about 134 MFLOP per CTA, so 18 us at per-SM peak and about 21-23 us realistic. The target of <= 24 is reachable. (2) unpooler m=128 at 128x64 gives 128 tiles and reads W once, so it is memory-bound at about 17 us. The 17.5 target is plausible. (3) unpooler m=256 at 128x128 gives 128 tiles. The 22 us target needs about 780 TFLOP/s against an incumbent already at about 693, so it is marginal. (4) pooler m=128 at 64x64 gives 128 tiles, but W is read twice across the 2 M-blocks unless L2 or B-multicast absorbs it. The 18 us target depends on co-scheduling and is plausible but uncertain. WEAKEST LINK: qkv_gate (n=2304) has no split-K in this kernel. At m=128 with BLOCK_N a multiple of 32 there are only 36-72 tiles. Each CTA re-reads the whole 128x8192 x slice (2 MB) from L2, so BLOCK_N=32 moves about 151 MB of L2 traffic and BLOCK_N=64 leaves about 18 us of per-SM compute. The <= 13.5 target is very likely unreachable. At m=256, 72 tiles at 128x64 is about 18 us at per-SM peak, and 144 tiles at 64x64 is 2 waves, so the <= 16 target is doubtful too. Expect both qkv cells to fail their gate. That is fine because the gate keeps master, but a qkv loss must not be read as a design kill. Expected-shape globs are slightly off: 'layer=unpooler load=verify_* m=*' also covers verify_1 m=8, which runs ouns_kernel and is not routed, so read it as verify_16 and verify_32 only. pooler verify_16 is targeted in the milestones but missing from expected. Prior check: bf16-verify-deepgemm-sm90-kernel-asserted-rewrite is correctly named, and its log confirms the claim. impls/ lists sm90_bf16_gemm.cuh, and the run failed with 'header parse failed (ValueError: no __global__ template kernel found in header); VOID RUN', so no pair was routed. The difference is real: explicit instantiation instead of the regex parse. It does not use the do-not-try API route (bf16-verify-deepgemm-bf16-gemm-nt-route, whose bf16_gemm_nt was bitwise cuBLAS on all 6 pairs in 4c224beb STEP A). Rewrite stages are exempt from do-not-try. One expectation to drop: DeepGEMM's JIT generator string lives in compiled csrc (jit_kernels/impls/*.hpp is built into the .so), and an installed wheel rarely ships it. Assume the 'reuse the JIT generator's argument order' branch finds nothing, and make the path that instantiates by printed template-parameter names the primary one.

### Confounders on the current kernel
- pooler-nn-layout-widen-m256 + restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (NN layout for pooler/unpooler at m 65..128 and pooler 129..256): these are the incumbents for 4 of the 6 target cells, and kernel_fn returns torch.mm(x, state[layer+'_t']) BEFORE the cuBLAS fallback. If the DG branch goes 'just before the cuBLAS fallback' as in stage 1 (which was built on e5548a7, before the NN block existed), pooler m=128/256 and unpooler m=128 never reach it. The flag-on and flag-off graphs are then identical, the gate reads about 1.0, and those cells silently measure master. The DG branch must sit above the NN_GATED block, gated by its own per-(layer, m) flag.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (B cell: qkv_gate m=128 on a prepare-fixed cuBLASLt plan via _lt_route_ok, plus the _cell_gate/_cell_kernels machinery): this defines the qkv m=128 incumbent (15.5, not the old 17.0) and the gate being copied. The same placement issue applies: _lt_route_ok returns before the fallback. Because the incumbent is already a tuned single node, the DG qkv m=128 cell must beat 15.5 by 3%.
- sglang-gemm-dispatch-parity: the fallback (_bf16_gemm_dispatch_impl) is the incumbent for qkv m=256 and unpooler m=256 (and unpooler m=130/200 via the NN path). Those rows' paired ratio against SGLang is a direct DeepGEMM-vs-nvjet comparison, which helps attribution and does not confound it.
- m<=32 and router landed items (swap-AB, bulk row-owner, ouns, router cluster/hilo): they share no launch sequence or memory traffic with the m=128/256 rows. They only share the prepare budget and add the usual ±1.5-4% row noise on untouched rows, so do not credit or blame untouched rows.

### Interference and A/B needs
No companion. This is a rewrite stage, and each target cell attributes itself when three things are true: the per-cell gate line prints both medians and IQRs; the judge profile's top_kernels for that row names sm90_bf16_gemm_impl and no nvjet; and the [dg] PATH CHOSEN/ROUTE TAKEN lines say the branch fired. An ab-revert of the NN layout or the Lt cell would only change the incumbent the gate compares against, not tell us whether DeepGEMM's kernel is fast. The kernel's own time against its floor is the milestone, and it is readable from the gate lines and the profile without another run. The real attribution risk is a non-result rather than an ambiguous one. That happens if the branch sits below the NN/Lt returns, if the build fails, or if the log loses the discovery lines. Fix those three in the build instead of spending a second check.

### Open questions
- Log tail: archived logs keep about the last 15 KB, and master's own log (741072dc) is already 10.9 KB and starts mid-line. Printing discovery FIRST, with 8 lines of context per match plus any JIT source, guarantees it is cut, especially if an nvcc error tail follows. Print discovery compactly (only the kernel's template parameter list and argument list, at most 2 KB), and re-print one-line summaries (signature, chosen config per cell, regs/spills, gate result) as the LAST lines of prepare.
- Does sm90_bf16_gemm.cuh's kernel take SHAPE_M as a template parameter that supports 0 (runtime M) in this fork? Stage 1 relied on SHAPE_M=0 with TMA OOB zero-fill for m=130/200. If this fork lacks runtime M, gate on the exact compiled m and leave 130/200 on master.
- What made the stage-1 regex miss __global__? Likely the template<...> and the __global__ void __launch_bounds__(...) name( declaration span lines, or a macro (CUTLASS_GLOBAL or an #if __CUDA_ARCH__ guard) wraps it. The explicit-instantiation path should match on the kernel name sm90_bf16_gemm_impl, not on '__global__ ... template'.
- Build budget: 6 cute C++20 instantiations, plus 2 rounds x 41 reps x 6 cells of gating and one profiler call each. Master's prepare is short, so 300 s should hold, but keep the 110 s skip-build guard and print [build-time].
- If DIAGNOSIS Action 1 (pooler K-halves restack) or Action 2 (per-cell Lt selector) lands first, the pooler incumbents fall to 26.75/21.43 and the DG pooler cells need about 25.9/20.8 to pass the 1.03 two-round gate. The milestone targets (24/18) still clear that.
- Two-round gate with IQR <= 1.5x master's: print per-round ratios so a cell that passes one round and fails the other is visible as a near-miss for stage 3 rather than a silent master.


## Implementation notes

Worktree `.worktrees/muse_bf16_proj/bf16-verify-deepgemm-sm90-explicit-instantiation-stage2`, `muse_bf16_proj/submission.py` only. The claim was on the parent's commit (stale_base), so I ran `git checkout master -- muse_bf16_proj/submission.py` (master 36c26199, 3212 lines) and ported the parent's `[dg]` scaffold by hand (the parent patch targeted an older base). Net diff vs master: +986/-5.

What changed:
- New section (10) before `_prepare_state`: `_dg_*` helpers (discovery, parse by name, config picker, source generator, load_inline build, self-check, two-round gate, per-cell PATH CHOSEN, route/forward, final summary). Called last in `_prepare_state` (after `_lt_prepare`, so the gate's incumbent is master's full routing incl. the Lt cell); state B inherits flags via `flags["_dg_route"]` and re-encodes its own weight descriptors.
- kernel_fn: `if layer in DG_LAYERS: _dg_route_ok(...)` is the first thing inside `if layer != "router":`, i.e. ABOVE the NN_GATED block and the `_lt_route_ok` return (the review's main trap; nothing earlier in kernel_fn catches m = 128 / 256 for these layers).
- Six per-cell flags `dg_<layer>_<m>_ok` (layers pooler, qkv_gate, unpooler; m 128, 256) added to `NEW_CELL_FLAGS`, so `_ab_flags` / the A/B self-check / the state-B failure path carry them. `_ab_component` maps a cell to its dg flag only while that flag is on. `AB_CASES` gains (qkv_gate 256) and (unpooler 256) so every DG cell gets the A,B,A,B graph check.
- Weight descriptors are CPU uint8 tensors in `state["dg_wdesc"]` (no module-global device or host state in the extension); x / out descriptors are encoded per call from data_ptr and shape.

Review fixes applied:
1. DG branch above the NN block and the Lt return, own flag per (layer, m).
2. No TMA multicast: BLOCK_M 64/128, BLOCK_N multiple of 32 dividing n, BLOCK_K 64, multicast 1, no cluster launch attribute (the stage-1 audit removed it: tx-byte mismatch = unbounded mbarrier hang in the shipped header). This corrects the spec's "multicast 1 or 2"; the review wins.
3. Discovery is compact (about 2 KB) and printed before any build: sm90 impls only (not the full listing), kernel name, template params, kernel params, runtime-M support, JIT-source hits, enum/epilogue names. On a parse failure it prints a capped context of the lines around the kernel name / `__global__` / `CUTLASS_GLOBAL` / `__launch_bounds__`. One-line summaries (signature, build time, regs/spills per variant, self-check results, per-cell PATH CHOSEN with both rounds' ratios, final flags) are re-printed as the LAST lines of prepare (`_dg_final_summary`, after FINAL FLAGS; the PATH CHOSEN lines there carry the final flag after the A/B check).
4. The kernel is found BY NAME (definition of `sm90_bf16_gemm_impl`, else any `sm90_*gemm*_impl`: name followed by balanced (...) then `{`, with the nearest preceding `template<...>` and only `__global__` / `void` / attribute words between; `__launch_bounds__(...)`, comments and preprocessor lines stripped). Tested offline on a mock header with CUTLASS_GLOBAL, a launch-bounds expression, an earlier helper template and `#if` guards. The "reuse the JIT generator's argument order" branch is print-only (the generator is in compiled csrc; its string has no machine-readable order): the primary and only instantiation path is by printed template-parameter NAME; an unknown name without a default prints "signature mismatch" and master runs; trailing parameters that all have defaults are left to the header.
5. Runtime M (open question): `runtime_m` = no SHAPE_M-like template parameter, or the header text contains `SHAPE_M != 0`. If True, group {130, 200, 256} is built with SHAPE_M = 0 (TMA OOB zero-fill, clipped stores); if False, that group is compiled for m = 256 only and 130 / 200 stay on master. Group {128} always compiles m = 128 exactly.
6. Gate: `_dg_gate` = the `_cell_gate` method (graph replay of kernel_fn flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline subtracted), run in TWO independent rounds (fresh x, fresh graphs, order phase flipped); route only if BOTH rounds give master/cand >= 1.03 AND cand IQR <= 1.5 x master IQR (master IQR floored at 0.1 us for timer granularity). Per-round medians, IQRs and ratios are printed, so a one-round near-miss is visible.
7. Profiler assert (`_cell_kernels` on the flag-on path): exactly one kernel, its name contains the DeepGEMM symbol, no nvjet / cublas; a profiler failure drops the cell. Master's kernel names are profiled too and printed in every PATH CHOSEN line, including kept-master cells.
8. Build/deadlines: the build is skipped past 110 s of prepare ("build skipped: deadline"), self-checks past 200 s and gates past 235 s print "skipped: deadline"; `[build-time]`, regs, local (spill) bytes, sass sm_90 are printed; a variant with sass != 90 or > 512 B local is rejected. A self-check miss or skipped m makes the whole variant unusable.

Deviations / clarifications:
- m = 130 and 200 are self-checked (M1) but NOT routed: only the scored m = 128 / 256 have a gate, so only they can be routed ("routed only at exact verified pairs" read narrowly). o stays on cuBLAS.
- Expected-shape globs follow the review: unpooler verify_1 (m=8) is not DeepGEMM (ouns path); read the cells as verify_16 / verify_32.
- Imports of `deep_gemm` are lazy (inside `_dg_discover`), so flag-off master behaviour is untouched when it is missing.
- Not runnable on this node (no GPU, torch or deep_gemm): nothing here has been compiled with nvcc; only py_compile, okc_lint and an offline test of the parser / config picker / source generator on a mock header.

Biggest risks: (a) nvcc compile of the header with our template args (cute C++20, six instantiations; a build error lands on master and the log's discovery lines still name the signature); (b) a hang inside DeepGEMM's unbounded mbarrier waits (host watchdog only logs, the run is then lost); (c) qkv_gate is expected to fail its gate (no split-K, 36-96 tiles), which is not a design kill.

## Rules check

- [x] CUDA-graph capturable: `_dg_route_ok` / `_dg_forward` (the only code added to kernel_fn's path) read cfg / shapes / state flags, call `torch.empty` and one extension launch; the tensor-map encode is host-only; no `.item()` / `.cpu()` / `synchronize()` / tensor print there (all syncs are in prepare-time `_dg_*` helpers, outside kernel_fn's call graph). Checked in the diff and by okc_lint.
- [x] Nothing derived from `data` is cached across calls: x / out descriptors are re-encoded each call from data_ptr and shape; the weight descriptors derive from weights in prepare().
- [x] Weights: no new weight copy, BF16 stays BF16 (the DeepGEMM kernel reads the existing `state[layer]`); router untouched.
- [x] Activations BF16 as SGLang keeps them; output BF16; fp32 accumulation inside the kernel.
- [x] n/a: attention sees every key: not an attention problem; no keys, no skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added; `grep` for subprocess / multiprocessing / Popen / os.fork finds nothing; load_inline is the same mechanism master already uses.
- [x] All work happens inside the timed call; no output caching or replay (a fresh `torch.empty` output per call).
- [x] Any build error, self-check miss, deadline or unsupported shape lands on master's exact path and never raises (every `_dg_prepare` branch is try/except; per-cell flags default False; a miss never touches an unrelated flag).
- [x] Bounded waits: DeepGEMM's mbarrier waits are in the shipped header and cannot be bounded; mitigated by the host watchdog (10 s per launch, prepare-time only), flushed LAUNCH / DONE prints before and after every launch, and no multicast.
- [x] Distinct symbol names: extension `muse_bf16_dgx_<hash>`, functions `dg_*` / `dgx_*`, kernel is DeepGEMM's own symbol.
- [x] sm_90a: `TORCH_CUDA_ARCH_LIST=9.0a` set around load_inline and restored; `-gencode=arch=compute_90a,code=sm_90a`; TU-level `#error` if a device pass lacks `__CUDA_ARCH_FEAT_SM90_ALL`; flags printed; `-lineinfo`.

okc_lint (`python3 -I .../okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`): `OK with warnings: submission.py` (warnings only: host-sync in prepare-time code outside kernel_fn's call graph and uint8 workspace / flush buffers, the same classes master already carries). `python3 -I -m py_compile` passes; `__pycache__` deleted.

## Pre-flight audit

Verdict: dispatch-with-fixes. Evidence scripts: scratchpad/audits26/dg_parser_test.py (parser on 9 header variants), scratchpad/audits26/dg_diff.txt (diff vs master).

- Parser/discovery: the builder's mock header was not saved. I re-tested the real `_dg_clean/_dg_parse_kernel` on upstream-style headers (namespace, helper template first, `__global__ __launch_bounds__(a+b,1) void\nname(`, CUTLASS_GLOBAL, attribute after void, `#if` guards, comment mentioning the name) and on absent/declaration-only/empty/garbage/`1 << 3` default headers: only ValueError, which `_dg_discover` prints and `_dg_prepare` turns into master. Unknown template or kernel argument names go to a printed "signature mismatch" and master. A newer fork with extra template params (this image is deep_gemm 0.2.1.post1+muse with sm120/mega_moe) lands on master, not a raise.
- Routing: every earlier kernel_fn branch is m <= 32; `_dg_route_ok` returns None unless m is 128 or 256, the cell flag is on and n/k/dtype/contiguity/alignment match, so it cannot shadow any cell at other m, and at 128/256 it shadows only gated incumbents (NN, Lt, dispatch).
- Profiler assert (`_cell_kernels`, exactly one kernel containing the DeepGEMM name, no nvjet/cublas) is only in `_dg_prepare_a`; dg_run launches a single kernel with no memset and no host sync.
- Fix 1: deadlines ran on `_prepare_state`'s own clock, which starts after master's import-time builds (ouns 37 s + router_cluster 37 s + bulk 37 s + lt 15 s, plus torch/triton/sglang import, about 150 s per the archived logs). Added `_DG_T_MODULE` (set before `import torch`) and moved the build/check/gate deadlines onto it, rebased to 260/360/400 s (was 110/200/235 s on the prepare clock, which is about 260/350/385 s of real time).
- Fix 2: `DG_IQR_FLOOR_US` 0.1 to 0.5 us so a timer-resolution IQR on master cannot veto a real win.
- okc_lint: OK with warnings (prepare-time syncs only); py_compile passes; __pycache__ removed.

Residual risk: nvcc has no timeout, so a very slow 6-instantiation C++20 build or a hang in the header's unbounded mbarrier waits can still cost the run; qkv_gate cells are expected to lose the gate.
