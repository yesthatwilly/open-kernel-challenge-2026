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
status: queued
review_at: 2026-10-25T212521Z
review_companions: null
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

