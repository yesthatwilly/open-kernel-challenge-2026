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
review: required
status: queued
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
