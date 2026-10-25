---
problem: muse_bf16_proj
slug: bf16-verify-deepgemm-sm90-kernel-asserted-rewrite
created: 2026-10-25T042040Z
base: master
area: bf16 GEMM dispatch (m=65..256, DeepGEMM sm90 BF16 kernel)
change: >
  Built on master. bf16-verify-deepgemm-bf16-gemm-nt-route (2d522d47) printed PATH CHOSEN deepgemm
  for all 16 (layer, m) pairs, every self-check at 0.0 s (no JIT ever happened), yet every m>=128
  profile row was nvjet: DeepGEMM's own sm90 BF16 kernel never ran. This stage makes it run and
  proves it. STEP A (import/prepare, untimed): for deep_gemm 0.2.1.post1+muse print
  inspect.getsource or the docstring of bf16_gemm_nt and any DG_* env/config getters, then profile
  ONE deep_gemm.bf16_gemm_nt(x, W, out) call per (layer, m in {128, 256}) under torch.profiler
  (in-process, prepare only) and print the launched kernel names. A layer is DG-eligible only if
  the kernel name contains 'sm90_bf16_gemm' (or another deep_gemm kernel symbol) and not
  'nvjet'/'cublas'/'gemmSN'. STEP B, if step A finds cuBLAS: build DeepGEMM's sm90 BF16 GEMM
  directly from the image's header tree (deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh with
  the bundled CUTLASS 4.2.1 / cute) via torch.utils.cpp_extension.load_inline:
  TORCH_CUDA_ARCH_LIST=9.0a set around the build and restored, extra_cuda_cflags = ['-std=c++20'
  (replacing, never appending to, any -std=c++17), '-gencode=arch=compute_90a,code=sm_90a',
  '-lineinfo', '--expt-relaxed-constexpr', '-I<deep_gemm include>', '-I<cutlass include>'], and a
  `#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)` #error guard so a stray
  sm_90 pass fails the build cleanly. Instantiate with the shapes as compile-time constants
  (qkv_gate n=2304 k=8192; pooler n=4096 k=8192; unpooler n=8192 k=4096; m=128 and 256) and
  DeepGEMM's own heuristic block sizes (BLOCK_M 64/128, BLOCK_N chosen to fill 132 SMs, TMA
  multicast 2 on the larger dimension); build the weight TMA descriptors once in prepare; encode
  the x descriptor per call on the host from x.data_ptr() and the cfg shape (a pointer and shape,
  never values; graph capture records it), no allocation, no sync. Print regs/spills/smem
  (cudaFuncGetAttributes); reject any variant with spills. Self-check per (layer, m in {128, 130,
  200, 256}): rel L2 <= 3e-3 vs fp64, one-hot probe, finite, CUDA graph capture+replay; then the
  step-A profiler name assertion on the built kernel. Route in kernel_fn just before the cuBLAS
  fallback: layer DG-ok and 65 <= cfg m <= 256 and full n. o stays on cuBLAS (short K). Distinct
  from bf16-verify-swapab-wgmma-cluster-dsmem-rewrite (our own swap-AB kernel): this tests
  DeepGEMM's shipped kernel; if both run, compare per row. Print '[dg] PATH CHOSEN <layer> m=<m>:
  deepgemm-api|deepgemm-header|master KERNEL=<name>'.
expected:
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: incumbent nvjet splitK 12.6 + splitKreduce 3.1 (two nodes), row 17.0, floor 8.4; M2 row <= 15.5, M3 <= 13
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: nvjet 19.5, row 19.7, floor 9.8; M2 <= 18.5
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: nvjet 24.5 at ~2.7 TB/s, row 25.0, floor 14.6; M2 <= 23
  - shapes: layer=pooler load=verify_32 m=256
    direction: faster
    why: nvjet 30.7, row 31.2, floor 17.4; M2 <= 29
  - shapes: layer=unpooler load=verify_*
    direction: flat
    why: nvjet 20.3/24.3 already ~700 TFLOP/s at m=256; m=128 may move
  - shapes: layer=o *
    direction: flat
    why: not routed
  - shapes: layer=* load=decode_*
    direction: flat
    why: m<=32 untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: cuBLAS nvjet (+cublasLt splitKreduce) for qkv_gate, pooler, unpooler at m=65..256
review: required
status: queued
---

# bf16-verify-deepgemm-sm90-kernel-asserted-rewrite

## Change

Built on master. bf16-verify-deepgemm-bf16-gemm-nt-route (2d522d47) printed PATH CHOSEN deepgemm for all 16 (layer, m) pairs, every self-check at 0.0 s (no JIT ever happened), yet every m>=128 profile row was nvjet: DeepGEMM's own sm90 BF16 kernel never ran. This stage makes it run and proves it. STEP A (import/prepare, untimed): for deep_gemm 0.2.1.post1+muse print inspect.getsource or the docstring of bf16_gemm_nt and any DG_* env/config getters, then profile ONE deep_gemm.bf16_gemm_nt(x, W, out) call per (layer, m in {128, 256}) under torch.profiler (in-process, prepare only) and print the launched kernel names. A layer is DG-eligible only if the kernel name contains 'sm90_bf16_gemm' (or another deep_gemm kernel symbol) and not 'nvjet'/'cublas'/'gemmSN'. STEP B, if step A finds cuBLAS: build DeepGEMM's sm90 BF16 GEMM directly from the image's header tree (deep_gemm/include/deep_gemm/impls/sm90_bf16_gemm.cuh with the bundled CUTLASS 4.2.1 / cute) via torch.utils.cpp_extension.load_inline: TORCH_CUDA_ARCH_LIST=9.0a set around the build and restored, extra_cuda_cflags = ['-std=c++20' (replacing, never appending to, any -std=c++17), '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '--expt-relaxed-constexpr', '-I<deep_gemm include>', '-I<cutlass include>'], and a `#if defined(__CUDA_ARCH__) && !defined(__CUDA_ARCH_FEAT_SM90_ALL)` #error guard so a stray sm_90 pass fails the build cleanly. Instantiate with the shapes as compile-time constants (qkv_gate n=2304 k=8192; pooler n=4096 k=8192; unpooler n=8192 k=4096; m=128 and 256) and DeepGEMM's own heuristic block sizes (BLOCK_M 64/128, BLOCK_N chosen to fill 132 SMs, TMA multicast 2 on the larger dimension); build the weight TMA descriptors once in prepare; encode the x descriptor per call on the host from x.data_ptr() and the cfg shape (a pointer and shape, never values; graph capture records it), no allocation, no sync. Print regs/spills/smem (cudaFuncGetAttributes); reject any variant with spills. Self-check per (layer, m in {128, 130, 200, 256}): rel L2 <= 3e-3 vs fp64, one-hot probe, finite, CUDA graph capture+replay; then the step-A profiler name assertion on the built kernel. Route in kernel_fn just before the cuBLAS fallback: layer DG-ok and 65 <= cfg m <= 256 and full n. o stays on cuBLAS (short K). Distinct from bf16-verify-swapab-wgmma-cluster-dsmem-rewrite (our own swap-AB kernel): this tests DeepGEMM's shipped kernel; if both run, compare per row. Print '[dg] PATH CHOSEN <layer> m=<m>: deepgemm-api|deepgemm-header|master KERNEL=<name>'.

## Expected

- `layer=qkv_gate load=verify_16 m=128` faster: incumbent nvjet splitK 12.6 + splitKreduce 3.1 (two nodes), row 17.0, floor 8.4; M2 row <= 15.5, M3 <= 13
- `layer=qkv_gate load=verify_32 m=256` faster: nvjet 19.5, row 19.7, floor 9.8; M2 <= 18.5
- `layer=pooler load=verify_16 m=128` faster: nvjet 24.5 at ~2.7 TB/s, row 25.0, floor 14.6; M2 <= 23
- `layer=pooler load=verify_32 m=256` faster: nvjet 30.7, row 31.2, floor 17.4; M2 <= 29
- `layer=unpooler load=verify_*` flat: nvjet 20.3/24.3 already ~700 TFLOP/s at m=256; m=128 may move
- `layer=o *` flat: not routed
- `layer=* load=decode_*` flat: m<=32 untouched
- `layer=router *` flat: untouched

## Risk

Milestones: M1 = a deep_gemm kernel name on >= 1 target row in the judge profile (else the run is master: read the [dg] lines and stop). M2 = qkv m=128 row <= 15.5 and pooler m=128 <= 23. Kill: M1 met and no target row beats its incumbent by > 3%. Header build may take 60-150 s; the sm_90 arch bug (LESSONS 2026-10-25) and the c++20 #error (bulk-tma build) are both pre-empted. Run as a check (a trap at self-check would poison the context).

## Milestones

Rewrite stage 1 of `cuBLAS nvjet (+cublasLt splitKreduce) for qkv_gate, pooler, unpooler at m=65..256`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
