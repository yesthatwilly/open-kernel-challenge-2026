---
problem: muse_bf16_proj
slug: bf16-cublaslt-backend-toggle
created: 2026-10-23T225303Z
base: master
area: bf16 GEMM dispatch
change: >
  A dispatch probe, pure PyTorch, no kernel of ours: in kernel_fn, for the four bf16 layers only,
  call torch.backends.cuda.preferred_blas_library('cublaslt') immediately before the
  _bf16_gemm_dispatch_impl(x, W[:n], None) call and restore
  torch.backends.cuda.preferred_blas_library('cublas') immediately after it (both are host-side
  settings, so inside the captured graph only the kernel chosen under the Lt backend is recorded;
  the restore guarantees SGLang's own measurement in the same process is taken with the default
  backend). ATen's torch.mm for bf16 goes through cublasGemmEx under the default backend and
  through cublasLtMatmul with PyTorch's own heuristic query and a 32 MB Lt workspace under
  'cublaslt'; the two can select different nvjet tiles and splitK factors for the same shape.
  Router path and prepare() unchanged. Kernels possibly moved: every nvjet_sm90_tst_* +
  splitKreduce pair on the 20 bf16 shapes; the run's profile shows which shapes changed kernel
  name at all.
expected:
  - shapes: "*layer=pooler load=decode_*"
    direction: faster
    why: the Lt heuristic with a larger workspace may pick a different splitK factor for the 4096x8192 skinny GEMM; the shapes at 1.5x floor have the most to gain
  - shapes: "*layer=unpooler load=decode_16 *"
    direction: faster
    why: single non-splitK nvjet today; Lt may choose a splitK variant
  - shapes: "*layer=router*"
    direction: flat
    why: router path untouched; control
  - shapes: "*layer=o*"
    direction: flat
    why: likely the same kernel either way; a second control
risk: low
depends_on: sglang-gemm-dispatch-parity
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# bf16-cublaslt-backend-toggle

## Change

A dispatch probe, pure PyTorch, no kernel of ours: in kernel_fn, for the four bf16 layers only, call torch.backends.cuda.preferred_blas_library('cublaslt') immediately before the _bf16_gemm_dispatch_impl(x, W[:n], None) call and restore torch.backends.cuda.preferred_blas_library('cublas') immediately after it (both are host-side settings, so inside the captured graph only the kernel chosen under the Lt backend is recorded; the restore guarantees SGLang's own measurement in the same process is taken with the default backend). ATen's torch.mm for bf16 goes through cublasGemmEx under the default backend and through cublasLtMatmul with PyTorch's own heuristic query and a 32 MB Lt workspace under 'cublaslt'; the two can select different nvjet tiles and splitK factors for the same shape. Router path and prepare() unchanged. Kernels possibly moved: every nvjet_sm90_tst_* + splitKreduce pair on the 20 bf16 shapes; the run's profile shows which shapes changed kernel name at all.

## Expected

- `*layer=pooler load=decode_*` faster: the Lt heuristic with a larger workspace may pick a different splitK factor for the 4096x8192 skinny GEMM; the shapes at 1.5x floor have the most to gain
- `*layer=unpooler load=decode_16 *` faster: single non-splitK nvjet today; Lt may choose a splitK variant
- `*layer=router*` flat: router path untouched; control
- `*layer=o*` flat: likely the same kernel either way; a second control

## Risk

Honest expectation: most shapes flat, because cublasGemmEx already consults the Lt heuristics internally on Hopper; this run exists to settle whether any PyTorch-level BLAS dispatch lever is left before all remaining bf16 effort goes into custom kernels. Reading: compare kernel names per shape against the parity profile; any shape whose nvjet name did not change and whose time moved is noise. Rule edges: a documented torch API call, not a monkeypatch of a loaded module and not harness state; the restore after each call keeps SGLang's baseline measurement unaffected (if the harness measured SGLang under our setting the paired ratios would be confounded, which is why the toggle is scoped to the single call). If the image's torch lacks preferred_blas_library or ignores it for bf16 mm, the run is a no-op at parity, costing one check. No correctness exposure: same cuBLAS math.
