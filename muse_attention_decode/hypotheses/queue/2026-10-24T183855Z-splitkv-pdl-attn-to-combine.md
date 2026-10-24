---
problem: muse_attention_decode
slug: splitkv-pdl-attn-to-combine
created: 2026-10-24T183855Z
base: master
area: launch overhead
change: >
  Programmatic dependent launch between master's kernel A (splitkv_attn_kernel) and kernel B
  (splitkv_combine_kernel), full decode 1 < batch <= 64. (1) In splitkv_combine (host), launch
  kernel B with cudaLaunchKernelEx: cudaLaunchConfig_t with gridDim (B, 16), blockDim 128, stream
  = current stream, and one attribute cudaLaunchAttributeProgrammaticStreamSerialization with
  programmaticStreamSerializationAllowed = 1. (2) First statement of splitkv_combine_kernel: asm
  volatile("griddepcontrol.wait;" ::: "memory"); this blocks until kernel A has completed and its
  writes are visible, so the combine's reads are as safe as today. (3) In splitkv_attn_kernel,
  after the main tile loop and before the final cp_wait<0>(), every thread executes asm
  volatile("griddepcontrol.launch_dependents;" :::); a CTA that exits early (bx >= total) triggers
  implicitly by exiting. Nothing else changes; CUDA graph capture records the programmatic edge
  (CUDA >= 12.3; the judge image is cu130). Rename the module muse_full_decode_splitkv_v2. If
  cudaLaunchKernelEx returns an error at the self-check, fall back to the plain <<<>>> launch and
  print '[splitkv] PDL off'; print '[splitkv] PDL on' otherwise.
expected:
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: faster
    why: "the combine's launch and block ramp overlap kernel A's tail instead of following it; the shape-minus-kernel-sum gap is ~1-2 us per graph node, so expect -1 to -2 us (~1%) on each shape, readable in the per-shape median, not the headline"
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: outside the split-KV path
risk: medium
depends_on: null
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# splitkv-pdl-attn-to-combine

## Change

Programmatic dependent launch between master's kernel A (splitkv_attn_kernel) and kernel B (splitkv_combine_kernel), full decode 1 < batch <= 64. (1) In splitkv_combine (host), launch kernel B with cudaLaunchKernelEx: cudaLaunchConfig_t with gridDim (B, 16), blockDim 128, stream = current stream, and one attribute cudaLaunchAttributeProgrammaticStreamSerialization with programmaticStreamSerializationAllowed = 1. (2) First statement of splitkv_combine_kernel: asm volatile("griddepcontrol.wait;" ::: "memory"); this blocks until kernel A has completed and its writes are visible, so the combine's reads are as safe as today. (3) In splitkv_attn_kernel, after the main tile loop and before the final cp_wait<0>(), every thread executes asm volatile("griddepcontrol.launch_dependents;" :::); a CTA that exits early (bx >= total) triggers implicitly by exiting. Nothing else changes; CUDA graph capture records the programmatic edge (CUDA >= 12.3; the judge image is cu130). Rename the module muse_full_decode_splitkv_v2. If cudaLaunchKernelEx returns an error at the self-check, fall back to the plain <<<>>> launch and print '[splitkv] PDL off'; print '[splitkv] PDL on' otherwise.

## Expected

- `*layer=full load=decode_16 * *layer=full load=decode_32 *` faster: the combine's launch and block ramp overlap kernel A's tail instead of following it; the shape-minus-kernel-sum gap is ~1-2 us per graph node, so expect -1 to -2 us (~1%) on each shape, readable in the per-shape median, not the headline
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: outside the split-KV path

## Risk

Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs). Below headline noise by design; judge on the full decode_16/32 medians against the paired SGLang ratio.
