---
problem: muse_attention_decode
slug: full-decode1-sk-pdl-from-preattn
created: 2026-10-25T042335Z
base: master
area: launch overhead (full decode_1, pre-attention -> stream-K kernel)
change: >
  Built on master (stream-K two-level ticket at full decode B=1, landed c3c1b01). Full decode_1 is
  22.6 us shape vs pre-attn 1.4 + sk kernel 18.6 us of kernels, so ~2.6 us is launch gap plus the
  serial ring fill. At batch 1 only (config): launch splitkv_attn_sk_kernel with
  cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 after master
  Triton _pre_attention_kernel (which needs no change: without an explicit trigger the dependent
  may launch as the primary drains). In the kernel, before cudaGridDependencySynchronize()
  (griddepcontrol.wait) do everything that does not read pre-attention outputs: smem/ticket setup,
  the tile-range computation, page_table index loads, and cp.async issue of the prologue ring
  stages for tiles whose keys are all < seq_b - 1 (old keys; the pre-attention kernel writes only
  key seq_b - 1 at cache_loc[b]). After the wait: load the FP8 q and issue any tile containing key
  seq_b - 1 (only the last tile of the request, normally the final CTA). Workspace and ticket
  writes stay after the wait. Batch > 1 unchanged (the running
  full-decode-splitkv-s3-fused-preattn targets B>1 pre-attention). Self-check: the landed B=1
  cases plus two graph replays with PDL on; fallback to plain launch on any error (PATH CHOSEN:
  full decode_1 two-level + PDL / no PDL (reason)). Print a prepare CUDA-event time of
  pre-attn+kernel with and without PDL and assert PDL is not slower.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: overlaps the sk kernel launch, index loads and ring fill with the 1.4 us pre-attention kernel; shape 22.6 -> ~20.5 us (-1 to -2.5 us)
  - shapes: "*layer=full load=decode_16* *layer=full load=decode_32* *layer=sliding* *load=verify_*"
    direction: flat
    why: outside the batch-1 gate
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# full-decode1-sk-pdl-from-preattn

## Change

Built on master (stream-K two-level ticket at full decode B=1, landed c3c1b01). Full decode_1 is 22.6 us shape vs pre-attn 1.4 + sk kernel 18.6 us of kernels, so ~2.6 us is launch gap plus the serial ring fill. At batch 1 only (config): launch splitkv_attn_sk_kernel with cudaLaunchKernelEx and cudaLaunchAttributeProgrammaticStreamSerialization = 1 after master Triton _pre_attention_kernel (which needs no change: without an explicit trigger the dependent may launch as the primary drains). In the kernel, before cudaGridDependencySynchronize() (griddepcontrol.wait) do everything that does not read pre-attention outputs: smem/ticket setup, the tile-range computation, page_table index loads, and cp.async issue of the prologue ring stages for tiles whose keys are all < seq_b - 1 (old keys; the pre-attention kernel writes only key seq_b - 1 at cache_loc[b]). After the wait: load the FP8 q and issue any tile containing key seq_b - 1 (only the last tile of the request, normally the final CTA). Workspace and ticket writes stay after the wait. Batch > 1 unchanged (the running full-decode-splitkv-s3-fused-preattn targets B>1 pre-attention). Self-check: the landed B=1 cases plus two graph replays with PDL on; fallback to plain launch on any error (PATH CHOSEN: full decode_1 two-level + PDL / no PDL (reason)). Print a prepare CUDA-event time of pre-attn+kernel with and without PDL and assert PDL is not slower.

## Expected

- `*layer=full load=decode_1 *` faster: overlaps the sk kernel launch, index loads and ring fill with the 1.4 us pre-attention kernel; shape 22.6 -> ~20.5 us (-1 to -2.5 us)
- `*layer=full load=decode_16* *layer=full load=decode_32* *layer=sliding* *load=verify_*` flat: outside the batch-1 gate

## Risk

Kill: decode_1 full shape not >= 1 us faster. splitkv-pdl-attn-to-combine (A->B) was neutral; this boundary differs because the dependent here can do useful gather work before the wait.
