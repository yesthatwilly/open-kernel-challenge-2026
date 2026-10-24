---
problem: muse_prefill_layer
slug: experts-dequant-overlap-attention
created: 2026-10-23T230250Z
base: master
area: w13 marlin + w2 marlin (dequant placement on a side stream)
change: >
  On the dequant path of experts-dequant-triton-moe-16k (tokens >= DEQUANT_MIN_TOKENS) move the
  weight dequant off the critical path: prepare() creates state['dequant_stream'] =
  torch.cuda.Stream(device). At the very start of kernel_fn, before the qkv GEMM is issued:
  dequant_stream.wait_stream(main); under torch.cuda.stream(dequant_stream) allocate w13_bf16 and
  w2_bf16 and launch the two dequant kernels; main.wait_stream(dequant_stream) immediately before
  the first invoke_fused_moe_kernel; record_stream both buffers on main when not capturing. The
  dequant depends only on the weights, so under graph capture it becomes a parallel branch that
  overlaps the attention block and pooler (qkv GEMM, FA3, o GEMM, boundary, pooler GEMM, router),
  which are compute-bound, against a bandwidth-bound dequant (0.77 GB read, 3.2 GB write). Nothing
  else changes; the Marlin path below the gate is untouched.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: hides most of the ~1-1.5 ms dequant behind 3-10 ms of compute-bound attention-block work; the share hidden is bounded by HBM contention with FA3 KV reads on the ctx 32K/64K full shapes
  - shapes: "*load=turn_1*"
    direction: flat
    why: below the gate (control)
  - shapes: "*load=turn_4*"
    direction: flat
    why: below the gate (control)
risk: medium
depends_on: experts-dequant-triton-moe-16k
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# experts-dequant-overlap-attention

## Change

On the dequant path of experts-dequant-triton-moe-16k (tokens >= DEQUANT_MIN_TOKENS) move the weight dequant off the critical path: prepare() creates state['dequant_stream'] = torch.cuda.Stream(device). At the very start of kernel_fn, before the qkv GEMM is issued: dequant_stream.wait_stream(main); under torch.cuda.stream(dequant_stream) allocate w13_bf16 and w2_bf16 and launch the two dequant kernels; main.wait_stream(dequant_stream) immediately before the first invoke_fused_moe_kernel; record_stream both buffers on main when not capturing. The dequant depends only on the weights, so under graph capture it becomes a parallel branch that overlaps the attention block and pooler (qkv GEMM, FA3, o GEMM, boundary, pooler GEMM, router), which are compute-bound, against a bandwidth-bound dequant (0.77 GB read, 3.2 GB write). Nothing else changes; the Marlin path below the gate is untouched.

## Expected

- `*load=first_turn*` faster: hides most of the ~1-1.5 ms dequant behind 3-10 ms of compute-bound attention-block work; the share hidden is bounded by HBM contention with FA3 KV reads on the ctx 32K/64K full shapes
- `*load=turn_1*` flat: below the gate (control)
- `*load=turn_4*` flat: below the gate (control)

## Risk

Medium risk: stream ordering only, no math change. Failure modes: the parent's dequant path must have landed as a gain or neutral (the dependency enforces it); forgetting record_stream in eager mode gives use-after-free in the tests; HBM contention could leave the gain below the interval on the ctx 64K full shape (attention there streams 21 MB per CTA group but is compute-bound). Rules: unchanged from the parent; the BF16 buffers still live only inside the timed call. Attribution: the delta against the parent is the dequant time hidden, nothing else moves.
