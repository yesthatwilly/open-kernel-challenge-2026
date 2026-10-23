---
problem: muse_moe_w4a16
slug: w13-marlin-atomic-add-small-m
created: 2026-10-23T225502Z
base: master
area: w13 marlin
change: >
  For num_tokens <= 32, pass use_atomic_add=True (and use_fp32_reduce=False, which Marlin requires
  alongside atomic add) to the w13 moe_wna16_marlin_gemm call only; the w2 call keeps SGLang's
  use_atomic_add=False / use_fp32_reduce=True. Marlin's non-atomic k-parallel reduction serializes
  slice partials through the workspace locks and an fp32 scratch (c_tmp) write+read; with atomic
  add each k-slice accumulates straight into the bf16 output, which must be zeroed first (one
  cache1.zero_() or memset launch, M*8 x 2I bf16: 128 KB at bs=1). This is the mode SGLang's dense
  Marlin path picks for small M (should_use_atomic_add_reduce); the MoE runner leaves it off.
  Shapes above 32 tokens untouched.
expected:
  - shapes: bs=1 *
    direction: faster
    why: at M=1 the w13 GEMM is one m-block with k-parallel slices; the lock-serialized fp32 reduce is a larger fraction of a ~30 us kernel than the memset it replaces
  - shapes: bs=8 *
    direction: faster
    why: same at bs=8
  - shapes: bs=16 *
    direction: faster
    why: same mechanism at bs=16; smaller share
  - shapes: bs=128 *
    direction: flat
    why: "control: untouched (M=128 > 32)"
risk: medium
depends_on: sglang-marlin-path-parity
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# w13-marlin-atomic-add-small-m

## Change

For num_tokens <= 32, pass use_atomic_add=True (and use_fp32_reduce=False, which Marlin requires alongside atomic add) to the w13 moe_wna16_marlin_gemm call only; the w2 call keeps SGLang's use_atomic_add=False / use_fp32_reduce=True. Marlin's non-atomic k-parallel reduction serializes slice partials through the workspace locks and an fp32 scratch (c_tmp) write+read; with atomic add each k-slice accumulates straight into the bf16 output, which must be zeroed first (one cache1.zero_() or memset launch, M*8 x 2I bf16: 128 KB at bs=1). This is the mode SGLang's dense Marlin path picks for small M (should_use_atomic_add_reduce); the MoE runner leaves it off. Shapes above 32 tokens untouched.

## Expected

- `bs=1 *` faster: at M=1 the w13 GEMM is one m-block with k-parallel slices; the lock-serialized fp32 reduce is a larger fraction of a ~30 us kernel than the memset it replaces
- `bs=8 *` faster: same at bs=8
- `bs=16 *` faster: same mechanism at bs=16; smaller share
- `bs=128 *` flat: control: untouched (M=128 > 32)

## Risk

Risks: bf16 atomic accumulation across k-slices rounds each partial to bf16 (SGLang itself accepts this for its dense small-M Marlin path; rel-L2 tolerance 1.5% leaves wide margin, per-row 5x bound is the one to watch on near-zero rows); the added memset may cost what the reduce saves at bs=1, in which case the result is flat and the knob is retired. If the sgl_kernel binding rejects the combination on MoE, record as rejected. Activations remain bf16 as SGLang keeps them.
