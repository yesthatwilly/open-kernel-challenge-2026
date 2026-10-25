---
problem: muse_moe_w4a16
slug: w13-marlin-atomic-add-t1-only
created: 2026-10-25T054545Z
base: master
area: w13 marlin
change: >
  Restack the one measured row of w13-marlin-atomic-add-on-master-small-m (job 171684977ec1470f,
  1.3558x check, on master 9080a71) with its gate narrowed to num_tokens == 1. In _forward, the
  w13 moe_wna16_marlin_gemm call gets common with use_atomic_add=True only when num_tokens == 1
  (decode_1 and the bs=1 test); every other T uses master's w13 kwargs byte for byte; w2 keeps
  master's atomic add at T<=32. Use a new constant W13_ATOMIC_MAX_TOKENS = 1 (do not reuse
  W2_ATOMIC_MAX_TOKENS: LESSONS, give each path its own bound). Keep the earlier run's prepare
  self-check of the w13 atomic mode at T=1 against master (rel-L2 vs fp32 dequant) and its
  fallback to master's kwargs. The Marlin kernel name is unchanged, so the bs=1 Marlin row is the
  attribution. Hard rules: CUDA-graph capturable; kernel_fn does no host sync
  (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except
  and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in
  registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes.
  The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may
  hold device pointers, TMA maps or buffers.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "the earlier run's bs=1 row: 44.02 vs 44.88 us adjusted, raw p10/p50 44.9/46.0 vs 46.1/47-48 in both master runs; profile Marlin pair 34.2 vs 42.1 us"
  - shapes: bs=8 *
    direction: flat
    why: "back to master: w13 atomic at bs=8 had raw p50 117 vs 109-110 us in master runs, so T=2..32 is dropped"
  - shapes: bs=16 *
    direction: flat
    why: master path
  - shapes: bs=32 *
    direction: flat
    why: master path (earlier run +1.2%)
risk: low
depends_on: w13-marlin-atomic-add-on-master-small-m
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# w13-marlin-atomic-add-t1-only

## Change

Restack the one measured row of w13-marlin-atomic-add-on-master-small-m (job 171684977ec1470f, 1.3558x check, on master 9080a71) with its gate narrowed to num_tokens == 1. In _forward, the w13 moe_wna16_marlin_gemm call gets common with use_atomic_add=True only when num_tokens == 1 (decode_1 and the bs=1 test); every other T uses master's w13 kwargs byte for byte; w2 keeps master's atomic add at T<=32. Use a new constant W13_ATOMIC_MAX_TOKENS = 1 (do not reuse W2_ATOMIC_MAX_TOKENS: LESSONS, give each path its own bound). Keep the earlier run's prepare self-check of the w13 atomic mode at T=1 against master (rel-L2 vs fp32 dequant) and its fallback to master's kwargs. The Marlin kernel name is unchanged, so the bs=1 Marlin row is the attribution. Hard rules: CUDA-graph capturable; kernel_fn does no host sync (.item/.cpu/synchronize), no allocation or branching on input values, no prints, no try/except and no per-call mutable state; nothing cached from inputs; NVFP4 weights stay NVFP4 (dequant in registers), activations BF16 where SGLang keeps BF16; attend every route; no child processes. The judge prepares TWO states (setup copies: 2) and rotates them, so no module/static global may hold device pointers, TMA maps or buffers.

## Expected

- `bs=1 *` faster: the earlier run's bs=1 row: 44.02 vs 44.88 us adjusted, raw p10/p50 44.9/46.0 vs 46.1/47-48 in both master runs; profile Marlin pair 34.2 vs 42.1 us
- `bs=8 *` flat: back to master: w13 atomic at bs=8 had raw p50 117 vs 109-110 us in master runs, so T=2..32 is dropped
- `bs=16 *` flat: master path
- `bs=32 *` flat: master path (earlier run +1.2%)

## Risk

Identical code to the 1.3558 run would dedupe, so this is the only way to restack it; the expected bs=1 gain (~-1 to -2 us, z~1-2) is near resolution, and the bs=8 recovery is the second half of the bet. If it reads flat, close the w13 atomic knob.
