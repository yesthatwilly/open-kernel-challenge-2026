---
problem: muse_moe_w4a16
slug: triton-w13-silu-epilogue
created: 2026-10-23T225501Z
base: master
area: silu-mul
change: >
  On top of triton-w13-grouped-gemv-small-m (small-M branch only), fold silu_and_mul into the w13
  Triton kernel's epilogue: change the w13_gemv grid to (32, I/BLOCK_N) and have program (e, n)
  accumulate two fp32 tiles per K step, gate from w13 rows [n*BLOCK_N, (n+1)*BLOCK_N) and up from
  rows I + [n*BLOCK_N, (n+1)*BLOCK_N) (both dequantized from the same expert's NVFP4 bank), then
  after the K loop apply the global scale to both, round each to bf16 (so the SwiGLU input matches
  Marlin's bf16 cache1 numerics), compute silu(gate)*up in fp32 and store the bf16 result straight
  into cache2 [M*8, I] at rows = slot. Remove the silu_and_mul launch and the cache1 buffer from
  the small-M branch; the w2 GEMM reads cache2 unchanged. Shapes with num_tokens > 8 untouched.
expected:
  - shapes: bs=1 *
    direction: faster
    why: "one launch fewer (silu_and_mul, ~3-4 us incl. its 64 KB read/write at bs=1) and cache1's 2I-wide write+read disappears; at ~40 us total that is visible"
  - shapes: bs=8 *
    direction: faster
    why: same; cache1 at bs=8 is 64 x 8192 x 2 B written then read
  - shapes: bs=16 *
    direction: flat
    why: "control: untouched"
risk: medium
depends_on: triton-w13-grouped-gemv-small-m
priority: 3
kind: primary
controls_for: null
review: required
status: hold
hold_reason: stacks on triton-w13-grouped-gemv-small-m, which regressed (2-2.6x slower than Marlin at small M)
---

# triton-w13-silu-epilogue

## Change

On top of triton-w13-grouped-gemv-small-m (small-M branch only), fold silu_and_mul into the w13 Triton kernel's epilogue: change the w13_gemv grid to (32, I/BLOCK_N) and have program (e, n) accumulate two fp32 tiles per K step, gate from w13 rows [n*BLOCK_N, (n+1)*BLOCK_N) and up from rows I + [n*BLOCK_N, (n+1)*BLOCK_N) (both dequantized from the same expert's NVFP4 bank), then after the K loop apply the global scale to both, round each to bf16 (so the SwiGLU input matches Marlin's bf16 cache1 numerics), compute silu(gate)*up in fp32 and store the bf16 result straight into cache2 [M*8, I] at rows = slot. Remove the silu_and_mul launch and the cache1 buffer from the small-M branch; the w2 GEMM reads cache2 unchanged. Shapes with num_tokens > 8 untouched.

## Expected

- `bs=1 *` faster: one launch fewer (silu_and_mul, ~3-4 us incl. its 64 KB read/write at bs=1) and cache1's 2I-wide write+read disappears; at ~40 us total that is visible
- `bs=8 *` faster: same; cache1 at bs=8 is 64 x 8192 x 2 B written then read
- `bs=16 *` flat: control: untouched

## Risk

Risks: per-program weight traffic doubles (gate+up tiles) with the same BLOCK_M, so register pressure may force a smaller BLOCK_N; the bf16 rounding of gate/up before silu must be kept to stay within the reference's margins (fp32 would be more accurate, not less, but match SGLang's activation precision). Gain is bounded by one launch per call, so the result may sit inside the +/-1.5% CI on verify_1; it is still worth the run because decode_1 is 1/6 of the score.
