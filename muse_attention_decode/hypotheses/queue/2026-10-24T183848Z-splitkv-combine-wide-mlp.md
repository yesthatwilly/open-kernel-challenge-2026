---
problem: muse_attention_decode
slug: splitkv-combine-wide-mlp
created: 2026-10-24T183848Z
base: master
area: split-KV combine
change: >
  Rework splitkv_combine_kernel (master full decode, 1 < batch <= 64) for more loads in flight;
  grid (B, 16) and the math stay. Today each (request, head) block has 4 warps; warp w walks
  partials w, w+4, ... (~20 per warp at 79 partials), each iteration loading m, then l and a
  float2 of acc, so the loop is a chain of L2 round trips. New: 8 warps (256 threads,
  __launch_bounds__(256), red_m/red_l/red_acc sized [8], loop 1 strides by 256). Loop 2: warp w
  handles partials c = w + 8 * i; process them in groups of 4: first load the 4 m values and 4 l
  values and the 4 float2 acc values into registers (masked by c < n, m = -inf when out of range),
  then compute the 4 weights and accumulate, so up to 12 independent loads are in flight per lane.
  The final warp-0 merge sums 8 entries; the epilogue (v_scale, scaleless RMSNorm with eps inside
  the sqrt, sigmoid(gate), bf16 store) is unchanged. Host launch uses 256 threads. Rename the
  module muse_full_decode_splitkv_v2.
expected:
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: faster
    why: "splitkv_combine_kernel costs 8.2 us at decode_16 and 8.4 us at decode_32 to merge ~5-10 MB of partials that sit in L2: it is latency-bound. 2x warps and 4-deep batched loads should bring it to ~3-4 us, -4% on the decode_16 shape, -2% on decode_32"
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: these shapes do not run the combine
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# splitkv-combine-wide-mlp

## Change

Rework splitkv_combine_kernel (master full decode, 1 < batch <= 64) for more loads in flight; grid (B, 16) and the math stay. Today each (request, head) block has 4 warps; warp w walks partials w, w+4, ... (~20 per warp at 79 partials), each iteration loading m, then l and a float2 of acc, so the loop is a chain of L2 round trips. New: 8 warps (256 threads, __launch_bounds__(256), red_m/red_l/red_acc sized [8], loop 1 strides by 256). Loop 2: warp w handles partials c = w + 8 * i; process them in groups of 4: first load the 4 m values and 4 l values and the 4 float2 acc values into registers (masked by c < n, m = -inf when out of range), then compute the 4 weights and accumulate, so up to 12 independent loads are in flight per lane. The final warp-0 merge sums 8 entries; the epilogue (v_scale, scaleless RMSNorm with eps inside the sqrt, sigmoid(gate), bf16 store) is unchanged. Host launch uses 256 threads. Rename the module muse_full_decode_splitkv_v2.

## Expected

- `*layer=full load=decode_16 * *layer=full load=decode_32 *` faster: splitkv_combine_kernel costs 8.2 us at decode_16 and 8.4 us at decode_32 to merge ~5-10 MB of partials that sit in L2: it is latency-bound. 2x warps and 4-deep batched loads should bring it to ~3-4 us, -4% on the decode_16 shape, -2% on decode_32
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: these shapes do not run the combine

## Risk

Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs).
