---
problem: muse_prefill_layer
slug: experts-rega-w13-fast-silu-epilogue
created: 2026-10-26T224003Z
base: master
area: silu-mul (rega w13 epilogue)
change: >
  Built on master (blob f4751fc0). rega_kernel<true, *> (w13) applies silu(gate) * up to its fp32
  accumulators in the epilogue through silu_mul(hg, hu) = hg / (1.f + expf(-hg)) * hu. The module
  is built without fast math (REGA_CUDA_CFLAGS: -O3 -std=c++20 -lineinfo), so each element costs a
  full-accuracy expf and an IEEE division (roughly 25-30 instructions), and every consumer thread
  evaluates NB/4 of them per item (32 at N = 128) between items while the tensor core idles: an
  estimated 3-5% of w13's time. CHANGE: derive a module muse_prefill_rega_fsilu from master's
  source by one asserted edit (master's module untouched; scaffolding as in
  done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's
  _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved
  rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines): silu_mul returns
  hg * __frcp_rn(1.f + __expf(-hg)) * hu (fp32 throughout, the same single bf16 rounding in
  pack_bf16; for very negative hg __expf overflows to +inf, the reciprocal is 0 and the product is
  -0, as the accurate form gives). w2's epilogue and everything else are master's. GATE per class
  (low T=753, high T=3012, ft T=16384): hs of fsilu vs master on random, skewed and pop routings:
  finite, rel L2 <= 2e-3, and the routed _experts output's error against the fp64 reference
  (master's _rega_reference_y path) at most 1.05x master's; then 2 interleaved rounds x 11 cold-L2
  graph alternations of the routed rega pair, master vs fsilu, on the pop routing; ON if <= 0.99
  in both rounds; state['rega_fsilu_ok'][cls]. Print w13 us both ways, hs rel L2 and max abs
  difference, PATH CHOSEN.
expected:
  - shapes: "*"
    direction: faster
    why: "w13 is two thirds of rega in every class; cutting most of the silu epilogue is ~2-3% of w13: sliding turn_1 ~-6 us, sliding turn_4 ~-15 to -20 us, sliding first_turn ~-70 to -110 us, ~0.7-1% of each shape"
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "experts-triton-w13-fused-silu-epilogue: that added silu to the old Triton engine's w13 epilogue (more epilogue work; lost 1-3% at first_turn); rega already fuses silu, and this only swaps the accurate expf and IEEE division for __expf and __frcp_rn"
review: required
status: queued
---

# experts-rega-w13-fast-silu-epilogue

## Change

Built on master (blob f4751fc0). rega_kernel<true, *> (w13) applies silu(gate) * up to its fp32 accumulators in the epilogue through silu_mul(hg, hu) = hg / (1.f + expf(-hg)) * hu. The module is built without fast math (REGA_CUDA_CFLAGS: -O3 -std=c++20 -lineinfo), so each element costs a full-accuracy expf and an IEEE division (roughly 25-30 instructions), and every consumer thread evaluates NB/4 of them per item (32 at N = 128) between items while the tensor core idles: an estimated 3-5% of w13's time. CHANGE: derive a module muse_prefill_rega_fsilu from master's source by one asserted edit (master's module untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines): silu_mul returns hg * __frcp_rn(1.f + __expf(-hg)) * hu (fp32 throughout, the same single bf16 rounding in pack_bf16; for very negative hg __expf overflows to +inf, the reciprocal is 0 and the product is -0, as the accurate form gives). w2's epilogue and everything else are master's. GATE per class (low T=753, high T=3012, ft T=16384): hs of fsilu vs master on random, skewed and pop routings: finite, rel L2 <= 2e-3, and the routed _experts output's error against the fp64 reference (master's _rega_reference_y path) at most 1.05x master's; then 2 interleaved rounds x 11 cold-L2 graph alternations of the routed rega pair, master vs fsilu, on the pop routing; ON if <= 0.99 in both rounds; state['rega_fsilu_ok'][cls]. Print w13 us both ways, hs rel L2 and max abs difference, PATH CHOSEN.

## Expected

- `*` faster: w13 is two thirds of rega in every class; cutting most of the silu epilogue is ~2-3% of w13: sliding turn_1 ~-6 us, sliding turn_4 ~-15 to -20 us, sliding first_turn ~-70 to -110 us, ~0.7-1% of each shape

## Risk

Not bit-exact (rare 1-ulp bf16 differences in hs), bounded by the gate against master and the fp64 reference. Size ~+0.5-0.8% headline: below the judge margin alone, so it lands through experts-rega-rotate-finebucket-fastsilu-combo or experts-rega-rotation-fastsilu-combo.
