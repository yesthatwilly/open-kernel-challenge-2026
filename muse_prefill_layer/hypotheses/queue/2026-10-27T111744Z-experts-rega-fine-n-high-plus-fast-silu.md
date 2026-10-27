---
problem: muse_prefill_layer
slug: experts-rega-fine-n-high-plus-fast-silu
created: 2026-10-27T111744Z
base: master
area: rega experts kernel (w13/w2 N buckets at the high class + w13 silu epilogue)
change: >
  Restack of experts-rega-fine-n-buckets (neutral, check c951d916; its patch is
  muse_prefill_layer/hypotheses/done/neutral/experts-rega-fine-n-buckets.patch) on current master,
  plus the queued experts-rega-w13-fast-silu-epilogue change, with no rotation. (1) Keep the
  fine-N module muse_prefill_rega_fine and its _rega_fine_phase exactly as in the parent (8 N
  buckets of 16*(b+1), Maps of 8 tensor maps, the fine-4map overhead control, the 200 s pre-build
  skip, bit-exact compare, the replay check), but gate and route the HIGH class (T=3012 class)
  only: low and ft stay master's rega_gemm without being timed (the parent measured low pop 1.015
  / random 1.043 and ft 1.010 / 1.011, both misses), which also saves their prepare seconds. (2)
  Add the fast-silu epilogue as a SECOND asserted string edit, applied to a separate derived
  module muse_prefill_rega_fsilu built from master's source (silu_mul returns hg * __frcp_rn(1.f +
  __expf(-hg)) * hu, fp32 throughout, the same single bf16 rounding in pack_bf16; w2 and
  everything else master's), gated per class (low T=753, high T=3012, ft T=16384) exactly as
  experts-rega-w13-fast-silu-epilogue specifies (hs vs master finite, rel L2 <= 2e-3, routed
  output error vs fp64 at most 1.05x master's; then 2 interleaved rounds x 11 cold-L2 alternations
  of the rega pair on pop; ON if <= 0.99 both rounds; random pooled <= 1.01 as a no-regression
  guard). (3) If both route at the high class, also build and time the combined module (fine-N
  buckets + fast silu, the two edits together) at the high class only, against master and against
  each single module, and route the fastest that clears its gate, so every part is attributable
  from the prepare-time KERNELS lines. Dispatch in _experts_rega: per class, the routed module's
  rega_gemm (combined, fine, fsilu, or master), with rega_low_variant precedence kept (no derived
  module at the low class unless rega_low_variant == 0). All phases run last in prepare, after
  _rega_lo_phase, with a pre-build elapsed check (skip with OFF lines past ~200 s; the parent
  reached 211 s total with one extra module, so build each derived module only when its class is
  still in play and print build seconds). PATH CHOSEN lines last.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: fine-N at the high class measured rega pair -41 us (pop) / -60 us (random) in-prepare (c951d916); fast silu adds ~2-3% of w13 (~-15 to -20 us); together ~-50 to -70 us on 2111-3706 us rows, ~1.5-2.5% of turn_4
  - shapes: "*load=turn_1*"
    direction: faster
    why: "fast silu only (fine-N stays OFF at low): ~-6 us, ~0.5-0.8%"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "fast silu only: ~-70 to -110 us of 11-18 ms, ~0.6-0.9%; fine-N stays OFF at ft"
risk: low
depends_on: experts-rega-fine-n-buckets
priority: 1
kind: primary
controls_for: null
prior: "experts-rega-fine-n-buckets: that run gated all three classes and only high routed (pop 0.961/0.969, random 0.920, bit-exact); this one keeps fine-N at high only, drops the losing classes' timing, and stacks the untested fast-silu epilogue behind its own gate, with no rotation (experts-rega-round-rotated-schedule was neutral)"
review: required
status: queued
---

# experts-rega-fine-n-high-plus-fast-silu

## Change

Restack of experts-rega-fine-n-buckets (neutral, check c951d916; its patch is muse_prefill_layer/hypotheses/done/neutral/experts-rega-fine-n-buckets.patch) on current master, plus the queued experts-rega-w13-fast-silu-epilogue change, with no rotation. (1) Keep the fine-N module muse_prefill_rega_fine and its _rega_fine_phase exactly as in the parent (8 N buckets of 16*(b+1), Maps of 8 tensor maps, the fine-4map overhead control, the 200 s pre-build skip, bit-exact compare, the replay check), but gate and route the HIGH class (T=3012 class) only: low and ft stay master's rega_gemm without being timed (the parent measured low pop 1.015 / random 1.043 and ft 1.010 / 1.011, both misses), which also saves their prepare seconds. (2) Add the fast-silu epilogue as a SECOND asserted string edit, applied to a separate derived module muse_prefill_rega_fsilu built from master's source (silu_mul returns hg * __frcp_rn(1.f + __expf(-hg)) * hu, fp32 throughout, the same single bf16 rounding in pack_bf16; w2 and everything else master's), gated per class (low T=753, high T=3012, ft T=16384) exactly as experts-rega-w13-fast-silu-epilogue specifies (hs vs master finite, rel L2 <= 2e-3, routed output error vs fp64 at most 1.05x master's; then 2 interleaved rounds x 11 cold-L2 alternations of the rega pair on pop; ON if <= 0.99 both rounds; random pooled <= 1.01 as a no-regression guard). (3) If both route at the high class, also build and time the combined module (fine-N buckets + fast silu, the two edits together) at the high class only, against master and against each single module, and route the fastest that clears its gate, so every part is attributable from the prepare-time KERNELS lines. Dispatch in _experts_rega: per class, the routed module's rega_gemm (combined, fine, fsilu, or master), with rega_low_variant precedence kept (no derived module at the low class unless rega_low_variant == 0). All phases run last in prepare, after _rega_lo_phase, with a pre-build elapsed check (skip with OFF lines past ~200 s; the parent reached 211 s total with one extra module, so build each derived module only when its class is still in play and print build seconds). PATH CHOSEN lines last.

## Expected

- `*load=turn_4*` faster: fine-N at the high class measured rega pair -41 us (pop) / -60 us (random) in-prepare (c951d916); fast silu adds ~2-3% of w13 (~-15 to -20 us); together ~-50 to -70 us on 2111-3706 us rows, ~1.5-2.5% of turn_4
- `*load=turn_1*` faster: fast silu only (fine-N stays OFF at low): ~-6 us, ~0.5-0.8%
- `*load=first_turn*` faster: fast silu only: ~-70 to -110 us of 11-18 ms, ~0.6-0.9%; fine-N stays OFF at ft

## Risk

Headline effect ~+0.8-1.3% if both route: around the 1.24% one-run margin, so read the prepare KERNELS lines. Risks: the fast-silu accuracy bound (rel L2 2e-3) must hold; the prepare budget (two or three derived module builds at ~40 s each) is the main hazard, hence the per-module build only when its class is in play and the 200 s skip.
