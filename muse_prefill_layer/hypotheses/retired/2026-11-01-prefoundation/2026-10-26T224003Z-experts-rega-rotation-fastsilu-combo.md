---
problem: muse_prefill_layer
slug: experts-rega-rotation-fastsilu-combo
created: 2026-10-26T224003Z
base: master
area: "w13 + w2 expert GEMMs (rega: rotated schedule + fast silu, one module)"
change: >
  Built on master (blob f4751fc0); no depends_on. The low-risk subset of
  experts-rega-rotate-finebucket-fastsilu-combo: no new wgmma shapes and no doubled compile, so it
  is the stack to trust if the fine-bucket build has trouble. One derived module
  muse_prefill_rega_rotfs (asserted edits of master's source; master's module untouched;
  scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits
  of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2
  interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines) with
  two runtime int arguments on rega_gemm: ROT = the round-rotated walk of
  experts-rega-round-rotated-schedule (it = r * G + (blockIdx.x + r * (n_tiles + 1)) % G in the
  producer and the consumers); FSILU = experts-rega-w13-fast-silu-epilogue (silu_mul through
  __expf and __frcp_rn). Implement each as its sibling's change text says. GATE per class (low
  T=753, high T=3012, ft T=16384): sets {00, 10, 01, 11} against master's module: bit-exact hs and
  y for ROT alone, the fsilu rel-L2 / fp64 rule for sets with FSILU; 2 interleaved rounds x 11
  cold-L2 graph alternations of the routed rega pair on the pop routing (_rega_pop_case with the
  alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as
  _rega_lo_phase calibrates) and on random; ship the fastest set with pop <= 0.98 in both rounds
  and random <= 1.00, else master's module; state['rega_rotfs'][cls]. Print every set's KERNELS
  w13 / w2 us and PATH CHOSEN.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "rotation (~4-10%) + silu (~2%) on rega 1098.5 / 1133.9 us: ~3-6% of each turn_4 shape"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "rotation (~2-4%) + silu (~2%) on rega 418.5 / 437.9 us: ~2-3% of each turn_1 shape"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "rotation (~2-4%) + silu (~2-3%) on rega 5067-5739 us: ~2-3% of each first_turn shape"
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "experts-rega-rotate-finebucket-fastsilu-combo: the same stack without the fine N buckets (no new wgmma shapes, half the compile), as the low-risk combination"
review: required
status: hold
hold_reason: "rotation was neutral on its own 2026-10-27 (d00770b6: pop x0.991-1.004, all classes OFF); a combo built on it would carry dead code. Re-mint fine-N (high class) + fast-silu without rotation via /hypothesize."
---

# experts-rega-rotation-fastsilu-combo

## Change

Built on master (blob f4751fc0); no depends_on. The low-risk subset of experts-rega-rotate-finebucket-fastsilu-combo: no new wgmma shapes and no doubled compile, so it is the stack to trust if the fine-bucket build has trouble. One derived module muse_prefill_rega_rotfs (asserted edits of master's source; master's module untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines) with two runtime int arguments on rega_gemm: ROT = the round-rotated walk of experts-rega-round-rotated-schedule (it = r * G + (blockIdx.x + r * (n_tiles + 1)) % G in the producer and the consumers); FSILU = experts-rega-w13-fast-silu-epilogue (silu_mul through __expf and __frcp_rn). Implement each as its sibling's change text says. GATE per class (low T=753, high T=3012, ft T=16384): sets {00, 10, 01, 11} against master's module: bit-exact hs and y for ROT alone, the fsilu rel-L2 / fp64 rule for sets with FSILU; 2 interleaved rounds x 11 cold-L2 graph alternations of the routed rega pair on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random; ship the fastest set with pop <= 0.98 in both rounds and random <= 1.00, else master's module; state['rega_rotfs'][cls]. Print every set's KERNELS w13 / w2 us and PATH CHOSEN.

## Expected

- `*load=turn_4*` faster: rotation (~4-10%) + silu (~2%) on rega 1098.5 / 1133.9 us: ~3-6% of each turn_4 shape
- `*load=turn_1*` faster: rotation (~2-4%) + silu (~2%) on rega 418.5 / 437.9 us: ~2-3% of each turn_1 shape
- `*load=first_turn*` faster: rotation (~2-4%) + silu (~2-3%) on rega 5067-5739 us: ~2-3% of each first_turn shape

## Risk

About 35 minutes. Size ~+1.5-2.8% headline if both estimates hold.
