---
problem: muse_prefill_layer
slug: experts-rega-rotate-finebucket-fastsilu-combo
created: 2026-10-26T224003Z
base: master
area: "w13 + w2 expert GEMMs (rega: rotated schedule + 16-row N buckets + fast silu, one module)"
change: >
  Built on master (blob f4751fc0); no depends_on, so the stack of the three rega edits queued
  beside it is measured today rather than after its parents harvest (dispatch stops ~23:45Z). One
  derived module muse_prefill_rega_combo (asserted edits of master's source; master's module
  untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted
  string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two
  states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare
  lines) with three runtime int arguments added to rega_gemm and passed to the kernel (uniform
  values, no template instantiations beyond the bucket set): ROT = the round-rotated walk of
  experts-rega-round-rotated-schedule (it = r * G + (blockIdx.x + r * (n_tiles + 1)) % G in the
  producer and the consumers); FINE = experts-rega-fine-n-buckets (Wg<NB> for 16..128 in steps of
  16, 8 tensor maps, m.bucket = nbk - 1 when FINE, else master's map onto bucket indices 0 / 1 / 3
  / 7, tx from 16 * (m.bucket + 1)); FSILU = experts-rega-w13-fast-silu-epilogue (silu_mul through
  __expf and __frcp_rn when set). Implement each exactly as its sibling's change text says; if a
  sibling's branch exists, take its code. GATE per class (low T=753, high T=3012, ft T=16384):
  flag sets {000, 100, 010, 001, 111} against master's own module: bit-exact hs and y for sets
  without FSILU, the fsilu sibling's rel-L2 / fp64 rule for sets with it; then 2 interleaved
  rounds x 11 cold-L2 graph alternations of the routed rega pair for every set and master's
  module, on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give
  for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random; ship the
  fastest set whose pop ratio vs master's module is <= 0.98 in both rounds and whose random ratio
  is <= 1.00, else master's module; state['rega_combo'][cls] = the set or None. Print every set's
  KERNELS w13 / w2 us so each flag's share is read from one log, and PATH CHOSEN. If prepare
  passes ~230 s, skip _rega_lo_phase (OFF in every run since combo-b).
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "rotation (~4-10%) + N buckets (~3-5%) + silu (~2%) on rega 1098.5 / 1133.9 us: ~4-8% of each turn_4 shape"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "N buckets (~4-8%) + rotation (~2-4%) + silu (~2%) on rega 418.5 / 437.9 us: ~4-7% of each turn_1 shape"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "rotation (~2-4%) + silu (~2-3%) + buckets (~1%) on rega 5067-5739 us: ~2-4% of each first_turn shape"
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "experts-rega-round-rotated-schedule: this carries that rotation together with experts-rega-fine-n-buckets and experts-rega-w13-fast-silu-epilogue in one module with a per-flag-set gate, so the combination is measured today instead of after the three parents finish"
review: required
status: queued
---

# experts-rega-rotate-finebucket-fastsilu-combo

## Change

Built on master (blob f4751fc0); no depends_on, so the stack of the three rega edits queued beside it is measured today rather than after its parents harvest (dispatch stops ~23:45Z). One derived module muse_prefill_rega_combo (asserted edits of master's source; master's module untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines) with three runtime int arguments added to rega_gemm and passed to the kernel (uniform values, no template instantiations beyond the bucket set): ROT = the round-rotated walk of experts-rega-round-rotated-schedule (it = r * G + (blockIdx.x + r * (n_tiles + 1)) % G in the producer and the consumers); FINE = experts-rega-fine-n-buckets (Wg<NB> for 16..128 in steps of 16, 8 tensor maps, m.bucket = nbk - 1 when FINE, else master's map onto bucket indices 0 / 1 / 3 / 7, tx from 16 * (m.bucket + 1)); FSILU = experts-rega-w13-fast-silu-epilogue (silu_mul through __expf and __frcp_rn when set). Implement each exactly as its sibling's change text says; if a sibling's branch exists, take its code. GATE per class (low T=753, high T=3012, ft T=16384): flag sets {000, 100, 010, 001, 111} against master's own module: bit-exact hs and y for sets without FSILU, the fsilu sibling's rel-L2 / fp64 rule for sets with it; then 2 interleaved rounds x 11 cold-L2 graph alternations of the routed rega pair for every set and master's module, on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random; ship the fastest set whose pop ratio vs master's module is <= 0.98 in both rounds and whose random ratio is <= 1.00, else master's module; state['rega_combo'][cls] = the set or None. Print every set's KERNELS w13 / w2 us so each flag's share is read from one log, and PATH CHOSEN. If prepare passes ~230 s, skip _rega_lo_phase (OFF in every run since combo-b).

## Expected

- `*load=turn_4*` faster: rotation (~4-10%) + N buckets (~3-5%) + silu (~2%) on rega 1098.5 / 1133.9 us: ~4-8% of each turn_4 shape
- `*load=turn_1*` faster: N buckets (~4-8%) + rotation (~2-4%) + silu (~2%) on rega 418.5 / 437.9 us: ~4-7% of each turn_1 shape
- `*load=first_turn*` faster: rotation (~2-4%) + silu (~2-3%) + buckets (~1%) on rega 5067-5739 us: ~2-4% of each first_turn shape

## Risk

About 45 minutes, not 30: three small edits and a five-set gate. Size ~+2.5-4% headline if the three estimates hold; the per-set KERNELS lines attribute each piece, so a win here also says which parent to keep.
