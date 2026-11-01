---
problem: muse_prefill_layer
slug: experts-rega-lpt-group-order
created: 2026-10-26T224008Z
base: master
area: "w13 + w2 expert GEMMs (rega persistent schedule: expert groups heaviest first)"
change: >
  Built on master (blob f4751fc0). rega walks expert groups in local-expert-id order, so the last
  round of the persistent loop holds whatever chunks the highest ids own: at turn_4 (~31 items per
  CTA at the judge-like load, one partial last round) and in w2 at turn_1 (8 items per CTA) a few
  CTAs finish on a heavy N = 128 item while the rest idle. CHANGE: derive a module
  muse_prefill_rega_lpt from master's source (master's module untouched; scaffolding as in
  done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's
  _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved
  rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines). After the
  per-expert chunk-group pass (s_gfirst, s_gcount), thread 0 computes for every local expert with
  chunks key = (sum of nbk over its chunks) / gcount (mean 16-route blocks per chunk, a proxy for
  per-item cost), sorts the experts by key descending (stable on expert id; at most 32), and
  writes s_pslot[e] (the expert's first chunk slot in that order: a prefix sum of gcount) and
  s_slot2e[slot] (uint8 expert id per chunk slot, Cap<BIG>::C entries, so the static shared memory
  stays under its 16384 B bound); __syncthreads. decode_item then takes cg = it / T; e =
  s_slot2e[cg]; f = s_gfirst[e]; k = s_gcount[e]; local = it - s_pslot[e] * T; c = f + local % k;
  j = local / k, in the producer and the consumers alike; the chunk tables and Item fields are
  unchanged. Same items, same per-item code, so outputs are bit-identical. GATE as
  experts-rega-round-rotated-schedule: bit-exact compare on random, skewed, one and pop routings,
  two-state replay check, 2 interleaved rounds x 11 cold-L2 alternations on the pop routing
  (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local
  routes, ~2.2, as _rega_lo_phase calibrates) and on random; ON if pop <= 0.98 in both rounds and
  random <= 1.00; state['rega_lpt_ok'][cls]. Print KERNELS w13 / w2 us both ways and PATH CHOSEN.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "the partial last round at ~31 items per CTA becomes the cheapest tail chunks: ~1-2% off rega 1098.5 / 1133.9 us"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "w2 runs 8 items per CTA, so the final item's cost spread matters: up to ~1-2% off rega w2 140.4 / 145.2 us"
  - shapes: "*load=first_turn*"
    direction: flat
    why: "~124 items per CTA: the tail is under 1% of rega"
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
prior: "experts-rega-round-rotated-schedule: that rotates which CTA takes which item inside each round; this reorders the expert groups heaviest-first so the last round holds the cheapest items, and composes with it"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: rotation cut imbalance but gate missed; round-rotated on Do-not-try"
---

# experts-rega-lpt-group-order

## Change

Built on master (blob f4751fc0). rega walks expert groups in local-expert-id order, so the last round of the persistent loop holds whatever chunks the highest ids own: at turn_4 (~31 items per CTA at the judge-like load, one partial last round) and in w2 at turn_1 (8 items per CTA) a few CTAs finish on a heavy N = 128 item while the rest idle. CHANGE: derive a module muse_prefill_rega_lpt from master's source (master's module untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines). After the per-expert chunk-group pass (s_gfirst, s_gcount), thread 0 computes for every local expert with chunks key = (sum of nbk over its chunks) / gcount (mean 16-route blocks per chunk, a proxy for per-item cost), sorts the experts by key descending (stable on expert id; at most 32), and writes s_pslot[e] (the expert's first chunk slot in that order: a prefix sum of gcount) and s_slot2e[slot] (uint8 expert id per chunk slot, Cap<BIG>::C entries, so the static shared memory stays under its 16384 B bound); __syncthreads. decode_item then takes cg = it / T; e = s_slot2e[cg]; f = s_gfirst[e]; k = s_gcount[e]; local = it - s_pslot[e] * T; c = f + local % k; j = local / k, in the producer and the consumers alike; the chunk tables and Item fields are unchanged. Same items, same per-item code, so outputs are bit-identical. GATE as experts-rega-round-rotated-schedule: bit-exact compare on random, skewed, one and pop routings, two-state replay check, 2 interleaved rounds x 11 cold-L2 alternations on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random; ON if pop <= 0.98 in both rounds and random <= 1.00; state['rega_lpt_ok'][cls]. Print KERNELS w13 / w2 us both ways and PATH CHOSEN.

## Expected

- `*load=turn_4*` faster: the partial last round at ~31 items per CTA becomes the cheapest tail chunks: ~1-2% off rega 1098.5 / 1133.9 us
- `*load=turn_1*` faster: w2 runs 8 items per CTA, so the final item's cost spread matters: up to ~1-2% off rega w2 140.4 / 145.2 us
- `*load=first_turn*` flat: ~124 items per CTA: the tail is under 1% of rega

## Risk

Bit-exact. Size ~+0.2-0.4% headline: lands only stacked; composes with experts-rega-round-rotated-schedule (that mixes chunks across CTAs inside each round; this orders the tail).
