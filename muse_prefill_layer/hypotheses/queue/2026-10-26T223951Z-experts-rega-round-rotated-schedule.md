---
problem: muse_prefill_layer
slug: experts-rega-round-rotated-schedule
created: 2026-10-26T223951Z
base: master
area: w13 + w2 expert GEMMs (rega persistent item->CTA schedule)
change: >
  Built on master (blob f4751fc0). Only the order in which rega_kernel's persistent CTAs walk
  items changes; every item is computed by master's code, so outputs are bit-identical. MECHANISM:
  the producer thread and the consumer warpgroups both walk for (it = blockIdx.x; it < total; it
  += gridDim.x) with gridDim = 132, and decode_item takes chunk c = f + (it - f*T) % k of an
  expert with k chunks (T = n_tiles = 64 for w13, 32 for w2; every group starts at a multiple of
  T). 132 and f*T are multiples of 4, so for every expert with k = 2 or 4 chunks CTA b always gets
  chunk b % k: at k = 2 the even CTAs take all the full 128-route chunks and the odd CTAs all the
  short tail chunks (N bucket 16-64, roughly 0.6x the cost), at k = 4 the CTAs with b % 4 == 3
  take every tail. In the low class (k = 1, 64-item groups, 132 = 2*64 + 4) CTAs below 64 see
  mostly even expert groups and CTAs 64-127 mostly odd ones. A launch lasts as long as its busiest
  CTA. CHANGE: derive a module muse_prefill_rega_rot from master's source (master's module and
  strings untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch:
  asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays
  over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last
  prepare lines) in which BOTH loops become: for (int r = 0;; ++r) { const int it = r * gridDim.x
  + (blockIdx.x + r * (n_tiles + 1)) % gridDim.x; if (it >= total) break; <master's loop body> }.
  Each round still covers items [r*G, r*G + G) (same set, same chunk-minor neighbours, same L2
  working set, a bijection inside the round), but the odd shift n_tiles + 1 moves every CTA by one
  group plus one item per round, so its chunk index and group parity change every round. The
  producer and the consumers must use the identical expression (the stage counter n still advances
  KS per item). GATE per class (low T=753, high T=3012, ft T=16384): bit-exact compare of hs and y
  with master's rega on random, skewed, one and pop routings, plus the two-state replay check;
  then 2 interleaved rounds x 11 cold-L2 graph alternations of master's routed rega pair vs the
  rot pair on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick
  give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random routing
  (random routing at T=3012 gives k = 1 for every expert, so it cannot show the effect; the pop
  routing gives k = 1-6). ON for the class only if pop <= 0.98 in both rounds and random <= 1.00.
  state['rega_rot_ok'][cls] is a plain bool per state; _experts_rega launches the rot module's
  rega_gemm when it is set. Print per class: experts by k (1, 2, 3, 4, >4), KERNELS w13 / w2 us
  master vs rot on both routings, PATH CHOSEN. A build or compare failure keeps master's kernel
  for that class.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "high class: the judge-like ~1.8x-skewed T=3012 routing has many k=2 experts whose full chunks all land on the even CTAs; rebalancing should take ~4-10% off rega w13+w2 (1098.5 us sliding, 1133.9 us full), ~2-5% of each turn_4 shape"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "low class: each CTA alternates between even and odd expert groups instead of mostly one parity; ~2-4% off rega 418.5 / 437.9 us, ~1-2% of each turn_1 shape"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "ft: k ~8 per expert and 132 = 4 mod 8, so CTAs with b % 8 in {3, 7} hold the tails; ~2-4% off rega 5067-5739 us, ~1-2% of each first_turn shape"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "experts-triton-persistent-grid-first-turn: that changed the old Triton engine's grid size at T > 4096; this keeps rega's 132-CTA persistent grid and only rotates the item->CTA map so CTAs stop aliasing onto chunk (b % k) of k=2/4 experts and onto one expert-group parity"
review: required
status: queued
---

# experts-rega-round-rotated-schedule

## Change

Built on master (blob f4751fc0). Only the order in which rega_kernel's persistent CTAs walk items changes; every item is computed by master's code, so outputs are bit-identical. MECHANISM: the producer thread and the consumer warpgroups both walk for (it = blockIdx.x; it < total; it += gridDim.x) with gridDim = 132, and decode_item takes chunk c = f + (it - f*T) % k of an expert with k chunks (T = n_tiles = 64 for w13, 32 for w2; every group starts at a multiple of T). 132 and f*T are multiples of 4, so for every expert with k = 2 or 4 chunks CTA b always gets chunk b % k: at k = 2 the even CTAs take all the full 128-route chunks and the odd CTAs all the short tail chunks (N bucket 16-64, roughly 0.6x the cost), at k = 4 the CTAs with b % 4 == 3 take every tail. In the low class (k = 1, 64-item groups, 132 = 2*64 + 4) CTAs below 64 see mostly even expert groups and CTAs 64-127 mostly odd ones. A launch lasts as long as its busiest CTA. CHANGE: derive a module muse_prefill_rega_rot from master's source (master's module and strings untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines) in which BOTH loops become: for (int r = 0;; ++r) { const int it = r * gridDim.x + (blockIdx.x + r * (n_tiles + 1)) % gridDim.x; if (it >= total) break; <master's loop body> }. Each round still covers items [r*G, r*G + G) (same set, same chunk-minor neighbours, same L2 working set, a bijection inside the round), but the odd shift n_tiles + 1 moves every CTA by one group plus one item per round, so its chunk index and group parity change every round. The producer and the consumers must use the identical expression (the stage counter n still advances KS per item). GATE per class (low T=753, high T=3012, ft T=16384): bit-exact compare of hs and y with master's rega on random, skewed, one and pop routings, plus the two-state replay check; then 2 interleaved rounds x 11 cold-L2 graph alternations of master's routed rega pair vs the rot pair on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random routing (random routing at T=3012 gives k = 1 for every expert, so it cannot show the effect; the pop routing gives k = 1-6). ON for the class only if pop <= 0.98 in both rounds and random <= 1.00. state['rega_rot_ok'][cls] is a plain bool per state; _experts_rega launches the rot module's rega_gemm when it is set. Print per class: experts by k (1, 2, 3, 4, >4), KERNELS w13 / w2 us master vs rot on both routings, PATH CHOSEN. A build or compare failure keeps master's kernel for that class.

## Expected

- `*load=turn_4*` faster: high class: the judge-like ~1.8x-skewed T=3012 routing has many k=2 experts whose full chunks all land on the even CTAs; rebalancing should take ~4-10% off rega w13+w2 (1098.5 us sliding, 1133.9 us full), ~2-5% of each turn_4 shape
- `*load=turn_1*` faster: low class: each CTA alternates between even and odd expert groups instead of mostly one parity; ~2-4% off rega 418.5 / 437.9 us, ~1-2% of each turn_1 shape
- `*load=first_turn*` faster: ft: k ~8 per expert and 132 = 4 mod 8, so CTAs with b % 8 in {3, 7} hold the tails; ~2-4% off rega 5067-5739 us, ~1-2% of each first_turn shape

## Risk

Bit-exact by construction. Size: ~+1-2% headline if the modeled imbalance holds, enough to clear the margin alone; if the pop ratio is above 0.98 the class stays OFF and the run is master re-measured, so read the KERNELS lines, not the headline. Adds one ~35 s module build to prepare (140 s today). This is not the consumer dequant/MMA overlap DIAGNOSIS bans: the per-item code is master's.
