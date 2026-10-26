---
problem: muse_prefill_layer
slug: experts-rega-fine-n-buckets
created: 2026-10-26T224003Z
base: master
area: w13 + w2 expert GEMMs (rega wgmma N buckets at 16-row granularity)
change: >
  Built on master (blob f4751fc0). rega sizes each item's wgmma N (tokens) and its slab TMA box
  from the chunk's 16-route block count nbk (1..8) with four buckets: nbk <= 1 -> 16, <= 2 -> 32,
  <= 4 -> 64, else 128. A chunk of 33-48 routes runs N = 64 and one of 65-112 runs N = 128, so up
  to 3/8 of those items' MMA columns and slab rows are padding. Judge-like skewed routing has many
  such chunks: turn_1 experts of ~33-112 routes, and the tail chunk of every k >= 2 expert at
  turn_4 and first_turn (uniform in 1-128 routes). Prepare's KERNELS lines put a stage at ~0.60 us
  at N = 32 (pingpong log, T=753 random: 307 us for 16 items x 32 stages) and ~0.93 us at N = 128
  (T=3012 random: 474 us), about 3.4 ns per N row per stage, so N = 16 * nbk saves up to ~0.16 us
  per stage on those items. CHANGE: derive a module muse_prefill_rega_fine from master's source
  (master's module untouched; scaffolding as in
  done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's
  _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved
  rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines): (1) generate
  Wg<NB> for NB in (16, 32, 48, 64, 80, 96, 112, 128) (_rega_wgmma_src already emits the m64nNk16
  RS wrapper per NB; bf16 wgmma takes any N that is a multiple of 8 up to 256); (2) struct Maps
  holds 8 CUtensorMaps and rega_gemm encodes 8 boxes {64, 16*(b+1)} with master's SWIZZLE_128B 2-D
  map (every box fits the 16 KB slab half); (3) decode_item sets m.bucket = nbk - 1 (clamped to
  0..7); (4) the producer's tx = WST + 2 * 16 * (m.bucket + 1) * 128; (5) the consumer switch has
  8 cases calling consume_item<16*(b+1), G>. NB/8 is even for every NB, so the epilogue loops are
  unchanged; the dequant, ring, smem layout and register budget (largest N still 128) are
  master's. GATE per class (low T=753, high T=3012, ft T=16384): bit-exact compare of hs and y
  with master's rega on random, skewed, one and pop routings (each output element is the same
  K-ordered dot product whatever N), plus the two-state replay check; then 2 interleaved rounds x
  11 cold-L2 graph alternations, master pair vs fine pair, on the pop routing (_rega_pop_case with
  the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as
  _rega_lo_phase calibrates) and on random; ON if pop <= 0.98 in both rounds and random <= 1.00;
  state['rega_fine_ok'][cls]. Print per class and routing the chunk histogram by nbk, KERNELS w13
  / w2 us both ways, PATH CHOSEN. The build has 32 consume_item instantiations instead of 16: if
  prepare passes ~230 s, skip _rega_lo_phase (its candidates have been OFF in every run since
  combo-b).
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "~42 routes per expert with a hot head (max ~228): chunks of 33-48 routes drop N 64->48 and of 65-112 drop 128->80-112; ~4-8% off rega 418.5 / 437.9 us, ~2-4% of each turn_1 shape"
  - shapes: "*load=turn_4*"
    direction: faster
    why: the tail chunk of every k>=2 expert (uniform 1-128 routes) is about half the items at the skewed T=3012 load; ~3-5% off rega 1098.5 / 1133.9 us, ~1.5-2.5% of each turn_4 shape
  - shapes: "*load=first_turn*"
    direction: faster
    why: "only the 1-in-8 tail chunks change at ~950 routes per expert: ~1% off rega, ~0.5% of each first_turn shape (inside row noise)"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "experts-turn-rega-s4-port-t4096: the port shipped rega with four power-of-two N buckets (16/32/64/128); this adds 48/80/96/112 so an item runs N = 16 * nbk; tile, dequant, ring and epilogue are unchanged"
review: required
status: queued
---

# experts-rega-fine-n-buckets

## Change

Built on master (blob f4751fc0). rega sizes each item's wgmma N (tokens) and its slab TMA box from the chunk's 16-route block count nbk (1..8) with four buckets: nbk <= 1 -> 16, <= 2 -> 32, <= 4 -> 64, else 128. A chunk of 33-48 routes runs N = 64 and one of 65-112 runs N = 128, so up to 3/8 of those items' MMA columns and slab rows are padding. Judge-like skewed routing has many such chunks: turn_1 experts of ~33-112 routes, and the tail chunk of every k >= 2 expert at turn_4 and first_turn (uniform in 1-128 routes). Prepare's KERNELS lines put a stage at ~0.60 us at N = 32 (pingpong log, T=753 random: 307 us for 16 items x 32 stages) and ~0.93 us at N = 128 (T=3012 random: 474 us), about 3.4 ns per N row per stage, so N = 16 * nbk saves up to ~0.16 us per stage on those items. CHANGE: derive a module muse_prefill_rega_fine from master's source (master's module untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines): (1) generate Wg<NB> for NB in (16, 32, 48, 64, 80, 96, 112, 128) (_rega_wgmma_src already emits the m64nNk16 RS wrapper per NB; bf16 wgmma takes any N that is a multiple of 8 up to 256); (2) struct Maps holds 8 CUtensorMaps and rega_gemm encodes 8 boxes {64, 16*(b+1)} with master's SWIZZLE_128B 2-D map (every box fits the 16 KB slab half); (3) decode_item sets m.bucket = nbk - 1 (clamped to 0..7); (4) the producer's tx = WST + 2 * 16 * (m.bucket + 1) * 128; (5) the consumer switch has 8 cases calling consume_item<16*(b+1), G>. NB/8 is even for every NB, so the epilogue loops are unchanged; the dequant, ring, smem layout and register budget (largest N still 128) are master's. GATE per class (low T=753, high T=3012, ft T=16384): bit-exact compare of hs and y with master's rega on random, skewed, one and pop routings (each output element is the same K-ordered dot product whatever N), plus the two-state replay check; then 2 interleaved rounds x 11 cold-L2 graph alternations, master pair vs fine pair, on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random; ON if pop <= 0.98 in both rounds and random <= 1.00; state['rega_fine_ok'][cls]. Print per class and routing the chunk histogram by nbk, KERNELS w13 / w2 us both ways, PATH CHOSEN. The build has 32 consume_item instantiations instead of 16: if prepare passes ~230 s, skip _rega_lo_phase (its candidates have been OFF in every run since combo-b).

## Expected

- `*load=turn_1*` faster: ~42 routes per expert with a hot head (max ~228): chunks of 33-48 routes drop N 64->48 and of 65-112 drop 128->80-112; ~4-8% off rega 418.5 / 437.9 us, ~2-4% of each turn_1 shape
- `*load=turn_4*` faster: the tail chunk of every k>=2 expert (uniform 1-128 routes) is about half the items at the skewed T=3012 load; ~3-5% off rega 1098.5 / 1133.9 us, ~1.5-2.5% of each turn_4 shape
- `*load=first_turn*` faster: only the 1-in-8 tail chunks change at ~950 routes per expert: ~1% off rega, ~0.5% of each first_turn shape (inside row noise)

## Risk

The n48 / n80 / n96 / n112 wgmma shapes are standard sm_90a forms but have not run on the judge in this repo; the separate module means a build or canary failure keeps master's rega. Size ~+1-1.7% headline if the judge-like bucket mix holds; read the nbk histogram and the KERNELS lines.
