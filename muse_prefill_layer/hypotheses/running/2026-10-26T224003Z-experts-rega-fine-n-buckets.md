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
review: proceed
status: running
review_at: 2026-10-27T103810Z
review_companions: null
stage: claimed
claimed_at: 2026-10-27T103810Z
base_sha: ec7851655c2c3f1ab2abfe571da8dc10d63062ea
base_blob: f4751fc054335a94d1b401af272e98a2d4760363
branch: hyp/muse_prefill_layer/experts-rega-fine-n-buckets
worktree: .worktrees/muse_prefill_layer/experts-rega-fine-n-buckets
mode: null
restack_of: null
stale_base: null
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

## Adversarial review

Reviewed 2026-10-27T103810Z by an adversarial subagent (model opus).

**Verdict: proceed.** The mechanism holds up: on padded chunks, a 16*nbk wgmma N plus a matching TMA box cuts MMA columns and, more importantly, L2->SM slab bytes, and slab ingest is rega's limiter (DIAGNOSIS/LESSONS; d00770b6 spans corr 0.93-0.98). The change is bit-exact and gated in prepare against master, so its KERNELS/GATE lines can be read alone. Before building, tighten the low-class precedence and the prepare-budget wording, and loosen the random-routing threshold to 1.01.

### Justification
master's decode_item uses four power-of-two buckets (16/32/64/128); a 65-112-route chunk moves 41 KB per stage when 21.5-37 KB would do, and a 33-48-route chunk 25.6 KB when 21.5 KB would do. MMA estimate ~4 ns per N row, matching the hypothesis's 3.4 ns/row and d00770b6's low-class figures. Item mix (d00770b6 histograms): low pop has ~45 routes per expert (the 33-48 bin is modal); high pop has many single and tail chunks in nbk 3/5/6/7. turn_1 and turn_4 faster at the claimed sizes (low 4-8%, high 3-5% of the rega pair) is plausible; first_turn should be FLAT (only ~32 of 252 chunks change). Headline ~+1% on two loads, inside the CI: read the prepare KERNELS/GATE lines and the nbk histogram. Weakest link: the 3.4 ns/row slope comes from two different runs/loads, and the judge's turn_1 histogram share at nbk=3 is inferred. Prior experts-turn-rega-s4-port-t4096 set the four buckets; experts-marlin-block-m-48-turn1 is a different engine and mechanism.

### Confounders on the current kernel
- combo-b-normgather-two-cta-low-regate (master): _rega_lo_phase chooses rega_low_variant per run; rega_lo_kernel has its own 4-bucket switch and does not use the fine module. Apply fine at the low class only when rega_low_variant == 0, gate it against the incumbent actually chosen, and print both.
- experts-turn-rega-s4-port-t4096 (+ first-turn-n128): source of the bucket set; the fine module is built by asserted string edits of master's _rega_src() (fragile if another rega change lands first; fine on master).
- combo-a / pre-expert-norm-fused-rega-gather: a smaller box reads a subset of the same slab rows; not a confounder.
- align-two-phase-all-t-rega: sets the chunk cut; not a confounder.

### Interference and A/B needs
No companion. Resolve the code-overhead ambiguity in the same prepare: add a third timed pair, the fine module with master's 4-bucket mapping (m.bucket = nbk<=1?0:nbk<=2?1:nbk<=4?3:7) through the 8-case switch: fine-4map vs master isolates overhead, fine vs fine-4map isolates the bucket effect.

### Open questions
- Low-class precedence when rega_low_variant is 1 or 2 (fine cannot apply).
- Use random <= 1.01, as the high class used in d00770b6.
- If the exact compare fails, print max |diff| and the first mismatching (class, routing, nbk), and fall back to master's fp32-reference calibrated bound.
- first_turn expected flat; record it in the expected list.
- Check the printed pop:2.2 nbk histogram against the judge's ~42-per-expert turn_1 figure.

