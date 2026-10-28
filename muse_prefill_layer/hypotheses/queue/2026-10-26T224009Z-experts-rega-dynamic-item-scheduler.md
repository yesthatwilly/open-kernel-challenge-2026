---
problem: muse_prefill_layer
slug: experts-rega-dynamic-item-scheduler
created: 2026-10-26T224009Z
base: master
area: "w13 + w2 expert GEMMs (rega persistent schedule: atomic work counter)"
change: >
  Built on master (blob f4751fc0). rega's persistent CTAs take items statically (it = blockIdx.x +
  i * 132), so a launch lasts as long as its busiest CTA, whose share depends on the N-bucket mix
  (~0.6-1.0x per item), on the CTA-index aliasing described in
  experts-rega-round-rotated-schedule, and on the partial last round. CHANGE: derive a module
  muse_prefill_rega_dyn from master's source (master's module untouched; scaffolding as in
  done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's
  _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved
  rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines). (1) A per-state
  int32 [4] device buffer state['rega_sched'] (w13 next, w13 done, w2 next, w2 done), zeroed once
  in prepare and passed to rega_gemm; the kernel always leaves it at zero, so graph replays are
  safe (LESSONS 167: one buffer per state, never module-global). (2) Producer (tid 0): per item it
  = atomicAdd(&next, 1); at the item's first stage it writes it into a static shared slot
  s_item[s] before mbar_expect_tx on full(s) (the arrive's release and the consumers' try_wait
  acquire order the write); when it >= total it waits empty(s) as for a normal stage, writes
  s_item[s] = -1, does a plain mbar_arrive on full(s) (count 1, no tx) and leaves its loop; then
  if (atomicAdd(&done, 1) == gridDim.x - 1) { next = 0; done = 0; } (every producer has stopped
  taking items by then). (3) Consumers: at each item start wait full(n % STAGES) with parity (n /
  STAGES) & 1, read it = s_item[n % STAGES], leave the loop on -1, else decode_item(it) and
  consume_item as now (its first mbar_wait on that completed phase returns at once). Per-item code
  is master's, so outputs are bit-identical. Keep master's bounded mbar_wait so a protocol bug
  becomes error word 1 and the gate turns the class OFF instead of hanging. GATE as
  experts-rega-round-rotated-schedule (bit-exact on random, skewed, one and pop routings; the
  two-state replay check, alternating states so the per-state counters are exercised; 2 rounds x
  11 cold-L2 alternations on the pop routing (_rega_pop_case with the alpha _rega_pop_counts /
  _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on
  random; ON if pop <= 0.98 in both rounds and random <= 1.00); state['rega_dyn_ok'][cls]. Print
  KERNELS w13 / w2 us both ways and PATH CHOSEN.
expected:
  - shapes: "*load=turn_4*"
    direction: faster
    why: "balances the k=2 full/tail split and the last partial round at run time: ~5-10% off rega 1098.5 / 1133.9 us"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "16 (w13) and 8 (w2) items per CTA of mixed N buckets: ~3-5% off rega 418.5 / 437.9 us"
  - shapes: "*load=first_turn*"
    direction: faster
    why: ~2-4% off rega at ~124 items per CTA
risk: high
depends_on: null
priority: 3
kind: primary
controls_for: null
prior: "experts-rega-round-rotated-schedule: that is a static rotation of the same item walk; this replaces the walk with an atomic work counter, so the bucket-cost spread, the aliasing and the partial last round are all balanced at run time"
review: required
status: hold
hold_reason: "screen 2026-10-28 1520Z (value bar, vs master): SUPERSEDED: stages/CTA max/min 928/896, imbalance premise fails"
---

# experts-rega-dynamic-item-scheduler

## Change

Built on master (blob f4751fc0). rega's persistent CTAs take items statically (it = blockIdx.x + i * 132), so a launch lasts as long as its busiest CTA, whose share depends on the N-bucket mix (~0.6-1.0x per item), on the CTA-index aliasing described in experts-rega-round-rotated-schedule, and on the partial last round. CHANGE: derive a module muse_prefill_rega_dyn from master's source (master's module untouched; scaffolding as in done/neutral/experts-rega-pdl-chain-gather-w13-w2.patch: asserted string edits of master's _rega_src(), its own module name, canary, bit-exact replays over two states, 2 interleaved rounds x 11 cold-L2 graph alternations, PATH CHOSEN as the last prepare lines). (1) A per-state int32 [4] device buffer state['rega_sched'] (w13 next, w13 done, w2 next, w2 done), zeroed once in prepare and passed to rega_gemm; the kernel always leaves it at zero, so graph replays are safe (LESSONS 167: one buffer per state, never module-global). (2) Producer (tid 0): per item it = atomicAdd(&next, 1); at the item's first stage it writes it into a static shared slot s_item[s] before mbar_expect_tx on full(s) (the arrive's release and the consumers' try_wait acquire order the write); when it >= total it waits empty(s) as for a normal stage, writes s_item[s] = -1, does a plain mbar_arrive on full(s) (count 1, no tx) and leaves its loop; then if (atomicAdd(&done, 1) == gridDim.x - 1) { next = 0; done = 0; } (every producer has stopped taking items by then). (3) Consumers: at each item start wait full(n % STAGES) with parity (n / STAGES) & 1, read it = s_item[n % STAGES], leave the loop on -1, else decode_item(it) and consume_item as now (its first mbar_wait on that completed phase returns at once). Per-item code is master's, so outputs are bit-identical. Keep master's bounded mbar_wait so a protocol bug becomes error word 1 and the gate turns the class OFF instead of hanging. GATE as experts-rega-round-rotated-schedule (bit-exact on random, skewed, one and pop routings; the two-state replay check, alternating states so the per-state counters are exercised; 2 rounds x 11 cold-L2 alternations on the pop routing (_rega_pop_case with the alpha _rega_pop_counts / _rega_pop_pick give for ~1.83x uniform local routes, ~2.2, as _rega_lo_phase calibrates) and on random; ON if pop <= 0.98 in both rounds and random <= 1.00); state['rega_dyn_ok'][cls]. Print KERNELS w13 / w2 us both ways and PATH CHOSEN.

## Expected

- `*load=turn_4*` faster: balances the k=2 full/tail split and the last partial round at run time: ~5-10% off rega 1098.5 / 1133.9 us
- `*load=turn_1*` faster: 16 (w13) and 8 (w2) items per CTA of mixed N buckets: ~3-5% off rega 418.5 / 437.9 us
- `*load=first_turn*` faster: ~2-4% off rega at ~124 items per CTA

## Risk

About 45 minutes. The run-time form of the two static schedule fixes (rotation, heaviest-first): if those come back flat while this gates ON, the imbalance is data-dependent. Size ~+1.5-2.5% headline if the modeled imbalance holds. A new producer/consumer handoff: the first measurement must be an okc check (LESSONS 144); build with -lineinfo and print after each canary launch.
