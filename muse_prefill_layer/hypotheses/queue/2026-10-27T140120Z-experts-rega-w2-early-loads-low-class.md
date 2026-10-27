---
problem: muse_prefill_layer
slug: experts-rega-w2-early-loads-low-class
created: 2026-10-27T140120Z
base: master
area: rega w2 epilogue (early gs/sid loads, arm a) routed at the LOW class
change: >
  Restack of experts-rega-w2-epilogue-prefetch-bulk-store (neutral, check 665c32dd; patch in
  muse_prefill_layer/hypotheses/done/neutral/) on current master, with three changes to its gate
  and dispatch only. (1) Keep arm (a) (early gsc / sorted_ids loads, the sid table in the staging
  tile's pad column, master's 16-B scatter loop reading it) and DROP arm (b) (the cp.async.bulk
  row stores): b was slower than master in every class and mode (x1.001-1.046), consistent with
  TMA contention against the producer. Removing b also removes its PTX from the fsilu module
  (fewer instantiations, less build time). (2) The LOW class (T=753) becomes routable: in 665c32dd
  arm a measured x0.975/0.978 cold and x0.975/0.981 warm on the judge-like pop:1 routing and
  x0.976/0.975 cold, x0.971/0.972 warm on random (w2 164.8 -> 161.0 us), all inside the <= 0.99
  bar, exactness torch.equal on all four routings; the review had called low 'HBM-bound, flat' and
  made it print-only, which the measurement contradicts. Gate low exactly as the parent gated high
  / ft (arm/master <= 0.99 every round, cold+warm, pop+random, then the two-state replay check),
  with rega_low_variant == 0 precedence kept. (3) High and ft: keep gating arm a (it measured
  x0.991-0.998, a near miss); route only if it clears the same bar. Everything else (fsilu module
  host, exactness checks, deadlines, compact PATH CHOSEN lines last) as in the parent.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "rega w2 at the low class x0.97-0.98 in-prepare (165 -> 161 us cold, 157 -> 153 us warm on pop:1): about -3 to -4 us on a ~760-1150 us turn_1 row, ~0.3-0.5% of each turn_1 shape; readable from the gate lines and the scored rega_kernel<false,*> row, not the headline"
  - shapes: "*load=turn_4*"
    direction: flat
    why: "arm a x0.991-0.995 at high: likely misses the 0.99 bar"
  - shapes: "*load=first_turn*"
    direction: flat
    why: arm a x0.997-0.999 at ft
risk: low
depends_on: experts-rega-w2-epilogue-prefetch-bulk-store
priority: 1
kind: primary
controls_for: null
prior: "experts-rega-w2-epilogue-prefetch-bulk-store: same arm a code; that run kept the low class print-only on the review's HBM-bound argument, but measured arm a at x0.971-0.981 there in every round; this run routes it at low and drops arm b, which lost everywhere"
review: required
status: queued
---

# experts-rega-w2-early-loads-low-class

## Change

Restack of experts-rega-w2-epilogue-prefetch-bulk-store (neutral, check 665c32dd; patch in muse_prefill_layer/hypotheses/done/neutral/) on current master, with three changes to its gate and dispatch only. (1) Keep arm (a) (early gsc / sorted_ids loads, the sid table in the staging tile's pad column, master's 16-B scatter loop reading it) and DROP arm (b) (the cp.async.bulk row stores): b was slower than master in every class and mode (x1.001-1.046), consistent with TMA contention against the producer. Removing b also removes its PTX from the fsilu module (fewer instantiations, less build time). (2) The LOW class (T=753) becomes routable: in 665c32dd arm a measured x0.975/0.978 cold and x0.975/0.981 warm on the judge-like pop:1 routing and x0.976/0.975 cold, x0.971/0.972 warm on random (w2 164.8 -> 161.0 us), all inside the <= 0.99 bar, exactness torch.equal on all four routings; the review had called low 'HBM-bound, flat' and made it print-only, which the measurement contradicts. Gate low exactly as the parent gated high / ft (arm/master <= 0.99 every round, cold+warm, pop+random, then the two-state replay check), with rega_low_variant == 0 precedence kept. (3) High and ft: keep gating arm a (it measured x0.991-0.998, a near miss); route only if it clears the same bar. Everything else (fsilu module host, exactness checks, deadlines, compact PATH CHOSEN lines last) as in the parent.

## Expected

- `*load=turn_1*` faster: rega w2 at the low class x0.97-0.98 in-prepare (165 -> 161 us cold, 157 -> 153 us warm on pop:1): about -3 to -4 us on a ~760-1150 us turn_1 row, ~0.3-0.5% of each turn_1 shape; readable from the gate lines and the scored rega_kernel<false,*> row, not the headline
- `*load=turn_4*` flat: arm a x0.991-0.995 at high: likely misses the 0.99 bar
- `*load=first_turn*` flat: arm a x0.997-0.999 at ft

## Risk

Headline ~0.1-0.2%: below single-run readability; worth a check only because the parent measured it directly (x0.971-0.981 in all 8 low-class rounds) and it removes the slower arm b from the module. Read it on the [rega-w2st] low GATE / PATH CHOSEN lines and the turn_1 w2 profile row.
