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
review: proceed
status: running
review_at: 2026-10-27T140421Z
review_companions: null
stage: claimed
claimed_at: 2026-10-27T140421Z
base_sha: 564d44539536723987b824306ca1ce595bbe50dd
base_blob: 42ded47196d5f0417c17f38574876508f0d822f1
branch: hyp/muse_prefill_layer/experts-rega-w2-early-loads-low-class
worktree: .worktrees/muse_prefill_layer/experts-rega-w2-early-loads-low-class
mode: null
restack_of: experts-rega-w2-epilogue-prefetch-bulk-store
stale_base: null
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

## Adversarial review

Reviewed 2026-10-27T140421Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companions. The parent's interleaved in-prepare gate measured arm a at the low class at x0.971-0.981 in all 8 rounds, bit-exact on every routing. The review that made low print-only called it HBM-bound on an arithmetic error: low w2 runs at ~2 TB/s (~42% of peak). The expected headline gain is ~0.15%, unreadable in one run; the evidence is the [rega-w2st] low GATE line plus the PATH CHOSEN line showing low routed with which=1.

### Justification
Arm a hoists gsc and the sorted_ids loads ahead of the k loop and serves the route id from the staging tile's pad column. Low class: ~2048 w2 items over 132 CTAs (~15 per CTA at ~10 us); saving ~0.25 us per item matches 164.8 -> 161.0 us cold and 156.7 -> 153.2 us warm on pop:1. Arm b (sharing the early loads) also measured x0.976-0.979 at low, so the gain is the early loads. The parent review's 604 MB is w13's byte count; w2 at low is ~300 MB in 149.9 us (2.01 TB/s) vs a 63 us floor: latency/consumer-bound. FINDINGS: rega_kernel<false,false> is 159.3 us (19.9%) of turn_1 sliding, matching the gate's pop:1 master w2. Expected ~0.46% of turn_1, ~0.15% headline; turn_4 / first_turn flat. Weakest link: the low gate margin is ~0.9% on one routing family, and routing depends on rega_low_variant == 0 (a borderline separate gate). Scored w2 rows are not readable (same-code jobs differ 1.4-14%), and without the SB template arm a's kernel name equals master's, so only PATH CHOSEN attributes it.

### Confounders on the current kernel
- experts-rega-w13-fast-silu-epilogue: arm a lives in the fsilu module; w2st is OFF if rega_fsilu_ext is None; check fsilu's PATH CHOSEN first.
- rega-lo V64/V32 variants: arm a applies only at rega_low_variant == 0 (guard 1.033 borderline); read the [rega-lo] PATH CHOSEN line first.
- align-two-phase-all-t-rega: produces the sorted_ids arm a preloads; exactness passed on all routings.
- experts-turn-rega-s4-port-t4096: the 4-stage ring decides how much epilogue is hidden.
- stack-tonight-five-prefill (zero_nonlocal_w2): arm a keeps the sid skip predicate; torch.equal with the sentinel covers unwritten rows.

### Interference and A/B needs
No companion: the same-process interleaved gate is the evidence. Build notes: (1) at low also time a heavier-skew guard routing (rega-lo's alpha 2.2, ~1356 routes) and require arm/master <= 1.0 there; (2) print low's which/ok next to rega_low_variant in PATH CHOSEN; (3) whichever of this, experts-rega-epilogue-wg-local-bulk-store and experts-rega-epilogue-overlap-next-stage lands second must re-gate against the other; experts-rega-low-2cta-fullslab-mc2 would replace the low kernel.

### Open questions
- Without arm b / SB, does the profile distinguish arm a's kernel from master's (same name, different module)?
- Is the rega-lo variant gate stable at variant 0 run to run?
- Which load is judge-like at turn_1: fsilu's pop:1 (~760 routes) or rega-lo's ~1350? Cover both.
- Dropping arm b changes the module hash (a fresh ~33 s build); confirm the deadline holds.

