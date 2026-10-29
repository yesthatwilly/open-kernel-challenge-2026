---
problem: muse_prefill_layer
slug: experts-rega-stage-antiphase-ordered-issue
created: 2026-10-29T161429Z
base: master
mint_blob: "c63801e7a1dcfdb7cbdd0d38bc9b295407be7489"
best_at_mint: 1.5008
area: "w13 + w2 expert GEMMs (rega consumer schedule: ordered wgmma issue / stage anti-phase of master's two consumer warpgroups), low / high / ft classes"
change: >
  Built on master (submission.py blob c63801e7, board 1.5008; master dispatches the
  muse_prefill_rega_fsilu_w2a module: fast-silu w13 in every class, w2st arm a w2 at low / high /
  ft; the LO phase is pinned off). Consumer schedule only: FA3-style ordered anti-phase of
  master's TWO existing consumer warpgroups at stage granularity, on master's shared 4-stage ring,
  at master's 40 / 232 setmaxnreg split, 384 threads, no K split, no reduction, full 128-route
  slab, master's items / chunking / buckets / producer / epilogue. MECHANISM: in master's
  consume_item both WGs (WG0 = consumer warps cw 0-3, weight rows 0-63; WG1 = cw 4-7, rows 64-127)
  run each stage in lockstep: mbar_wait(full) -> dq_stage -> fence_acc -> wgmma_fence -> 8
  Wg<NB>::mma -> commit -> wgmma_wait<0> -> fence_acc -> lane 0 mbar_arrive(empty); both issue
  their 8 wgmmas at once, the tensor core interleaves them so both groups retire late, and both
  dequantize the next stage while the tensor core idles (rega-anat cb22e610: t0 exceeds
  max(MMA-only, dequant-only) by g 0.18-0.34 in every class, not ingest-bound). CHANGE: order only
  the wgmma ISSUE between the two WGs with two named barriers, leaving dq_stage outside any
  barrier: WG0 issues stage n, then WG1 issues stage n, then WG0 stage n+1, ... In steady state
  WG0 retires its group first and dequantizes stage n+1 while WG1's wgmmas of n run, and WG1
  dequantizes n+1 while WG0's wgmmas of n+1 run (then swap). NEW MODULE muse_prefill_rega_ap
  derived from _rega_fsilu_src() (never edit master's strings) by asserted (old, new) edits, each
  old occurring exactly once (else '[rega-ap] OFF: edit k'); first assert the source has no
  'bar.arrive' and no named_bar id other than 1. (E1) after named_bar(): add bar_arrive(int id,
  int count) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(count) : "memory"); } and
  constexpr int AP_BAR_W0 = 2 (WG0 may issue), AP_BAR_W1 = 3 (WG1 may issue); bar 0 stays
  __syncthreads, bar 1 the epilogue. (E2) consume_item gains trailing params bool wg1, bool&
  started. (E3) in its k loop, between fence_acc<NA>(acc) and wgmma_fence(): 'if (wg1)
  named_bar(AP_BAR_W1, 256); else { if (started) named_bar(AP_BAR_W0, 256); started = true; }';
  between wgmma_commit() and wgmma_wait<0>(): 'bar_arrive(wg1 ? AP_BAR_W0 : AP_BAR_W1, 256);'
  (FA3's warp_scheduler_barrier_arrive placement). Everything else in the loop is master's:
  mbar_wait(full) and dq_stage BEFORE the barrier, wgmma_wait<0> every stage (register-A variant
  A: a0 is never rewritten while a group is in flight, no deferred wait), then fence_acc and lane
  0's mbar_arrive(empty). (E4) in rega_kernel's consumer branch after 'const int ctid = tid - 128,
  cw = ctid >> 5;': 'const bool wg1 = cw >= 4; bool started = false;'; WG_CONSUME passes wg1,
  started; after the item loop: 'if (!wg1 && started) named_bar(AP_BAR_W0, 256);' (absorbs WG1's
  last arrive). Launch markers, error words, rega_init / rega_gemm signatures and the 4
  rega_kernel<G, BIG> instantiations stay master's (ft served too). BARRIER PROTOCOL: count 256 =
  128 syncing + 128 arriving threads per phase; wg1 is warp-uniform and started is uniform across
  WG0 (both WGs walk the same items, KS = K / BK stages each). Initial phase: WG0 holds the issue
  token at launch (no sync on its first stage), WG1 waits bar 3, so WG0's group of stage 0 issues
  first; both dequantize stage 0 together (both wait full[0]) and the anti-phase forms from stage
  1; each item's epilogue (named_bar(1, 256) x 2) re-aligns the pair, costing one in-phase stage
  per item (1 / KS). Counts over a launch with S consumed stages: WG0 bar-2 syncs S - 1 + 1 final
  = S = WG1 bar-2 arrives; WG1 bar-3 syncs S = WG0 bar-3 arrives; a CTA with 0 items does no AP
  op. A barrier's next phase cannot open early: WG0's arrive(3) for n+1 follows its sync(2) for
  n+1, which needs WG1's arrive(2) for n, which follows WG1's sync(3) for n (and symmetrically).
  WG-issue skew <= 1 stage < STAGES = 4, so full / empty parity cannot alias; full[s] (count 1,
  expect_tx) and empty[s] (count 8, one per consumer warp after its own wait<0>) are unchanged.
  Epilogue: WG0 reaches bar 1 after its last arrive(3), WG1 after its last sync(3) (already
  satisfied): no deadlock; WG0's next-item sync(2) pairs with WG1's arrive(2) issued before bar 1.
  The mbar timeout 'dead' path skips only mbar waits, never AP ops. Outputs bit-identical by
  construction (each WG's instruction sequence and accumulation order unchanged). DIFFERS FROM
  CLAIMS:44 (experts-rega-pingpong-consumers-all-t, 5fbaec9a): that build was also stage-granular
  on the shared ring (not item ping-pong, no split ring) but put bar.sync BEFORE mbar_wait(full) +
  dq_stage, so dq_WG1(s) -> dq_WG2(s) -> dq_WG1(s+1) were forced to alternate (stage >= 2 x one-WG
  dequant + 2 hops) and the two latency-bound dequants per SMSP that master runs concurrently were
  serialized; it never probed ptxas (high w13 474.2 -> 577.9 us, 5.80 -> 4.76 TB/s was the
  consequence, not the cause; the ingest-cap reading was retired by rega-anat t4/t0 0.41-0.65).
  Here only the issue order is constrained: when the wgmmas are latency-bound (small NB) WG1
  issues right after WG0 and the groups still run concurrently (degenerates to master's lockstep);
  when tensor-throughput-bound (NB 128) WG0 retires first and its dequant hides under WG1's group.
  DIFFERS FROM the refuted K-pair (bbca0105): no extra WGs (384 not 640 threads), registers stay
  232 (kp 112), no split-K smem reduction / extra named_bar(512) rounds / 512-thread epilogue,
  empty count stays 8; it forces the anti-phase instead of relying on free drift (kp stagger 0 vs
  1 differed <= 0.003). BUILD: once, load_inline with REGA_CUDA_CFLAGS + ['-Xptxas', '-Werror'] as
  the kp probe did (37 s), only if prepare() has used < 200 s (REGA_FSILU_START_DEADLINE_S); a
  ptxas failure prints up to 4 lines containing 'C75' or 'ptxas' as '[rega-ap] ptxas: ...' and
  leaves the phase OFF (that answers whether bar.arrive inside the wgmma stage makes ptxas
  serialize). Print attrs as _rega_fsilu_attrs (need r <= 168 at launch, local 0, master's dyn
  smem, 1 CTA/SM; any spill -> OFF). PREPARE PHASE _rega_ap_phase(state, weights, device,
  t_prepare0, fsilu_info), after _rega_w2st_phase and before 'del fsilu_info'; never raises (try /
  except / finally: synchronize, drop graphs / tensors, empty_cache). Canary first: per
  instantiation one eager launch at T = 210 'one' (CTAs with 0 and 1 items) and T = 300 random,
  flushed print before, synchronize + print after (named barriers have no timeout). Per class low
  (753), high (3012), ft (16384), only classes with rega_ok, in that order, no class step after
  265 s (SKIPPED lines): (a) exactness: the ap w13 + w2 pair vs the pair master dispatches at that
  class (fsilu w13 / w2st-or-master w2 per state flags) on master's xs (norm-gather prenorm tables
  as kernel_fn) for random / skewed / one / pop routings: torch.equal on hs and on y prefilled
  with REGA_W2ST_SENTINEL, error word 0, both markers; (b) two-state replay check
  (REGA_FSILU_REPLAY_PLAN[cls]) with ap ON; (c) gated race: one CUDA graph per pair through
  _rega_fsilu_bench, REGA_FSILU_ROUNDS 2 x REGA_FSILU_ALTERNATIONS 11, cold and warm L2, on pop
  (fsilu_info['alpha'][cls]), random (_rega_case 'random') and a skew guard (low: rega-lo's guard
  alpha nearest REGA_LO_GUARD_ROUTES ~1350 routes; high / ft: _rega_case 'skewed'): ON only if ap
  / master <= 0.97 on pop and <= 1.00 on random and guard, both L2 modes, every round; also print
  w13-alone and w2-alone pop-warm ratios (attribution only). State: state['rega_ap_ext'] (module
  or None), state['rega_ap_ok'] = {cls: bool}. DISPATCH in _experts_rega: ap = (not variant and
  state.get('rega_ap_ext') is not None and state['rega_ap_ok'].get(_rega_class(num_tokens),
  False)); when ap, both w13 and w2 launch rega_ap_ext.rega_gemm(0 / 1, ...) with master's
  arguments and grid; precedence variant > ap > fsilu / w2st > master. Print '[rega-ap] PATH
  CHOSEN <cls>: ap|master (pop warm/cold, random warm/cold, guard warm/cold max ratios; w13 / w2
  alone; attrs)' per class, summary under ~2 KB, before prepare()'s final echo block. Nothing else
  in kernel_fn, the align, the gather, or the fsilu / v2 modules changes.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "low class (rega-anat cb22e610 pop warm): w13 t0 293.6 / max(t1,t2) 194.4 us, w2 146.5 / 99.1 (g 0.33); capturing ~1/3 of the gap -> w13 ~261, w2 ~131 us; the low ceiling (all of g at sliding + full turn_1) is ~5.2% headline, so ~1.7%. Flat if the gate keeps master here (NB 32/64 wgmmas latency-bound: the protocol degenerates to lockstep)"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "high class: w13 746.5 / 589.2, w2 393.2 / 310.7 (g 0.21); 1/3 of the gap -> w13 ~694, w2 ~366 us; ceiling ~3% headline -> ~1.0%"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "ft class: w13 3570.5 / 2927.0, w2 1908.4 / 1502.4 (g 0.18-0.21); 1/3 -> w13 ~3356, w2 ~1773 us; ceiling ~2.9% (sliding first_turn share 22.6% x rega 48% x g 0.19 + full first_turn rows 10.8% x ~33% x 0.2) -> ~1.0%. Sum low + high + ft = ~3.7%, quoted as --gain 3.5"
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "experts-rega-pingpong-consumers-all-t: CLAIMS:44's build (5fbaec9a) was ALSO stage-granular on master's shared 4-stage ring, not item ping-pong and not a split ring (the K-pair prior misdescribes it); it placed bar.sync BEFORE mbar_wait(full) + dq_stage, forcing the two WGs' latency-bound dequants to alternate, and never probed ptxas. Here only the wgmma issue is ordered (FA3 warp-scheduler barrier around the 8 wgmmas, dequant outside, arrive between commit and wait<0>), so the worst case degenerates to master's lockstep; ptxas -Werror probe; and the cause CLAIMS:44 recorded (ingest cap, 4.76 TB/s) is retired by rega-anat cb22e610 (t4/t0 0.41-0.65, SERIAL). Unlike the refuted K-pair (bbca0105): 2 WGs at 232 regs, no split-K reduction, forced (not free-drift) anti-phase"
target: check
value_gain: 3.5
value_p: 0.2
value_ev: 0.7
value_cost: 350k tokens, 1 judge runs
evidence: >
  check cb22e6104be142ee [rega-anat] MODEL (warm, pop, mode-0 fidelity 0.98-1.00, spill 0): low
  w13 t0 293.6 / MMA-only 194.4 / dequant-only 191.1 / producer 145.1 us (g 0.338), low w2 146.5 /
  99.1 / 96.4 / 74.2 (g 0.324); high w13 746.5 / 589.2 / 355.9 / 340.3 (g 0.211), w2 g 0.210; ft
  w13 3570.5 / 2927.0 / 2308.4 / 2311.0 (g 0.180), w2 g 0.213; t4/t0 0.41-0.65 (not ingest-bound);
  CLAIMS [lead] re-scaled ceilings low ~5.2%, high ~3%. Against: 5fbaec9a (CLAIMS:44) stage
  ping-pong with bar.sync before dequant: high w13 474.2 -> 577.9 us x1.219, gate 747.0 vs 880.5
  us x1.179 (archive log [rega-pp] KERNELS / GATE); 6edf1cb002bf42b9 wait<1> ring x1.075 slower;
  bbca0105449c4217 K-pair [refuted]: -Werror probe clean, r96 l0, kp/master x1.104-1.140 at low
  T=753 and x1.087-1.133 at high T=3012 (pop/random, cold/warm), stagger 0 vs 1 within 0.003
user_requested: true
review: proceed
status: running
review_at: 2026-10-29T162053Z
review_companions: null
review_best: 1.5007957663757285
review_blob: c63801e7a1dcfdb7cbdd0d38bc9b295407be7489
stage: claimed
claimed_at: 2026-10-29T162054Z
base_sha: cab523d330d2015479dc4d0f602a30a1722974e0
base_blob: c63801e7a1dcfdb7cbdd0d38bc9b295407be7489
branch: hyp/muse_prefill_layer/experts-rega-stage-antiphase-ordered-issue
worktree: .worktrees/muse_prefill_layer/experts-rega-stage-antiphase-ordered-issue
mode: null
restack_of: null
stale_base: null
---

# experts-rega-stage-antiphase-ordered-issue

## Change

Built on master (submission.py blob c63801e7, board 1.5008; master dispatches the muse_prefill_rega_fsilu_w2a module: fast-silu w13 in every class, w2st arm a w2 at low / high / ft; the LO phase is pinned off). Consumer schedule only: FA3-style ordered anti-phase of master's TWO existing consumer warpgroups at stage granularity, on master's shared 4-stage ring, at master's 40 / 232 setmaxnreg split, 384 threads, no K split, no reduction, full 128-route slab, master's items / chunking / buckets / producer / epilogue. MECHANISM: in master's consume_item both WGs (WG0 = consumer warps cw 0-3, weight rows 0-63; WG1 = cw 4-7, rows 64-127) run each stage in lockstep: mbar_wait(full) -> dq_stage -> fence_acc -> wgmma_fence -> 8 Wg<NB>::mma -> commit -> wgmma_wait<0> -> fence_acc -> lane 0 mbar_arrive(empty); both issue their 8 wgmmas at once, the tensor core interleaves them so both groups retire late, and both dequantize the next stage while the tensor core idles (rega-anat cb22e610: t0 exceeds max(MMA-only, dequant-only) by g 0.18-0.34 in every class, not ingest-bound). CHANGE: order only the wgmma ISSUE between the two WGs with two named barriers, leaving dq_stage outside any barrier: WG0 issues stage n, then WG1 issues stage n, then WG0 stage n+1, ... In steady state WG0 retires its group first and dequantizes stage n+1 while WG1's wgmmas of n run, and WG1 dequantizes n+1 while WG0's wgmmas of n+1 run (then swap). NEW MODULE muse_prefill_rega_ap derived from _rega_fsilu_src() (never edit master's strings) by asserted (old, new) edits, each old occurring exactly once (else '[rega-ap] OFF: edit k'); first assert the source has no 'bar.arrive' and no named_bar id other than 1. (E1) after named_bar(): add bar_arrive(int id, int count) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(count) : "memory"); } and constexpr int AP_BAR_W0 = 2 (WG0 may issue), AP_BAR_W1 = 3 (WG1 may issue); bar 0 stays __syncthreads, bar 1 the epilogue. (E2) consume_item gains trailing params bool wg1, bool& started. (E3) in its k loop, between fence_acc<NA>(acc) and wgmma_fence(): 'if (wg1) named_bar(AP_BAR_W1, 256); else { if (started) named_bar(AP_BAR_W0, 256); started = true; }'; between wgmma_commit() and wgmma_wait<0>(): 'bar_arrive(wg1 ? AP_BAR_W0 : AP_BAR_W1, 256);' (FA3's warp_scheduler_barrier_arrive placement). Everything else in the loop is master's: mbar_wait(full) and dq_stage BEFORE the barrier, wgmma_wait<0> every stage (register-A variant A: a0 is never rewritten while a group is in flight, no deferred wait), then fence_acc and lane 0's mbar_arrive(empty). (E4) in rega_kernel's consumer branch after 'const int ctid = tid - 128, cw = ctid >> 5;': 'const bool wg1 = cw >= 4; bool started = false;'; WG_CONSUME passes wg1, started; after the item loop: 'if (!wg1 && started) named_bar(AP_BAR_W0, 256);' (absorbs WG1's last arrive). Launch markers, error words, rega_init / rega_gemm signatures and the 4 rega_kernel<G, BIG> instantiations stay master's (ft served too). BARRIER PROTOCOL: count 256 = 128 syncing + 128 arriving threads per phase; wg1 is warp-uniform and started is uniform across WG0 (both WGs walk the same items, KS = K / BK stages each). Initial phase: WG0 holds the issue token at launch (no sync on its first stage), WG1 waits bar 3, so WG0's group of stage 0 issues first; both dequantize stage 0 together (both wait full[0]) and the anti-phase forms from stage 1; each item's epilogue (named_bar(1, 256) x 2) re-aligns the pair, costing one in-phase stage per item (1 / KS). Counts over a launch with S consumed stages: WG0 bar-2 syncs S - 1 + 1 final = S = WG1 bar-2 arrives; WG1 bar-3 syncs S = WG0 bar-3 arrives; a CTA with 0 items does no AP op. A barrier's next phase cannot open early: WG0's arrive(3) for n+1 follows its sync(2) for n+1, which needs WG1's arrive(2) for n, which follows WG1's sync(3) for n (and symmetrically). WG-issue skew <= 1 stage < STAGES = 4, so full / empty parity cannot alias; full[s] (count 1, expect_tx) and empty[s] (count 8, one per consumer warp after its own wait<0>) are unchanged. Epilogue: WG0 reaches bar 1 after its last arrive(3), WG1 after its last sync(3) (already satisfied): no deadlock; WG0's next-item sync(2) pairs with WG1's arrive(2) issued before bar 1. The mbar timeout 'dead' path skips only mbar waits, never AP ops. Outputs bit-identical by construction (each WG's instruction sequence and accumulation order unchanged). DIFFERS FROM CLAIMS:44 (experts-rega-pingpong-consumers-all-t, 5fbaec9a): that build was also stage-granular on the shared ring (not item ping-pong, no split ring) but put bar.sync BEFORE mbar_wait(full) + dq_stage, so dq_WG1(s) -> dq_WG2(s) -> dq_WG1(s+1) were forced to alternate (stage >= 2 x one-WG dequant + 2 hops) and the two latency-bound dequants per SMSP that master runs concurrently were serialized; it never probed ptxas (high w13 474.2 -> 577.9 us, 5.80 -> 4.76 TB/s was the consequence, not the cause; the ingest-cap reading was retired by rega-anat t4/t0 0.41-0.65). Here only the issue order is constrained: when the wgmmas are latency-bound (small NB) WG1 issues right after WG0 and the groups still run concurrently (degenerates to master's lockstep); when tensor-throughput-bound (NB 128) WG0 retires first and its dequant hides under WG1's group. DIFFERS FROM the refuted K-pair (bbca0105): no extra WGs (384 not 640 threads), registers stay 232 (kp 112), no split-K smem reduction / extra named_bar(512) rounds / 512-thread epilogue, empty count stays 8; it forces the anti-phase instead of relying on free drift (kp stagger 0 vs 1 differed <= 0.003). BUILD: once, load_inline with REGA_CUDA_CFLAGS + ['-Xptxas', '-Werror'] as the kp probe did (37 s), only if prepare() has used < 200 s (REGA_FSILU_START_DEADLINE_S); a ptxas failure prints up to 4 lines containing 'C75' or 'ptxas' as '[rega-ap] ptxas: ...' and leaves the phase OFF (that answers whether bar.arrive inside the wgmma stage makes ptxas serialize). Print attrs as _rega_fsilu_attrs (need r <= 168 at launch, local 0, master's dyn smem, 1 CTA/SM; any spill -> OFF). PREPARE PHASE _rega_ap_phase(state, weights, device, t_prepare0, fsilu_info), after _rega_w2st_phase and before 'del fsilu_info'; never raises (try / except / finally: synchronize, drop graphs / tensors, empty_cache). Canary first: per instantiation one eager launch at T = 210 'one' (CTAs with 0 and 1 items) and T = 300 random, flushed print before, synchronize + print after (named barriers have no timeout). Per class low (753), high (3012), ft (16384), only classes with rega_ok, in that order, no class step after 265 s (SKIPPED lines): (a) exactness: the ap w13 + w2 pair vs the pair master dispatches at that class (fsilu w13 / w2st-or-master w2 per state flags) on master's xs (norm-gather prenorm tables as kernel_fn) for random / skewed / one / pop routings: torch.equal on hs and on y prefilled with REGA_W2ST_SENTINEL, error word 0, both markers; (b) two-state replay check (REGA_FSILU_REPLAY_PLAN[cls]) with ap ON; (c) gated race: one CUDA graph per pair through _rega_fsilu_bench, REGA_FSILU_ROUNDS 2 x REGA_FSILU_ALTERNATIONS 11, cold and warm L2, on pop (fsilu_info['alpha'][cls]), random (_rega_case 'random') and a skew guard (low: rega-lo's guard alpha nearest REGA_LO_GUARD_ROUTES ~1350 routes; high / ft: _rega_case 'skewed'): ON only if ap / master <= 0.97 on pop and <= 1.00 on random and guard, both L2 modes, every round; also print w13-alone and w2-alone pop-warm ratios (attribution only). State: state['rega_ap_ext'] (module or None), state['rega_ap_ok'] = {cls: bool}. DISPATCH in _experts_rega: ap = (not variant and state.get('rega_ap_ext') is not None and state['rega_ap_ok'].get(_rega_class(num_tokens), False)); when ap, both w13 and w2 launch rega_ap_ext.rega_gemm(0 / 1, ...) with master's arguments and grid; precedence variant > ap > fsilu / w2st > master. Print '[rega-ap] PATH CHOSEN <cls>: ap|master (pop warm/cold, random warm/cold, guard warm/cold max ratios; w13 / w2 alone; attrs)' per class, summary under ~2 KB, before prepare()'s final echo block. Nothing else in kernel_fn, the align, the gather, or the fsilu / v2 modules changes.

## Expected

- `*load=turn_1*` faster: low class (rega-anat cb22e610 pop warm): w13 t0 293.6 / max(t1,t2) 194.4 us, w2 146.5 / 99.1 (g 0.33); capturing ~1/3 of the gap -> w13 ~261, w2 ~131 us; the low ceiling (all of g at sliding + full turn_1) is ~5.2% headline, so ~1.7%. Flat if the gate keeps master here (NB 32/64 wgmmas latency-bound: the protocol degenerates to lockstep)
- `*load=turn_4*` faster: high class: w13 746.5 / 589.2, w2 393.2 / 310.7 (g 0.21); 1/3 of the gap -> w13 ~694, w2 ~366 us; ceiling ~3% headline -> ~1.0%
- `*load=first_turn*` faster: ft class: w13 3570.5 / 2927.0, w2 1908.4 / 1502.4 (g 0.18-0.21); 1/3 -> w13 ~3356, w2 ~1773 us; ceiling ~2.9% (sliding first_turn share 22.6% x rega 48% x g 0.19 + full first_turn rows 10.8% x ~33% x 0.2) -> ~1.0%. Sum low + high + ft = ~3.7%, quoted as --gain 3.5

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** check cb22e6104be142ee [rega-anat] MODEL (warm, pop, mode-0 fidelity 0.98-1.00, spill 0): low w13 t0 293.6 / MMA-only 194.4 / dequant-only 191.1 / producer 145.1 us (g 0.338), low w2 146.5 / 99.1 / 96.4 / 74.2 (g 0.324); high w13 746.5 / 589.2 / 355.9 / 340.3 (g 0.211), w2 g 0.210; ft w13 3570.5 / 2927.0 / 2308.4 / 2311.0 (g 0.180), w2 g 0.213; t4/t0 0.41-0.65 (not ingest-bound); CLAIMS [lead] re-scaled ceilings low ~5.2%, high ~3%. Against: 5fbaec9a (CLAIMS:44) stage ping-pong with bar.sync before dequant: high w13 474.2 -> 577.9 us x1.219, gate 747.0 vs 880.5 us x1.179 (archive log [rega-pp] KERNELS / GATE); 6edf1cb002bf42b9 wait<1> ring x1.075 slower; bbca0105449c4217 K-pair [refuted]: -Werror probe clean, r96 l0, kp/master x1.104-1.140 at low T=753 and x1.087-1.133 at high T=3012 (pop/random, cold/warm), stagger 0 vs 1 within 0.003
- **Gain if it works:** 3.5% headline; **p** 0.2; **EV** 0.7% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (EV 0.7% < 1.5%), minted because the user described it.
- **Cost:** 350k tokens, 1 judge runs.

## Risk

Medium risk. HANG: named barriers have no timeout; the change gives the count invariants and the 0-item guard (low class leaves CTAs with no item); reviewer should hand-simulate KS stages with 0, 1, 2 items per CTA and an item boundary. VALUE CASE HONESTY: three overlap / TLP builds on this tile lost in their own gates (stage ping-pong x1.18-1.47, wait<1> ring x1.075, K-pair x1.09-1.14 at a clean ptxas build), and the stage model does NOT explain the old ping-pong's loss at high (there MMA-only >> dequant-only, so even its forced alternation should have won), so an unmodelled cost of overlapping register-A wgmma with another warp's ALU work on the same SMSP is plausible; K-pair's free drift (pairs up to 3 stages apart, overlap available) also did not pay, though its reduction / 112-reg overheads confound that. Hence p 0.2. Anti-phase is the cleanest remaining test of the [lead]: it adds only two bar ops per stage per WG to master's exact kernel, so a flat or slower result closes overlap levers on the current rega tile (proposed CLAIMS line), and a ptxas C7510/C7512 failure says the FA3 arrive placement serializes. Also lesson (fine-N): read the scored profile's rega rows, not only the in-prepare gate. Restack note: if any rega module lands first, gate ap against the pair master then dispatches.

## Adversarial review

Reviewed 2026-10-29T162053Z by an adversarial subagent (model opus).

**Verdict: proceed.** E passes: CLAIMS.md:44 (ping-pong, refuted) is reopened for ping-pong-like schedules by the newer [lead] CLAIMS.md:70-71, which names this exact sibling; board/blob unmoved. The barrier protocol is hang-safe (same counts as 5fbaec9a, which ran exact on the judge). One flat run is only interpretable with an in-run indicator showing whether the anti-phase formed, and the build should add __syncwarp before the AP bar.sync.

### Justification
Protocol checked by hand: WG0's bar-2 syncs (S-1 plus 1 final) equal WG1's S bar-2 arrives; WG1's bar-3 syncs equal WG0's bar-3 arrives; each arrive for phase n+1 is causally after the partner's sync for phase n, so no phase aliasing; 0-item CTAs do no AP op, 1-item ends with the final sync(2); no cycle with epilogue bar 1; mbarrier skew <= 1 stage < 4; producer empty[] dependency met. 5fbaec9a used ids 2/3, count 256, 'started' skip, final sync and arrive-after-commit and ran exact; only the sync moves from before mbar_wait+dq to after dq. Prior accurate. WEAKEST LINK: with this placement the barriers bind only once per item (KS=32 stages); afterwards anti-phase persists only if each SMSP's tensor pipe retires the two warps' wgmma groups in issue order; if the pipe round-robins, the kernel silently falls back to master's lockstep plus 2 bar ops per stage. Per-WG wgmma_wait<0> does not block overlap. VALUE: low (NB mostly 32, MMA-only 0.379 us/stage) is where the model predicts lockstep, yet carries 1.7 of the 3.5% gain; high and ft (NB128) are where a gain is predicted, ~2% headline at the 1/3-of-gap capture rate; p 0.2 honest; EV 0.7% user-requested. ap phase starts ~181 s + ~37-39 s build: fits the 200 / 265 s deadlines.

### Confounders on the current kernel
- experts-rega-w13-fast-silu-epilogue and w2st arm a (CLAIMS.md:54/56): shared by both gate arms; epilogue length sets WG0's idle at bar 1 (small at KS=32). No revert A/B needed.
- experts-turn-rega-s4-port-t4096 / -first-turn-n128 (base engine): 4-stage ring, register-A variant A, 232-register split unchanged; result applies to this tile only.
- combo-b rega-lo two-CTA arm: never routes (skipped since 350343bc); no interaction.

### Interference and A/B needs
No companion run. The ambiguity is internal: flat could mean (a) anti-phase never formed or (b) it formed and overlap does not pay; only (b) supports 'closes overlap levers on this tile'. Require a cheap in-run indicator: a prepare-only skew probe (CTA 0, lane 0 of consumer warps cw0 and cw4 accumulate clock64 at issue after commit and at retire after wait<0>; print mean own-group latency and retire skew as a fraction of the stage period per class and N bucket; ~0 = lockstep fallback, ~0.3-0.6 = formed), behind a runtime guard on the diag launch only (magic value in err[0], err buffer >= 3+8 ints), not a second module (a second build breaks the 265 s deadline). If too invasive, print per-N-bucket ap/master ratios (synthetic all-NB128 vs NB32 routings) and treat a uniform x1.00 +- 0.005 as inert/undetermined, not refuted. If lockstep, the follow-up is a binding variant (WG0's arrive(3) after its wait<0>), a next hypothesis.

### Open questions
- Does Hopper's per-SMSP tensor pipe retire two warps' queued wgmma groups in issue order? The skew probe answers it.
- At low (NB 32) are the dequant warps issue-bound or latency-bound? If latency-bound, the low booking is ~0 and success is ~2% (high + ft).
- Why did 5fbaec9a lose x1.18-1.47 at high? An unmodelled per-handoff cost would apply here too.
- Judge by the scored profile's rega_kernel rows at sliding turn_4 / first_turn, not only the in-prepare gate (CLAIMS.md:50).

