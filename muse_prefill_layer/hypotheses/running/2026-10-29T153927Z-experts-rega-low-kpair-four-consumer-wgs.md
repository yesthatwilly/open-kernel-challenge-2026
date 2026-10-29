---
problem: muse_prefill_layer
slug: experts-rega-low-kpair-four-consumer-wgs
created: 2026-10-29T153927Z
base: master
mint_blob: "c63801e7a1dcfdb7cbdd0d38bc9b295407be7489"
best_at_mint: 1.5008
area: "w13 + w2 expert GEMMs at 200 < T <= 1024 (rega low class, turn_1): K-pair rega kernel, four consumer warpgroups in one CTA splitting each stage's two 64-k records over master's full 128-route slab"
change: >
  Built on master (submission.py blob c63801e7, board 1.5008; fast-silu w13 and w2st arm a are
  what master dispatches at the low class, the LO phase is pinned off by REGA_LO_SKIP). GOAL: the
  ~10% TLP effect that two CTAs/SM showed at the low class (V64 occ2 0.887x vs its occ1 twin
  0.984x, LESSONS:180) WITHOUT V64's 64-route slab, whose chunk split lost under skew. One CTA/SM,
  master's item order, chunking (8 blocks of 16 = 128 routes per chunk), slab TMA boxes, weight
  records and N buckets are kept exactly; only the consumer side widens. NEW MODULE
  muse_prefill_rega_kp, derived from _rega_fsilu_src() (master's _rega_src() + the fast-silu swap
  + REGA_W2ST_EDITS, i.e. the fsilu_w2a source master dispatches) by asserted (old, new) edits
  that each occur exactly once (else '[rega-kp] OFF'), instantiating only rega_kernel<G, false>
  for G in {true, false} (no BIG / ft instantiation: less nvcc time). CTA SHAPE: THREADS 384 ->
  640 = producer WG (warps 0-3, unchanged: tid 0 issues bulk_g2s of the 9216-B weight stage + the
  two SLAB_HALF tma_2d boxes per stage, exactly master's) + four consumer WGs (warps 4-19).
  __launch_bounds__(640, 1); the launch must get 96 registers/thread (pool 96 x 640 = 61440);
  producer setmaxnreg.dec 32, consumers setmaxnreg.inc 112 (32 x 128 + 112 x 512 = 61440 exactly;
  master is 40 / 232 at 384 threads). PAIRS: consumer WG q (0-3) belongs to pair p = q >> 1;
  within a pair, ptid = (tid - 128) & 255 and cw = ptid >> 5 (0-7) index the 16-row weight slice
  exactly as master's ctid / cw do, so each pair covers all 128 weight rows of the item. PER STAGE
  (consume_item): every consumer waits full[s] as master; pair p dequantizes ONLY record r = p of
  the stage (dq_stage's r loop collapsed to r = p: a0[4][4], 16 registers instead of 32; codes at
  st + p * REC + cw * 512 + lane * 16, scales at st + p * REC + REC_CODES + cw * 64 + g * 8) and
  issues the 4 Wg<NB>::mma k16 steps of that record against slab + p * SLAB_HALF + k * 32 (master
  issues 8: r = 0, 1), commit, wgmma_wait<0>, then lane 0 of each warp arrives on empty[s]. So
  each pair accumulates a partial 128-row x NB-token sum over half of K (records 0 of every stage
  vs records 1); acc stays NA = NB / 2 floats per thread (64 at NB = 128). No forced stagger: each
  pair waits only on full[s] and the stage is released by all 16 warps, so the pairs may drift up
  to STAGES - 1 stages apart and one pair's dequant overlaps the other's wgmma, the same way two
  CTAs/SM interleave. BARRIERS: full[s] init count 1 (producer expect_tx, unchanged); empty[s]
  init count 16 (one arrive per consumer warp; master 8). STAGES stays 4, STAGE_BYTES 41984, the
  ring and its tx bytes unchanged. REDUCTION (split K, not N): the partial of pair 1 is added into
  pair 0's registers through shared memory, in thread-fragment order (thread ptid of pair 1 holds
  the same (row, token) fragment slots as thread ptid of pair 0, so no layout transform): buffer
  red[i][ptid] (float, i = accumulator index, 256 consecutive floats per i: conflict-free)
  ALIASING the epilogue region ep, whose size becomes 128 * PITCH2 = 34816 B for both G (w13 grows
  from 18432; total dynamic smem 1024 + 4 x 41984 + 34816 = 203776 for both instantiations,
  master's w2 figure, under the 227 KB limit with the <= 16384 B static tables). At NB <= 64 one
  round (NA <= 32 floats x 256 threads = 32 KB); at NB = 128 two rounds of 32 accumulators each
  (acc[0..31] then acc[32..63]). Per item: named_bar(1, 512) (all consumers done with the previous
  item's staging / sid table) -> per round: pair 1 stores its 32 (or NA) floats, named_bar(1,
  512), pair 0 adds them into acc, named_bar(1, 512) before the next round's stores ->
  named_bar(2, 256) among pair 0 (the last red reads done) -> pair 0 runs master's epilogue
  staging verbatim (fast silu_mul with gsc for G, gsc_w2 scaling for w2, stmatrix into ep, and
  w2st arm a's sid table in the pad column, MOVED to after the reduction because red overlaps the
  pad bytes; its sid_r / gsc_w2 early loads are done by pair 0 threads only) -> named_bar(1, 512)
  -> the coalesced 16-B store loops run over all 512 consumer threads (v = tid - 128; v += 512)
  instead of 256. Pair 1 does no epilogue math. Items, decode_item, the persistent it =
  blockIdx.x; it += gridDim.x loop and the error words are master's; grid = state['rega_grid'] (SM
  count, 1 CTA/SM). HOST: rega_init / rega_gemm with master's signatures (640 threads, dynamic
  smem 203776 set by cudaFuncSetAttribute), so _rega_fsilu_gemm-style callers work unchanged.
  PREPARE PHASE _rega_kp_phase(state, weights, device, t_prepare0, fsilu_info), called right after
  _rega_w2st_phase and before 'del fsilu_info'; never raises (try / except / finally: synchronize,
  drop its graphs and tensors); summary lines kept short and printed before the w2st PATH CHOSEN
  lines so those stay the last prepare() output. Build only if prepare() has used < 200 s
  (REGA_FSILU_START_DEADLINE_S); no step after 265 s. ON criteria, all required: (a) attrs of both
  instantiations before any launch: regs >= 96 (pool covers 32 x 128 + 112 x 512, else
  setmaxnreg.inc hangs), local bytes 0 (no spills at 112), occupancy >= 1, static + dynamic smem
  <= REGA_SMEM_LIMIT, sm_90a image; (b) accuracy at the low class (T 753) on random / skewed / one
  / the judge-like pop routing (_rega_pop_case at fsilu_info['alpha']['low']): hs vs master's
  dispatched w13 rel L2 <= REGA_FSILU_HS_REL (2e-3; the K split reorders fp32 sums, not bit exact)
  and routed output error vs the fp64 reference <= REGA_FSILU_REF_MULT (1.05) x master's, y
  sentinel rows (REGA_W2ST_SENTINEL) identical to master's, error words 0 and both launch magics
  set after graph replay (REGA_FSILU_REPLAY_PLAN['low']); (c) gate: the w13 + w2 pair of the kp
  module vs the pair master dispatches at the low class (fsilu w13 and w2st-or-master w2 per
  state['rega_fsilu_ok'] / ['rega_w2st_ok']), CUDA graphs on master's own xs / hs tables
  (norm-gather prenorm as kernel_fn dispatches) through _rega_fsilu_bench, REGA_FSILU_ROUNDS 2 x
  REGA_FSILU_ALTERNATIONS 11, cold and warm L2: ON only when kp / master <= 0.97 on pop routing in
  BOTH L2 modes and <= 1.00 on random in both, every round. Optional second class 'high' (T 3012,
  alpha['high']), same criteria, only if prepare() has used < 240 s when low is done; ft is never
  gated (no BIG instantiation). State: state['rega_kp_ext'] (module or None) and
  state['rega_kp_ok'] = {'low': bool, 'high': bool}, plain prepare-time values. DISPATCH in
  _experts_rega: kp = (not variant and state.get('rega_kp_ext') is not None and
  state['rega_kp_ok'].get(_rega_class(num_tokens), False)); when kp, BOTH the w13 and the w2
  launch go to rega_kp_ext.rega_gemm(0 / 1, ...) with master's arguments and grid; precedence
  variant (never on) > kp > fsilu / w2st > master. Fallback = master on any miss. Print '[rega-kp]
  PATH CHOSEN low: kp|master (pop warm x.xxx cold x.xxx, random warm x.xxx cold x.xxx; regs/local
  per instantiation)' and the same for high. Nothing else in kernel_fn, _experts, the align, the
  gather, the fsilu / w2st modules or master's v2 module changes.
expected:
  - shapes: "*layer=sliding*load=turn_1*"
    direction: faster
    why: "rega w13 + w2 = 249.9 + 127.4 = 377.3 us of 708.8 (53%) at sliding turn_1 (b2941ea profile 0); if K-pair recovers the occ2 TLP effect (0.887 / 0.984 = 0.90, LESSONS:180) the pair drops ~38 us: row x0.946; at share 0.217 that is ~+1.2% headline"
  - shapes: "*layer=full*load=turn_1*"
    direction: faster
    why: "same rega time (~377 us) of 1083.7 us (~35%): row x0.965, share 0.117: ~+0.4% headline. Low class alone ~1.6%"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "only if the optional 'high' gate turns ON: rega 703.9 + 372.6 of 2246.9 us (48%) at sliding turn_4; at half the low-class TLP effect (5% of rega) ~+0.5% headline (sliding share 0.208 + full 0.125). Low ~1.6 + high ~0.5 -> --gain 2.1; flat when high stays OFF"
  - shapes: "*load=first_turn*"
    direction: flat
    why: "ft never gated (no BIG instantiation); master's rega_kernel<G, true> / fsilu path unchanged"
risk: high
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "experts-rega-low-class-two-ctas-per-sm: V64/V32 occ2 got the TLP from a second CTA per SM but halved the slab to 64/32 routes, so popular experts split into 4-8 chunks and lost under skew (guard 1.035, judge pin 1.4776); K-pair keeps 1 CTA/SM, master's 128-route slab and chunking, and gets the TLP from four consumer warpgroups splitting each stage by 64-k record on one ring - not experts-rega-pingpong-consumers-all-t's alternation over items, which cut L2->SM ingest to 4.76 TB/s; meets CLAIMS:55's reopen rule (a different low-class kernel, not V64 / V32)"
target: check
value_gain: 2.1
value_p: 0.2
value_ev: 0.42
value_cost: 400k tokens, 1 judge runs
evidence: >
  LESSONS:180 / FINDINGS (experts-rega-low-class-two-ctas-per-sm, impl dfb755d3): two CTAs/SM at
  the low class ran 0.887x master (V64, random routing) while its occ1 V64 twin ran 0.984x,
  isolating a ~10% TLP effect; it lost only because the 64-route slab split popular experts
  (0.994x under skew; combo-b 73b40cd1: V64 0.878 random but 1.035 on the 1.86x-skew guard; V64
  pinned on the judge, check 56752c8d 1.4776 vs 1.4897, CLAIMS:55). K-pair keeps the 128-route
  slab, so the chunking is master's. Size: b2941ea23dc743a3 profile 0: rega w13 + w2 = 249.9 +
  127.4 us of 708.8 us at sliding turn_1 (53%), 703.9 + 372.6 of 2246.9 at sliding turn_4.
  Counter-evidence: two-WG ping-pong over items cut L2->SM ingest 5.8-6.05 -> 4.76 TB/s and ran
  1.18-1.53x slower (5fbaec9a, LESSONS:183); the in-WG register ring pf-rega-low-stream ran 1.075x
  slower (6edf1cb0, CLAIMS:68); stage anatomy (experts-rega-stage-anatomy-diag-r2) still running
decision: >
  Only meaningful after experts-rega-stage-anatomy-diag-r2: SERIAL at low -> run this check;
  OVERLAPPED or ingest-bound at low -> close unrun and append [refuted] for the K-pair shelf line;
  INVALID / mixed -> run only with a spare check. After the run: kp ON at low and the turn_1 rows
  faster by >= the margin -> promote; kp ON but rows flat -> [refuted] (in-prepare TLP does not
  transfer to the judge's warm-L2 pop routing); kp OFF by its own gate -> [refuted] with the pop /
  random ratios; OFF by spill -> close, the 112-register budget is the blocker.
user_requested: true
review: proceed
status: running
review_at: 2026-10-29T154811Z
review_companions: null
review_best: 1.5007957663757285
review_blob: c63801e7a1dcfdb7cbdd0d38bc9b295407be7489
stage: claimed
claimed_at: 2026-10-29T154812Z
base_sha: 06e1361569ec34702fbc0577933b82193553a264
base_blob: c63801e7a1dcfdb7cbdd0d38bc9b295407be7489
branch: hyp/muse_prefill_layer/experts-rega-low-kpair-four-consumer-wgs
worktree: .worktrees/muse_prefill_layer/experts-rega-low-kpair-four-consumer-wgs
mode: null
restack_of: null
stale_base: null
---

# experts-rega-low-kpair-four-consumer-wgs

## Change

Built on master (submission.py blob c63801e7, board 1.5008; fast-silu w13 and w2st arm a are what master dispatches at the low class, the LO phase is pinned off by REGA_LO_SKIP). GOAL: the ~10% TLP effect that two CTAs/SM showed at the low class (V64 occ2 0.887x vs its occ1 twin 0.984x, LESSONS:180) WITHOUT V64's 64-route slab, whose chunk split lost under skew. One CTA/SM, master's item order, chunking (8 blocks of 16 = 128 routes per chunk), slab TMA boxes, weight records and N buckets are kept exactly; only the consumer side widens. NEW MODULE muse_prefill_rega_kp, derived from _rega_fsilu_src() (master's _rega_src() + the fast-silu swap + REGA_W2ST_EDITS, i.e. the fsilu_w2a source master dispatches) by asserted (old, new) edits that each occur exactly once (else '[rega-kp] OFF'), instantiating only rega_kernel<G, false> for G in {true, false} (no BIG / ft instantiation: less nvcc time). CTA SHAPE: THREADS 384 -> 640 = producer WG (warps 0-3, unchanged: tid 0 issues bulk_g2s of the 9216-B weight stage + the two SLAB_HALF tma_2d boxes per stage, exactly master's) + four consumer WGs (warps 4-19). __launch_bounds__(640, 1); the launch must get 96 registers/thread (pool 96 x 640 = 61440); producer setmaxnreg.dec 32, consumers setmaxnreg.inc 112 (32 x 128 + 112 x 512 = 61440 exactly; master is 40 / 232 at 384 threads). PAIRS: consumer WG q (0-3) belongs to pair p = q >> 1; within a pair, ptid = (tid - 128) & 255 and cw = ptid >> 5 (0-7) index the 16-row weight slice exactly as master's ctid / cw do, so each pair covers all 128 weight rows of the item. PER STAGE (consume_item): every consumer waits full[s] as master; pair p dequantizes ONLY record r = p of the stage (dq_stage's r loop collapsed to r = p: a0[4][4], 16 registers instead of 32; codes at st + p * REC + cw * 512 + lane * 16, scales at st + p * REC + REC_CODES + cw * 64 + g * 8) and issues the 4 Wg<NB>::mma k16 steps of that record against slab + p * SLAB_HALF + k * 32 (master issues 8: r = 0, 1), commit, wgmma_wait<0>, then lane 0 of each warp arrives on empty[s]. So each pair accumulates a partial 128-row x NB-token sum over half of K (records 0 of every stage vs records 1); acc stays NA = NB / 2 floats per thread (64 at NB = 128). No forced stagger: each pair waits only on full[s] and the stage is released by all 16 warps, so the pairs may drift up to STAGES - 1 stages apart and one pair's dequant overlaps the other's wgmma, the same way two CTAs/SM interleave. BARRIERS: full[s] init count 1 (producer expect_tx, unchanged); empty[s] init count 16 (one arrive per consumer warp; master 8). STAGES stays 4, STAGE_BYTES 41984, the ring and its tx bytes unchanged. REDUCTION (split K, not N): the partial of pair 1 is added into pair 0's registers through shared memory, in thread-fragment order (thread ptid of pair 1 holds the same (row, token) fragment slots as thread ptid of pair 0, so no layout transform): buffer red[i][ptid] (float, i = accumulator index, 256 consecutive floats per i: conflict-free) ALIASING the epilogue region ep, whose size becomes 128 * PITCH2 = 34816 B for both G (w13 grows from 18432; total dynamic smem 1024 + 4 x 41984 + 34816 = 203776 for both instantiations, master's w2 figure, under the 227 KB limit with the <= 16384 B static tables). At NB <= 64 one round (NA <= 32 floats x 256 threads = 32 KB); at NB = 128 two rounds of 32 accumulators each (acc[0..31] then acc[32..63]). Per item: named_bar(1, 512) (all consumers done with the previous item's staging / sid table) -> per round: pair 1 stores its 32 (or NA) floats, named_bar(1, 512), pair 0 adds them into acc, named_bar(1, 512) before the next round's stores -> named_bar(2, 256) among pair 0 (the last red reads done) -> pair 0 runs master's epilogue staging verbatim (fast silu_mul with gsc for G, gsc_w2 scaling for w2, stmatrix into ep, and w2st arm a's sid table in the pad column, MOVED to after the reduction because red overlaps the pad bytes; its sid_r / gsc_w2 early loads are done by pair 0 threads only) -> named_bar(1, 512) -> the coalesced 16-B store loops run over all 512 consumer threads (v = tid - 128; v += 512) instead of 256. Pair 1 does no epilogue math. Items, decode_item, the persistent it = blockIdx.x; it += gridDim.x loop and the error words are master's; grid = state['rega_grid'] (SM count, 1 CTA/SM). HOST: rega_init / rega_gemm with master's signatures (640 threads, dynamic smem 203776 set by cudaFuncSetAttribute), so _rega_fsilu_gemm-style callers work unchanged. PREPARE PHASE _rega_kp_phase(state, weights, device, t_prepare0, fsilu_info), called right after _rega_w2st_phase and before 'del fsilu_info'; never raises (try / except / finally: synchronize, drop its graphs and tensors); summary lines kept short and printed before the w2st PATH CHOSEN lines so those stay the last prepare() output. Build only if prepare() has used < 200 s (REGA_FSILU_START_DEADLINE_S); no step after 265 s. ON criteria, all required: (a) attrs of both instantiations before any launch: regs >= 96 (pool covers 32 x 128 + 112 x 512, else setmaxnreg.inc hangs), local bytes 0 (no spills at 112), occupancy >= 1, static + dynamic smem <= REGA_SMEM_LIMIT, sm_90a image; (b) accuracy at the low class (T 753) on random / skewed / one / the judge-like pop routing (_rega_pop_case at fsilu_info['alpha']['low']): hs vs master's dispatched w13 rel L2 <= REGA_FSILU_HS_REL (2e-3; the K split reorders fp32 sums, not bit exact) and routed output error vs the fp64 reference <= REGA_FSILU_REF_MULT (1.05) x master's, y sentinel rows (REGA_W2ST_SENTINEL) identical to master's, error words 0 and both launch magics set after graph replay (REGA_FSILU_REPLAY_PLAN['low']); (c) gate: the w13 + w2 pair of the kp module vs the pair master dispatches at the low class (fsilu w13 and w2st-or-master w2 per state['rega_fsilu_ok'] / ['rega_w2st_ok']), CUDA graphs on master's own xs / hs tables (norm-gather prenorm as kernel_fn dispatches) through _rega_fsilu_bench, REGA_FSILU_ROUNDS 2 x REGA_FSILU_ALTERNATIONS 11, cold and warm L2: ON only when kp / master <= 0.97 on pop routing in BOTH L2 modes and <= 1.00 on random in both, every round. Optional second class 'high' (T 3012, alpha['high']), same criteria, only if prepare() has used < 240 s when low is done; ft is never gated (no BIG instantiation). State: state['rega_kp_ext'] (module or None) and state['rega_kp_ok'] = {'low': bool, 'high': bool}, plain prepare-time values. DISPATCH in _experts_rega: kp = (not variant and state.get('rega_kp_ext') is not None and state['rega_kp_ok'].get(_rega_class(num_tokens), False)); when kp, BOTH the w13 and the w2 launch go to rega_kp_ext.rega_gemm(0 / 1, ...) with master's arguments and grid; precedence variant (never on) > kp > fsilu / w2st > master. Fallback = master on any miss. Print '[rega-kp] PATH CHOSEN low: kp|master (pop warm x.xxx cold x.xxx, random warm x.xxx cold x.xxx; regs/local per instantiation)' and the same for high. Nothing else in kernel_fn, _experts, the align, the gather, the fsilu / w2st modules or master's v2 module changes.

## Expected

- `*layer=sliding*load=turn_1*` faster: rega w13 + w2 = 249.9 + 127.4 = 377.3 us of 708.8 (53%) at sliding turn_1 (b2941ea profile 0); if K-pair recovers the occ2 TLP effect (0.887 / 0.984 = 0.90, LESSONS:180) the pair drops ~38 us: row x0.946; at share 0.217 that is ~+1.2% headline
- `*layer=full*load=turn_1*` faster: same rega time (~377 us) of 1083.7 us (~35%): row x0.965, share 0.117: ~+0.4% headline. Low class alone ~1.6%
- `*load=turn_4*` faster: only if the optional 'high' gate turns ON: rega 703.9 + 372.6 of 2246.9 us (48%) at sliding turn_4; at half the low-class TLP effect (5% of rega) ~+0.5% headline (sliding share 0.208 + full 0.125). Low ~1.6 + high ~0.5 -> --gain 2.1; flat when high stays OFF
- `*load=first_turn*` flat: ft never gated (no BIG instantiation); master's rega_kernel<G, true> / fsilu path unchanged

## Value case

- **Target:** check: a candidate to beat the board in one judge check.
- **Evidence:** LESSONS:180 / FINDINGS (experts-rega-low-class-two-ctas-per-sm, impl dfb755d3): two CTAs/SM at the low class ran 0.887x master (V64, random routing) while its occ1 V64 twin ran 0.984x, isolating a ~10% TLP effect; it lost only because the 64-route slab split popular experts (0.994x under skew; combo-b 73b40cd1: V64 0.878 random but 1.035 on the 1.86x-skew guard; V64 pinned on the judge, check 56752c8d 1.4776 vs 1.4897, CLAIMS:55). K-pair keeps the 128-route slab, so the chunking is master's. Size: b2941ea23dc743a3 profile 0: rega w13 + w2 = 249.9 + 127.4 us of 708.8 us at sliding turn_1 (53%), 703.9 + 372.6 of 2246.9 at sliding turn_4. Counter-evidence: two-WG ping-pong over items cut L2->SM ingest 5.8-6.05 -> 4.76 TB/s and ran 1.18-1.53x slower (5fbaec9a, LESSONS:183); the in-WG register ring pf-rega-low-stream ran 1.075x slower (6edf1cb0, CLAIMS:68); stage anatomy (experts-rega-stage-anatomy-diag-r2) still running
- **Gain if it works:** 2.1% headline; **p** 0.2; **EV** 0.42% (bar: gain >= 2.5%, EV >= 1.5%); **below the bar** (gain 2.1% < 2.5%; EV 0.42% < 1.5%), minted because the user described it.
- **Cost:** 400k tokens, 1 judge runs.
- **Decision rule:** Only meaningful after experts-rega-stage-anatomy-diag-r2: SERIAL at low -> run this check; OVERLAPPED or ingest-bound at low -> close unrun and append [refuted] for the K-pair shelf line; INVALID / mixed -> run only with a spare check. After the run: kp ON at low and the turn_1 rows faster by >= the margin -> promote; kp ON but rows flat -> [refuted] (in-prepare TLP does not transfer to the judge's warm-L2 pop routing); kp OFF by its own gate -> [refuted] with the pop / random ratios; OFF by spill -> close, the 112-register budget is the blocker.

## Risk

DEPENDS ON THE STAGE ANATOMY (read before claiming): experts-rega-stage-anatomy-diag-r2 (running/, prepare-only) measures whether master's rega stages are SERIAL (dequant then wgmma, t0 >= 0.90 (t1 + t2 - t4)), OVERLAPPED or ingest-bound at low / high. K-pair's value is the serial gap: its pre-registered decision re-mints / sizes this K-pair claim only on SERIAL, and appends [refuted] for it on OVERLAPPED. hyp.py's depends_on is a hard stacking base (churn would build on the diag's branch), so it is NOT set; churn should hold this file until the diag's [rega-anat] VERDICT low line is read: SERIAL -> run as specced (re-size --gain by g = (t0 - max(t1, t2)) / t0 x the rega share if it differs); OVERLAPPED or ingest-bound (t4 >= 0.85 t0) at low -> close without running (no gap for TLP to hide); INVALID / mixed -> run only if a check is spare. RISKS: (1) register cap 112 at NB 128 (64 acc + 16 a0 + addressing; V64's consumer fit 32 acc + 32 a0 in 104): a spill is OFF by criterion (a), not a slow run; if it spills, the fallback design is NOT an N split (each pair would dequantize the full stage: 2x dequant ALU per SM, the very cost K-pair hides) but dropping the producer to setmaxnreg 24 does not help either (the 96 x 640 pool caps consumers at 112), so report and close. (2) Lockstep: both pairs wait on the same full barrier; if they stay phase-locked (both dequant, then both MMA) the only gain is latency hiding from 4 warps per SMSP instead of 2; the drift up to 3 stages is what reproduces 2-CTA interleaving. A follow-up arm, only if this one is flat and the diag said SERIAL, is a one-stage start skew of pair 1 (it costs one stage per item at the reduction, ~1.5% of a 64-stage w13 item). (3) Not bit-exact: K split reorders the fp32 sum (two half-K partials added once); accuracy gates as fsilu's. (4) Prepare time: one more nvcc module (2 instantiations) after w2st; deadline-guarded, OFF if late. (5) The kp w13 carries fast silu: if master's fsilu low gate is OFF on a run the gate compares kp+fsilu vs master; the PATH CHOSEN line prints master's low flags. Distinct from the refuted mechanisms: not ping-pong over items (one item, one ring, same slab bytes per stage, no ingest cut), not the in-WG register ring (no wait<1>), not 2 CTAs/SM (no slab split), not mc2 multicast. Read the [rega-kp] PATH CHOSEN / attrs lines before crediting any turn_1 row (row noise +-3-4%).

## Adversarial review

Reviewed 2026-10-29T154811Z by an adversarial subagent (model opus).

**Verdict: proceed.** E passes: CLAIMS.md:70 [lead] (anatomy cb22e610 SERIAL at low, g 0.338/0.324, t4/t0 0.49, not ingest-bound) reopens the K-pair shelf CLAIMS.md:69; CLAIMS.md:55's reopen (a different low-class kernel) is met; board/blob unmoved. K-pair is a sound TLP/latency lever for what was measured, but a flat or OFF result is readable only if the gate also runs a stagger arm, prints w13 and w2 separately, and captures ptxas wgmma-serialization warnings.

### Justification
Low w13 warm pop:1: t0 293.6, t1 194.4, t2 191.1, t3 290.6, t4 145.1, t5 158.6 us; ~512 stages/CTA at 0.573 us. The super-serial reading (t0 > t1+t2-t5) is at least partly a model artifact: modes 1 and 2 are each floored by the producer's pace (ft: t2 0.586 us/stage vs t4 0.587), so t1+t2-t5 undercounts the consumer chain; the data fit a plain serial per-WG chain (dq_stage, 8 dependent k16 wgmmas, wgmma_wait<0>, mbarrier round trip), latency-bound with the producer idle half the time (t4/t0 0.41-0.65): the regime a TLP lever attacks. K-pair halves each WG's dependent MMA chain (8 -> 4 k16) and doubles independent consumer chains per SMSP; its floor is ~t4 or the dequant issue rate, not max(t1,t2). V64 occ2 (LESSONS:180, 0.887 vs 0.984) is a fair analog. Prior differences are real (one shared ring, one item, 128-route slab kept; not CLAIMS:44 item ping-pong, not CLAIMS:68 wait<1> ring). WEAKEST LINK: pairs likely phase-locked (same full[s], producer ahead, named_bar(1,512) per item re-syncs), leaving only chain-halving and latency hiding. SECOND RISK: 112 regs at NB 128; ptxas may silently serialize wgmmas (C7510/C7512) without a spill. COST: reduction 64 KB/item + ~6 named barriers ~0.5 us vs ~37 (w13) / ~18 (w2) us per item, 1.5-3%. Prepare fine (fsilu+w2st end ~176 s). VALUE RE-SIZE: CLAIMS:70's 13.32% applies a 40% all-row share to a class touching only turn_1 rows; recovering all of g at sliding turn_1 (x0.823, share 0.217) and full turn_1 (x0.884, share 0.117) caps low at ~5.2% headline; high (g 0.21, rega 1076.5 of 2246.9 us) caps ~3%. gain ~2.1 (1.6 if high OFF) stands; p 0.2 -> ~0.3, EV ~0.6%, user-requested.

### Confounders on the current kernel
- experts-rega-w13-fast-silu-epilogue (9fbde72c, CLAIMS:51): kp w13 carries fast silu; not confounding while rega_fsilu_ok['low'] is ON; PATH CHOSEN must echo master's low flags.
- w2st arm a (95af1c5f, CLAIMS:54, x0.97 on w2 at low): kp moves the sid table after the reduction and limits early loads to pair 0, possibly giving back part of w2st's gain; print w13 and w2 ratios separately and prefer a per-GEMM gate (kp w13 with master's w2st w2 when kp w2 is not <= 1.00).
- combo-b two-CTA low re-gate (61ce79bb, CLAIMS:46/55): REGA_LO_SKIP pins variant 0; the skew guard that killed V64 is missing from kp's gate.
- two-phase align (1932a9c4) and norm-gather (33aa4083): unchanged tables; no confound.

### Interference and A/B needs
No companion judge run. Fold into _rega_kp_phase: (1) a runtime STAGGER argument (uniform branch, no new instantiation) gated as a second arm with stagger 0: pair 1 starts each item one stage late, or the pairs alternate dequant/MMA via an ordered named-barrier pair; (2) capture and print the kp build's ptxas stderr plus a SASS grep for WARPGROUP.DEPBAR count per k loop, with regs and local (CLAIMS:68's reopen evidence); (3) w13 and w2 ratios printed separately with per-GEMM dispatch; (4) add the rega-lo/w2st skew-guard routing (pop:2.2, ~1.83x uniform, <= 1.00 every round) to the gate. If (1) is not folded in and the run is flat, the follow-up is the stagger arm, not a [refuted] line.

### Open questions
- Is the super-serial excess real contention or the producer floor in modes 1 and 2? Compare t1/t2 per stage to t4 per stage (low 0.380 / 0.373 vs 0.283; ft t2 = t4). If the floor model holds, g underestimates what a TLP lever can recover (floor ~t4) and overestimates a pure overlap lever (floor max(t1,t2)).
- Sibling worth minting: ordered anti-phase of master's two existing consumer WGs at stage granularity (FA3-style named barriers): WG0 issues wgmmas while WG1 dequantizes, at 232 regs, no K split, no reduction, bit-exact, full slab and ring kept; attacks the overlap gap toward max(t1,t2) = 194 us.
- Deferred wgmma_wait (wait<1> with double-buffered a0) inside one WG fits at 232 regs; CLAIMS.md:68 holds it refuted until SASS shows no C7510/C7512 serialization; was pf-rega-low-stream's SASS ever checked?
- 'kp ON but rows flat -> [refuted]' should read the stagger arm and the ptxas warnings first.
- If the high gate runs, report its ratios even when OFF (g 0.21 bounds ~3%).

