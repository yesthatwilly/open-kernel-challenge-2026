---
problem: muse_prefill_layer
slug: experts-rega-stage-anatomy-diag-r2
created: 2026-10-29T150952Z
base: master
mint_blob: "c63801e7a1dcfdb7cbdd0d38bc9b295407be7489"
best_at_mint: 1.5008
area: "w13 + w2 expert GEMMs (prepare-only diagnostic: where a rega stage goes)"
change: >
  Built on master (submission.py blob c63801e7, board 1.5008: fast-silu w13 and w2st arm a landed,
  LO phase pinned off by REGA_LO_SKIP). A prepare-only diagnostic: kernel_fn, _experts,
  _experts_rega and every function the timed call reaches stay byte-for-byte master's, and no
  state key kernel_fn reads is set or changed. QUESTION: DIAGNOSIS (historical, 1.4677 board) bans
  consumer-side dequant/MMA overlap on the rega tile (128 weight rows = two consumer warpgroups x
  4 warps x 16 rows, by up to 128 routes at bucket 128, BK = 128 per stage) until the L2->SM
  ingest cap is removed, but both cap-removal attempts ran slower (mc2 slab multicast cut issued
  L2->SM bytes 24-39% and ran 1.31-1.81x slower; bm256 halved slab bytes per FLOP and ran
  1.16-1.20x slower), and master's consume_item is serial by construction inside a warpgroup
  (dq_stage into a0, wgmma x8, commit, wgmma_wait<0>, release) with both consumer warpgroups
  waiting on the same full barrier. This run measures where a stage goes. Add one prepare-only
  phase _rega_anat_phase(state, weights, device, t_prepare0, fsilu_info), called in prepare()
  right after _rega_w2st_phase and BEFORE 'del fsilu_info', its summary lines printed at once and
  kept under ~3 KB so the align2 / fa3c / lo / fsilu / w2st echo block stays the last prepare()
  output. It derives one module muse_prefill_rega_anat from _rega_fsilu_src() (the
  muse_prefill_rega_fsilu_w2a source master dispatches when rega_fsilu_ok / rega_w2st_ok are set:
  master's _rega_src() + the fast-silu swap + REGA_W2ST_EDITS), by asserted (old, new) edits each
  occurring exactly once (else '[rega-anat] OFF'), adding a runtime int MODE kernel argument
  (uniform branch, no new template instantiation: the four rega_kernel<G, BIG> instantiations stay
  four) read by the consumer warpgroups: 0 = unchanged; 1 = no dequant (a0 filled once per item
  from constant bf16x2 pairs instead of dq_stage; codes and scales still land through the producer
  and the full / empty barriers run as master's); 2 = no MMA (dq_stage runs every stage and every
  a0 register is folded into acc by one dependent add so the compiler keeps the dequant; the 8
  Wg<NB>::mma calls, commit and wait skipped); 3 = no epilogue math (silu_mul / gsc scaling
  replaced by packing zeros; named_bar, stmatrix staging, the w2st sid table and every store
  kept); 4 = producer only (consumers wait on each full barrier and arrive on its empty barrier,
  nothing else; epilogue skipped). Print the anat kernels' attributes (regs, local bytes, smem,
  CTAs/SM per instantiation, as _rega_fsilu_attrs) next to fsilu_w2a's. Then per class low / high
  / ft (T = 753, 3012, 16384, REGA_FSILU_TOKENS; only classes with rega_ok, low only when
  rega_low_variant == 0), per GEMM (w13 = rega_gemm(0, ...), K = HIDDEN, KS = 64; w2 =
  rega_gemm(1, ...), K = D_EXPERT, KS = 32) on master's routed inputs (xs / hs from master's own
  _experts_rega pipeline for that class's flags, so the align2 / norm-gather tables are what
  kernel_fn would launch on), on two routings: the judge-like pop routing (_rega_pop_case at
  fsilu_info['alpha'][cls], the alpha _rega_fsilu_calibrate bisected into REGA_FSILU_BAND_US
  starting from REGA_FSILU_POP_MULT = 1.83x the uniform local routes; if missing,
  _rega_pop_pick(_rega_pop_counts(...), 1.83 x uniform)) and random (_ssdq_case 'random'), time
  master's launch (state['rega_fsilu_ext'] if the class's flag is on, else state['rega_ext']) and
  anat modes 0-4 as CUDA graphs through _rega_fsilu_bench (cold L2: 256 MB read flush; warm L2:
  one unmeasured replay + REGA_FSILU_WARM_REPS; empty-graph median subtracted), REGA_FSILU_ROUNDS
  = 2 rounds x REGA_FSILU_ALTERNATIONS = 11 alternations. Print per class, GEMM, routing and L2
  mode: master us and t0..t4 us; max stages per CTA (ceil(items / grid) x KS, items from
  _rega_tiles' chunk count x output tiles under the static it = blockIdx.x; it += gridDim.x map);
  per-stage us t / max stages; and the model check t0 vs t1 + t2 - t4 (serial) vs max(t1, t2)
  (overlapped) vs t4 (ingest floor). Outputs of modes 1-4 are garbage and never leave the phase.
  Deadlines: no nvcc build once prepare() has used 205 s, no timing step after 255 s (ft last;
  skipped classes print SKIPPED); never raises (try / except / finally: synchronize, drop the anat
  module's tensors and graphs, empty_cache). End with one '[rega-anat] VERDICT <cls>' line per
  class (warm L2, pop routing, w13 and w2 each): INVALID when anat mode 0 is outside 0.97-1.03x
  master's launch or any anat kernel spills; ingest-bound when t4 >= 0.85 t0; overlapped when t0
  <= 1.10 max(t1, t2); serial when t0 >= 0.90 (t1 + t2 - t4) and t0 > 1.10 max(t1, t2); else mixed
  (fractions printed).
expected:
  - shapes: "*"
    direction: flat
    why: "diagnostic only: the timed path is master's c63801e7, so every row is master re-measured (row noise +-1.5-2.5%); the value is the [rega-anat] per-mode stage times. Lever sized (--gain 4.0): rega w13 + w2 are 48-53% of the sliding rows and 28-43% of the full rows (b2941ea profile, ~40% mean), so an overlap / TLP gap of ~10% of rega time is ~4% headline; p 0.15 because every overlap try so far lost in its own gate and the K-pair shelf estimate is 2.1%"
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "experts-rega-stage-anatomy-diag: re-minted on current master blob c63801e7 at the user's request (10-29), was f4751fc0: derives from the fsilu_w2a module master now dispatches (fast silu + w2st arm a) not v2, pop alpha from _rega_fsilu_calibrate since _rega_lo_phase is pinned off, KS 64 for w13 (was 32 for both), warm L2 timed too, mode-0 fidelity check, prepare deadlines, and a pre-registered decision"
target: diag
value_gain: 4.0
value_p: 0.15
value_ev: 0.6
value_cost: 200k tokens, 1 judge runs
evidence: >
  board b2941ea23dc743a3 profile
  (campaigns/2026-10-26-expert-bandwidth/runs/b2941ea23dc743a3.profile.txt): rega w13 + w2 = 249.9
  + 127.4 of 708.8 us at sliding turn_1 (53%), 703.9 + 372.6 of 2246.9 us at sliding turn_4 (48%),
  3441.1 + 1799.0 of 10967.6 us at sliding first_turn (48%). Ingest: master's tile moves 5.8-6.05
  TB/s L2->SM at T=3012 (LESSONS:182); mc2 cut issued L2 bytes 24-39% and ran 1.31-1.81x slower
  (DISTILL:185, 0ca60bb7 / a6f60630); per-CTA cycles max/mean 1.06-1.13 vs bytes 1.09-1.20
  (d00770b6, LESSONS:195). TLP: V64 occ2 0.887x vs occ1 0.984x master (LESSONS:180). Overlap tries
  lost in their own gates: pingpong 1.18-1.47x, scale-expand 1.03-1.10x (LESSONS:182),
  pf-rega-low-stream ring x1.075 (6edf1cb002bf42b9, CLAIMS)
decision: >
  Pre-registered on the warm-L2 pop-routing VERDICT lines (the judged condition), per class;
  INVALID or SKIPPED in every class = no decision, re-run once only if a check is spare. SERIAL
  (t0 >= 0.90 (t1 + t2 - t4) and t0 > 1.10 max(t1, t2) at low and/or high): the stages do not hide
  dequant under MMA, so DIAGNOSIS's ingest-cap premise for the overlap ban is retired and the
  overlap gap g = (t0 - max(t1, t2)) / t0 bounds every overlap / TLP lever; if g x the rega share
  of the rows (b2941ea: 53% turn_1 sliding, 48% turn_4 sliding, 48% first_turn sliding) sums to >=
  2.5% headline, re-mint the [shelf] rega K-pair claim (experts-rega-low-kpair-four-consumer-wgs)
  as a check sized by g, as the next rung; else close it. Pingpong / scale-expand / the
  pf-rega-low-stream ring stay refuted as built (their own gates lost). OVERLAPPED (t0 <= 1.10
  max(t1, t2)): overlap levers stay dead; append [refuted] for the K-pair shelf claim (no
  dequant/MMA gap left to hide); the gap is elsewhere, named by which single mode sits near t0: t2
  (dequant only) ~ t0 -> a cheaper dequant (fewer INT32 ops per code, the decode dq3 direction) is
  the next lead; t1 (MMA + slab only) ~ t0 -> the wgmma / slab path; neither -> epilogue (t0 - t3)
  or tail. INGEST-BOUND (t4 >= 0.85 t0): the reopen condition of the three [refuted]
  slab-multicast claims (mc4 / mc2 / bm256, experts-rega-low-2cta-fullslab-mc2,
  experts-rega-mc2-bm256-stack-high-ft) is met: next rung is experts-rega-low-2cta-fullslab-mc2
  re-minted at the class with the highest t4 / t0 (low first) citing this VERDICT, and the K-pair
  claim closes (TLP cannot beat a cap). MIXED: no lever reopens; append one CLAIMS line with the
  per-class fractions; K-pair stays [shelf].
instrument: >
  prepare-time CUDA-graph timings in the judge check's own prepare() (no H100): '[rega-anat]
  attrs' (regs / local bytes / smem / CTAs per SM per instantiation, anat vs fsilu_w2a),
  '[rega-anat] TIME <cls> <w13|w2> <pop:alpha|random> <cold|warm>: master / m0 / m1 / m2 / m3 / m4
  us', '[rega-anat] STAGE <cls> <gemm>: max stages per CTA, per-stage us per mode', '[rega-anat]
  MODEL <cls> <gemm>: t0, t1 + t2 - t4, max(t1, t2), t4 / t0', and one '[rega-anat] VERDICT <cls>:
  serial|overlapped|ingest-bound|mixed|INVALID' per class; '[rega-anat] OFF / SKIPPED' lines when
  a deadline or an edit assertion stops it
user_requested: true
review: required
status: queued
---

# experts-rega-stage-anatomy-diag-r2

## Change

Built on master (submission.py blob c63801e7, board 1.5008: fast-silu w13 and w2st arm a landed, LO phase pinned off by REGA_LO_SKIP). A prepare-only diagnostic: kernel_fn, _experts, _experts_rega and every function the timed call reaches stay byte-for-byte master's, and no state key kernel_fn reads is set or changed. QUESTION: DIAGNOSIS (historical, 1.4677 board) bans consumer-side dequant/MMA overlap on the rega tile (128 weight rows = two consumer warpgroups x 4 warps x 16 rows, by up to 128 routes at bucket 128, BK = 128 per stage) until the L2->SM ingest cap is removed, but both cap-removal attempts ran slower (mc2 slab multicast cut issued L2->SM bytes 24-39% and ran 1.31-1.81x slower; bm256 halved slab bytes per FLOP and ran 1.16-1.20x slower), and master's consume_item is serial by construction inside a warpgroup (dq_stage into a0, wgmma x8, commit, wgmma_wait<0>, release) with both consumer warpgroups waiting on the same full barrier. This run measures where a stage goes. Add one prepare-only phase _rega_anat_phase(state, weights, device, t_prepare0, fsilu_info), called in prepare() right after _rega_w2st_phase and BEFORE 'del fsilu_info', its summary lines printed at once and kept under ~3 KB so the align2 / fa3c / lo / fsilu / w2st echo block stays the last prepare() output. It derives one module muse_prefill_rega_anat from _rega_fsilu_src() (the muse_prefill_rega_fsilu_w2a source master dispatches when rega_fsilu_ok / rega_w2st_ok are set: master's _rega_src() + the fast-silu swap + REGA_W2ST_EDITS), by asserted (old, new) edits each occurring exactly once (else '[rega-anat] OFF'), adding a runtime int MODE kernel argument (uniform branch, no new template instantiation: the four rega_kernel<G, BIG> instantiations stay four) read by the consumer warpgroups: 0 = unchanged; 1 = no dequant (a0 filled once per item from constant bf16x2 pairs instead of dq_stage; codes and scales still land through the producer and the full / empty barriers run as master's); 2 = no MMA (dq_stage runs every stage and every a0 register is folded into acc by one dependent add so the compiler keeps the dequant; the 8 Wg<NB>::mma calls, commit and wait skipped); 3 = no epilogue math (silu_mul / gsc scaling replaced by packing zeros; named_bar, stmatrix staging, the w2st sid table and every store kept); 4 = producer only (consumers wait on each full barrier and arrive on its empty barrier, nothing else; epilogue skipped). Print the anat kernels' attributes (regs, local bytes, smem, CTAs/SM per instantiation, as _rega_fsilu_attrs) next to fsilu_w2a's. Then per class low / high / ft (T = 753, 3012, 16384, REGA_FSILU_TOKENS; only classes with rega_ok, low only when rega_low_variant == 0), per GEMM (w13 = rega_gemm(0, ...), K = HIDDEN, KS = 64; w2 = rega_gemm(1, ...), K = D_EXPERT, KS = 32) on master's routed inputs (xs / hs from master's own _experts_rega pipeline for that class's flags, so the align2 / norm-gather tables are what kernel_fn would launch on), on two routings: the judge-like pop routing (_rega_pop_case at fsilu_info['alpha'][cls], the alpha _rega_fsilu_calibrate bisected into REGA_FSILU_BAND_US starting from REGA_FSILU_POP_MULT = 1.83x the uniform local routes; if missing, _rega_pop_pick(_rega_pop_counts(...), 1.83 x uniform)) and random (_ssdq_case 'random'), time master's launch (state['rega_fsilu_ext'] if the class's flag is on, else state['rega_ext']) and anat modes 0-4 as CUDA graphs through _rega_fsilu_bench (cold L2: 256 MB read flush; warm L2: one unmeasured replay + REGA_FSILU_WARM_REPS; empty-graph median subtracted), REGA_FSILU_ROUNDS = 2 rounds x REGA_FSILU_ALTERNATIONS = 11 alternations. Print per class, GEMM, routing and L2 mode: master us and t0..t4 us; max stages per CTA (ceil(items / grid) x KS, items from _rega_tiles' chunk count x output tiles under the static it = blockIdx.x; it += gridDim.x map); per-stage us t / max stages; and the model check t0 vs t1 + t2 - t4 (serial) vs max(t1, t2) (overlapped) vs t4 (ingest floor). Outputs of modes 1-4 are garbage and never leave the phase. Deadlines: no nvcc build once prepare() has used 205 s, no timing step after 255 s (ft last; skipped classes print SKIPPED); never raises (try / except / finally: synchronize, drop the anat module's tensors and graphs, empty_cache). End with one '[rega-anat] VERDICT <cls>' line per class (warm L2, pop routing, w13 and w2 each): INVALID when anat mode 0 is outside 0.97-1.03x master's launch or any anat kernel spills; ingest-bound when t4 >= 0.85 t0; overlapped when t0 <= 1.10 max(t1, t2); serial when t0 >= 0.90 (t1 + t2 - t4) and t0 > 1.10 max(t1, t2); else mixed (fractions printed).

## Expected

- `*` flat: diagnostic only: the timed path is master's c63801e7, so every row is master re-measured (row noise +-1.5-2.5%); the value is the [rega-anat] per-mode stage times. Lever sized (--gain 4.0): rega w13 + w2 are 48-53% of the sliding rows and 28-43% of the full rows (b2941ea profile, ~40% mean), so an overlap / TLP gap of ~10% of rega time is ~4% headline; p 0.15 because every overlap try so far lost in its own gate and the K-pair shelf estimate is 2.1%

## Value case

- **Target:** diag: a prepare-only judge diagnostic that sizes a lever.
- **Evidence:** board b2941ea23dc743a3 profile (campaigns/2026-10-26-expert-bandwidth/runs/b2941ea23dc743a3.profile.txt): rega w13 + w2 = 249.9 + 127.4 of 708.8 us at sliding turn_1 (53%), 703.9 + 372.6 of 2246.9 us at sliding turn_4 (48%), 3441.1 + 1799.0 of 10967.6 us at sliding first_turn (48%). Ingest: master's tile moves 5.8-6.05 TB/s L2->SM at T=3012 (LESSONS:182); mc2 cut issued L2 bytes 24-39% and ran 1.31-1.81x slower (DISTILL:185, 0ca60bb7 / a6f60630); per-CTA cycles max/mean 1.06-1.13 vs bytes 1.09-1.20 (d00770b6, LESSONS:195). TLP: V64 occ2 0.887x vs occ1 0.984x master (LESSONS:180). Overlap tries lost in their own gates: pingpong 1.18-1.47x, scale-expand 1.03-1.10x (LESSONS:182), pf-rega-low-stream ring x1.075 (6edf1cb002bf42b9, CLAIMS)
- **Gain if it works:** 4.0% headline; **p** 0.15; **EV** 0.6% (bar: gain >= 3.0%, EV >= 1.5%); **below the bar** (EV 0.6% < 1.5%), minted because the user described it.
- **Cost:** 200k tokens, 1 judge runs.
- **Decision rule:** Pre-registered on the warm-L2 pop-routing VERDICT lines (the judged condition), per class; INVALID or SKIPPED in every class = no decision, re-run once only if a check is spare. SERIAL (t0 >= 0.90 (t1 + t2 - t4) and t0 > 1.10 max(t1, t2) at low and/or high): the stages do not hide dequant under MMA, so DIAGNOSIS's ingest-cap premise for the overlap ban is retired and the overlap gap g = (t0 - max(t1, t2)) / t0 bounds every overlap / TLP lever; if g x the rega share of the rows (b2941ea: 53% turn_1 sliding, 48% turn_4 sliding, 48% first_turn sliding) sums to >= 2.5% headline, re-mint the [shelf] rega K-pair claim (experts-rega-low-kpair-four-consumer-wgs) as a check sized by g, as the next rung; else close it. Pingpong / scale-expand / the pf-rega-low-stream ring stay refuted as built (their own gates lost). OVERLAPPED (t0 <= 1.10 max(t1, t2)): overlap levers stay dead; append [refuted] for the K-pair shelf claim (no dequant/MMA gap left to hide); the gap is elsewhere, named by which single mode sits near t0: t2 (dequant only) ~ t0 -> a cheaper dequant (fewer INT32 ops per code, the decode dq3 direction) is the next lead; t1 (MMA + slab only) ~ t0 -> the wgmma / slab path; neither -> epilogue (t0 - t3) or tail. INGEST-BOUND (t4 >= 0.85 t0): the reopen condition of the three [refuted] slab-multicast claims (mc4 / mc2 / bm256, experts-rega-low-2cta-fullslab-mc2, experts-rega-mc2-bm256-stack-high-ft) is met: next rung is experts-rega-low-2cta-fullslab-mc2 re-minted at the class with the highest t4 / t0 (low first) citing this VERDICT, and the K-pair claim closes (TLP cannot beat a cap). MIXED: no lever reopens; append one CLAIMS line with the per-class fractions; K-pair stays [shelf].
- **Instrument:** prepare-time CUDA-graph timings in the judge check's own prepare() (no H100): '[rega-anat] attrs' (regs / local bytes / smem / CTAs per SM per instantiation, anat vs fsilu_w2a), '[rega-anat] TIME <cls> <w13|w2> <pop:alpha|random> <cold|warm>: master / m0 / m1 / m2 / m3 / m4 us', '[rega-anat] STAGE <cls> <gemm>: max stages per CTA, per-stage us per mode', '[rega-anat] MODEL <cls> <gemm>: t0, t1 + t2 - t4, max(t1, t2), t4 / t0', and one '[rega-anat] VERDICT <cls>: serial|overlapped|ingest-bound|mixed|INVALID' per class; '[rega-anat] OFF / SKIPPED' lines when a deadline or an edit assertion stops it

## Risk

Ships master: the headline is noise. Prepare budget (estimate): master's prepare should end ~140-150 s with the LO phase skipped (the w2st run ended at 182.6 s with LO, LO ~40 s); one nvcc build ~32-42 s (fsilu build 32.2 s) + ~25 s of timing fits under the 205 s / 255 s deadlines; a late prepare prints SKIPPED instead of risking the 300 s limit. Mode 2's sink must keep the dequant live and MODE as a runtime int may shift register allocation: the attrs line and the mode-0 vs master fidelity check (INVALID outside 0.97-1.03x) guard both. Checks left today: prefill 20.
