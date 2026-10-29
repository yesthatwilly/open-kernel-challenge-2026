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
review: proceed
status: running
review_at: 2026-10-29T151524Z
review_companions: null
review_best: 1.5007957663757285
review_blob: c63801e7a1dcfdb7cbdd0d38bc9b295407be7489
stage: implemented
claimed_at: 2026-10-29T151524Z
base_sha: 9047e4ba7caa809ca4b9aae6a1a556243f61a13f
base_blob: c63801e7a1dcfdb7cbdd0d38bc9b295407be7489
branch: hyp/muse_prefill_layer/experts-rega-stage-anatomy-diag-r2
worktree: .worktrees/muse_prefill_layer/experts-rega-stage-anatomy-diag-r2
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-29T153453Z
impl_commit: 7679e727d73f7e91048dd97a08d9180f90f3581a
submission_sha256: b7d3e604c299b3cd70beeaffbe997b46cee7b5540e95da5275f4b424257ae1ac
norm_sha256: 183a04a464325f26c8d9458cf1b71e1471515edd14226c6d2b9ce16f7cb6ea78
submission_blob: 7cc2179e987ab39cdacbf1caeb14d4e64cbfa4b6
lint_warnings: 118
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

## Adversarial review

Reviewed 2026-10-29T151524Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed with no companions. One run gives a readable answer: the timed path is master's c63801e7, mode 0 is checked against master's launch, and the deadlines leave about 50 s of slack before the build cutoff. Three things to fix before building: mode 2's add-based sink inflates t2; mode 4 drops the epilogue, so t1 + t2 - t4 counts the epilogue twice; and OVERLAPPED does not justify refuting the K-pair TLP claim.

### Justification
A. The mechanism question is well posed and matches master's code (consume_item, submission.py:2747: dq_stage, then 8 wgmma, commit, wgmma_wait<0>, release; serial inside each warpgroup). KS is right: w13 64, w2 32. Mode 0 against master's launch at 0.97-1.03x, plus the attrs/spill check, is a proper in-run control. The prior is accurate. PREPARE BUDGET is realistic: LO-skipped master ended w2st at 149.7 s (7db6fa1a) and 155.4 s (b2941ea); fsilu builds took 32-33 s; anat timing about 15-20 s, ending around 200-235 s, under the 255 s step deadline and 300 s. WEAKEST LINKS: (1) g = (t0 - max(t1,t2))/t0 does NOT bound TLP levers: t1 is the MMA path as master schedules it (wgmma_wait<0> per stage + full-barrier wait), not the tensor-core floor; LESSONS:180 shows a ~9-10% TLP gain with this same serial structure. Print each GEMM's wgmma math floor next to t1 and t4; refute K-pair only if t1 is also within ~10% of max(math floor, t4). (2) Mode 2's sink folds 32 a0 registers into acc with real adds, which can add 10-25% to t2; use asm volatile("" :: "r"(a0[r][k])) per register instead. (3) Mode 4 skips the epilogue while t1 and t2 keep it, so t1 + t2 - t4 double-counts the epilogue; keep mode 3's zero-pack epilogue (with stores) in mode 4, or add mode 5 = producer + zero epilogue and use t5 in the serial formula; keep t4 as the pure ingest floor. (4) The runtime MODE branch near wgmma may make ptxas serialize the wgmma pipeline (C7510/C7512, CLAIMS:68); branch on MODE once per item outside the k-loop, grep the build log for C7510/C7512 and print it on the attrs line. Smaller: mode 1's constant A should not be all zeros; print the per-class N-bucket histogram with the STAGE line; the decision's 2.5% sum should use the all-row rega share (~40%), not sliding-only (48-53%).

### Confounders on the current kernel
- experts-rega-w13-fast-silu-epilogue (CLAIMS:51): mode 0 matches master only while rega_fsilu_ok[cls] is ON; compare mode 0 with state['rega_fsilu_ext'] directly and print master's dispatch separately.
- w2st arm a (CLAIMS:54, 56): master's w2 is picked by rega_w2st_ok, not rega_fsilu_ok; the builder must use rega_w2st_ok for w2.
- combo-b / REGA_LO_SKIP (CLAIMS:46, 56): low class always on master's rega_kernel. No confound.
- align2 two-phase (CLAIMS:42) and norm-gather (CLAIMS:46): tables and xs come from master's own pipeline. No confound.

### Interference and A/B needs
No companion is warranted. The timed path is byte-identical master; the result is the internal per-mode times with in-run controls (mode 0 vs master's launch, attrs/spill, two routings x two L2 modes). The only follow-up is one re-run if every class is INVALID or SKIPPED.

### Open questions
- The r1 file 2026-10-26T224009Z-experts-rega-stage-anatomy-diag.md is still queued (hold) and stale; retire or supersede it.
- Log tail: ~6.2 KB left for the anat summary; print MODEL and VERDICT last and repeat the 3 VERDICT lines after the w2st echo.
- INGEST-BOUND branch: mc2 cut issued bytes 24-39% yet ran 1.31-1.81x slower, suggesting a cluster-sync or multicast cost; an ingest-bound verdict should re-mint a multicast design addressing that cost.
- Is the anat nvcc build bounded by a timeout in _rega_build? A build hang past ~95 s after starting near 205 s would exceed 300 s.


## Implementation notes

**Mechanism:** a prepare-only diagnostic phase `_rega_anat_phase` builds a separate `muse_prefill_rega_anat` module (the fsilu_w2a source plus a runtime MODE 0-5 argument that removes dequant, MMA, epilogue math or consumer work from `consume_item`). It times master's dispatched rega w13 and w2 against modes 0-5 on the low, high and ft classes. The timed path is master's c63801e7, byte for byte.

Built 2026-10-29T1533Z in the worktree's `muse_prefill_layer/submission.py`. All 645 lines are insertions; no line was deleted or changed. `kernel_fn` through EOF is identical to master, and so is everything the timed call reaches.

**Where:**
- `prepare()`: `_rega_anat_phase(state, weights, device, t_prepare0, fsilu_info)` is called right after `_rega_w2st_phase` and before `del fsilu_info`, with a `torch.cuda.synchronize()` before it. Its returned VERDICT lines are printed again after the w2st echo block.
- New block before `kernel_fn`:
  - constants `REGA_ANAT_*`
  - `REGA_ANAT_CPP_EDIT` plus 9 `REGA_ANAT_EDITS`. Each old text occurs exactly once in `_rega_fsilu_src()`; this was verified offline by AST-extracting master's strings, and the v2 hash still matches.
  - functions `_rega_anat_src / _audit / _build / _ptxas / _attrs / _attr_text / _launch / _chunks / _work / _rel / _step / _step_lines / _verdict / _report / _phase`.

**CUDA edits** (on the fsilu_w2a source; the four `rega_kernel<G, BIG>` instantiations stay four):
- The host `rega_gemm` gets `int64_t mode`, with `TORCH_CHECK(0 <= mode <= 5)`. The kernel gets `int mode` as its last argument, `WG_CONSUME` passes it, and `consume_item` gets `int mode`.
- `consume_item` branches on mode once per item, outside the k loop (review fix 4). Master's loop is the final `else` arm and is unchanged (modes 0 and 3):
  - mode 1 has its own wgmma loop with a constant A;
  - mode 2 has its own dequant-only loop;
  - modes 4 and 5 have a wait-full / arrive-empty loop.
- After the shared `wgmma_wait<0>(); fence_acc`, mode 4 returns (no epilogue). Modes 3 and 5 take a new zero-pack arm in both the w13 and w2 staging blocks. That arm keeps master's stmatrix layout, named_bar, the w2st sid table and every store.

**Review fixes:**
1. **Mode 2 sink (deviation, partial).** Each a0 register is an input of a zero-instruction `asm volatile("" :: "r"(a0[i][j]))`, as the review asked. An empty asm emits no PTX, though. ptxas performs dead-code elimination, so on its own that sink would let ptxas drop `dq_stage`, and t2 would collapse to about t5. So the 32 registers also feed ONE asm whose only instructions are a `setp` and 8 `st.shared.v4.b32` predicated on the pipeline's `dead` flag. That flag is false in every healthy run, so nothing is stored; the target address stays inside the `ep` staging tile. There are no adds and no dependency on `acc`. The cost is about 9 issue slots per stage against about 250-300 dequant instructions, so t2 may read up to ~3% high.
2. **Mode 5 added.** Mode 5 = producer + mode 3's zero epilogue with its stores; mode 4 stays the pure producer, the ingest floor. The serial prediction is `t1 + t2 - t5`. Remaining bias: t1 and t2 carry the real epilogue math and t5 does not, so the prediction overcounts by the epilogue math. That amount (`t0 - t3`) is printed on every MODEL line.
3. **Math floor.** Per GEMM, the floor is the sum over work items of (bucket-padded tokens `16 << bucket`) x 128 weight rows x K x 2 FLOP. It is reported two ways: all items over 989 TF/s, and the busiest CTA under the kernel's static `it = blockIdx.x; it += gridDim.x` map in `decode_item` order, over 989/132 = 7.49 TF/s per SM. The larger of the two is used. It is printed on the STAGE line (constant shown) and on the MODEL line, as `t1/max(floor,t4)`. The VERDICT's OVERLAPPED label says `K-pair refutable` only when `t1 <= 1.10 x max(floor, t4)`, else `K-pair NOT refuted`. See the decision text below.
4. **C7510 / C7512.**
   - The first build is an `-Xptxas -Werror` probe, `muse_prefill_rega_anat_p`; the moe precedent is the 31c27e4b log. If it passes, it is the module, at no extra cost.
   - If it fails, the C75xx codes and up to 2 ptxas lines are printed, then a plain build runs if prepare is still under 205 s. The result is printed on the `ptxas` line and in short form on the `attrs` line.
   - Caveat: an info-level C75xx note does not fail `-Werror`, and load_inline discards a good build's output. A passing probe therefore rules out a ptxas warning only.
   - We start no process of our own and redirect no file descriptor.
5. **Mode 1 constant A.** A is a non-zero bf16x2 near 1.0 (`0x3F803F80 ^ ((lane+4i+j)&7)*0x00010001`), set once per item and laundered through `asm volatile("" : "+r")`. The STAGE line carries a per-class N-bucket chunk histogram (N16/32/64/128), plus items, max items of one CTA x KS, and warm us per stage per mode.
6. **Master's actual dispatch.**
   - Per class, "master" means: w13 = `state['rega_fsilu_ext']` when `rega_fsilu_ok[cls]`, else `rega_ext`; w2 = `rega_fsilu_ext` (arm a) when `rega_w2st_ok[cls]`, else `rega_ext`.
   - The w2 input hs is computed by that dispatched w13 on master's align plus gather or norm-gather tables (`_rega_fsilu_tables` with `norm_gather_ok[cls]`).
   - Fidelity is m0/m on warm L2 with the pop routing, required within 0.97-1.03; cold is printed too.
   - Addition: a mode-0 exactness check on the pop step, comparing the mode-0 output with master's dispatched output. It accepts `eq`, or rel L2 <= 2e-3 (REGA_FSILU_HS_REL), and also requires error word 0 and both launch markers. A miss means INVALID: it catches a misrouted MODE.
   - "Spill" means the anat local bytes exceed the reference module's for the instantiation the class uses. The reference is fsilu_w2a when it was built, else master's v2. A strict local > 0 test was not used because master's own w13 rows are not required to have 0 local bytes.
7. **Nothing kept.** The phase uses its own `err` tensor (`state['rega_err']` is never touched). All tensors, graphs and closures are local to `_rega_anat_step` and are dropped in its `finally`. The ext handle is dropped in the phase's `finally`, followed by a synchronize and `empty_cache`, all wrapped in try/except. Nothing is written to `state`.
8. **Output and deadlines.**
   - MODEL lines, then VERDICT lines, are the phase's last output. The VERDICT lines are printed again after the w2st echo, so w2st's PATH CHOSEN lines are no longer the very last prepare lines. Estimated anat output is about 5.5-6.5 KB.
   - No build (probe or plain) starts after 205 s; if it cannot start, a SKIPPED line is printed. No timing step starts after 255 s, and no ft step after 240 s (an ft step takes about 4-6 s). Skipped steps or classes print SKIPPED.
   - **Not bounded: a hung nvcc.** Neither `_rega_build` nor `load_inline` takes a timeout, so a build that starts before 205 s and hangs past about 95 s would push prepare past 300 s. Master's fsilu build carries the same exposure.
   - The anat module has roughly twice the consume_item code of master's, so the build may take about 40-55 s rather than 32 s.
9. **Decision sizing.** The decision's 2.5% test uses the all-row rega share of 40% (`REGA_ANAT_SHARE`). The VERDICT line prints sum(t0 - max(t1,t2)) over w13 + w2, as a fraction of their summed t0, x 40%, against the 2.5% bar.

**Other deviations:**
- **KS.** The spec, and the review's "KS is right", say w13 K = HIDDEN, KS 64. Master's w13 launch actually passes `K = size_k = x.shape[1] = D_EXPERT = 4096` (the pooler output), so KS = 4096/128 = 32 for BOTH GEMMs. n_tiles is 64 for w13 and 32 for w2. The phase uses master's real launch arguments.
- **Order.** The pop routing runs first for low, high and ft, then random for low, high and ft. The verdict routing comes first, and ft is last within each routing. The pop alpha is `fsilu_info['alpha'][cls]`, or else `_rega_pop_pick(_rega_pop_counts(T), 1.83 x uniform)`.
- **Verification without CUDA on this node.** The device code was syntax-checked by `g++ -std=c++20 -fsyntax-only -Wall -Wextra` on stubbed CUDA built-ins, instantiating all four `rega_kernel<G, BIG>`: both fsilu and anat came out clean. Inline-asm bodies were swallowed by the stub and were checked by eye only. The pure-Python helpers (chunks, work, verdict, report, step_lines, ptxas grep) were exercised against a numpy-backed fake torch.

**Updated decision text** (replaces the `decision` field for this run; the rest is unchanged). Pre-registered on the warm-L2 pop-routing VERDICT lines, per class. INVALID or SKIPPED in every class = no decision; re-run once only if a check is spare.
- **INGEST-BOUND** (t4 >= 0.85 t0): as written. Re-mint experts-rega-low-2cta-fullslab-mc2 at the class with the highest t4/t0 (low first), and address the mc2 cluster-sync / multicast cost. The K-pair claim closes.
- **OVERLAPPED** (t0 <= 1.10 max(t1, t2)): overlap levers stay dead.
  - Append [refuted] for the K-pair shelf claim ONLY when the line says `K-pair refutable` (t1 <= 1.10 x max(math floor, t4)).
  - When it says `K-pair NOT refuted`, the MMA path as master schedules it (wait<0> per stage + full-barrier wait) sits above both the math floor and the ingest floor. The K-pair claim stays [shelf], sized by t1/max(floor, t4) - 1 x the 40% share.
  - The `near t0` field names the next lead: t2 -> a cheaper dequant; t1 -> the wgmma / slab path; neither -> the epilogue (t0 - t3) or the tail.
- **SERIAL** (t0 >= 0.90 (t1 + t2 - t5) and t0 > 1.10 max(t1, t2)): DIAGNOSIS's ingest-cap premise for the overlap ban is retired. If the VERDICT line's headline (gap fraction x 40% all-row share) is >= 2.5% at low and/or high, re-mint experts-rega-low-kpair-four-consumer-wgs as a check sized by that number; else close it. Pingpong, scale-expand and the pf-rega-low-stream ring stay refuted as built.
- **MIXED:** no lever reopens. Append one CLAIMS line with the per-class fractions; K-pair stays [shelf].

## Rules check

Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-stage-anatomy-diag-r2 diff`: 645 insertions, 0 deletions, in 3 hunks (prepare() x2, and the new block before kernel_fn).

- [x] CUDA-graph capturable: kernel_fn and everything it reaches are byte-identical to master (`sed -n '/^def kernel_fn/,$p'` diffs clean; no existing line changed). The new `.cpu()` / `.tolist()` / `synchronize()` calls are all in prepare-only functions (lint: outside kernel_fn's call graph).
- [x] Nothing derived from `data` is cached across calls: the phase uses synthetic cases only, sets no state key, never stores the module, graphs or tensors, and uses its own err tensor.
- [x] Weights: n/a, no new weight storage. The anat module reads master's `rega_bank` (NVFP4 codes and scales, unchanged), launched in prepare only.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: n/a, the timed path is unchanged (master's).
- [x] Attention sees every key: n/a, attention is untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the `-Xptxas -Werror` probe is an extra load_inline flag only. There is no subprocess, fd redirection or monkeypatching; the module is a new load_inline extension and no loaded module is edited.
- [x] All work happens inside the timed call; no output caching or replay: the diagnostic outputs (garbage for modes 1-5) live in phase-local buffers freed before prepare returns.
- [x] First line still `# okc: problem=muse_prefill_layer`; no triton.autotune added; no host sync in kernel_fn.

`okc_lint.py ... --problem muse_prefill_layer` (exit 0):
```
OK with warnings: .../experts-rega-stage-anatomy-diag-r2/muse_prefill_layer/submission.py
118 warn lines (master: 112). The 6 new ones are all prepare-side host-sync, e.g.:
warn  host-sync    L7080: used = min(int(n_padded.cpu()[0]) // REGA_BSM, block_experts.numel())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7081: be = block_experts[:used].cpu().tolist()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7160: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7195: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L7396: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
(+ the prepare() synchronize before the anat call)
```
`python3 -I -m py_compile` passes; the __pycache__ it left was removed.
