---
problem: muse_prefill_layer
slug: combo-b-normgather-two-cta-low-regate
created: 2026-10-25T161233Z
base: master
area: "w13 + w2 expert GEMMs at 200 < T <= 1024 (two CTAs/SM V64/V32 port, re-gated on master's random routing with a skew guard) stacked on combo A"
change: >
  COMBINATION B (conjectural: re-gate the two-CTA/SM low class on master's own gate routing,
  stacked on A). Start from the same merge as combination A
  (/home/user/git/musespark/muse_prefill_layer/hypotheses/archive/pre-expert-norm-fused-rega-gather/master_plus_normgather.merged.py:
  master 1.4587 + pre-expert-norm-fused-rega-gather bd60e2db) and port the low-class
  two-CTAs-per-SM kernels from experts-rega-low-class-two-ctas-per-sm, branch
  hyp/muse_prefill_layer/experts-rega-low-class-two-ctas-per-sm, impl commit dfb755d3 (check
  1a77da43b3304b07). PORT, from git show dfb755d3:muse_prefill_layer/submission.py: _REGA_LO_CPP,
  _REGA_LO_SRC (rega_lo_kernel<G, NMAX, STG, OCC>, rega_init_lo, rega_gemm_lo), REGA_LO_VARIANTS
  restricted to V64 = (64, 3, OCC 2) and V32 = (32, 4, OCC 2) (drop the V64-occ1 diag to save nvcc
  time), REGA_LO_REGS_OCC2 80, REGA_LO_POOL, REGA_LO_MAXBLK, REGA_LO_STATIC_BOUND, _rega_pop_case,
  _rega_pop_alpha, _rega_lo_tiles, _rega_lo_gate, the per-candidate attribute check (numRegs ==
  80, local 0, 2 CTAs/SM, binaryVersion 90, static <= 12288) and per-candidate self-checks
  (T=300/753 random/skewed/one, 753 pop, 1024 local-saturated; A/B/C copy-replay with refilled
  inputs). DO NOT port its module layout: dfb755d3 renamed master's extension to
  muse_prefill_rega_v3 (master v2 source + LO). Here master's muse_prefill_rega_v2 build, its v1
  fallback and its sha checks stay byte-identical, and the LO source compiles as a SEPARATE
  extension muse_prefill_rega_lo containing only _REGA_HEAD + _rega_wgmma_src() + the namespace
  part of _REGA_BODY needed by the LO template + _REGA_LO_SRC (the pattern
  experts-rega-producer-scale-expand-all-t used for muse_prefill_rega_se: master templates never
  instantiated, only the 4 LO instantiations). Same REGA_CUDA_CFLAGS, TORCH_CUDA_ARCH_LIST=9.0a
  around load_inline and restored. DISPATCH: _experts_rega (as merged with norm-gather) reads
  state['rega_low_variant'] (0 master, 1 V64, 2 V32) and state['rega_low_grid'] only when T <=
  REGA_LOW_MAX and state['rega_ok']['low']; it calls state['rega_lo_ext'].rega_gemm_lo(which,
  variant, ...) for w13 (on the xs that the fused norm-gather or master's gather produced) and w2,
  else master's rega_gemm. The align tables come from master's align2 path (bit-identical to
  _align_parallel's), so the LO decode_item / chunk-cut logic consumes them unchanged; assert that
  in the self-check by comparing _rega_lo_tiles on align2 vs _align_parallel tables. ROUTING: T in
  (200,1024] -> V64 or V32 if its gate passes, else master rega; every other T unchanged from A
  (norm-gather ON where A has it). GATE (the conjectural change): dfb755d3 kept master because it
  took the WORSE of two routings at <= 0.95 (V64: random 0.887, pop:2 0.994). Master's own
  low-class rega gate decides on the random routing ('random (soft)'). New rule, per candidate,
  after master's setup and after norm-gather's setup, on the norm-gather-ON expert block (so the
  gate sees the same prologue the judge runs): cold-L2 graph medians, >= 9 interleaved
  alternations, empty-graph baseline subtracted, at T=753 on (a) random routing: ON only at <=
  0.95 x master, AND (b) the pop routing calibrated to 1.8x random local routes: guard <= 1.00 x
  master (never slower under skew). Pick the lower random-routing median of passing candidates.
  PRINT-ONLY diagnostic: also time V64 and master on pop routings calibrated to 1.2x and 1.5x
  local routes and print the three ratios as '[rega-lo] SKEW CURVE 1.0x r0 1.2x r1 1.5x r2 1.8x
  r3', plus routes, touched experts and max chunks per expert at each. Print w13 / w2 kernel-alone
  us and TB/s per candidate, and '[rega-lo] PATH CHOSEN low: master|V64|V32' as the last prepare()
  line, beside norm-gather's and fa3c's PATH CHOSEN lines. PREPARE BUDGET: master ~124.8 s, fa3c
  1.1 s, norm-gather 0.7 s -> ~125.5 s; LO nvcc + load ~30-35 s (4 instantiations; scale-expand's
  4 took 39.0 s) -> ~161 s; attribute checks + self-checks ~1.5 s; pop-alpha calibration and gates
  (random, pop 1.8x, plus diag 1.2x/1.5x) ~1.5 s -> ~164 s. Own deadlines:
  REGA_LO_START_DEADLINE_S = 185 s (no LO build after this), REGA_LO_GATE_DEADLINE_S = 250 s (no
  candidate step, and no switch away from master, after this). REGA_FT_START_DEADLINE_S 150 s and
  REGA_COPY_DEADLINE_S 255 s do not move; the LO phase starts only after _rega_setup, _fa3c_setup
  and _norm_gather_setup have returned. A build failure, attribute mismatch, self-check miss or
  deadline leaves rega_low_variant 0 (master), and A's paths are unaffected. RULES: kernel_fn adds
  one int lookup and one grid int keyed by the T class; no syncs, prints, try/except or
  input-sized allocation. setmaxnreg.inc hangs on a short pool, so no LO candidate is launched
  before its attribute check passes. No module, static or device globals (run _rega_audit_cpp on
  the LO source); TMA maps encoded per call from that call's tensors; the judge's two state copies
  each get their own ints, no data_ptr equality. NVFP4 weights stay NVFP4 (register dequant only).
expected:
  - shapes: "*layer=sliding*load=turn_1*"
    direction: faster
    why: "if the judge's turn_1 routing is closer to random than to pop:2: low rega block 484 -> ~430 us (V64 0.887 gate), sliding turn_1 720 -> ~660-675 us on top of A's ~711 (-5 to -7%)"
  - shapes: "*layer=full*load=turn_1*"
    direction: faster
    why: "same expert block saving on a 1132 us row: ~-4%"
  - shapes: "*load=turn_4*"
    direction: unchanged
    why: "high class keeps master's kernels; same as A"
  - shapes: "*load=first_turn*"
    direction: unchanged
    why: "ft class keeps master's kernels; same as A"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-25T161800Z
review_companions: null
---

# combo-b-normgather-two-cta-low-regate

## Change

COMBINATION B (conjectural: re-gate the two-CTA/SM low class on master's own gate routing, stacked on A). Start from the same merge as combination A (/home/user/git/musespark/muse_prefill_layer/hypotheses/archive/pre-expert-norm-fused-rega-gather/master_plus_normgather.merged.py: master 1.4587 + pre-expert-norm-fused-rega-gather bd60e2db) and port the low-class two-CTAs-per-SM kernels from experts-rega-low-class-two-ctas-per-sm, branch hyp/muse_prefill_layer/experts-rega-low-class-two-ctas-per-sm, impl commit dfb755d3 (check 1a77da43b3304b07). PORT, from git show dfb755d3:muse_prefill_layer/submission.py: _REGA_LO_CPP, _REGA_LO_SRC (rega_lo_kernel<G, NMAX, STG, OCC>, rega_init_lo, rega_gemm_lo), REGA_LO_VARIANTS restricted to V64 = (64, 3, OCC 2) and V32 = (32, 4, OCC 2) (drop the V64-occ1 diag to save nvcc time), REGA_LO_REGS_OCC2 80, REGA_LO_POOL, REGA_LO_MAXBLK, REGA_LO_STATIC_BOUND, _rega_pop_case, _rega_pop_alpha, _rega_lo_tiles, _rega_lo_gate, the per-candidate attribute check (numRegs == 80, local 0, 2 CTAs/SM, binaryVersion 90, static <= 12288) and per-candidate self-checks (T=300/753 random/skewed/one, 753 pop, 1024 local-saturated; A/B/C copy-replay with refilled inputs). DO NOT port its module layout: dfb755d3 renamed master's extension to muse_prefill_rega_v3 (master v2 source + LO). Here master's muse_prefill_rega_v2 build, its v1 fallback and its sha checks stay byte-identical, and the LO source compiles as a SEPARATE extension muse_prefill_rega_lo containing only _REGA_HEAD + _rega_wgmma_src() + the namespace part of _REGA_BODY needed by the LO template + _REGA_LO_SRC (the pattern experts-rega-producer-scale-expand-all-t used for muse_prefill_rega_se: master templates never instantiated, only the 4 LO instantiations). Same REGA_CUDA_CFLAGS, TORCH_CUDA_ARCH_LIST=9.0a around load_inline and restored. DISPATCH: _experts_rega (as merged with norm-gather) reads state['rega_low_variant'] (0 master, 1 V64, 2 V32) and state['rega_low_grid'] only when T <= REGA_LOW_MAX and state['rega_ok']['low']; it calls state['rega_lo_ext'].rega_gemm_lo(which, variant, ...) for w13 (on the xs that the fused norm-gather or master's gather produced) and w2, else master's rega_gemm. The align tables come from master's align2 path (bit-identical to _align_parallel's), so the LO decode_item / chunk-cut logic consumes them unchanged; assert that in the self-check by comparing _rega_lo_tiles on align2 vs _align_parallel tables. ROUTING: T in (200,1024] -> V64 or V32 if its gate passes, else master rega; every other T unchanged from A (norm-gather ON where A has it). GATE (the conjectural change): dfb755d3 kept master because it took the WORSE of two routings at <= 0.95 (V64: random 0.887, pop:2 0.994). Master's own low-class rega gate decides on the random routing ('random (soft)'). New rule, per candidate, after master's setup and after norm-gather's setup, on the norm-gather-ON expert block (so the gate sees the same prologue the judge runs): cold-L2 graph medians, >= 9 interleaved alternations, empty-graph baseline subtracted, at T=753 on (a) random routing: ON only at <= 0.95 x master, AND (b) the pop routing calibrated to 1.8x random local routes: guard <= 1.00 x master (never slower under skew). Pick the lower random-routing median of passing candidates. PRINT-ONLY diagnostic: also time V64 and master on pop routings calibrated to 1.2x and 1.5x local routes and print the three ratios as '[rega-lo] SKEW CURVE 1.0x r0 1.2x r1 1.5x r2 1.8x r3', plus routes, touched experts and max chunks per expert at each. Print w13 / w2 kernel-alone us and TB/s per candidate, and '[rega-lo] PATH CHOSEN low: master|V64|V32' as the last prepare() line, beside norm-gather's and fa3c's PATH CHOSEN lines. PREPARE BUDGET: master ~124.8 s, fa3c 1.1 s, norm-gather 0.7 s -> ~125.5 s; LO nvcc + load ~30-35 s (4 instantiations; scale-expand's 4 took 39.0 s) -> ~161 s; attribute checks + self-checks ~1.5 s; pop-alpha calibration and gates (random, pop 1.8x, plus diag 1.2x/1.5x) ~1.5 s -> ~164 s. Own deadlines: REGA_LO_START_DEADLINE_S = 185 s (no LO build after this), REGA_LO_GATE_DEADLINE_S = 250 s (no candidate step, and no switch away from master, after this). REGA_FT_START_DEADLINE_S 150 s and REGA_COPY_DEADLINE_S 255 s do not move; the LO phase starts only after _rega_setup, _fa3c_setup and _norm_gather_setup have returned. A build failure, attribute mismatch, self-check miss or deadline leaves rega_low_variant 0 (master), and A's paths are unaffected. RULES: kernel_fn adds one int lookup and one grid int keyed by the T class; no syncs, prints, try/except or input-sized allocation. setmaxnreg.inc hangs on a short pool, so no LO candidate is launched before its attribute check passes. No module, static or device globals (run _rega_audit_cpp on the LO source); TMA maps encoded per call from that call's tensors; the judge's two state copies each get their own ints, no data_ptr equality. NVFP4 weights stay NVFP4 (register dequant only).

## Expected

- `*layer=sliding*load=turn_1*` faster: if the judge's turn_1 routing is closer to random than to pop:2: low rega block 484 -> ~430 us (V64 0.887 gate), sliding turn_1 720 -> ~660-675 us on top of A's ~711 (-5 to -7%)
- `*layer=full*load=turn_1*` faster: same expert block saving on a 1132 us row: ~-4%
- `*load=turn_4*` unchanged: high class keeps master's kernels; same as A
- `*load=first_turn*` unchanged: ft class keeps master's kernels; same as A

## Risk

## Why this pairing (and not the two suggested ones)
The prepare logs show that neither suggested pairing has a working half to combine. Scale-expand lost in every class in its own gate (w13 sx/master 1.135 at T=753, block 1.099). It also needs +8 KB of smem per CTA, so V64 w2 at 2 CTAs/SM no longer fits (2 x (95232 + 8192 + 12288 + 1024) B > 228 KB); only V32 would fit, and V32 measured 1.233 under skew. Ping-pong lost 1.18-1.47x in every class. Fusing it with scale-expand cannot close a 47% loss at low or a 24% loss at ft. Two-CTAs is the only one of the three whose kernels were faster than master on some routing: V64 was 0.887 on random, the routing master's own low gate uses, and 0.994 on pop:2. It was kept OFF only by its worse-of-two rule at <= 0.95.

## Conjecture
The interaction tested is gate-surrogate versus judge routing at turn_1, under the in-graph regime. In the current master profile, the low class's w13 runs 249 us at sliding turn_1, while prepare's cold-L2 gate runs 305 us on random routing (752 routes, all 32 experts touched) and 344 us on pop:2 (1291 routes). The judge's turn_1 routing is skewed by design (reference: 'a few experts take most tokens'; local_start is the busiest 32-expert range), so in-graph it probably touches fewer experts, each carrying more routes. V64 cuts an expert into a new 64-route chunk every 4 blocks. Sibling chunks of a tile are adjacent items on adjacent CTAs (decode_item: c = f + local % k), so a split expert's weights are re-read from L2 rather than from HBM, but each split still pays an extra dequant of the same weight tile. The conjecture is that TLP (the occ1 diag isolates it at ~9%) outweighs the split cost at the judge's actual routes-per-expert, because the judge's lower touched-expert count in-graph means fewer weight bytes per CTA, which leaves the latency-bound regime that two CTAs per SM hide. Confirm: the profile shows rega_lo_kernel<...,64,3,2> rows at sliding turn_1 with w13 + w2 below master's 249 + 123 us (and the full turn_1 counterparts below 226 + ~115), and both turn_1 rows beat A's paired run. Refute: those rows are at or above master's, which means the judge's routing sits at or beyond the 1.8x surrogate. The printed SKEW CURVE then gives the routes-per-expert crossover that any future low-class kernel must beat. This calibrates every later low-class gate, not just this one. A second, smaller interaction: the gate now times the norm-gather-ON block, so V64's w13 reads xs written by the fused kernel immediately before. If V64's whole-block ratio is worse than its w13 + w2 kernel-alone ratios suggest, the fused gather's tail and the 264-CTA ramp contend, and the KERNELS versus GATE lines will show it.

## Expected value and risk
Upside is about +1.2% headline over A, from 46+16 weight on the turn_1 rows. The pop guard at <= 1.00 bounds the downside to the skew beyond 1.8x. Risk is medium: the kernels already passed every self-check and replay in the judge (1a77da43), but the module split and the norm-gather xs are new. A hang is impossible before the attribute gate, and every failure falls back to master's low class with A's paths intact. B and A are best submitted together: A is B's paired control.

## Adversarial review

Reviewed 2026-10-25T161800Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed as a measurement, not a likely gain. The 'pop 1.8x' guard is the same alpha=2.0 routing (1291 routes, 1.72x) on which V64 measured 0.994, so B will probably ship V64 by a 0.6% margin that is inside gate noise. The judge's turn_1 routing, recovered from the TASK.md floor, sits at about 1.78-1.81x (1314-1359 routes), past the guard point, where the measured trend puts V64 at about parity (~1.00-1.01). Expect roughly a null on the headline and read the result from the profile kernel rows against A's. Promote B only if those rows win on both turn_1 shapes.

### Justification
MECHANISM. TLP from two 80-register CTAs per SM is real. In the 1a77da43 gate, V64 at OCC2 vs OCC1 on the same slab measured 0.887 vs 0.984 on random routing and 0.994 vs 1.082 on pop:2, so about 9% comes from occupancy. The advantage shrinks as routes per expert grow, because V64 cuts an expert into a new chunk every 64 routes and each extra chunk pays its own dequant: 0.887 at 752 routes, 0.994 at 1291 routes. V32 collapses to 1.233. The mechanism is plausible. What matters is where the judge sits on that curve.

WEAKEST LINK: the expected values assume the judge is 'closer to random than to pop:2'. TASK.md's floor refutes that. sliding turn_1's 257.0 us floor cannot be bytes-bound: solving speed_of_light_ns for bytes gives ROUTING_SHARE*router_distinct = 34 touched experts, more than the 32 that exist. It is therefore flop-bound, and solving the flop term gives 0.75*router_load = 1019, so router_load is about 1359 local routes. full turn_1's 496.7 us gives about 1314. Both are about 1.75-1.81x of uniform (752). The '-5 to -7% sliding turn_1' and '+1.2% headline' come from the random-routing 0.887 and do not apply. Extrapolating the measured 0.887 -> 0.994 line to 1340-1360 routes gives a V64 block ratio of about 1.003-1.008, so the realistic row effect is about -1% to +0.5% and the headline effect is within +/-0.2%.

The one fact that keeps the conjecture alive is in the profile. Master's in-judge low-class w13 is 249 us (sliding) / 259.5 us (full), faster than the gate's own 303 us on random and 344 us on pop:2, even though the judge carries about 1.8x the routes. So the judge's routing is concentrated differently from the Zipf(0.8)-over-32-experts surrogate: probably fewer touched experts and fewer weight bytes. V64's crossover depends on chunks per expert, not only on route count, so the surrogate cannot predict the judge's ratio in either direction. That is the conjecture, and only an in-judge V64 profile row can settle it.

GATE vs USER CONCERN. The guard is not looser in where it measures: _rega_pop_alpha's grid is (1.0, 1.5, 2.0, 2.5, 3.0, 4.0) with target 1.8, and it already picked alpha 2.0 = 1291 routes = 1.72x in 1a77da43. 'pop:2' and 'pop 1.8x' are the same routing unless the alpha grid is refined. The guard is looser in threshold: <= 1.00 instead of <= 0.95. Last measured margin: 0.994, against roughly 1% gate-to-gate drift (master random block 484.4 vs 486.3 us between two gates in the same log). Yes, it can ship a kernel that is about 0.5-1% slower on the judge's real routing: the judge sits slightly past the guard point (1.78-1.81x vs 1.72x), and the pass margin is noise-sized. The bounded downside is about 2.5 us on a ~480 us block, roughly 0.3% of the sliding turn_1 row. The board keeps the best run, so a slightly slower scored run costs only the attempt. The real hazard is promoting B on a noisy headline. Suggested tightening if the builder has time: refine REGA_LO_POP_ALPHAS around 2.0-2.4 so the calibrated count actually reaches about 1350 routes (1.8x, the floor-implied judge load), and pick the passing candidate by its pop-1.8x median, not its random median. Random is the routing the judge provably does not use, and V32's 0.882 random median would win that tiebreak if it ever passed the guard.

PREPARE BUDGET. Master+fa3c+norm-gather measured 124.8 s of rega phases in edf68fa3 (before-rega 76.9 s), plus about 1.8 s. But before-rega was 36.9 s in 1a77da43, a 40 s swing between runs. LO nvcc + load is 35-39 s (v3 build 34.8 s; scale-expand's 4 instantiations 39.0 s). The LO phase therefore starts at about 127 s, which leaves 58 s before REGA_LO_START_DEADLINE_S 185, and ends at about 165-170 s, well inside the 250/255 s gate/copy deadlines and the 300 s prepare cap. REGA_FT_START_DEADLINE_S 150 is untouched because ft runs before LO. The new deadlines (185/250) are tighter than dfb755d3's (200/255), which is fine. The budget is not a risk, but a slow before-rega pushing LO past 185 s would silently make B identical to A, so the deadline-skip print must be explicit.

### Confounders on the current kernel
- pre-expert-norm-fused-rega-gather (bd60e2db, the merge base shared with A, BEST.json slug): the fused norm-gather writes xs immediately before w13, which changes the L2 state and the launch tail V64's 264-CTA ramp starts into. The 0.887/0.994 ratios were measured on master's separate gather in 1a77da43, so they transfer only roughly. The plan to re-gate on the norm-gather-ON block is correct. A's paired run controls for it at the row level.
- align-two-phase-all-t-rega (landed after 1a77da43): kernel_fn now feeds LO align2 tables, while dfb755d3's self-checks and _rega_lo_tiles used _align_parallel. Per-expert counts are identical, but the intra-expert row order may differ. The self-check must run LO on align2-produced tables through the real _experts_rega path, not only compare tile counts. A tiles-equality miss turns LO OFF and makes B identical to A.
- experts-turn-rega-s4-port-t4096 / experts-turn-rega-s4-first-turn-n128: master's low-class rega_kernel<.,false> NMAX128/STG4/OCC1 is the incumbent LO is compared against. It is byte-identical in A and B, so it is the baseline and not a confounder. The separate muse_prefill_rega_lo module keeps v2's sha checks intact.
- fa3-full-contiguous-kv-triton-gather-gated: disjoint (full-layer attention), but it sits on the full turn_1 row (35% FA3), so that row's noise is dominated by attention. Judge LO's effect there from kernel rows only. A carries it too.

### Interference and A/B needs
No companion. A (master + norm-gather) is the paired control, and the readable signal is kernel-level: rega_lo_kernel<G,64,3,2> w13+w2 rows in B's profile against A's rega_kernel<true,false>+<false,false> rows (about 249+123 us sliding, 259.5+~115 us full). Kernel-row noise is smaller than row noise. The 1a77da43 run moved sliding turn_1 by +3.4% with LO OFF, so a row-level B-vs-A delta of the realistic size (<1%) is not interpretable, and the headline is not interpretable at all. With the gate as written V64 will probably be selected, since 0.994 passes <= 1.00, so the conjecture does get an in-judge measurement. A forced-V64 control is needed only if B's gate keeps master. That outcome is the one ambiguous branch, but it costs another scored submit. Mint it then from B's branch, with the gate print-only and rega_low_variant forced to 1, instead of pre-committing it now. Decision rule for harvest: promote B over A only if both turn_1 profiles show LO w13+w2 below A's master rows. Otherwise record the SKEW CURVE and the in-judge LO rows as the calibration result.

### Open questions
- Does the refined guard routing actually reach ~1.8x? With REGA_LO_POP_ALPHAS unchanged, 'pop 1.8x' is alpha 2.0 = 1.72x, identical to the routing that measured 0.994. Print the achieved routes/random ratio on the guard line.
- Floor inversion puts the judge at ~1314-1359 local routes on turn_1 (flop-bound floor). Should the guard target that count directly (≈1.79x) rather than 1.8x of a random draw whose count itself varies (752)?
- Why is master's in-judge w13 (249 us) faster than both surrogates (303 random, 344 pop:2) despite ~1.8x routes? Print touched experts and max routes/expert for pop curves; if the judge touches far fewer than 32 experts, a Zipf-over-32 surrogate is the wrong shape and a lower-distinct surrogate (e.g. hot-set of ~16-24 experts) would gate better.
- Prepare could gate on the run's actual router: weights (router + router_bias of the judge-chosen router_seed) are available in prepare(), so routing synthetic h through them would reproduce the judge's expert popularity far better than a Zipf surrogate. Is that within the house rules for synthetic prepare-time cases (it uses only weights, not data)?
- If B's gate keeps master (deadline skip, tiles miss, or guard > 1.00), is a forced-V64 scored control worth one submit to get the in-judge LO rows?

