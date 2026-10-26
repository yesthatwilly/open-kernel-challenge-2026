---
problem: muse_prefill_layer
slug: experts-rega-s4-bm256-wide-tile-high-ft
created: 2026-10-25T152138Z
base: master
area: "w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft classes): 256-weight-row CTA tile with in-warpgroup dequant/MMA overlap"
change: >
  Rewrite stage 4 of the rega line, built on master (1932a9c4; stage 2
  experts-turn-rega-s4-first-turn-n128 landed, stage 3 experts-rega-low-class-two-ctas-per-sm was
  neutral and touched only the low class). Target: the 'ft' (T=16384) and 'high' (T=3012) classes,
  where rega runs at 53-58% of the 989 TFLOP/s peak (sliding first_turn ~30.5K local routes: w13
  2.05 TFLOP in 3669 us, w2 1.02 TFLOP in 1783 us; floor at peak ~2070 / ~1035 us). MECHANISM:
  master's tile is 128 weight rows x <= 128 tokens, so each k-stage loads 9216 B of NVFP4 weight
  but 32768 B of bf16 activations per CTA; activations are re-read from L2 once per 128-row tile
  (64 times per route for w13, 32 for w2). At first_turn that is ~41 KB per stage per SM every
  ~1530 clk, ~6 TB/s aggregate L2->SM, near Hopper's L2 limit, and the 16 wgmmas per stage are
  issued only after a serialized dequant. NEW KERNEL rega_wide_kernel<G, BIG> (appended; master's
  strings untouched): a CTA tile of 256 weight rows = two ADJACENT master tiles j0 = 2J and j1 =
  2J + 1 of the existing retiled bank (no bank change: two bulk copies per stage, one from each
  tile's records), x the same <= 128-token chunk (same align, block tables, chunk tables,
  decode_item with n_tiles halved: w13 32 wide tiles of 128 silu columns, w2 16 wide tiles of 256
  columns), same 4 token buckets. Stage = 2 x 9216 B weight + 32768 B activation slab = 51200 B;
  STAGES = 3 (153600 B); epilogue staging 128 tokens x 272 B pitch = 34816 B (w13 stages its 128
  columns in one pass; w2 stages its 256 columns in two 128-column passes through the same buffer
  with named_bar(1, 256) between); dynamic smem 1024 + 153600 + 34816 = 189440 B. Each consumer
  warp cw keeps two register-A fragment sets aA (tile j0, its 16 rows as master) and aB (tile j1)
  and two accumulator sets accA / accB (NB/2 fp32 each). Per k-stage n: mbar_wait(full); dq(aA);
  fence; 8 wgmma(aA -> accA); commit; [n > first stage of the item: wgmma_wait<1> (B(n-1)
  retired), arrive(empty) for stage n-1]; dq(aB); fence; 8 wgmma(aB -> accB); commit;
  wgmma_wait<1> (A(n) retired, so aA may be rewritten next stage). After the item's last stage:
  wgmma_wait<0>, arrive(empty) for it, then the epilogue. So each register set is never rewritten
  while its own group is in flight (master's variant-A safety rule, per set), and the dequant of
  one row block overlaps the other block's wgmmas inside the warpgroup. Per output element the k16
  accumulation sequence is identical to master's, so outputs must be bit-identical. Effects:
  activation bytes per FLOP halved (25.6 KB per 1024 clk of MMA instead of 41 KB), half the items
  (w13 ~8.6K at first_turn), dequant hidden under MMA inside each warpgroup. Registers: accA +
  accB = 128 fp32 at NB = 128 plus aA + aB = 64 plus addressing, under setmaxnreg 232; print regs
  / spill; if the NB = 128 instantiation spills, ship the wide kernel for w2 only (G = false has
  no silu epilogue temporaries) and say so, or cap it as the next stage's fix. Also check ptxas
  output for 'wgmma serialization' warnings and print them: a serialized build is correct but
  loses the overlap. Dispatch: _experts_rega uses rega_gemm_wide for a class when
  state['rega_wide'][class] (prepare flag per 'high' / 'ft'; 'low' stays master's, it is turn_1's
  weight-streaming regime). BUILD: do NOT edit master's _REGA_HEAD / _REGA_BODY strings (the v1
  build-failure fallback derives its source from them by text substitution and checks
  REGA_V1_SHA256). Append the new code as a separate string after master's body (new __global__
  kernel(s) + a new host entry declared in _REGA_CPP), compile _rega_src() + that string under a
  bumped module name (muse_prefill_rega_v3), with master's exact flags and sm_90a guards; if that
  build or its import fails, build master's v2 source under its master name exactly as today and
  leave the new variant OFF. Print nvcc seconds and regs / local bytes / smem / CTAs per SM for
  every new instantiation (any spill turns that variant OFF). Prepare is ~124 s today (build 38.6
  s); keep the new phases inside master's REGA budget/deadline constants. PREPARE GATE (only for
  classes whose rega_ok is ON): (1) exactness on master's self-check routings at the high / ft
  sizes (3012 / 4096 random, skewed, one; 4500 local; 8192 random, local; 16384 random, skewed,
  one, local): max |wide - master| == 0 on hs and y (if nonzero, print the first differing bucket
  and column and turn OFF), error word 0, launch markers; (2) master's _rega_replay_check copy
  plan with wide on; (3) cold-L2 graph medians, REGA_ALTERNATIONS interleaved, empty graph
  subtracted, wide block vs master's rega block at 3012 and 16384 soft; ON for a class only if <=
  0.97x. Print per class the w13 and w2 kernels alone for both variants with TFLOP/s, then
  '[rega-wide] PATH CHOSEN <class>: wide|master' as the last lines of prepare(). Any failure
  prints 'OFF: <reason>' and master's rega runs. RULES: kernel_fn stays CUDA-graph capturable with
  no host sync, no allocation sized by input values, no data-dependent host control flow, no
  prints, no try/except, no timing and no per-call mutable Python state; it dispatches only on the
  token count and on flags prepare() set. Nothing derived from inputs is cached. NVFP4 expert
  weights stay NVFP4/E4M3 bytes in global memory and are dequantized in registers (any bf16 scale
  values live only in per-stage shared memory inside one launch); activations BF16, fp32
  accumulation; every route is computed. No child processes (in-process load_inline only). The
  judge prepares TWO state copies: no module, static or C++ global may hold device pointers, TMA
  maps or buffers; keep master's per-call tensor-map encoding (__grid_constant__ Maps) and
  per-state bank / error word. First measurement is an okc check, never a scored submit.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "incumbent rega w13 3669 us / w2 1783 us on sliding first_turn (558 / 574 TFLOP/s; floor ~2070 / ~1035 us at 30.5K routes). Halved activation traffic per FLOP plus in-warpgroup dequant/MMA overlap: stage-4 target w13 <= 2900 us, w2 <= 1450 us (~720 TFLOP/s); sliding first_turn 11120 -> ~10000 us at weight 138"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "high class, bucket 128 at ~173 routes per expert: w13 712 -> <= 590 us, w2 348 -> <= 290 us"
  - shapes: "*load=turn_1*"
    direction: unchanged
    why: "control: the low class (T <= 1024) keeps master's rega_kernel"
risk: high
depends_on: experts-turn-rega-s4-first-turn-n128
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: rega_kernel<true,*> + rega_kernel<false,*> (128-row tile) in the high (T=3012) and ft (T=16384) classes
review: proceed
status: done
review_at: 2026-10-25T212524Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T212551Z
base_sha: d375dd60d9615046903966bfa1143a876e566b8a
base_blob: c2e3e595def2e6b5949e0fd350ce98ed59729147
branch: hyp/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft
worktree: .worktrees/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-26T011825Z
impl_commit: 97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2
submission_sha256: 3916dfe084128c3d8d79d4115f45dd195a4d3c7161a4dca38e75297943092836
norm_sha256: 0c499f6341769f494a929dd9b5fdb8c2a8b71d1fea622a4e6f5687d48fc94a1b
submission_blob: 82449fd7e7fae52ba13d26109cc48a9fd256b40c
lint_warnings: 106
attempt_file: 2026-10-26T012540Z-check-experts-rega-s4-bm256-wide-tile-high-ft.md
prepared_at: 2026-10-26T012540Z
job: a6f6063051f14750
dispatched_at: 2026-10-26T012820Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.4757672292732262
speedup_ci: [1.453630720834128, 1.4979037377123245]
harvested_at: 2026-10-26T023858Z
best_at_harvest: 1.4677493909229
promotion: null
confirm_queued: experts-rega-s4-bm256-wide-tile-high-ft-cf
---

# experts-rega-s4-bm256-wide-tile-high-ft

## Change

Rewrite stage 4 of the rega line, built on master (1932a9c4; stage 2 experts-turn-rega-s4-first-turn-n128 landed, stage 3 experts-rega-low-class-two-ctas-per-sm was neutral and touched only the low class). Target: the 'ft' (T=16384) and 'high' (T=3012) classes, where rega runs at 53-58% of the 989 TFLOP/s peak (sliding first_turn ~30.5K local routes: w13 2.05 TFLOP in 3669 us, w2 1.02 TFLOP in 1783 us; floor at peak ~2070 / ~1035 us). MECHANISM: master's tile is 128 weight rows x <= 128 tokens, so each k-stage loads 9216 B of NVFP4 weight but 32768 B of bf16 activations per CTA; activations are re-read from L2 once per 128-row tile (64 times per route for w13, 32 for w2). At first_turn that is ~41 KB per stage per SM every ~1530 clk, ~6 TB/s aggregate L2->SM, near Hopper's L2 limit, and the 16 wgmmas per stage are issued only after a serialized dequant. NEW KERNEL rega_wide_kernel<G, BIG> (appended; master's strings untouched): a CTA tile of 256 weight rows = two ADJACENT master tiles j0 = 2J and j1 = 2J + 1 of the existing retiled bank (no bank change: two bulk copies per stage, one from each tile's records), x the same <= 128-token chunk (same align, block tables, chunk tables, decode_item with n_tiles halved: w13 32 wide tiles of 128 silu columns, w2 16 wide tiles of 256 columns), same 4 token buckets. Stage = 2 x 9216 B weight + 32768 B activation slab = 51200 B; STAGES = 3 (153600 B); epilogue staging 128 tokens x 272 B pitch = 34816 B (w13 stages its 128 columns in one pass; w2 stages its 256 columns in two 128-column passes through the same buffer with named_bar(1, 256) between); dynamic smem 1024 + 153600 + 34816 = 189440 B. Each consumer warp cw keeps two register-A fragment sets aA (tile j0, its 16 rows as master) and aB (tile j1) and two accumulator sets accA / accB (NB/2 fp32 each). Per k-stage n: mbar_wait(full); dq(aA); fence; 8 wgmma(aA -> accA); commit; [n > first stage of the item: wgmma_wait<1> (B(n-1) retired), arrive(empty) for stage n-1]; dq(aB); fence; 8 wgmma(aB -> accB); commit; wgmma_wait<1> (A(n) retired, so aA may be rewritten next stage). After the item's last stage: wgmma_wait<0>, arrive(empty) for it, then the epilogue. So each register set is never rewritten while its own group is in flight (master's variant-A safety rule, per set), and the dequant of one row block overlaps the other block's wgmmas inside the warpgroup. Per output element the k16 accumulation sequence is identical to master's, so outputs must be bit-identical. Effects: activation bytes per FLOP halved (25.6 KB per 1024 clk of MMA instead of 41 KB), half the items (w13 ~8.6K at first_turn), dequant hidden under MMA inside each warpgroup. Registers: accA + accB = 128 fp32 at NB = 128 plus aA + aB = 64 plus addressing, under setmaxnreg 232; print regs / spill; if the NB = 128 instantiation spills, ship the wide kernel for w2 only (G = false has no silu epilogue temporaries) and say so, or cap it as the next stage's fix. Also check ptxas output for 'wgmma serialization' warnings and print them: a serialized build is correct but loses the overlap. Dispatch: _experts_rega uses rega_gemm_wide for a class when state['rega_wide'][class] (prepare flag per 'high' / 'ft'; 'low' stays master's, it is turn_1's weight-streaming regime). BUILD: do NOT edit master's _REGA_HEAD / _REGA_BODY strings (the v1 build-failure fallback derives its source from them by text substitution and checks REGA_V1_SHA256). Append the new code as a separate string after master's body (new __global__ kernel(s) + a new host entry declared in _REGA_CPP), compile _rega_src() + that string under a bumped module name (muse_prefill_rega_v3), with master's exact flags and sm_90a guards; if that build or its import fails, build master's v2 source under its master name exactly as today and leave the new variant OFF. Print nvcc seconds and regs / local bytes / smem / CTAs per SM for every new instantiation (any spill turns that variant OFF). Prepare is ~124 s today (build 38.6 s); keep the new phases inside master's REGA budget/deadline constants. PREPARE GATE (only for classes whose rega_ok is ON): (1) exactness on master's self-check routings at the high / ft sizes (3012 / 4096 random, skewed, one; 4500 local; 8192 random, local; 16384 random, skewed, one, local): max |wide - master| == 0 on hs and y (if nonzero, print the first differing bucket and column and turn OFF), error word 0, launch markers; (2) master's _rega_replay_check copy plan with wide on; (3) cold-L2 graph medians, REGA_ALTERNATIONS interleaved, empty graph subtracted, wide block vs master's rega block at 3012 and 16384 soft; ON for a class only if <= 0.97x. Print per class the w13 and w2 kernels alone for both variants with TFLOP/s, then '[rega-wide] PATH CHOSEN <class>: wide|master' as the last lines of prepare(). Any failure prints 'OFF: <reason>' and master's rega runs. RULES: kernel_fn stays CUDA-graph capturable with no host sync, no allocation sized by input values, no data-dependent host control flow, no prints, no try/except, no timing and no per-call mutable Python state; it dispatches only on the token count and on flags prepare() set. Nothing derived from inputs is cached. NVFP4 expert weights stay NVFP4/E4M3 bytes in global memory and are dequantized in registers (any bf16 scale values live only in per-stage shared memory inside one launch); activations BF16, fp32 accumulation; every route is computed. No child processes (in-process load_inline only). The judge prepares TWO state copies: no module, static or C++ global may hold device pointers, TMA maps or buffers; keep master's per-call tensor-map encoding (__grid_constant__ Maps) and per-state bank / error word. First measurement is an okc check, never a scored submit.

## Expected

- `*load=first_turn*` faster: incumbent rega w13 3669 us / w2 1783 us on sliding first_turn (558 / 574 TFLOP/s; floor ~2070 / ~1035 us at 30.5K routes). Halved activation traffic per FLOP plus in-warpgroup dequant/MMA overlap: stage-4 target w13 <= 2900 us, w2 <= 1450 us (~720 TFLOP/s); sliding first_turn 11120 -> ~10000 us at weight 138
- `*load=turn_4*` faster: high class, bucket 128 at ~173 routes per expert: w13 712 -> <= 590 us, w2 348 -> <= 290 us
- `*load=turn_1*` unchanged: control: the low class (T <= 1024) keeps master's rega_kernel

## Risk

High risk: new tile shape, two accumulator sets near the 232-register ceiling, a 3-stage ring and a wait<1> schedule across two register sets that ptxas may serialize. Correctness is an exact compare with master (identical k16 order per output), so any nonzero diff is a bug, not rounding. Smem: 189440 B dynamic + <= 14.3 KB static (small class) fits 227 KB. Milestones for this stage: (1) builds with 0 spill and passes the exact compare; (2) w13 and w2 at or below master's per-kernel time at 16384 and 3012; (3) within ~1.4x of the 989-TFLOP/s floor at first_turn. If the per-kernel TFLOP/s rises only for w2, the w13 silu epilogue registers are the next fix. Stage 5 candidates: fuse rega_gather (184 us at first_turn) into the activation load (producer warps 1-3 cp.async rows of x by sorted_ids into the swizzled slab), and stack the ping-pong / scale-expand primaries if they land. Read with experts-rega-pingpong-consumers-all-t: PP flat + wide faster = L2 activation traffic was the limiter.

## Milestones

Rewrite stage 4 of `rega_kernel<true,*> + rega_kernel<false,*> (128-row tile) in the high (T=3012) and ft (T=16384) classes`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T212524Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed. This is a sound stage-4 rewrite aimed at the limiter DIAGNOSIS names: L2->SM activation ingest on rega's 128x128 tile. It is the reviewer-named successor to the flat pingpong run. One build-plan defect could make a failure unreadable, so it must be fixed before the build: folding the new kernel into master's rega module puts master's own ft and high rega at risk through build time and failure.

### Justification
Mechanism. Master's rega_kernel works like this (submission.py ~2812-2970): a 128-weight-row tile (8 consumer warps x 16 rows); per k-stage 9216 B of weight plus up to 32768 B of bf16 slab (STAGE_BYTES 41984); the dequant is serialised with the MMA (wgmma_wait<0> right after each commit, 'variant A'); the slab is re-read once per 128-row tile. Pairing two adjacent tiles over the same chunk halves the activation bytes issued per FLOP (about 25 B/clk of MMA instead of about 41) without touching the bank, align or chunk tables. The wait<1> schedule across two register-A sets is correct: each set is rewritten only after its own group has retired, and each stage is released after B(n-1) retires. The k16 order per output is unchanged, so bit-exactness is the right gate. The resources fit. Dynamic smem is 189440 B plus at most about 14.3 KB static, under 227 KB. setmaxnreg 40x128 + 232x256 = 64512 registers. Three stages of 2x MMA work buffer about 6K clk, against master's about 4K. The profile backs the targets. rega<true,true> is 3652-3848 us and rega<false,true> 1851-1890 us on the first_turn rows; at turn_4 rega w13/w2 are 741/357 us (sliding) and 763/370 us (full). That is about 1.75x the 989-TFLOP/s floor, so a <=2900/<=1450 us target is physically reachable. turn_1 stays on the low class, so 'unchanged' there is a valid control. Weakest link: the evidence that L2->SM ingest is a hard ceiling is inferential. Pingpong being slower at 4.76 TB/s fits either an ingest cap or a latency/issue cap. The 2-CTA V64 low class (LESSONS) reached about 6.4 TB/s and ran 12% faster, so 5.8-6.05 TB/s was not a hard wall; part of the cap may be per-CTA latency/occupancy, which a bigger tile with fewer resident warps does not fix. Second weakest: two 64-fp32 accumulator sets + 64 A registers + addressing + the silu epilogue, under 232 registers. The w13 NB=128 instantiation may spill or be wgmma-serialised by ptxas; the plan already prints both and turns the variant OFF. Prior check: the hypothesis has no 'prior' field. Its closest relatives are the pingpong (consumer overlap on the 128-row tile, Do-not-try 'until actions 1 or 2 remove the cap') and producer-scale-expand runs, both flat because the tile was ingest-capped. This change attacks the cap itself; DIAGNOSIS explicitly names it as action 2 and exempts it, so the difference is real and it is not a repeat. Stale facts in the brief: it says base 1932a9c4 and prepare ~124 s. Master is now 61ce79bb (combo B over combo A: fused norm-gather, the rega_lo module, FA3 contiguous gather), and prepare reaches 139.9 s before the rega-lo phase ends. _experts_rega now takes prenorm/variant arguments. The builder must rebase onto the current master.

### Confounders on the current kernel
- experts-turn-rega-s4-port-t4096 + experts-turn-rega-s4-first-turn-n128 (the incumbent rega module): the plan compiles _rega_src() + the wide string under a bumped name (muse_prefill_rega_v3). Master's high and ft rega then depend on the new code compiling and on the longer nvcc time. If it fails, the plan rebuilds v2 (about 38 s more). Either way prepare can pass REGA_FT_START_DEADLINE_S = 150 s (ft-self-check already ends at about 120 s, and combo B added time). That would turn master's ft rega OFF and send first_turn back to Triton (0.63x slower), a large regression that looks like the wide kernel failing. Fix in the change: build the wide kernel as a SEPARATE module (as rega_lo is) AFTER master's rega/ft setup, leaving muse_prefill_rega_v2 byte-identical. Wide must never be a prerequisite of master's path.
- combo-b-normgather-two-cta-low-regate: adds the rega_lo build and gate (33.2 s; prepare total 139.9 s) and does not interact with the high/ft kernels. It is the main source of prepare-budget pressure for the new build and gates under REGA_BUDGET_S 200 / REGA_COPY_DEADLINE_S 255. Print the prepare total before and after the wide phase.
- combo-a-master-normgather-naive-stack (pre-expert norm fused into rega gather): the wide kernel reads the same xs slab through the same TMA maps, so there is no numeric interaction. Its gate should time the norm-gather-ON block (as rega-lo did) or the GEMM kernels alone, so the ratio is not diluted by gather time.
- align-two-phase-all-t-rega: supplies the chunk/block tables the wide kernel reuses unchanged. Halving n_tiles halves the item count with longer items, so the persistent-grid tail at high-class w2 (about 1024 mixed bucket-128/64 items over 132 CTAs) roughly doubles in us. This is an interaction with the chunking, not with align itself, and the per-class gate absorbs it.

### Interference and A/B needs
No companion. This is kind: rewrite, and its result is read from per-kernel times and PATH CHOSEN lines, which the gate prints for both variants per class. A companion check would add nothing that prepare cannot print for free. The one attribution question is whether a gain comes from halved ingest or from the in-warpgroup dequant/MMA overlap. That is best answered inside prepare: if build time allows, time a diagnostic instantiation of the wide kernel with wait<0> after each group (no overlap; print-only, never ON) at ft for w2. The landed confounders are all handled by fixing the build coupling (separate module) rather than by an ab-revert. The queued experts-rega-mc2-slab-multicast-all-t attacks the same limiter by another route and should run in the same batch, per DIAGNOSIS: mc2 faster + wide faster confirms ingest; wide faster + pingpong flat confirms the overlap was ingest-masked.

### Open questions
- Will the build go into a separate module after master's ft setup (recommended) instead of the bumped muse_prefill_rega_v3 that replaces v2? If not, what is the measured ft start time with the larger TU, and the time on the build-failure-then-v2 path, against the 150 s REGA_FT_START_DEADLINE_S?
- Does w13 NB=128 fit under 232 registers with no spill and no 'wgmma serialization' warning? fence_acc must touch only the accumulator set NOT in flight, or ptxas will serialise. If w13 spills, does w2-only still clear 0.97 at ft?
- At high class (T=3012), does the doubled per-item cost with about 1024 mixed-bucket w2 items on 132 CTAs eat the gain through tail imbalance? Print items per CTA, min/max.
- Is about 6 TB/s really the L2->SM ceiling? Print the wide kernel's issued TB/s next to master's. If wide stays at about 6 TB/s while its time drops, ingest was the cap. If its TB/s drops and time is flat, the limiter is latency/issue and stage 5 should pivot (e.g. to the PDL chain or the 2-CTA/multicast idea).
- The gate routings at 3012/16384 are soft/random. Chunking is unchanged, so skew sensitivity should be small, but print one skewed-routing ratio as a guard, as combo B's lesson suggests.


## Implementation notes

Worktree `.worktrees/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft`, `muse_prefill_layer/submission.py` only. The claim was on a stale parent commit, so master's file was brought in first (`git checkout master -- muse_prefill_layer/submission.py`, staged) and every change is an unstaged pure insertion on top of it (954 lines added, 0 removed; `_REGA_HEAD`, `_REGA_BODY`, `_REGA_CPP`, `_REGA_LO_*`, the v1 fallback and both sha256 constants are byte-identical).

What changed:
- New separate module `muse_prefill_rega_wide` (`_REGA_WIDE_CPP`, `_REGA_WIDE_SRC`, `_rega_wide_src()` = master's head + wgmma wrappers + the namespace part of master's body + the new source, the same assembly rega_lo uses). It is built by `_rega_wide_phase`, called in prepare() AFTER `_rega_setup` (v2 build + low/high/ft gates), `_fa3c_setup`, `_norm_gather_setup` and `_rega_lo_phase`. Master's `muse_prefill_rega_v2` is neither rebuilt nor renamed, so the wide kernel is never a prerequisite of master's path (the review's main required fix). The phase checks that master's v2 source still hashes to `REGA_V2_SHA256` and the v1 fallback still derives.
- New kernel `rega_wide_kernel<G, BIG>` (distinct symbol), host entries `rega_init_wide` / `rega_gemm_wide` (distinct names; per-call tensor-map encoding into a stack `Maps`, no C++/module globals). Tile = 256 weight rows = master tiles 2J, 2J+1 of the unchanged bank x the same <= 128-token chunk (master's align, chunk and block tables, `decode_item` with n_tiles halved). Stage 51200 B (2 x 9216 weight + 32768 slab), 3 stages, epilogue staging 128 x 272 B, dynamic smem 189440 B (static_asserts), setmaxnreg 40/232. Consumer warp: two register-A sets (aA, aB), two accumulator sets; schedule exactly as the md: wait full; dq(aA); 8 wgmma -> accA; commit; [kb>0: wait<1>, release stage n-1]; dq(aB); 8 wgmma -> accB; commit; wait<1>; after the last stage wait<0> + release. `fence_acc` only touches the accumulator set that is not in flight. Per-output k16 order equals master's, so outputs must be bit-identical. w13 stages both 64-column halves into one 128-column row (single pass); w2 does two 128-column passes through the same buffer with `named_bar(1, 256)` between.
- Dispatch: `_experts_rega` reads `state['rega_wide'][class]`, a per-state int mask (bit 1 = w13, bit 2 = w2) set by prepare, only for T > REGA_LOW_MAX (low stays master's); kernel_fn adds no print/sync/try/allocation on values.
- Prepare phase (own start deadline 175 s, own step deadline 250 s, both print `skipped: deadline`): source audit; build with master's flags and `TORCH_CUDA_ARCH_LIST=9.0a` (verbose=True so ptxas 'wgmma serialization' warnings reach the log); attributes of all four instantiations before any launch (regs, local bytes = spill => OFF, smem, CTAs/SM, register pool); per class, w2-only candidate mask if the w13 instantiation spills (hypothesis fallback). Exactness (high: 3012 / 4096 random, skewed, one; ft: 4500 local, 8192 random + local, 16384 random, skewed, one, local): master vs wide on the same align/gather, hs and y bitwise equal, error word 0, launch markers, first differing bucket/column printed, each launch announced + synced. Gate (two interleaved rounds x 21 alternations, cold L2, empty graph subtracted): master's ACTUALLY ROUTED block (`_experts_rega`, wide off, fused norm-gather when `norm_gather_ok[class]`) vs the wide block, ON only if ratio <= 0.97 in BOTH rounds and wide IQR <= 1.5x master's in both; per round medians/IQRs, per-kernel us, TFLOP/s, issued TB/s (L2->SM bytes), items and items per CTA (min-max) for both variants, plus a print-only skewed-routing ratio at the same T. Then master's `_rega_replay_check` copy plan with wide on (per class, only for classes that passed). Last lines of prepare: compact `[rega-wide] GATE` / SUMMARY / `PATH CHOSEN high|ft: wide|master` lines, with prepare totals before and after the phase printed.

Review fixes applied: (1) separate module after master's rega/ft setup, master's module untouched (required fix); (2) fence_acc discipline + w13 NB=128 spill/serialization handling (mask fallback, regs/local bytes printed); (3) items per CTA printed; (4) issued TB/s printed next to master's; (5) one skewed-routing ratio printed; (6) rebased on current master (norm-gather block timed as routed; prepare totals printed; REGA_ALTERNATIONS replaced by the brief's two rounds). Where the stale hypothesis text conflicts (bumped `muse_prefill_rega_v3`, 1932a9c4, 124 s prepare) the review wins.

Deviations: (a) the optional print-only wait<0> no-overlap diagnostic instantiation was NOT built (it adds nvcc time inside a 300 s prepare; the per-kernel TB/s and TFLOP/s lines answer the ingest-vs-overlap question indirectly). (b) The spec said 'REGA_ALTERNATIONS interleaved'; the brief's two-round rule replaced it (21 per round). (c) ptxas warnings cannot be re-printed from Python without redirecting fds (rules risk), so they appear in the build output of the log only (verbose=True); regs/spills are read back from cudaFuncGetAttributes. (d) Nothing could be compiled or run on this node (no nvcc/torch/GPU): brace/paren balance of the CUDA source and py_compile were checked only.

Biggest risk: w13 NB=128 register pressure (128 fp32 accumulators + 64 A registers + silu epilogue under 232); a spill turns w13 off for the class (w2-only), a serialized build is correct but loses the overlap and will fail the 0.97 gate.

## Rules check

- [x] CUDA-graph capturable: `_experts_rega` adds only a dict lookup of a prepare-set int by the token count and one launch; no `.item()`/`.cpu()`/`synchronize()`/prints/allocation on values in kernel_fn (all syncs/prints are in prepare-only `_rega_wide_*`).
- [x] Nothing derived from `data` is cached: state gets only `rega_wide` (ints) and `rega_wide_ext` (module); tensor maps are encoded per call; no module/static/C++ global holds pointers (source audit `_rega_wide_audit` + `_rega_audit_cpp`).
- [x] Weights: unchanged bank from `_rega_bank`; NVFP4 codes and E4M3 scales dequantized in registers inside the kernel; nothing repacked or stored at another precision.
- [x] Activations BF16 (slab and outputs bf16), fp32 accumulation; KV cache and queries untouched.
- [x] Attention untouched; every route is computed by the wide kernel (same chunk/tile coverage as master, exact-compare gate).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses (in-process `load_inline` only, as master's builds) or network.
- [x] All work inside the timed call; no output caching or replay (the copy/replay check refills inputs per state).
- [x] Builds only inside prepare; build error, attribute miss, spill, exact-compare miss, deadline or gate miss leaves master's `rega_gemm` path (flag 0); wide has its own per-state flag and never disables an unrelated landed flag.
- [x] Bounded waits: `mbar_wait` (2^26-cycle bound, error word 1) on every mbarrier; no spin elsewhere; launch markers err[1]/err[2]; non-sm_90a image sets error word 2.
- [x] Prepare budget: wide start deadline 175 s, step deadline 250 s; prints `skipped: deadline`.

okc_lint: `OK with warnings` (0 errors; 101 host-sync warnings, all in prepare-only code outside kernel_fn's call graph, and 5 pre-existing dtype warnings). `python3 -I -m py_compile`: ok; `__pycache__` removed.

## Pre-flight audit

Verdict: dispatch-with-fixes (read-only audit plus numpy emulation; nothing could be compiled or run on this node).

- Smem: dynamic 1024 + 3 x 51200 + 128 x 272 = 189440 B (matches REGA_WIDE_SMEM and the static_asserts). Static, small class: 48 (bars) + 9216 (s_be) + 4096 (s_cblk, s_cnb) + 384 + 48 + 4 = 13796 B; first-turn class 9192 B. Total about 203.2 KB plus 1 KB driver reserve, against 232448 B (227 KB). Headroom is about 28 KB.
- Barriers: full = 1 arrival plus tx WST2 + 2 x rows x 128 (two 9216 B bulk copies plus two slab boxes), which matches the issued bytes. empty = 8 (one arrive per consumer warp). Release of stage n-1 follows wait<1> after A(n) is committed, and the last stage is released after wait<0>. The wait<1> group order was checked by hand, and the 3-stage ring cannot deadlock. fence_acc touches only the idle accumulator set (accA at A-block, accB at B-block), and both sets only after wait<0>.
- w13 / w2 registers: the live set during dq(aB) is accA 64 + accB 64 + aA 32 (in flight) + aB 32 + temps, which is very tight against 232. The main loop is identical for G = true and false, so a spill is likely to hit both kernels, not w13 alone. Any spill leaves local bytes > 0 and turns the variant OFF, so master keeps running. The w2-only mask path is correct: bit 1 / bit 2 are tested separately in _experts_rega, the exact check and the gate launch only the masked kernels, and the w13 call stays on master's rega_gemm.
- Emulation: /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/audits26/wide_tile_emu.py (output /tmp/claude-1000/-home-user-git-musespark/73199a37-afd5-408c-8bef-97b87f4411ce/scratchpad/audits26/wide_tile_emu.out). At T = 3012 and 16384 (random, skewed, one, local) plus 4096 / 4500 / 8192, it checks decode_item with n_tiles/2, tile pairing 2J/2J+1, the w13 staging column map, w2 column blocks, nvalid tails and sentinel rows, the block / chunk table capacities, and per-CTA item balance. Every (slot, master tile) is written exactly once, by the same expert as master's kernel. All checks pass.
- Deadlines: t_prepare0 is the first statement of prepare(), which is the clock the 300 s prepare limit uses. This module builds nothing at import time, so the sibling import-clock problem does not apply. Master prepare is about 125-140 s before the wide phase (recent logs: 138.9 s, 139.9 s). Start deadline 175 s and step deadline 250 s leave about 35 s of slack at the start. I estimate the phase at 90-110 s, ending near 230-250 s, with a worst-case overrun of one step past 250 s, so still under 300 s.
- Fix: the IQR veto had no absolute floor (wide IQR <= 1.5 x master's IQR). Added REGA_WIDE_IQR_FLOOR_US = 1.0, so the test is wide IQR <= 1.5 x max(master IQR, 1.0 us).
- okc_lint: OK with warnings (0 errors); py_compile ok; __pycache__ removed.

Residual risk: NB = 128 register pressure (spill or ptxas wgmma serialization turns the path off or fails the 0.97 gate), and the prepare total landing near 250 s.

## Result

- job `a6f6063051f14750` (check), status done, passed True, tests 5/5
- speedup 1.4758x, 95% CI [1.4536, 1.4979], score 262510.165773762 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.4677493909229, "ci": [1.4457331500590564, 1.4897656317867434], "job": "73b40cd13fdf4bb0"}; base result: muse_prefill_layer/hypotheses/done/gain/experts-turn-rega-s4-first-turn-n128.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-s4-bm256-wide-tile-high-ft.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-s4-bm256-wide-tile-high-ft.patch`; branch `hyp/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 777.02 | 769.33 | 1.0664 | gained |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1136.06 | 1145.38 | 1.0143 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 2251.37 | 2148.97 | 1.0493 | gained |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 3785.25 | 3725.86 | 1.0411 | gained |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 12148.27 | 10973.2 | 1.0621 | gained |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 11658.83 | 10828.31 | 1.0547 | gained |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 15589.38 | 14827.97 | 1.0644 | gained |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 19167.43 | 18238.85 | 1.043 | gained |

gained on sliding first_turn, turn_1, turn_4; full first_turn, turn_4. Expected hit: 3/3. Noise ±1.5%.

## Follow-ups

- rewrite stage 4 of `rega_kernel<true,*> + rega_kernel<false,*> (128-row tile) in the high (T=3012) and ft (T=16384) classes`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_prefill_layer "<what to fix or tune>" --kind rewrite --depends-on experts-rega-s4-bm256-wide-tile-high-ft --stage 5`; a slow or failed first cut is the expected start, not a do-not-retry.
- unproven-vs-best: faster than the best but inside the drift margin; churn queues one confirm rerun when the margin allows (CONFIRM_MIN), and pools the two runs.

