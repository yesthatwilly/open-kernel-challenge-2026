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
status: running
review_at: 2026-10-25T212524Z
review_companions: null
stage: claimed
claimed_at: 2026-10-25T212551Z
base_sha: d375dd60d9615046903966bfa1143a876e566b8a
base_blob: c2e3e595def2e6b5949e0fd350ce98ed59729147
branch: hyp/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft
worktree: .worktrees/muse_prefill_layer/experts-rega-s4-bm256-wide-tile-high-ft
mode: check
restack_of: null
stale_base: true
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

