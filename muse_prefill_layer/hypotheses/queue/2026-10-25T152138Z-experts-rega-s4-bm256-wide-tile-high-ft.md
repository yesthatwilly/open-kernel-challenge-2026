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
review: required
status: queued
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
