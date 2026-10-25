---
problem: muse_prefill_layer
slug: experts-rega-pingpong-consumers-all-t
created: 2026-10-25T152138Z
base: master
area: "w13 + w2 expert GEMMs (rega_kernel consumer schedule: two-warpgroup ping-pong), every rega class"
change: >
  Built on master (1932a9c4, align-two-phase-all-t-rega). Consumer schedule of rega_kernel only.
  ARITHMETIC FIRST (the judge's router seed fixes the busiest GPU's load; backing it out of
  TASK.md's floors with reference.speed_of_light_ns gives ~30.5K local routes at sliding
  first_turn, ~5.5K at turn_4, ~1.36K at turn_1): sliding first_turn w13 = 2 x 30.5K x 8192 x 4096
  = 2.05 TFLOP in 3669 us = 558 TFLOP/s (56% of 989), w2 1.02 TFLOP in 1783 us = 574 TFLOP/s
  (58%); turn_4 w13 0.37 TFLOP in 712 us = 523 TFLOP/s (53%). So at T >= 3012 rega is math-side
  but leaves ~45% of the tensor core idle. Per CTA at first_turn: ~270 chunks x 64 w13 tiles =
  ~17.3K items / 132 CTAs x 32 k-stages = ~4190 k-stages in 3669 us = ~1530 clk per k-stage,
  against 1024 clk of wgmma for the two warpgroups' 2 x 8 m64n128k16 (2 x 1.05 M FMA at ~2048
  FMA/clk/SM). MECHANISM: in master's consume_item both consumer warpgroups (WG1 = consumer warps
  cw 0-3, weight rows 0-63 of the tile; WG2 = cw 4-7, rows 64-127) run in lockstep on the same
  stage: mbar_wait(full) -> dq_stage (~180 ALU instr per warp per stage) -> 8 register-A wgmma ->
  commit -> wgmma_wait<0> -> arrive(empty). The tensor core idles during both dequants and the
  ALUs idle during the 16 wgmmas; the ~500 exposed clk per stage match a serialized dequant.
  CHANGE (port of muse_moe_w4a16's queued cuda-nvfp4-wgmma-rega-pingpong-consumers, adapted to
  prefill's rega): a new consume_item_pp that keeps master's variant A inside each warpgroup (no
  register-A fragment rewritten while a group is in flight) and offsets the two warpgroups by half
  a stage with two named barriers PP_A = 2 and PP_B = 3, each count 256 (bar 1 stays the epilogue
  barrier). Per k-stage, WG1: [bar.sync PP_A, skipped only on its first k-stage of the launch]
  mbar_wait(full); dq_stage; fence; 8 wgmma; commit; bar.arrive PP_B; wgmma_wait<0>; fence_acc;
  arrive(empty). WG2: bar.sync PP_B; mbar_wait(full); dq_stage; fence; 8 wgmma; commit; bar.arrive
  PP_A; wgmma_wait<0>; fence_acc; arrive(empty). So WG2 dequantizes stage s under WG1's wgmmas of
  s, and WG1 dequantizes s+1 under WG2's. After the item loop WG1 does one final bar.sync PP_A,
  ONLY if the CTA consumed at least one k-stage (a CTA with no items must do no PP op at all, or
  WG1 hangs). Count invariant: both warpgroups walk the same items and KS per item, so WG1's PP_A
  syncs (stages - 1 + 1 final) equal WG2's PP_A arrives and WG2's PP_B syncs equal WG1's PP_B
  arrives; WG2 can never arrive PP_A twice before WG1 syncs once (its next PP_B sync needs WG1's
  next arrive). The epilogue (two named_bar(1, 256), stmatrix staging, row stores) is unchanged
  and re-aligns the pair at each item boundary. The mbar timeout ('dead') path must still execute
  every PP bar op. Expose it as rega_kernel_pp<G, BIG> (4 instantiations: w13/w2 x
  small/first-turn class) behind a new host entry rega_gemm_pp with master's rega_gemm signature;
  _experts_rega calls rega_gemm_pp instead of rega_gemm for a class when state['rega_pp'][class]
  is True (a prepare flag per class 'low' / 'high' / 'ft', default False). Producer, ring (4
  stages), bank, align, gather, buckets, chunking, epilogues and the post-expert weighted sum are
  untouched. Outputs are bit-identical by construction (same per-warpgroup instruction sequence;
  only the cross-warpgroup issue order moves). BUILD: do NOT edit master's _REGA_HEAD / _REGA_BODY
  strings (the v1 build-failure fallback derives its source from them by text substitution and
  checks REGA_V1_SHA256). Append the new code as a separate string after master's body (new
  __global__ kernel(s) + a new host entry declared in _REGA_CPP), compile _rega_src() + that
  string under a bumped module name (muse_prefill_rega_v3), with master's exact flags and sm_90a
  guards; if that build or its import fails, build master's v2 source under its master name
  exactly as today and leave the new variant OFF. Print nvcc seconds and regs / local bytes / smem
  / CTAs per SM for every new instantiation (any spill turns that variant OFF). Prepare is ~124 s
  today (build 38.6 s); keep the new phases inside master's REGA budget/deadline constants.
  PREPARE GATE (after master's rega setup; only for classes whose rega_ok is already ON): (1)
  exactness: for each class run master's rega and the PP kernel eagerly on master's self-check
  routings at its sizes (753 and 1024 random/skewed/one; 3012 and 4096 random/skewed/one; 4500
  local, 8192 random/local, 16384 random/skewed/one/local) and require max |PP - master| == 0 on
  hs and y, error word 0 and both launch markers; (2) copy safety: master's _rega_replay_check
  plan (states A, deepcopy B, tensor clone C, refilled graph replays) with PP on; (3) speed:
  cold-L2 graph medians, REGA_ALTERNATIONS interleaved, empty graph subtracted, PP block vs
  master's rega block at the class gate T (753, 3012, 16384 soft as master's gates use); PP ON for
  a class only if its median <= 0.98x master's rega. Also print, per class, the w13 and w2 kernels
  timed alone (graph of one launch) for both variants with their TFLOP/s from the local route
  count, so the run attributes the change per kernel. Print one '[rega-pp] PATH CHOSEN <class>:
  pp|master' line per class as the last lines of prepare(). Any failure prints 'OFF: <reason>' and
  master's rega runs unchanged. RULES: kernel_fn stays CUDA-graph capturable with no host sync, no
  allocation sized by input values, no data-dependent host control flow, no prints, no try/except,
  no timing and no per-call mutable Python state; it dispatches only on the token count and on
  flags prepare() set. Nothing derived from inputs is cached. NVFP4 expert weights stay NVFP4/E4M3
  bytes in global memory and are dequantized in registers (any bf16 scale values live only in
  per-stage shared memory inside one launch); activations BF16, fp32 accumulation; every route is
  computed. No child processes (in-process load_inline only). The judge prepares TWO state copies:
  no module, static or C++ global may hold device pointers, TMA maps or buffers; keep master's
  per-call tensor-map encoding (__grid_constant__ Maps) and per-state bank / error word.
expected:
  - shapes: "*load=first_turn*"
    direction: faster
    why: "rega w13 3669 us (558 TFLOP/s, 56% of peak, floor ~2070 us at 30.5K routes) and w2 1783 us (58%): if ~500 of the ~1530 clk per k-stage are the serialized dequant, hiding it under the other warpgroup's wgmmas targets w13 <= 3100 us and w2 <= 1500 us; sliding first_turn 11120 -> ~10300 us (weight 138)"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "same class of tile (bucket 128, ~173 routes per expert): w13 712 -> <= 620 us, w2 348 -> <= 300 us"
  - shapes: "*load=turn_1*"
    direction: faster
    why: "small: buckets 32/64, the wgmma chain per stage is short, so little to hide; w13 283 / w2 144 us move <= 5%. Flat here is not a failure"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# experts-rega-pingpong-consumers-all-t

## Change

Built on master (1932a9c4, align-two-phase-all-t-rega). Consumer schedule of rega_kernel only. ARITHMETIC FIRST (the judge's router seed fixes the busiest GPU's load; backing it out of TASK.md's floors with reference.speed_of_light_ns gives ~30.5K local routes at sliding first_turn, ~5.5K at turn_4, ~1.36K at turn_1): sliding first_turn w13 = 2 x 30.5K x 8192 x 4096 = 2.05 TFLOP in 3669 us = 558 TFLOP/s (56% of 989), w2 1.02 TFLOP in 1783 us = 574 TFLOP/s (58%); turn_4 w13 0.37 TFLOP in 712 us = 523 TFLOP/s (53%). So at T >= 3012 rega is math-side but leaves ~45% of the tensor core idle. Per CTA at first_turn: ~270 chunks x 64 w13 tiles = ~17.3K items / 132 CTAs x 32 k-stages = ~4190 k-stages in 3669 us = ~1530 clk per k-stage, against 1024 clk of wgmma for the two warpgroups' 2 x 8 m64n128k16 (2 x 1.05 M FMA at ~2048 FMA/clk/SM). MECHANISM: in master's consume_item both consumer warpgroups (WG1 = consumer warps cw 0-3, weight rows 0-63 of the tile; WG2 = cw 4-7, rows 64-127) run in lockstep on the same stage: mbar_wait(full) -> dq_stage (~180 ALU instr per warp per stage) -> 8 register-A wgmma -> commit -> wgmma_wait<0> -> arrive(empty). The tensor core idles during both dequants and the ALUs idle during the 16 wgmmas; the ~500 exposed clk per stage match a serialized dequant. CHANGE (port of muse_moe_w4a16's queued cuda-nvfp4-wgmma-rega-pingpong-consumers, adapted to prefill's rega): a new consume_item_pp that keeps master's variant A inside each warpgroup (no register-A fragment rewritten while a group is in flight) and offsets the two warpgroups by half a stage with two named barriers PP_A = 2 and PP_B = 3, each count 256 (bar 1 stays the epilogue barrier). Per k-stage, WG1: [bar.sync PP_A, skipped only on its first k-stage of the launch] mbar_wait(full); dq_stage; fence; 8 wgmma; commit; bar.arrive PP_B; wgmma_wait<0>; fence_acc; arrive(empty). WG2: bar.sync PP_B; mbar_wait(full); dq_stage; fence; 8 wgmma; commit; bar.arrive PP_A; wgmma_wait<0>; fence_acc; arrive(empty). So WG2 dequantizes stage s under WG1's wgmmas of s, and WG1 dequantizes s+1 under WG2's. After the item loop WG1 does one final bar.sync PP_A, ONLY if the CTA consumed at least one k-stage (a CTA with no items must do no PP op at all, or WG1 hangs). Count invariant: both warpgroups walk the same items and KS per item, so WG1's PP_A syncs (stages - 1 + 1 final) equal WG2's PP_A arrives and WG2's PP_B syncs equal WG1's PP_B arrives; WG2 can never arrive PP_A twice before WG1 syncs once (its next PP_B sync needs WG1's next arrive). The epilogue (two named_bar(1, 256), stmatrix staging, row stores) is unchanged and re-aligns the pair at each item boundary. The mbar timeout ('dead') path must still execute every PP bar op. Expose it as rega_kernel_pp<G, BIG> (4 instantiations: w13/w2 x small/first-turn class) behind a new host entry rega_gemm_pp with master's rega_gemm signature; _experts_rega calls rega_gemm_pp instead of rega_gemm for a class when state['rega_pp'][class] is True (a prepare flag per class 'low' / 'high' / 'ft', default False). Producer, ring (4 stages), bank, align, gather, buckets, chunking, epilogues and the post-expert weighted sum are untouched. Outputs are bit-identical by construction (same per-warpgroup instruction sequence; only the cross-warpgroup issue order moves). BUILD: do NOT edit master's _REGA_HEAD / _REGA_BODY strings (the v1 build-failure fallback derives its source from them by text substitution and checks REGA_V1_SHA256). Append the new code as a separate string after master's body (new __global__ kernel(s) + a new host entry declared in _REGA_CPP), compile _rega_src() + that string under a bumped module name (muse_prefill_rega_v3), with master's exact flags and sm_90a guards; if that build or its import fails, build master's v2 source under its master name exactly as today and leave the new variant OFF. Print nvcc seconds and regs / local bytes / smem / CTAs per SM for every new instantiation (any spill turns that variant OFF). Prepare is ~124 s today (build 38.6 s); keep the new phases inside master's REGA budget/deadline constants. PREPARE GATE (after master's rega setup; only for classes whose rega_ok is already ON): (1) exactness: for each class run master's rega and the PP kernel eagerly on master's self-check routings at its sizes (753 and 1024 random/skewed/one; 3012 and 4096 random/skewed/one; 4500 local, 8192 random/local, 16384 random/skewed/one/local) and require max |PP - master| == 0 on hs and y, error word 0 and both launch markers; (2) copy safety: master's _rega_replay_check plan (states A, deepcopy B, tensor clone C, refilled graph replays) with PP on; (3) speed: cold-L2 graph medians, REGA_ALTERNATIONS interleaved, empty graph subtracted, PP block vs master's rega block at the class gate T (753, 3012, 16384 soft as master's gates use); PP ON for a class only if its median <= 0.98x master's rega. Also print, per class, the w13 and w2 kernels timed alone (graph of one launch) for both variants with their TFLOP/s from the local route count, so the run attributes the change per kernel. Print one '[rega-pp] PATH CHOSEN <class>: pp|master' line per class as the last lines of prepare(). Any failure prints 'OFF: <reason>' and master's rega runs unchanged. RULES: kernel_fn stays CUDA-graph capturable with no host sync, no allocation sized by input values, no data-dependent host control flow, no prints, no try/except, no timing and no per-call mutable Python state; it dispatches only on the token count and on flags prepare() set. Nothing derived from inputs is cached. NVFP4 expert weights stay NVFP4/E4M3 bytes in global memory and are dequantized in registers (any bf16 scale values live only in per-stage shared memory inside one launch); activations BF16, fp32 accumulation; every route is computed. No child processes (in-process load_inline only). The judge prepares TWO state copies: no module, static or C++ global may hold device pointers, TMA maps or buffers; keep master's per-call tensor-map encoding (__grid_constant__ Maps) and per-state bank / error word.

## Expected

- `*load=first_turn*` faster: rega w13 3669 us (558 TFLOP/s, 56% of peak, floor ~2070 us at 30.5K routes) and w2 1783 us (58%): if ~500 of the ~1530 clk per k-stage are the serialized dequant, hiding it under the other warpgroup's wgmmas targets w13 <= 3100 us and w2 <= 1500 us; sliding first_turn 11120 -> ~10300 us (weight 138)
- `*load=turn_4*` faster: same class of tile (bucket 128, ~173 routes per expert): w13 712 -> <= 620 us, w2 348 -> <= 300 us
- `*load=turn_1*` faster: small: buckets 32/64, the wgmma chain per stage is short, so little to hide; w13 283 / w2 144 us move <= 5%. Flat here is not a failure

## Risk

Medium risk: named barriers have no timeout, so a count mismatch hangs the launch; the change text gives the invariants and the no-item guard (persistent grid = 132 CTAs, and the low class can leave CTAs with no item). A reviewer should hand-simulate KS = 32 (both w13 and w2 have K = 4096 -> 32 stages of 128) with 0, 1 and 2 items per CTA. Exactness is the cheap proof: max |PP - master| must be 0. Attribution: if PP is flat at first_turn while the per-kernel TFLOP/s stays ~560, the limiter is not the dequant/MMA serialization but L2->SM activation traffic (41 KB per stage per SM, ~6 TB/s aggregate at first_turn), which is what the rewrite experts-rega-s4-bm256-wide-tile-high-ft attacks. Independent of the sibling experts-rega-producer-scale-expand-all-t (fewer dequant instructions); stack only after both are read. Judge on the rega_kernel<true,true>/<false,true> rows of the first_turn profiles.
