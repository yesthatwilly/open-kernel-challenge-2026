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
review: proceed
status: running
review_at: 2026-10-25T152526Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T152526Z
base_sha: f4ace8661739740d326729699aeb3d2f785ae596
base_blob: a2fe2a75893c60d4f68cf94fcce8ce8f9b708a71
branch: hyp/muse_prefill_layer/experts-rega-pingpong-consumers-all-t
worktree: .worktrees/muse_prefill_layer/experts-rega-pingpong-consumers-all-t
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T154139Z
impl_commit: ffe531f6cdb87040edc100c7d0eab62f28bbdf33
submission_sha256: 1133f5f6fc4f9ab1c355909cee138d98b39e7f6294f22a9ef56accf377d8cd67
norm_sha256: fb27e2247d1d41bfe5e2d6d91ba782778f8daf7ac947a3d54ec11eb6c77b05b7
submission_blob: d3e7b75d044c8d3262672f0fb473e87a4f68f7af
lint_warnings: 83
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

## Adversarial review

Reviewed 2026-10-25T152526Z by an adversarial subagent (model opus).

**Verdict: proceed.** Mechanism is real in master's code: consume_item runs both consumer warpgroups in lockstep with variant A's wgmma_wait<0>, so the dequant and the 16 wgmmas serialize. The in-prepare exact-compare and interleaved A/B against master's own rega make a single scored run interpretable without a companion. The biggest threat to the first_turn target is an L2-to-SM ingest ceiling, not correctness. Keep it separate from producer-scale-expand; build a stack only later, as a third gated kernel.

### Justification
A. The mechanism checks out against submission.py. consume_item (~line 2693) does mbar_wait(full) -> dq_stage -> 8 register-A wgmma -> commit -> wgmma_wait<0> -> arrive(empty). Both warpgroups (cw 0-3 = rows 0-63, cw 4-7 = rows 64-127) walk the same n with no offset, so per SM the tensor core idles during both groups' dequant, and the ALUs idle while 2 x 8 m64nNBk16 issue back to back. At NB=128, 2 x 8 x 131K FMA / ~2048 FMA/clk/SM = 1024 clk per stage of pure MMA. The profile's first_turn rega rows (w13 3556-3851 us, w2 1848-1925 us across the four first_turn shapes in FINDINGS) give the stated ~1530 clk/stage at ~30.5K local routes. That leaves ~500 clk that matches one serialized dequant of 2 warps per SMSP x ~180 instructions plus smem latency. Moving the two warpgroups half a stage apart with named barriers 2/3 is the standard CUTLASS ping-pong idea, and it keeps variant A inside each warpgroup, so there is no new register-A hazard. Outputs are bit-identical by construction: the per-warpgroup accumulation order is unchanged, so an exactness gate of max diff == 0 is a sound, cheap proof. I hand-simulated the barrier counts with KS=32 for 0, 1 and 2 items per CTA. With 0 items there are no PP ops. With 1 item WG1 does 31 in-loop PP_A syncs plus 1 final = 32, against WG2's 32 PP_A arrives, and 32 PP_B arrives against 32 syncs. With 2 items it is 64/64. WG2 cannot arrive PP_A twice in one phase, because its next PP_B sync needs WG1's next arrive, which follows WG1's PP_A sync. The epilogue's named_bar(1,256) cannot deadlock against PP: WG2 reaches bar 1 after its last PP_B sync, which WG1 has already satisfied. The expected directions fit the shape weights: the sliding first_turn shape (weight 138) carries most of the gain, turn_4 at bucket 128/64 gets a smaller share, and turn_1 at buckets 32/64 (~42 routes/expert, mostly NB=64, 256 clk of MMA per stage for both groups) has little MMA to hide behind, so flat there is correctly allowed. WEAKEST LINK: the claim that the ~500 exposed clk/stage are dequant serialization rather than L2->SM ingest. Each stage moves 41,984 B per CTA (9216 weight + 2 x 16 KB activation slab at NB=128). 132 CTAs at ~1530 clk/stage is already ~6.3-6.5 TB/s of L2->SM traffic at ~1.75-1.85 GHz. The 1024-clk ideal needs ~9.5 TB/s, and the w13 <= 3100 us target needs ~7.6 TB/s. Both are at or above what H100 L2 usually delivers to SMs without cluster multicast. So the first_turn gain may be capped at ~1300 clk/stage (w13 ~3200-3300 us) or come out flat. The hypothesis names this in its Risk section, but its per-kernel print gives TFLOP/s only. It should also print achieved L2->SM bytes/s (KS x stage bytes x items / kernel time), so that a flat first_turn reads as 'ingest-capped' (which hands the problem to bm256-wide-tile) rather than as 'mechanism wrong'. Secondary weak point: the arithmetic rests on route counts backed out from floors, not measured. The prepare gate's own per-kernel timings at the gate T fix that, as long as the route count used for TFLOP/s is computed from n_padded/sorted_ids in prepare.

### Confounders on the current kernel
- experts-turn-rega-s4-port-t4096: introduced rega_kernel with variant A (wgmma_wait<0> right after commit). PP's entire headroom is the serialization variant A creates. A gain measures 'PP vs variant-A lockstep', not ping-pong vs a well-pipelined baseline. Because the A/B is in-prepare against this exact kernel, this does not confound the measurement, only how far it generalizes.
- experts-turn-rega-s4-first-turn-n128: the BIG (ft) class reads the block-expert table via __ldg from global and owns the first_turn rows where most of the expected gain lies. PP is tested on exactly this kernel at the T=16384 soft gate, so there is no confound, but the ft gate distribution (soft) must match what master's ft gate uses, as the hypothesis states.
- align-two-phase-all-t-rega: sets chunking, item count and CTA load balance (including CTAs with zero items in the low class, which the no-PP-op guard must handle). PP changes per-stage time roughly uniformly, so it interacts with tail balance only marginally, and the in-prepare A/B runs on the same align. Not a confounder of the measurement; it is the reason the 0-item guard is mandatory.
- stack-tonight-five-prefill / sglang-pipeline-parity / hybrid / Triton-line landings: they do not touch rega_kernel's consumer schedule or its memory traffic. No confound.
- (not landed, but relevant) experts-rega-low-class-two-ctas-per-sm: also hides consumer dequant latency, at turn_1. If it is ever landed, it and PP would substitute for each other in the low class. PP's low-class result is valid only for the 1-CTA/SM master.

### Interference and A/B needs
No companion is warranted. The change carries its own control: prepare() runs master's rega and the PP kernel on identical routings with an exact compare (max diff 0), then cold-L2 graph medians over REGA_ALTERNATIONS interleaved alternations at each class's gate T, and turns PP ON per class only at <= 0.98x. It prints per-class PATH CHOSEN lines and per-kernel w13/w2 timings for both variants. The scored profile's rega_kernel<*,true>/<*,false> rows then show what the judge saw. That makes every outcome readable from this one run. Headline gain with PP ON means attributed to PP. Flat with PP OFF means the gate refused it, and the per-kernel prints say by how much. Flat with PP ON and ~560 TFLOP/s per kernel means the ingest-capped reading, provided the bytes/s print is added. A loss with PP ON would be a gate-vs-judge discrepancy that the profile rows show directly. An ab-revert of any landed hypothesis would not disambiguate anything the in-prepare A/B leaves open, and a control run isolating a sub-change is meaningless because PP is a single change. Spending a second scored run is not justified. On the sibling experts-rega-producer-scale-expand-all-t: KEEP SEPARATE for now. (1) Both specify the same module name muse_prefill_rega_v3 and the same 'append after _REGA_BODY' slot, and both rewrite the consumer stage loop, so a naive stack is a merge conflict, not a sum. (2) The mechanisms overlap at first_turn: SX shortens the very dequant PP tries to hide. If PP hides it fully, SX adds ~0 there, and if the kernel is ingest-capped, neither moves, so a stacked headline could not say which one mattered. (3) They are complementary mostly at turn_1, where SX targets issue count and PP has little MMA to hide behind. (4) A stack is a third kernel (PP barriers plus the sfull wait in one consumer). It needs its own exactness gate, and it adds a further ~4 instantiations of nvcc time to a prepare that already holds master's ft-start (150 s) and copy (255 s) deadlines. Read both results, then mint a depends_on-both combo kernel gated against the better single variant, and only for the classes where each showed a per-kernel win. If scored submits are so scarce that only one run is possible, the fallback is a single build carrying both variants as separate per-class options (master | pp | sx, best chosen per class, no combined kernel). That stays attributable from the prepare prints, but it costs ~8 extra instantiations of compile time.

### Open questions
- What L2->SM bytes/s does master's ft rega already reach (stage bytes x k-stages / kernel time)? If it is already ~6.5 TB/s, how much ping-pong headroom exists before the ingest ceiling? Add this print next to TFLOP/s in the per-kernel lines.
- Is the ~1530 clk/stage figure net of the per-item epilogue (~131 items per CTA at first_turn, ~28 us each)? The epilogue's named_bar(1,256) re-aligns the pair every item, adding a half-stage drain per item (~1.5% at KS=32). Is that accounted for in the targets?
- At turn_4 (~173 routes/expert, a mix of NB=128 and NB=64 chunks), does PP win on the NB=64 items, where each group's MMA (~256 clk) is about as long as its dequant? The per-class gate decides, but a per-bucket split in the print would say whether a bucket-gated PP is the follow-up.
- Is the route count used for the TFLOP/s print measured in prepare (from n_padded / sorted_ids at the gate routing) or backed out from TASK.md floors? Only the measured number makes the per-kernel efficiency comparable across variants and shapes.


## Implementation notes

Built on base f4ace866 (master 47eaa27e has the same submission.py blob a2fe2a75, so no merge needed). Only `muse_prefill_layer/submission.py` in the worktree changed (+629/-12).

**What changed**
- New section before `kernel_fn` with constants `REGA_PP_*`, `_REGA_PP_CPP`, `_REGA_PP_PRE`, `_REGA_PP_INIT`, `_REGA_PP_SUBS`, plus `_rega_class`, `_rega_pp_src`, `_rega_pp_diff/_compare/_replay_check/_medians/_kernel_times/_gate` and `_rega_pp_setup`.
- CUDA: a separate extension **muse_prefill_rega_pp**. It holds only `rega_kernel_pp<G, BIG>` (4 instantiations), `rega_pp_init` and `rega_gemm_pp`, which has master's `rega_gemm` signature. Its source is **derived** from master's `_REGA_HEAD` / `_REGA_BODY` by slicing at 7 anchors and 14 exact substitutions, each required to match exactly once. Master's strings are not edited. A CPU harness confirmed that the v1 fallback still hashes to `REGA_V1_SHA256`, that the derived source has no master kernel names left, and that `_rega_audit_cpp` finds 0 globals in it. The gather stays master's `rega_ext.rega_gather`.
  - `consume_item_pp`: in each k-stage, WG2 (`cw >= 4`) runs `pp_sync(3)`. WG1 runs `if (started) pp_sync(2); started = true;`. Then come master's mbar_wait / dq / 8 wgmma / commit, then `pp_arrive(wg2 ? 2 : 3)`, then master's `wgmma_wait<0>` / fence_acc / `if (lane == 0) mbar_arrive(empty)`. The epilogue is master's text, unchanged.
  - `rega_kernel_pp`: master's kernel text with these changes: launch marker 0x50504d4d (instead of master's 0x57474d4d), `const bool wg2 = cw >= 4; bool started = false;`, the `consume_item_pp` call, and after the item loop `if (!wg2 && started) pp_sync(PP_BAR_A);`.
  - `pp_sync` / `pp_arrive` call `__syncwarp()` and then `bar.sync` / `bar.arrive id, 256`.
- `_experts_rega(..., pp=None, keep=None)`. When `pp` is None (the timed call) it uses `state.get("rega_pp", {}).get(_rega_class(T), False)`, which is a shape class plus a prepare-time flag. When pp is set it calls `state["rega_pp_ext"].rega_gemm_pp`. `keep` (prepare only) receives hs, y and the align tables.
- `_rega_build` gained optional `cpp=` / `functions=` arguments. Their defaults are master's, so master's call is unchanged.
- `prepare()` calls `_rega_pp_setup` **after** `_rega_setup` returns, then prints the align2 lines, then the PP lines (gate lines, `OFF <cls>: reason`, and one `[rega-pp] PATH CHOSEN <cls>` line per class) as the last lines.

**Binding review / caller items**
1. Build order: master's whole setup is untouched and runs in its original order, including the ft start deadline (150 s) and copy deadline (255 s). The PP build starts only afterwards, under its own deadlines: no build after `REGA_PP_START_DEADLINE_S` = 180 s, and no new class step after `REGA_PP_STEP_DEADLINE_S` = 240 s. Classes are processed ft, high, low, and only those whose master `rega_ok` is ON. A build failure, spill, attribute or register-pool failure, canary miss, exactness miss, replay miss, gate miss or exception leaves that class on master's rega. PP is also skipped when master is running its v1 fallback (`len(rega_init()) != 40`).
2. `_REGA_HEAD` / `_REGA_BODY` are not edited (SHA check reproduced on CPU).
3. Hang safety: every item in the review checklist is applied. `started` is a per-thread register. The final PP_A sync uses the same predicate in all 128 WG1 threads. PP ops are unconditional with respect to `dead`. Bars 2/3 are used nowhere else (0 is `__syncthreads`, 1 is the epilogue). `__syncwarp` precedes each bar op, and lane 0's mbar_arrive sits outside them. A canary runs before any other PP launch: T=210 `one` (2 chunks; w2 has 64 items on 132 CTAs, so CTAs get 0 or 1 item; w13 has 128 items) and then T=300 random (7-16 items per CTA). Every launch has a flushed print before it and a sync plus print after it.
4. The per-kernel line `[rega-pp] KERNELS T=...` shows w13 and w2 timed alone (one-launch graphs, cold L2, interleaved) for master and pp. Each shows us, TFLOP/s and **TB/s L2->SM**. The traffic counts 9216 B of weight plus 2 x NB x 128 B of activation per item-stage, with NB taken from each chunk's block count. The route count is measured from the gate routing (`(lids >= 0).sum()`). The line also prints padded slots and the chunk histogram per N bucket.
5. Per-class gate: (a) exactness: max diff 0 between pp and master's rega, eager, on master's self-check routings (low: 753/1024 random/skewed/one; high: 3012/4096 random/skewed/one; ft: 4500 local, 8192 random/local, 16384 random/skewed/one/local). This uses `torch.equal` on hs (rows of used local blocks) and y (local-route rows) and on the align tables, and requires error word 0 and the right markers on both. (b) Copy check: states A, B (deepcopy with both modules in the memo) and C (tensor clone), one graph each, replays A, B, C, A, B with refilled inputs. Each replay must be bit-identical to master's eager rega on the same inputs, with that state's own error word 0 and both PP markers. TMA maps are encoded per call from that call's tensors, as in master, and nothing sits at file scope. (c) Speed: whole expert block, cold-L2 graph medians over `REGA_ALTERNATIONS` with the empty graph subtracted, at master's gate T (753 / 3012 / 16384 soft). ON only if pp <= 0.98 x master's rega.

**Barrier count table** (all four instantiations share one warp layout: 384 threads, producer warps 0-3, WG1 = CTA warps 4-7 (cw 0-3), WG2 = CTA warps 8-11 (cw 4-7); NB only changes the wgmma width; KS = 4096/128 = 32 for both w13 and w2; S = stages consumed by the CTA = 32 x items):

| items/CTA | S | PP_A (bar 2): WG1 bar.sync | PP_A: WG2 bar.arrive | PP_B (bar 3): WG1 bar.arrive | PP_B: WG2 bar.sync | bar 1 (epilogue, 256) |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 (`started` false, no final) | 0 | 0 | 0 | 0 |
| 1 | 32 | 31 in-loop + 1 final = 32 | 32 | 32 | 32 | 2 (both WGs) |
| 2 | 64 | 63 + 1 = 64 | 64 | 64 | 64 | 4 |
| k >= 1 | 32k | 32k - 1 + 1 = 32k | 32k | 32k | 32k | 2k |

Every phase is 128 syncing threads plus 128 arriving threads = 256, the count. WG2 cannot arrive PP_A twice inside one phase, because its next PP_B sync needs WG1's next PP_B arrive, which comes after WG1's PP_A sync. WG1 cannot arrive PP_B twice either, because its next arrive comes after a PP_A sync that needs WG2's arrive, which comes after WG2's PP_B sync. At an item boundary, WG1 reaches the epilogue's bar 1 after its last PP_B arrive. WG2 reaches bar 1 after its last PP_B sync, which is already satisfied, so it cannot deadlock. WG1's next-item PP_A sync pairs with WG2's arrive from before bar 1. The mbar timeout path (`dead`) skips only mbar waits, never PP ops. WG1 runs at most one stage ahead of WG2 on the 4-stage ring, so full/empty parity aliasing cannot occur.

**Deviations from the spec**
- The module is named `muse_prefill_rega_pp`, not v3 (caller's instruction; the sibling uses v3). It does **not** recompile master's source plus PP: it compiles only the 4 PP instantiations, which saves about half the nvcc time. If the build fails, PP stays OFF and master's already-built module keeps running. Nothing is rebuilt.
- Exactness and the copy check compare against master's rega bit for bit, not against the fp32 reference, which master's own checks already cover. Rows the kernels never write (torch.empty padding) are excluded. The final weighted-sum output's max diff is printed for information but not gated, so a non-deterministic SGLang reduction cannot veto the class.
- Per-bucket split (open question 3): only the chunk histogram per N bucket is printed. Per-bucket timing is not.
- Open question 2 (the epilogue's half-stage drain per item): not modelled. The in-prepare gate measures it.

## Rules check

- [x] CUDA-graph capturable: `kernel_fn` adds only a dict lookup on a prepare-time flag keyed by a token-count class. There are no syncs, prints, try/except or allocations sized by input values. The PP launch is `rega_gemm_pp`, with master's signature and per-call map encoding.
- [x] Nothing derived from `data` is cached: `state["rega_pp"]` holds bools set in prepare and `state["rega_pp_ext"]` is a module. `keep` is passed only by prepare.
- [x] Weights: the same `rega_bank` (lossless NVFP4/E4M3 retile from master). Dequant happens in registers, unchanged.
- [x] Activations BF16, fp32 accumulation; outputs bit-identical to master's rega (gated).
- [x] n/a: attention is untouched (every key).
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network: only in-process `load_inline`.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Two-state safety: no module, static or C++ globals (audit: 0 findings on the derived source). The A/B/C refilled-replay check runs with PP on.

```
$ python3 -I .claude/skills/churnsubs/bin/okc_lint.py <wt>/muse_prefill_layer/submission.py --problem muse_prefill_layer
warn  host-sync ... (prepare-only; many, all "outside kernel_fn's call graph; fine in prepare()")
warn  dtype        L3427/L3428 (master's rega bank), L3686 (master's pad), L4364 (PP replay-check pad, same as master's)
OK with warnings: muse_prefill_layer/submission.py
```
`python3 -I -m py_compile` ok; `__pycache__` removed. A CPU dry run of `_rega_pp_setup` exercised these branches: master all OFF, start deadline, v1 fallback, and build failure. Each printed per-class OFF reasons and `PATH CHOSEN ... master`.

## Pre-flight audit

Audited 2026-10-25 by an Opus subagent. **Verdict: dispatch (no fixes).** The focus was hang safety.

- **Derived source.** The derivation was re-run on CPU by AST-extracting master's strings. All 7 anchors and 14 substitutions match exactly once. `diff` of master's `_rega_src()` against the derived PP source shows only these changes: the PP barrier helpers; `consume_item_pp` (its signature, the pre-`mbar_wait` sync block and the post-commit `pp_arrive`); the kernel name, launch marker, `wg2`/`started` declarations, the call site and the post-loop final sync; `info_one_pp`/`rega_pp_init`/`rega_gemm_pp` renames; and the gather kernel and entry dropped. `_rega_src_v1()` still hashes to `REGA_V1_SHA256`. Master's `_rega_src()` is byte-identical to master (blob a2fe2a75). Nothing was compiled: there is no nvcc here.
- **Barrier counts, re-derived independently.**
  - The layout gives exactly 128 threads on each side of every PP bar op: WG1 = CTA warps 4-7, WG2 = warps 8-11, and the producer branch (`warp < 4`) contains no bar 1/2/3 op.
  - Both warpgroups see the same item sequence, KS and bucket: `total` and KS come from shared memory after `__syncthreads`, `decode_item` is deterministic, and the bucket only changes NB.
  - For S = 32·items stages: WG1 does S-1 in-loop syncs on A plus 1 final sync, matching WG2's S arrives on A; WG1's S arrives on B match WG2's S syncs on B. With S = 0 neither warpgroup does any PP op.
  - `dead` never gates a PP op, and every `mbar_wait` is bounded, so the timeout path ends instead of hanging. The kernel has no early return.
  - `wg2` and `started` are uniform within each warp, and `__syncwarp` precedes every bar op. Lane 0's `mbar_arrive` sits after the op.
  - Epilogue bar 1: WG2's last sync on B is satisfied before WG1 reaches bar 1, so the two cannot deadlock.
- **Simulation.** A random-schedule simulation with hardware barrier semantics (count 256, release and reset, double-arrive detection) covered items 0-5 × KS {1, 2, 3, 32} × 300 schedules and found 0 faults. Negative controls confirm the simulation can fail: removing the `started` guard deadlocks it, and removing the final sync leaves 128 arrivals pending.
- **Ring.** WG1 runs at most one stage ahead of WG2, so there is no parity aliasing on the 4-stage ring.
- **Rules.** kernel_fn adds only a dict lookup on prepare-time bools keyed by `_rega_class(T)`. That class matches master's kernel_fn class split (low ≤ 1024 < high ≤ 4096 < ft ≤ 16384). There are no prints, try/except, syncs or allocations. TMA maps are encoded per call (the code is master's). There are no module or C++ globals, and the PP Python constants are never mutated. Master's checks run before `state['rega_pp']` exists, so they exercise master's kernel.
- **Gates.** The register-pool, attribute and smem check is master's, plus local bytes == 0 (master logs 0 B). Exactness uses bit-equality on rows the kernels write, which is sound because each output is owned by one CTA with a fixed accumulation order and no atomics. Prepare budget: master ends at ~124 s; PP starts by 180 s, takes no new step after 240 s, and each step runs in seconds, so prepare stays well under its 300 s limit.
- **Lint and compile.** `okc_lint` gives OK with warnings (all pre-existing kinds). `py_compile` passes; `__pycache__` removed.

Residual risk: nvcc has not been run on the derived source (the edits are syntactically trivial; a build failure or spill leaves every class on master's rega and prints PATH CHOSEN master).
