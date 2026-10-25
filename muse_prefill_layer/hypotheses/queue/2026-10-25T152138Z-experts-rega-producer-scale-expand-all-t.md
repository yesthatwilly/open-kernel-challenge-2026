---
problem: muse_prefill_layer
slug: experts-rega-producer-scale-expand-all-t
created: 2026-10-25T152138Z
base: master
area: "w13 + w2 expert GEMMs (rega_kernel dequant: E4M3 scale expansion moved to idle producer warps), every rega class"
change: >
  Built on master (1932a9c4). Consumer instruction count of rega_kernel only. ARITHMETIC: turn_1
  (~1.36K local routes, ~42 per expert, buckets 32/64) streams w13's 604 MB in 283 us = 2.1 TB/s
  and w2's 302 MB in 144 us = 2.1 TB/s (44% of HBM), so at small N the per-stage dequant issue
  dominates (one stage = 2 x 8 m64n32k16 = 256 clk of wgmma vs ~180 dequant instructions per
  consumer warp, 2 warps per SMSP); the two-CTAs-per-SM run (LESSONS) already showed latency/issue
  hiding pays 9% there. In dq_stage every consumer lane spends ~7 of ~24 instructions per k16 step
  on scales: a uint2 load, cvt.rn.f16x2.e4m3x2 x2, then per k16 step __half22float2, two fmul by
  2^119 and two __float2bfloat162_rn. Producer warps 1-3 (96 threads, setmaxnreg.dec 40) are idle.
  CHANGE (port of muse_moe_w4a16's queued cuda-nvfp4-wgmma-rega-producer-scale-expand to prefill's
  rega): (1) a scale buffer per ring stage in dynamic smem, 2 records x 1024 B = 2048 B per stage,
  8192 B for 4 stages, placed after the epilogue staging tile (Smem<false>::BYTES 203776 ->
  211968; update the static_assert to <= 227 KB with the 16384 static bound: 228352 <= 232448).
  (2) STAGES more mbarriers sfull[s] (static), count 3. Producer warps 1-3 walk the same item /
  k-stage sequence as tid 0 (same decode_item, same stage counter n and parity): for each fill
  they mbar_wait(full[s], parity), then convert the stage's 2 x 512 scale bytes (st + r*REC +
  REC_CODES) into bf16 in the SAME byte order: sx[s][r*512 + i] = bf16(float(e4m3 byte i) *
  2^119), computed exactly as master's consumers do (cvt.rn.f16x2.e4m3x2 -> __half2float -> *
  2^119 -> __float2bfloat16_rn; exact, since e4m3 has 3 mantissa bits and 448 * 2^119 < FLT_MAX),
  256 32-bit scale words spread over the 96 threads, then __syncwarp and lane 0 arrives on
  sfull[s]. (3) In the new consumer, wait full[s] then sfull[s] (same parity), and in dq_stage
  replace the uint2 scale load + e4m3x4_to_f16x2x2 + per-k convert with ONE uint4 load per record
  at sx[s] + r*1024 + cw*128 + g*16 (the 8 bf16 = rows g / g+8 for k16 steps 0-3, the same
  positions master's uint2 at cw*64 + g*8 holds); per k16 step sg = prmt(w_k, w_k, 0x1010) (row g)
  and sg8 = prmt(w_k, w_k, 0x3232) (row g+8): 2 instructions instead of ~7. dequant8, the wgmma
  sequence, wgmma_wait<0>, the empty arrive, epilogues, producer copies, ring depth, buckets,
  chunking, gather and the weighted sum are unchanged. Reuse is safe: helpers overwrite sx[s] for
  fill f only after full[s] phase f, which the producer issued only after the consumers' empty[s]
  arrives for fill f-1, which follow their dq_stage reads of sx[s]. The dead/timeout path must
  still arrive on sfull. Expose it as rega_kernel_sx<G, BIG> (4 instantiations) behind a host
  entry rega_gemm_sx with rega_gemm's signature; _experts_rega uses it for a class when
  state['rega_sx'][class] is True (prepare flag per class 'low' / 'high' / 'ft', default False).
  Outputs are bit-identical to master's rega. BUILD: do NOT edit master's _REGA_HEAD / _REGA_BODY
  strings (the v1 build-failure fallback derives its source from them by text substitution and
  checks REGA_V1_SHA256). Append the new code as a separate string after master's body (new
  __global__ kernel(s) + a new host entry declared in _REGA_CPP), compile _rega_src() + that
  string under a bumped module name (muse_prefill_rega_v3), with master's exact flags and sm_90a
  guards; if that build or its import fails, build master's v2 source under its master name
  exactly as today and leave the new variant OFF. Print nvcc seconds and regs / local bytes / smem
  / CTAs per SM for every new instantiation (any spill turns that variant OFF). Prepare is ~124 s
  today (build 38.6 s); keep the new phases inside master's REGA budget/deadline constants.
  PREPARE GATE (only for classes whose rega_ok is ON): (1) exactness on master's self-check
  routings at every class size (753 / 1024 / 3012 / 4096 random, skewed, one; 4500 local; 8192
  random, local; 16384 random, skewed, one, local): max |SX - master| == 0 on hs and y, error word
  0, launch markers; (2) master's _rega_replay_check copy plan with SX on; (3) cold-L2 graph
  medians, REGA_ALTERNATIONS interleaved, empty graph subtracted, SX block vs master's rega block
  at each class gate T; SX ON for a class only if <= 0.98x. Print per class the w13 and w2 kernels
  alone (graph of one launch) for both variants with TB/s (weight bytes) and TFLOP/s, then one
  '[rega-sx] PATH CHOSEN <class>: sx|master' line per class as the last lines of prepare(). Any
  failure prints 'OFF: <reason>' and master's rega runs unchanged. RULES: kernel_fn stays
  CUDA-graph capturable with no host sync, no allocation sized by input values, no data-dependent
  host control flow, no prints, no try/except, no timing and no per-call mutable Python state; it
  dispatches only on the token count and on flags prepare() set. Nothing derived from inputs is
  cached. NVFP4 expert weights stay NVFP4/E4M3 bytes in global memory and are dequantized in
  registers (any bf16 scale values live only in per-stage shared memory inside one launch);
  activations BF16, fp32 accumulation; every route is computed. No child processes (in-process
  load_inline only). The judge prepares TWO state copies: no module, static or C++ global may hold
  device pointers, TMA maps or buffers; keep master's per-call tensor-map encoding
  (__grid_constant__ Maps) and per-state bank / error word.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "dequant-issue-bound at buckets 32/64 (2.1 TB/s of 4.8): cutting ~25-30% of dequant instructions targets w13 283 -> <= 250 us and w2 144 -> <= 128 us; sliding turn_1 704 -> ~655 us (weight 46)"
  - shapes: "*load=turn_4*"
    direction: faster
    why: "bucket 128 at ~173 routes per expert: the dequant is a smaller share of each ~1500-clk stage; w13 712 -> <= 680 us, w2 348 -> <= 335 us"
  - shapes: "*load=first_turn*"
    direction: faster
    why: "small: ~100 of ~1530 clk per stage are scale conversion on the serialized dequant path; w13 3669 -> <= 3560 us, w2 1783 -> <= 1730 us (~3%); flat would point at the activation-traffic limit instead"
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-25T152518Z
review_companions: null
---

# experts-rega-producer-scale-expand-all-t

## Change

Built on master (1932a9c4). Consumer instruction count of rega_kernel only. ARITHMETIC: turn_1 (~1.36K local routes, ~42 per expert, buckets 32/64) streams w13's 604 MB in 283 us = 2.1 TB/s and w2's 302 MB in 144 us = 2.1 TB/s (44% of HBM), so at small N the per-stage dequant issue dominates (one stage = 2 x 8 m64n32k16 = 256 clk of wgmma vs ~180 dequant instructions per consumer warp, 2 warps per SMSP); the two-CTAs-per-SM run (LESSONS) already showed latency/issue hiding pays 9% there. In dq_stage every consumer lane spends ~7 of ~24 instructions per k16 step on scales: a uint2 load, cvt.rn.f16x2.e4m3x2 x2, then per k16 step __half22float2, two fmul by 2^119 and two __float2bfloat162_rn. Producer warps 1-3 (96 threads, setmaxnreg.dec 40) are idle. CHANGE (port of muse_moe_w4a16's queued cuda-nvfp4-wgmma-rega-producer-scale-expand to prefill's rega): (1) a scale buffer per ring stage in dynamic smem, 2 records x 1024 B = 2048 B per stage, 8192 B for 4 stages, placed after the epilogue staging tile (Smem<false>::BYTES 203776 -> 211968; update the static_assert to <= 227 KB with the 16384 static bound: 228352 <= 232448). (2) STAGES more mbarriers sfull[s] (static), count 3. Producer warps 1-3 walk the same item / k-stage sequence as tid 0 (same decode_item, same stage counter n and parity): for each fill they mbar_wait(full[s], parity), then convert the stage's 2 x 512 scale bytes (st + r*REC + REC_CODES) into bf16 in the SAME byte order: sx[s][r*512 + i] = bf16(float(e4m3 byte i) * 2^119), computed exactly as master's consumers do (cvt.rn.f16x2.e4m3x2 -> __half2float -> * 2^119 -> __float2bfloat16_rn; exact, since e4m3 has 3 mantissa bits and 448 * 2^119 < FLT_MAX), 256 32-bit scale words spread over the 96 threads, then __syncwarp and lane 0 arrives on sfull[s]. (3) In the new consumer, wait full[s] then sfull[s] (same parity), and in dq_stage replace the uint2 scale load + e4m3x4_to_f16x2x2 + per-k convert with ONE uint4 load per record at sx[s] + r*1024 + cw*128 + g*16 (the 8 bf16 = rows g / g+8 for k16 steps 0-3, the same positions master's uint2 at cw*64 + g*8 holds); per k16 step sg = prmt(w_k, w_k, 0x1010) (row g) and sg8 = prmt(w_k, w_k, 0x3232) (row g+8): 2 instructions instead of ~7. dequant8, the wgmma sequence, wgmma_wait<0>, the empty arrive, epilogues, producer copies, ring depth, buckets, chunking, gather and the weighted sum are unchanged. Reuse is safe: helpers overwrite sx[s] for fill f only after full[s] phase f, which the producer issued only after the consumers' empty[s] arrives for fill f-1, which follow their dq_stage reads of sx[s]. The dead/timeout path must still arrive on sfull. Expose it as rega_kernel_sx<G, BIG> (4 instantiations) behind a host entry rega_gemm_sx with rega_gemm's signature; _experts_rega uses it for a class when state['rega_sx'][class] is True (prepare flag per class 'low' / 'high' / 'ft', default False). Outputs are bit-identical to master's rega. BUILD: do NOT edit master's _REGA_HEAD / _REGA_BODY strings (the v1 build-failure fallback derives its source from them by text substitution and checks REGA_V1_SHA256). Append the new code as a separate string after master's body (new __global__ kernel(s) + a new host entry declared in _REGA_CPP), compile _rega_src() + that string under a bumped module name (muse_prefill_rega_v3), with master's exact flags and sm_90a guards; if that build or its import fails, build master's v2 source under its master name exactly as today and leave the new variant OFF. Print nvcc seconds and regs / local bytes / smem / CTAs per SM for every new instantiation (any spill turns that variant OFF). Prepare is ~124 s today (build 38.6 s); keep the new phases inside master's REGA budget/deadline constants. PREPARE GATE (only for classes whose rega_ok is ON): (1) exactness on master's self-check routings at every class size (753 / 1024 / 3012 / 4096 random, skewed, one; 4500 local; 8192 random, local; 16384 random, skewed, one, local): max |SX - master| == 0 on hs and y, error word 0, launch markers; (2) master's _rega_replay_check copy plan with SX on; (3) cold-L2 graph medians, REGA_ALTERNATIONS interleaved, empty graph subtracted, SX block vs master's rega block at each class gate T; SX ON for a class only if <= 0.98x. Print per class the w13 and w2 kernels alone (graph of one launch) for both variants with TB/s (weight bytes) and TFLOP/s, then one '[rega-sx] PATH CHOSEN <class>: sx|master' line per class as the last lines of prepare(). Any failure prints 'OFF: <reason>' and master's rega runs unchanged. RULES: kernel_fn stays CUDA-graph capturable with no host sync, no allocation sized by input values, no data-dependent host control flow, no prints, no try/except, no timing and no per-call mutable Python state; it dispatches only on the token count and on flags prepare() set. Nothing derived from inputs is cached. NVFP4 expert weights stay NVFP4/E4M3 bytes in global memory and are dequantized in registers (any bf16 scale values live only in per-stage shared memory inside one launch); activations BF16, fp32 accumulation; every route is computed. No child processes (in-process load_inline only). The judge prepares TWO state copies: no module, static or C++ global may hold device pointers, TMA maps or buffers; keep master's per-call tensor-map encoding (__grid_constant__ Maps) and per-state bank / error word.

## Expected

- `*load=turn_1*` faster: dequant-issue-bound at buckets 32/64 (2.1 TB/s of 4.8): cutting ~25-30% of dequant instructions targets w13 283 -> <= 250 us and w2 144 -> <= 128 us; sliding turn_1 704 -> ~655 us (weight 46)
- `*load=turn_4*` faster: bucket 128 at ~173 routes per expert: the dequant is a smaller share of each ~1500-clk stage; w13 712 -> <= 680 us, w2 348 -> <= 335 us
- `*load=first_turn*` faster: small: ~100 of ~1530 clk per stage are scale conversion on the serialized dequant path; w13 3669 -> <= 3560 us, w2 1783 -> <= 1730 us (~3%); flat would point at the activation-traffic limit instead

## Risk

Low risk: no new MMA form, no register-A hazard, the new sync is an mbarrier (bounded wait, error word), and correctness reduces to an exact compare with master (max diff 0). Rules: scales stay E4M3 bytes in global memory; the bf16 expansion exists only in per-stage shared memory inside one launch, as master already builds the same values in registers, so it is not a stored BF16 weight copy. Main risk: the helper hop (wait full, convert, arrive sfull) adds latency in front of each stage; with 4 stages it should hide, but if the ring runs dry the kernel is flat or slower and the gate keeps master. Second risk: shared memory at 211968 B dynamic + up to ~14.3 KB static in the small class leaves ~6 KB; print both. Sibling experts-rega-pingpong-consumers-all-t tests the other consumer hypothesis (serialization, not instruction count); each is attributable on its own rega rows; stack only after both are read.

## Adversarial review

Reviewed 2026-10-25T152518Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound, exactness-preserving and self-attributing (its own kernel name in the profile plus per-class PATH CHOSEN lines), so no companion is needed. The expected sizes are optimistic because the conversion moves onto SMSPs the consumers share rather than vanishing. One build-order change is mandatory before this scored run: as written, the bigger v3 build can push prepare past REGA_FT_START_DEADLINE_S=150 s and silently turn master's ft class OFF, which would regress the 138-weight first_turn shape.

### Justification
Mechanism: plausible. In master's dq_stage every lane re-converts its row-pair scales: per record it does 1 uint2 LDS and 2 cvt.f16x2.e4m3x2, and per k16 step it does __half22float2 (2 cvt), 2 FMUL by 2^119 and 2 __float2bfloat162_rn. That is about 6 instructions per k16 step plus 3 per record, roughly 54 per consumer warp per stage. The replacement costs 2 PRMT per k16 step plus the same one LDS (a uint4 now) per record. The uint4 at cw*128+g*16 is 4-lane broadcast over 128 contiguous bytes, so it is a single wavefront with no bank conflict. Exactness holds. e4m3 to f16 to f32 is exact, x2^119 stays in range (448*2^119 < FLT_MAX; the min subnormal 2^-9 becomes 2^110), and a 3-bit mantissa fits bf16, so the producer-computed bf16 values are bit-identical to what consumers build today. The smem math is also right: 211968 dynamic plus at most 14336 static (small class) plus 32 B of new barriers is under 232448, at 1 CTA/SM. Profile support: the rega pair is 381 us of 704 us (54%) on sliding turn_1 (w13 259.3, w2 122.1) and 1054 of 2098 us on sliding turn_4. The weight stream runs at about 2.1 TB/s of 4.8 at turn_1, and the 2-CTA/SM run gained on sliding turn_1, so consumer latency/issue is the limiter there. Direction: faster at turn_1, smaller at turn_4/ft. That is consistent with the weights, but the magnitudes are optimistic. WEAKEST LINK: 'producer warps 1-3 are idle' is true of the warps but not of the issue slots. Warps 1/2/3 sit on SMSPs 1/2/3, the same schedulers as two consumer warps each. A wgmma needs all four warps of a warpgroup, which spans all four SMSPs, so the slowest SMSP gates. Per stage the helpers add about 3 x (3 iterations x ~16 instr + wait/arrive) = 50-55 warp-instructions to SMSPs 1-3, and save about 2 x 36 = 72 there. So if rega is issue-bound, the net cut on the gating SMSPs is about 15-20 instructions per stage (a few %), not 25-30%. Only if it is latency-bound (the serialized dq_stage -> wgmma -> wait<0> chain) does the full ~36-instruction shortening of each consumer's critical path show up. That would be about 5-10% on w13/w2 at turn_1 and about 1-3% at turn_4/ft, so expect the low class, if anything, to clear the 0.98 gate, and turn_4/ft to land near it. The hypothesis also quotes turn_1 w13 at 283 us, but the current best profile shows 259.3 (sliding) and 226.4 (full). Its targets should be read against the profile's numbers. RISK TO MASTER (the scored-run concern): the BUILD section compiles _rega_src() plus the new string as ONE module, muse_prefill_rega_v3. That means 8 rega instantiations instead of 4, so nvcc goes from 38.6 s to roughly 65-80 s. In the master logs ft-self-check starts at about 119 s of prepare against REGA_FT_START_DEADLINE_S = 150 s. A 30-40 s longer build crosses that deadline, ft goes OFF ('prepare already used ... ft start deadline'), and first_turn falls back to Triton. Master's ft gate was 0.630, so that is about a 1.6x expert slowdown on every first_turn shape. 'Keep the new phases inside master's budget/deadline constants' is exactly what triggers it. Required fix: build master's v2 module UNCHANGED (same name, same source, same 38.6 s) and run master's whole setup including ft first. Then build ONLY the sx kernels (+ host entry rega_gemm_sx) as a separate small extension (e.g. muse_prefill_rega_sx_v1, prepended with _REGA_HEAD and the shared helpers so it shares no symbols at link time). Then run SX self-checks, the replay check and the gates under a separate SX deadline. A slow or failed SX build can then only cost SX, never master's ft. If the implementer keeps one module, they must at least raise REGA_FT_START_DEADLINE_S to cover the measured build delta and print it, but the separate-module order is strictly safer.

### Confounders on the current kernel
- experts-turn-rega-s4-port-t4096 / experts-turn-rega-s4-first-turn-n128: they define rega_kernel, the dq_stage being edited, the ring depth and the smem budget. They are the base, not a confound. The change is a strict sub-kernel edit with bit-identical outputs, and prepare gates SX against these exact kernels at the same T. Their prepare-time build sits on the critical path to the ft deadline, though (see justification).
- align-two-phase-all-t-rega: decides the chunk/block tables the kernel walks, i.e. how many k-stages each CTA runs and how unbalanced they are. SX is gated and profiled on that same align, and the change is per-stage, so its relative effect does not depend on the align. Not a confound.
- stack-tonight-five-prefill (align_par_ok / align2 precondition, router hi-lo) and experts-turn4-ssdq-s1b: rega wins the low/high gates, so ssdq and Marlin are untimed fallbacks. No interaction with the timed path.
- Not landed but relevant for later: experts-rega-low-class-two-ctas-per-sm (neutral 1.4173, gained sliding turn_1). The +8 KB SX scale ring makes 2 CTAs/SM harder to fit, and both target the same turn_1 latency/issue slack, so their gains would overlap. Do not assume they add.

### Interference and A/B needs
No companion. Outputs are proven bit-identical in prepare, so every non-rega profile row should stay flat. SX is a distinct kernel symbol (rega_kernel_sx<G,BIG>), so the profile gives its µs per shape directly against the 259/122 (turn_1), 710/344 (turn_4) and ~3700/1850 (ft) master rows in FINDINGS. The prepare log prints per-class SX-vs-master graph medians and a PATH CHOSEN line. A headline gain or loss is therefore attributable from this one run: a loss is either the gate letting through a slower kernel (visible in the rows) or the ft-deadline failure above (visible as 'OFF ft'). A revert/tune companion would cost a second scored run without disambiguating anything the log does not already show. SIBLING experts-rega-pingpong-consumers-all-t: keep it SEPARATE, not stacked. Both edit consume_item's per-stage loop and both add 4 instantiations, so stacked they roughly double the build-time risk to the ft deadline. Their gains are sub-additive by design: ping-pong hides dequant under the other warpgroup's wgmma, which is exactly the time SX shortens. They add up only where dequant exceeds the wgmma time (the turn_1 low class at N 32/64). A stacked headline could not say which one moved. Stack afterwards as a third run, consume_item_pp calling the SX dq_stage, only if both clear their own class gates. Expect the stack to beat the better single one mainly at turn_1.

### Open questions
- Print per SMSP-relevant counts if possible (ncu not available in judge). At minimum, print w13/w2 kernel-alone medians for both variants at each class gate T. If low-class SX is under 0.98 but turn_4/ft is flat, that confirms latency-bound at small N and issue/tensor-bound at large N.
- Could warp 0 lanes 1-31 (idle beside the tid-0 producer) or a different helper distribution keep SMSP 1-3 load balanced? Or is a consumer-only cheaper conversion better? Example: build the bf16 scale straight from the e4m3 bits ((b & 0x7F) << 4 | sign), which is exactly e4m3 x 2^-120 for normals and subnormals alike, with the 2^-120 to 2^119 offset folded into the dequant8 code constant or the fp32 global scale. That would cut consumer scale work to about 2-3 integer ops per pair with no helpers, no extra smem and no extra barrier, but it changes dequant8's constant and needs its own exactness proof. A candidate follow-up if SX is flat.
- Does the helper hop (wait full -> convert -> arrive sfull) ever run the ring dry at the start of a CTA's first item or after the epilogue barrier? Print the SX first-item latency, or compare a 1-item graph.
- The hypothesis quotes turn_1 w13 at 283 us, but the current best profile shows 259.3 (sliding) and 226.4 (full). Re-anchor the expected targets to the profile before reading the result.

