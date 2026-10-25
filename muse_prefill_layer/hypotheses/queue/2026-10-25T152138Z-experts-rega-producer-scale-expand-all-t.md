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
review: required
status: queued
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
