---
problem: muse_attention_decode
slug: attn-fast-math-epilogues
created: 2026-10-27T132752Z
base: master
area: "output norm-gate (CUDA epilogue math: o/L, RMSNorm reciprocal sqrt, sigmoid) in the routed attention kernels"
change: >
  Built on master 2eb03f76 (no depends_on). LESSONS 2026-10-27 (prefill
  experts-rega-w13-fast-silu-epilogue: accurate expf + IEEE divide cost 6-8% of w13, 5-10x the
  instruction estimate) asks to check other non-fast-math epilogues. Every attention load_inline
  module is built without --use_fast_math (flags -O3 [-lineinfo]); the softmax and merge
  exponentials already use ex2.approx.ftz (the ex2 helper in each module), but every ROUTED output
  epilogue still uses accurate forms per element: o / L * v_scale (IEEE div x4 per thread), 1.f /
  sqrtf(ss / D + eps) (IEEE sqrt + IEEE div per row), and y = o * r / (1.f + expf(-g)) (accurate
  expf + IEEE div per element). Replace ONLY those lines in the four routed epilogues with: inv =
  v_scale * rcp_approx(L) once (rcp.approx.ftz.f32 via asm; L > 0 as the code comments state; keep
  the L > 0 ? .. : 0 guard where it exists), r = rsqrtf(ss * (1.f / D) + eps), sig =
  rcp_approx(1.f + ex2(-g * LOG2E)) (ex2(+inf) = inf gives rcp 0, so sigmoid saturates to 0), y =
  o * inv * r * sig, same bf16 rounding and store. Sites: (a) _SK_CUDA finalize_store (full decode
  1<B<=64 ticket tail, PSEUDO sliding verify_1, sd sliding decode_32); (b) _SLIDE_PH_CUDA
  sliding_cluster_kernel PERHEAD reduce-scatter epilogue (sliding decode_1); (c)
  c320_combine_kernel warp-0 epilogue (full decode_1, the routed <320, 16, PDL, ONEPASS>
  instance); (d) _SW_CUDA sw_combine_kernel<PDL=true> epilogue (sliding verify B=2..32). Leave
  untouched: prologue RMSNorm / RoPE (accurate sincosf, hidden under the gather), every unrouted
  kernel (skF, C=4 cluster, merge_head / tk, stage-1 splitkv, Triton). Incumbent fallback by
  template twins only, so landed instances stay byte-identical: a bool FE = false flag reaching
  the epilogue; new instances sk variants 11 / 12 / 13 = FE twins of 0 / 4 / 6 (indices chosen to
  leave 8-10 to sk-ticket-merge-onepass), sliding_cluster_kernel<true, FE>,
  c320_combine_kernel<320, 16, true, true, FE>, sw_combine_kernel<true, FE>; print regs / local
  bytes for each twin beside its incumbent. Per twin: the incumbent own self-check cases +
  two-state replay, plus the whole-output rel L2 of twin vs incumbent printed (expect ~1e-6; fail
  above 1e-3). Per routed cell (full decode B=16 / B=32, sliding verify_1, sd decode_32, sliding
  decode_1, full decode_1, sliding verify B=16 / B=32): interleaved cold-L2 graph timing incumbent
  vs twin through the real pipeline (two rounds of 21 reps); FE is a strict shortening of the same
  dependent chain, so route it UNLESS slower than 0.995x in either round (regression veto), freeze
  per-cell choices at import, print PATH CHOSEN: fast epilogue [cell]: on|off (r1, r2). Each new
  phase starts only if import clock + 20 s <= 430 s, else the cell keeps its incumbent. Precision:
  fp32 approximations at ~2 ulp under a bf16 output; queries stay E4M3, KV FP8, fp32 accumulation,
  every key attended.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "PH epilogue is the last serial step of a latency-bound 9.6 us kernel: -0.05..-0.15 us at 1.03%/us = +0.05..0.15%"
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: "finalize_store on each PSEUDO merge winner: -0.05..-0.1 us at 0.68%/us (likely reads flat on the headline; judge the PATH CHOSEN ratio)"
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: "c320 combine warp-0 epilogue, 2 elements per lane: -0.05..-0.1 us at 0.36%/us, inside noise"
  - shapes: "*layer=sliding load=verify_[13][62] *"
    direction: flat
    why: "sw_combine 4 elements per thread once per row: inside noise"
  - shapes: "*layer=full load=decode_[13][62] *"
    direction: flat
    why: ticket-tail epilogue -0.1 us of 104-191 us
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
prior: "epilogue-gate-l2-prefetch-cuda: neutral (within-ci); it prefetched the gate operand to L2 and changed no arithmetic, while this changes only the epilogue arithmetic (accurate expf / IEEE div / 1.f/sqrtf to ex2.approx / rcp.approx / rsqrtf), per the 2026-10-27 prefill fast-silu lesson, with per-cell twins and gates"
review: required
status: queued
---

# attn-fast-math-epilogues

## Change

Built on master 2eb03f76 (no depends_on). LESSONS 2026-10-27 (prefill experts-rega-w13-fast-silu-epilogue: accurate expf + IEEE divide cost 6-8% of w13, 5-10x the instruction estimate) asks to check other non-fast-math epilogues. Every attention load_inline module is built without --use_fast_math (flags -O3 [-lineinfo]); the softmax and merge exponentials already use ex2.approx.ftz (the ex2 helper in each module), but every ROUTED output epilogue still uses accurate forms per element: o / L * v_scale (IEEE div x4 per thread), 1.f / sqrtf(ss / D + eps) (IEEE sqrt + IEEE div per row), and y = o * r / (1.f + expf(-g)) (accurate expf + IEEE div per element). Replace ONLY those lines in the four routed epilogues with: inv = v_scale * rcp_approx(L) once (rcp.approx.ftz.f32 via asm; L > 0 as the code comments state; keep the L > 0 ? .. : 0 guard where it exists), r = rsqrtf(ss * (1.f / D) + eps), sig = rcp_approx(1.f + ex2(-g * LOG2E)) (ex2(+inf) = inf gives rcp 0, so sigmoid saturates to 0), y = o * inv * r * sig, same bf16 rounding and store. Sites: (a) _SK_CUDA finalize_store (full decode 1<B<=64 ticket tail, PSEUDO sliding verify_1, sd sliding decode_32); (b) _SLIDE_PH_CUDA sliding_cluster_kernel PERHEAD reduce-scatter epilogue (sliding decode_1); (c) c320_combine_kernel warp-0 epilogue (full decode_1, the routed <320, 16, PDL, ONEPASS> instance); (d) _SW_CUDA sw_combine_kernel<PDL=true> epilogue (sliding verify B=2..32). Leave untouched: prologue RMSNorm / RoPE (accurate sincosf, hidden under the gather), every unrouted kernel (skF, C=4 cluster, merge_head / tk, stage-1 splitkv, Triton). Incumbent fallback by template twins only, so landed instances stay byte-identical: a bool FE = false flag reaching the epilogue; new instances sk variants 11 / 12 / 13 = FE twins of 0 / 4 / 6 (indices chosen to leave 8-10 to sk-ticket-merge-onepass), sliding_cluster_kernel<true, FE>, c320_combine_kernel<320, 16, true, true, FE>, sw_combine_kernel<true, FE>; print regs / local bytes for each twin beside its incumbent. Per twin: the incumbent own self-check cases + two-state replay, plus the whole-output rel L2 of twin vs incumbent printed (expect ~1e-6; fail above 1e-3). Per routed cell (full decode B=16 / B=32, sliding verify_1, sd decode_32, sliding decode_1, full decode_1, sliding verify B=16 / B=32): interleaved cold-L2 graph timing incumbent vs twin through the real pipeline (two rounds of 21 reps); FE is a strict shortening of the same dependent chain, so route it UNLESS slower than 0.995x in either round (regression veto), freeze per-cell choices at import, print PATH CHOSEN: fast epilogue [cell]: on|off (r1, r2). Each new phase starts only if import clock + 20 s <= 430 s, else the cell keeps its incumbent. Precision: fp32 approximations at ~2 ulp under a bf16 output; queries stay E4M3, KV FP8, fp32 accumulation, every key attended.

## Expected

- `*layer=sliding load=decode_1 *` faster: PH epilogue is the last serial step of a latency-bound 9.6 us kernel: -0.05..-0.15 us at 1.03%/us = +0.05..0.15%
- `*layer=sliding load=verify_1 *` faster: finalize_store on each PSEUDO merge winner: -0.05..-0.1 us at 0.68%/us (likely reads flat on the headline; judge the PATH CHOSEN ratio)
- `*layer=full load=decode_1 *` flat: c320 combine warp-0 epilogue, 2 elements per lane: -0.05..-0.1 us at 0.36%/us, inside noise
- `*layer=sliding load=verify_[13][62] *` flat: sw_combine 4 elements per thread once per row: inside noise
- `*layer=full load=decode_[13][62] *` flat: ticket-tail epilogue -0.1 us of 104-191 us

## Risk

Sized honestly: master attention epilogues run 2-4 elements per thread ONCE per output row (prefill w13 ran the accurate silu over every element of every tile), and the softmax / merge exponentials are already ex2.approx, so the expected weighted effect is only ~+0.1..0.3%: this can ONLY land stacked (combo-tail-onepass-push-fastepi carries it). Its standalone value is the per-cell printed ratios, which say on 8 cells in one check whether the LESSONS 5-10x multiplier also holds for short serialized tails. Precision risk is nil next to the 3% output bound; the only real risk is compile time from 6 extra instances (3 big sk twins + 1 big PH twin), so print every [module] compiled-in line vs master and keep the import-clock guard. Do not apply fast math module-wide (--use_fast_math would also turn the accurate prologue sincosf at positions up to 80000 rad into __sinf / __cosf, whose absolute error grows with the argument and can break RoPE).
