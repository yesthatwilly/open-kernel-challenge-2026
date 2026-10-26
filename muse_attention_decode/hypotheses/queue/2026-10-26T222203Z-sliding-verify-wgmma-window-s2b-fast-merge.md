---
problem: muse_attention_decode
slug: sliding-verify-wgmma-window-s2b-fast-merge
created: 2026-10-26T222203Z
base: master
area: sliding-layer attention (verify B=2..32, wgmma window kernel merge)
change: >
  Stage 2, the alternative fix to sliding-verify-wgmma-window-s2-split-combine (each answered by
  its own check), on the stage-1 branch hyp/muse_attention_decode/sliding-verify-wgmma-window-s1
  (job cd01da3909cc4f4c), whatever its score. s1's gate printed a 'merge tail' of 14.9-81.7 us
  (grids 32..264) on top of a 14.9-16.1 us kernel, so the in-kernel ticket path, not the kernel,
  kept the route off. Keep the in-kernel ticket merge but make it cheap, and attribute what
  remains. (1) merge_head (module muse_sliding_verify_wgmma_v1, renamed _v2b): one int64 division
  before the loop and start_of(c + 1) of iteration c reused as start_of(c) of c + 1, so neither
  loop divides; ONE pass over the request's slots [c_lo, c_hi] in batches of 4: the m, l and 8
  float4 loads of the batch issued before any use, then an online LSE (running max M; rescale L
  and o by ex2(M_old - M_new); m = -inf weighs 0), then the unchanged epilogue (v_scale, scaleless
  RMSNorm, sigmoid gate, bf16). Same arithmetic as s1 up to fp32 summation order. (2) A
  prepare-only DIAG next to s1's quick-pass line: a third instantiation TICKET_ONLY (the
  segment-end __threadfence, named barriers, atomicAdd and ticket reset kept, merge_head compiled
  out, output not written) timed with the others and printed as 'ticket-only tail X us', so the
  tail splits into fence / atomic / barrier cost versus merge cost. Never routed, never in
  kernel_fn. Everything else unchanged: kernel, grids, bands B=2..16 / B=17..32, eager correctness
  runs, two-state replay, final rounds >= 1.05x in both with the IQR bound, PATH CHOSEN.
expected:
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: if the tail is merge_head (two non-unrolled slot loops with two int64 divisions per slot), pre-attn ~3 + kernel ~15-16 + merge ~2 us = ~20-21 us vs master 24.4 us net; 0.21%/us
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: same mechanism in band B=17..32; 0.124%/us
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
risk: medium
depends_on: sliding-verify-wgmma-window-s1
priority: 3
kind: rewrite
controls_for: null
prior: "sliding-verify-wgmma-window-s1: s1's in-kernel merge tail (14.9-81.7 us) kept every grid off; this keeps the ticket but rewrites merge_head (no int64 division per slot, one batched online pass) and adds a ticket-only DIAG to locate the rest of the tail"
rewrite_stage: 2
replaces: "sliding verify (q_len 8, page 1, tp8) at 2<=B<=32: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang norm_gate"
review: required
status: queued
---

# sliding-verify-wgmma-window-s2b-fast-merge

## Change

Stage 2, the alternative fix to sliding-verify-wgmma-window-s2-split-combine (each answered by its own check), on the stage-1 branch hyp/muse_attention_decode/sliding-verify-wgmma-window-s1 (job cd01da3909cc4f4c), whatever its score. s1's gate printed a 'merge tail' of 14.9-81.7 us (grids 32..264) on top of a 14.9-16.1 us kernel, so the in-kernel ticket path, not the kernel, kept the route off. Keep the in-kernel ticket merge but make it cheap, and attribute what remains. (1) merge_head (module muse_sliding_verify_wgmma_v1, renamed _v2b): one int64 division before the loop and start_of(c + 1) of iteration c reused as start_of(c) of c + 1, so neither loop divides; ONE pass over the request's slots [c_lo, c_hi] in batches of 4: the m, l and 8 float4 loads of the batch issued before any use, then an online LSE (running max M; rescale L and o by ex2(M_old - M_new); m = -inf weighs 0), then the unchanged epilogue (v_scale, scaleless RMSNorm, sigmoid gate, bf16). Same arithmetic as s1 up to fp32 summation order. (2) A prepare-only DIAG next to s1's quick-pass line: a third instantiation TICKET_ONLY (the segment-end __threadfence, named barriers, atomicAdd and ticket reset kept, merge_head compiled out, output not written) timed with the others and printed as 'ticket-only tail X us', so the tail splits into fence / atomic / barrier cost versus merge cost. Never routed, never in kernel_fn. Everything else unchanged: kernel, grids, bands B=2..16 / B=17..32, eager correctness runs, two-state replay, final rounds >= 1.05x in both with the IQR bound, PATH CHOSEN.

## Expected

- `*layer=sliding load=verify_16 *` faster: if the tail is merge_head (two non-unrolled slot loops with two int64 divisions per slot), pre-attn ~3 + kernel ~15-16 + merge ~2 us = ~20-21 us vs master 24.4 us net; 0.21%/us
- `*layer=sliding load=verify_32 *` faster: same mechanism in band B=17..32; 0.124%/us
- `*layer=full *` flat: untouched

## Risk

Weakest link: at grid 32 a merge reads only ~2 slots yet the tail was 14.9 us, so part of the tail is likely outside merge_head (segment-end fences, barriers, atomics); then this stays unrouted and the ticket-only DIAG says where the time goes (same s5-family epilogue that costs s6's full-verify kernel ~107 us at B=16). Built on s1's base, so the headline lacks v3chain's FD1 chain; read SV16/SV32 rows and the DIAG lines.

## Milestones

Rewrite stage 2 of `sliding verify (q_len 8, page 1, tp8) at 2<=B<=32: prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang norm_gate`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
