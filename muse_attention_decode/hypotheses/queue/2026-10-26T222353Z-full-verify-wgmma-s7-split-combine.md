---
problem: muse_attention_decode
slug: full-verify-wgmma-s7-split-combine
created: 2026-10-26T222353Z
base: master
area: full-layer attention (verify B=2..64, wgmma stream-K kernel merge)
change: >
  Stage 7 on the s6 branch (hyp/muse_attention_decode/full-verify-wgmma-s6-split-ring-2cta, job
  4899ab49d74f4df7), whatever its score; the same fix as
  sliding-verify-wgmma-window-s2-split-combine, applied to full verify B=16/32. s6's DIAG fitted
  its best full kernel, s5's CTA-wide ring (variant 2, w6_kernel<RING=0, 4 stages, 2 CTAs/SM>), at
  T = 162.6 us fixed + 1.76 us per tile per CTA (285.7 / 431.4 us at B=16 / 32), and its
  gather-only pair put the segment-end ticket epilogue at +107 us (B=16) and +64 us (B=32): the
  B-independent cost is the in-kernel ticket merge, the same tail that kept s1 off at sliding
  verify. CHANGE: (1) give w6_kernel an epilogue mode STORE_ONLY (the segment-end fp32 partial
  store to slot bx + cb, then no __threadfence / ticket / merge_head; today's EPI = true path
  stays byte-identical) and instantiate RING=0, 4 stages, MINB 2, OVL false, STORE_ONLY in the
  ROUTED module muse_full_verify_wgmma_s6 (renamed _s7); (2) add one combine kernel
  w7_combine_kernel, grid (B * QL, HKV), 128 threads, one CTA per (request b, verify token j, kv
  head g), thread t: head h = g * G + t / 16, dims (t % 16) * 4 .. +3: recompute pre_s / T exactly
  as w6_kernel's prologue (shared __device__ helper), c_lo / c_hi = owner_of(pre[b]) /
  owner_of(pre[b+1] - 1) with the launch grid, one pass over the overlapping slots (start_of
  computed incrementally, loads of 4 slots issued before use, online LSE, m = -inf weighs 0), then
  merge_head's epilogue (v_scale, scaleless RMSNorm over 64, sigmoid gate, bf16 into the [B * 8,
  HQ * 64] output). (3) Gate this pair as candidate 'v2-split' beside s6's variants 0 / 1 with
  s6's harness unchanged: s5's 19-case bisect table, two-state replay, two interleaved rounds >=
  1.05x vs master paired FA3 (pre-attention + prepare + FA3 + combine + paired norm-gate) at B=16
  (band 2..24) and B=32 (band 25..64), IQR bound, rel L2 < 3e-2, error word 0, PATH CHOSEN with
  both numbers; add 'v2-split kernel only' and 'combine' times to the DIAG print so the no-ticket
  per-tile cost is measured directly. kernel_fn's paired FA3 path stays the fallback.
expected:
  - shapes: "*layer=full load=verify_16 *"
    direction: faster
    why: "routes only if the no-ticket s5-ring kernel + combine beats paired FA3's ~166 us by 1.05x; the gather-only epilogue cost (+107 us) suggests ~180 us, the full-kernel fit (162.6 us fixed) suggests ~132 us: the run decides; 0.073%/us"
  - shapes: "*layer=full load=verify_32 *"
    direction: faster
    why: "same at B=32: ~266-367 us vs FA3 ~304 us; 0.043%/us"
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
risk: medium
depends_on: full-verify-wgmma-s6-split-ring-2cta
priority: 4
kind: rewrite
controls_for: null
prior: "full-verify-wgmma-s6-split-ring-2cta: s6 changed only pipeline ownership and kept the in-kernel ticket merge whose B-independent tail (~107 us at B=16 in its own gather-only DIAG) kept every variant at 0.46-0.57x of FA3; this routes the s5-ring kernel with partials only plus one parallel one-pass combine"
rewrite_stage: 7
replaces: "full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel (master paired FA3)"
review: required
status: queued
---

# full-verify-wgmma-s7-split-combine

## Change

Stage 7 on the s6 branch (hyp/muse_attention_decode/full-verify-wgmma-s6-split-ring-2cta, job 4899ab49d74f4df7), whatever its score; the same fix as sliding-verify-wgmma-window-s2-split-combine, applied to full verify B=16/32. s6's DIAG fitted its best full kernel, s5's CTA-wide ring (variant 2, w6_kernel<RING=0, 4 stages, 2 CTAs/SM>), at T = 162.6 us fixed + 1.76 us per tile per CTA (285.7 / 431.4 us at B=16 / 32), and its gather-only pair put the segment-end ticket epilogue at +107 us (B=16) and +64 us (B=32): the B-independent cost is the in-kernel ticket merge, the same tail that kept s1 off at sliding verify. CHANGE: (1) give w6_kernel an epilogue mode STORE_ONLY (the segment-end fp32 partial store to slot bx + cb, then no __threadfence / ticket / merge_head; today's EPI = true path stays byte-identical) and instantiate RING=0, 4 stages, MINB 2, OVL false, STORE_ONLY in the ROUTED module muse_full_verify_wgmma_s6 (renamed _s7); (2) add one combine kernel w7_combine_kernel, grid (B * QL, HKV), 128 threads, one CTA per (request b, verify token j, kv head g), thread t: head h = g * G + t / 16, dims (t % 16) * 4 .. +3: recompute pre_s / T exactly as w6_kernel's prologue (shared __device__ helper), c_lo / c_hi = owner_of(pre[b]) / owner_of(pre[b+1] - 1) with the launch grid, one pass over the overlapping slots (start_of computed incrementally, loads of 4 slots issued before use, online LSE, m = -inf weighs 0), then merge_head's epilogue (v_scale, scaleless RMSNorm over 64, sigmoid gate, bf16 into the [B * 8, HQ * 64] output). (3) Gate this pair as candidate 'v2-split' beside s6's variants 0 / 1 with s6's harness unchanged: s5's 19-case bisect table, two-state replay, two interleaved rounds >= 1.05x vs master paired FA3 (pre-attention + prepare + FA3 + combine + paired norm-gate) at B=16 (band 2..24) and B=32 (band 25..64), IQR bound, rel L2 < 3e-2, error word 0, PATH CHOSEN with both numbers; add 'v2-split kernel only' and 'combine' times to the DIAG print so the no-ticket per-tile cost is measured directly. kernel_fn's paired FA3 path stays the fallback.

## Expected

- `*layer=full load=verify_16 *` faster: routes only if the no-ticket s5-ring kernel + combine beats paired FA3's ~166 us by 1.05x; the gather-only epilogue cost (+107 us) suggests ~180 us, the full-kernel fit (162.6 us fixed) suggests ~132 us: the run decides; 0.073%/us
- `*layer=full load=verify_32 *` faster: same at B=32: ~266-367 us vs FA3 ~304 us; 0.043%/us
- `*layer=sliding *` flat: untouched

## Risk

Low route probability (about 15%) but the biggest single prize on the board if it routes (+2-4% headline: verify at near-decode bandwidth is DIAGNOSIS's +5-6.5% ceiling), and either way it measures the s5-family kernel without its ticket tail, which decides whether the wgmma full-verify line is worth more stages. Built on s6's base (no v3chain), so read the FV16/FV32 rows, PATH CHOSEN and DIAG, not the headline. Do-not-try check: wgmma, not mma.sync; all threads issue loads (not a single producer warp).

## Milestones

Rewrite stage 7 of `full-layer verify at 1<batch<=64: FA3 main + prepare_varlen_num_blocks + FlashAttnFwdCombine + _paired_norm_gate_kernel (master paired FA3)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
