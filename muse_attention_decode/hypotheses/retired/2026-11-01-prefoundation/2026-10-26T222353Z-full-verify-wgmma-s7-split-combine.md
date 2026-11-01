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
review: hold
status: hold
review_at: 2026-10-28T114619Z
review_companions: null
hold_reason: "adversarial review: Hold. Not on master (no s7 module; full verify at B>1 still runs paired FA3 at 164/304 us), but s6's own DIAG (archive full-verify-wgmma-s6-split-ring-2cta/4899"
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

## Adversarial review

Reviewed 2026-10-28T114619Z by an adversarial subagent (model opus).

**Verdict: hold.** Hold. Not on master (no s7 module; full verify at B>1 still runs paired FA3 at 164/304 us), but s6's own DIAG (archive full-verify-wgmma-s6-split-ring-2cta/4899ab49d74f4df7.log) already answers it: subtracting the measured ticket epilogue (156/107/64 us at B=8/16/32) from the s5-ring full kernel gives a no-ticket kernel of ~179/367 us at B=16/32; pre-attn + kernel + combine projects to ~0.87x / 0.80x of the FA3 route, so it cannot clear the 1.05x gate. A win would port mechanically to muse_decode_layer's full verify rows; none is expected.

### Justification
s6 DIAG: s5-ring full kernel v2 234.7/285.7/431.4 us at B=8/16/32; gather-only + ticket epilogue (v3) 198.3/188.7/236.7; gather-only, no epilogue (v4) 42.6/81.9/172.5. The ticket epilogue (v3-v4) is 155.7/106.8/64.2 us and falls with B (not B-independent as the hypothesis assumes); v2 minus it gives ~79/179/367 us. The hypothesis reads v2's fit intercept (162.6 us) as the ticket cost and expects ~132/266 us, which would need the ticket to cost 1.28x / 2.4x more in the full kernel than in its gather-only twin. Incumbent (bfb0457c profile): FA3 main 148.6/285.9 + FlashAttnFwdCombine 6.4/6.8 + prepare 3.0/3.4 + pre-attn 2.7/4.1 + norm-gate 2.6/3.8 us; s7 route projects 188-192 us (0.87x) at B=16 and 378-381 us (0.80x) at B=32. The gap is the main loop (compute adds ~1.3 us per tile per CTA at 2 CTAs/SM over a 1.15 us/tile gather), not the merge. Closer prior not cited: full-verify-cuda-streamk-splitkv (regression 1.2314x) lost in its main loop. Secondary risk: a third load_inline module on s6's tight import clock.

### Confounders on the current kernel
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup (+ ab-tune) define the incumbent; the gate times route vs route in one process and the incumbent is the same on both bases, so no confound.
- fused-preattention-triton is shared by both routes, so no confound.
- The s6 base predates everything landed after 10-25, so the headline cannot be compared with the 1.634 board; a routed result would still need a restack check onto current master.

### Interference and A/B needs
No companions: nothing landed is entangled with the full-verify B>1 route, and the hypothesis's own DIAG rows already isolate its two sub-changes.

### Open questions
- If the direct number is still wanted: one DIAG-only w6_kernel instantiation (RING=0, 4 stages, MINB 2, full compute, EPI off / STORE_ONLY) in the next full-verify stage settles additivity at no check cost (prediction ~179/367 us at B=16/32).
- The next full-verify stage should target the main loop (gather alone 1.29x / 1.36x floor at 81.9/172.5 us; FA3 main 148.6/285.9 us), with STORE_ONLY + combine as its epilogue from the start.
- Port to muse_decode_layer: mechanical (S3 c3 cells run the same paired FA3 + combine + norm-gate on the same geometry; FA3 main is 151.9 us of the 466.96 us verify_16 row and 288.2 us of the 700.90 us verify_32 row in 9afcccc3), needing its own in-file gate.
- Bookkeeping: muse_attention_decode/hypotheses/BEST.result.json holds job a8d647772fa945e5 (a 0.0023 check), not the 1.634 board job 4e386647 named in BEST.json.

