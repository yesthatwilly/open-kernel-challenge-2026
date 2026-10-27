---
problem: muse_attention_decode
slug: sk-ticket-merge-onepass
created: 2026-10-27T132640Z
base: master
area: "stream-K ticket merge (splitkv_attn_sk_kernel merge_range: full decode 1<B<=64, PSEUDO sliding verify_1, sk-window sliding decode_32)"
change: >
  Built on master 2eb03f76 (no depends_on). ONE mechanism, the load structure of the in-kernel
  ticket merge in _SK_CUDA splitkv_attn_sk_kernel. Today merge_range (called through merge_slots
  by the winner CTA of request c_b on the single-level ticket path, and twice on the two-level
  path) makes TWO dependent passes over slots s_lo..s_hi with runtime-bounded loops: pass 1 loads
  every slot m from ws_ml to find M, pass 2 reloads m, l and the float4 of ws_acc and folds with
  ex2(m - Mu). nvcc unrolls such loops about 4x at most, so the winner pays roughly 2*ceil(n/4)
  dependent L2 round trips (n = slots per request: ~17.5 at full decode_16 grid 2xSM, ~9 at full
  decode_32 and at PSEUDO sliding verify_1 grid p8, ~4 at sd sliding decode_32 grid p4) while its
  next stream-K segment, or the kernel end, waits; sk-diag priced the ticket epilogue at 5.8 us
  (B=16) and 2.8 us (B=32) over no-ticket, and DEFER (moving the draw) did not recover it. NEW
  merge_range_onepass<MB> with MB = 8: for c0 = s_lo; c0 <= s_hi; c0 += MB, issue ALL loads of the
  batch first (m and l via __ldcg, the float4 acc via __ldcg; slot c0+u > s_hi gives m = -inf, l =
  0, a = 0), then fold in registers with an online LSE exactly as sw_combine_kernel and c320
  ONEPASS do (Mn = fmaxf(M, m_u); sc = ex2(M - Mn), ex2(-inf) = 0; w = m_u == -inf ? 0 : ex2(m_u -
  Mn); L = L*sc + w*l_u; o = o*sc + w*a_u; M = Mn), returning Mout = the final raw M (-inf when
  every slot is empty) so the two-level group-slot store keeps its meaning. Add template flag bool
  ONEPASS = false to splitkv_attn_sk_kernel that selects merge_range_onepass at every merge site;
  finalize_store and everything else unchanged. Instantiate ONLY three new twins in sk_pick: 8 =
  <4, ticket, ONEPASS> (full decode 1<B<=64), 9 = <4, ticket, PSEUDO, ONEPASS> (sliding verify
  B=1), 10 = <4, ticket, PSEUDO, TOK=1, ONEPASS> (sliding decode B=17..32); variants 0-7 stay
  byte-identical (grow sk_prepare done[] and sk_smem). Print sk_attrs (regs, local bytes) for 8-10
  beside 0/4/6; if a twin spills where its incumbent does not, rebuild with MB = 4. IN-PREPARE
  GATE per routed cell, after that cell incumbent route is decided: run the twin through the
  incumbent own correctness cases (_sk_case set for 8; _ps_case at the routed grid for 9; _sd_case
  at the routed grid for 10) and its two-state replay check, then time incumbent vs twin
  interleaved through the real pair (pre-attention + kernel; cold-L2 flush, empty graph
  subtracted, two rounds of 21 reps) at B=16 and B=32 (full decode bands 2..24 and 25..64), B=1
  (PSEUDO) and B=32 (sd band 3); route the twin for a cell only if it wins >= 1.01x in BOTH rounds
  with IQR <= 1.5x the incumbent (0.5 us floor), by swapping the variant index in that cell frozen
  table (_SK_VARIANT becomes an import-frozen per-band MappingProxyType read by _full_decode_sk;
  the _PS_ROUTE / _SD_ROUTE entries take 9 / 10). Print PATH CHOSEN: sk ticket merge [cell]:
  onepass|incumbent (t_inc vs t_one us both rounds, slots per request). Full decode_1 (c320
  chain), FA3, sw and cluster paths unchanged. Same fp32 arithmetic up to summation order: KV FP8,
  E4M3 q, every key attended.
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: "PSEUDO kernel 9.9 us, ~9 slots per pseudo-row at grid p8: ~6 merge round trips -> 2 at each segment-end merge; -0.5..-1.0 us at 0.68%/us = +0.35..0.7% headline"
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "~17.5 slots per request: ~10 round trips -> 3 on the winner; -1..-2 us of sk 104.1 us at 0.112%/us = +0.1..0.2%"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "~9 slots: -0.5..-1 us of 191 us at 0.074%/us (small, may read flat)"
  - shapes: "*layer=sliding load=decode_32 *"
    direction: flat
    why: "sd route at p4 has ~4 slots (2 round trips -> 1): -0.1..-0.3 us at 0.21%/us, inside noise"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: c320 chain and PERHEAD cluster untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
prior: "c320-combine-onepass: the same one-pass batched-load online-LSE fold (routed 1.119x on the full decode_1 combine, landed) applied to the stream-K ticket merge inside splitkv_attn_sk_kernel, which still does two dependent runtime-bounded passes; full-decode-sk-s5-deferred-ticket-fault-fix (neutral) moved WHEN the ticket draws (DEFER), not how the winner loads the slots"
review: proceed
status: running
review_at: 2026-10-27T134016Z
review_companions: null
stage: claimed
claimed_at: 2026-10-27T134016Z
base_sha: bb6c492319e7d6ef89caaf609741556209472f17
base_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
branch: hyp/muse_attention_decode/sk-ticket-merge-onepass
worktree: .worktrees/muse_attention_decode/sk-ticket-merge-onepass
mode: null
restack_of: null
stale_base: null
---

# sk-ticket-merge-onepass

## Change

Built on master 2eb03f76 (no depends_on). ONE mechanism, the load structure of the in-kernel ticket merge in _SK_CUDA splitkv_attn_sk_kernel. Today merge_range (called through merge_slots by the winner CTA of request c_b on the single-level ticket path, and twice on the two-level path) makes TWO dependent passes over slots s_lo..s_hi with runtime-bounded loops: pass 1 loads every slot m from ws_ml to find M, pass 2 reloads m, l and the float4 of ws_acc and folds with ex2(m - Mu). nvcc unrolls such loops about 4x at most, so the winner pays roughly 2*ceil(n/4) dependent L2 round trips (n = slots per request: ~17.5 at full decode_16 grid 2xSM, ~9 at full decode_32 and at PSEUDO sliding verify_1 grid p8, ~4 at sd sliding decode_32 grid p4) while its next stream-K segment, or the kernel end, waits; sk-diag priced the ticket epilogue at 5.8 us (B=16) and 2.8 us (B=32) over no-ticket, and DEFER (moving the draw) did not recover it. NEW merge_range_onepass<MB> with MB = 8: for c0 = s_lo; c0 <= s_hi; c0 += MB, issue ALL loads of the batch first (m and l via __ldcg, the float4 acc via __ldcg; slot c0+u > s_hi gives m = -inf, l = 0, a = 0), then fold in registers with an online LSE exactly as sw_combine_kernel and c320 ONEPASS do (Mn = fmaxf(M, m_u); sc = ex2(M - Mn), ex2(-inf) = 0; w = m_u == -inf ? 0 : ex2(m_u - Mn); L = L*sc + w*l_u; o = o*sc + w*a_u; M = Mn), returning Mout = the final raw M (-inf when every slot is empty) so the two-level group-slot store keeps its meaning. Add template flag bool ONEPASS = false to splitkv_attn_sk_kernel that selects merge_range_onepass at every merge site; finalize_store and everything else unchanged. Instantiate ONLY three new twins in sk_pick: 8 = <4, ticket, ONEPASS> (full decode 1<B<=64), 9 = <4, ticket, PSEUDO, ONEPASS> (sliding verify B=1), 10 = <4, ticket, PSEUDO, TOK=1, ONEPASS> (sliding decode B=17..32); variants 0-7 stay byte-identical (grow sk_prepare done[] and sk_smem). Print sk_attrs (regs, local bytes) for 8-10 beside 0/4/6; if a twin spills where its incumbent does not, rebuild with MB = 4. IN-PREPARE GATE per routed cell, after that cell incumbent route is decided: run the twin through the incumbent own correctness cases (_sk_case set for 8; _ps_case at the routed grid for 9; _sd_case at the routed grid for 10) and its two-state replay check, then time incumbent vs twin interleaved through the real pair (pre-attention + kernel; cold-L2 flush, empty graph subtracted, two rounds of 21 reps) at B=16 and B=32 (full decode bands 2..24 and 25..64), B=1 (PSEUDO) and B=32 (sd band 3); route the twin for a cell only if it wins >= 1.01x in BOTH rounds with IQR <= 1.5x the incumbent (0.5 us floor), by swapping the variant index in that cell frozen table (_SK_VARIANT becomes an import-frozen per-band MappingProxyType read by _full_decode_sk; the _PS_ROUTE / _SD_ROUTE entries take 9 / 10). Print PATH CHOSEN: sk ticket merge [cell]: onepass|incumbent (t_inc vs t_one us both rounds, slots per request). Full decode_1 (c320 chain), FA3, sw and cluster paths unchanged. Same fp32 arithmetic up to summation order: KV FP8, E4M3 q, every key attended.

## Expected

- `*layer=sliding load=verify_1 *` faster: PSEUDO kernel 9.9 us, ~9 slots per pseudo-row at grid p8: ~6 merge round trips -> 2 at each segment-end merge; -0.5..-1.0 us at 0.68%/us = +0.35..0.7% headline
- `*layer=full load=decode_16 *` faster: ~17.5 slots per request: ~10 round trips -> 3 on the winner; -1..-2 us of sk 104.1 us at 0.112%/us = +0.1..0.2%
- `*layer=full load=decode_32 *` faster: ~9 slots: -0.5..-1 us of 191 us at 0.074%/us (small, may read flat)
- `*layer=sliding load=decode_32 *` flat: sd route at p4 has ~4 slots (2 round trips -> 1): -0.1..-0.3 us at 0.21%/us, inside noise
- `*load=decode_1 *` flat: c320 chain and PERHEAD cluster untouched

## Risk

Weighted effect ~+0.5..1.0%: below the ~1.2% one-run margin, so on its own it lands only if sliding verify_1 reaches the top of its range; otherwise it lands stacked (combo-tail-onepass-push-fastepi). Judge it on the PATH CHOSEN ratios and the splitkv_attn_sk_kernel rows (verify_1 sliding 9.9 us, decode_16 full 104.1 us, decode_32 full 191.0 us, decode_32 sliding 13.0 us). Risks: register pressure at the merge site under launch_bounds(256, 2) (the 8-slot batch holds 48 live regs of loads; MB = 4 fallback); three extra big instantiations add compile time (print the [sk] compiled-in line vs master, keep the new gate phases behind the import-clock guard, deadline 430 s). Composes with the queued sliding-verify1-pseudo-fused-preattn-rope (prologue) and sk-full-line-5stage-select (same per-band variant table: whichever lands second re-keys it).

## Adversarial review

Reviewed 2026-10-27T134016Z by an adversarial subagent (model opus).

**Verdict: proceed.** A real, untested mechanism: the ticket winner loads its slots in one batched pass instead of two dependent runtime-bounded passes. Neither prior covers it (c320 ONEPASS was a standalone combine kernel; DEFER only moved the ticket draw). Its in-prepare interleaved A/B makes one run readable without companions; judge it on the PATH CHOSEN ratios and the splitkv_attn_sk_kernel rows, not the headline (+0.5..1.0% expected, below the ~1.2% margin).

### Justification
merge_range runs pass 1 (an __ldcg of m per slot to find M) then pass 2 (m, l and float4 acc per slot), both runtime-bounded: ~2*ceil(n/4) dependent L2 round trips on the winner CTA. The batched form routed c320-combine-onepass at 1.119x (18.94 -> 16.93 us) and moved the stage-1 combine row 8.2 -> 6.0 us. It applies to full decode B=2..64 (v2 <4,ticket>, grid 2xSM), sliding verify_1 (PSEUDO, grid p8) and sliding decode B=17..32 (PSEUDO TOK=1, grid p4). %/us coefficients checked: 0.68 (sl verify_1), 0.112 (fd16), 0.074 (fd32), 0.21 (sl decode_32). The strongest case is sliding verify_1 (latency-bound tail). Weakest link: how much of the ticket gap is merge loads vs fence/atomic/finalize is unmeasured, and the 'nvcc unrolls ~4x' premise is unverified. The prior omits splitkv-combine-wide-mlp (do-not-try; same idea on a different kernel; not a duplicate, and its kernel row supports this).

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket: the host of the change; slot count n set by its grid choice.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn: FUSED_PRE re-gates every run (~0.99-1.01x); the twin-8 gate must run after _skf_setup and skip any band in _FUSED_ROUTE.
- restack-sliding-decode-sk-window-on-cluster-master: the sd grid was picked with the two-pass merge; gating twin 10 only at p4 underestimates it.
- sk-full-line-gather: the live sk module is SK_FULLLINE when its canary passes, else the half-line rebuild; twins compile into whichever is live.

### Interference and A/B needs
No companion; the in-run A/B (incumbent vs ONEPASS twin, interleaved cold-L2, 2 rounds x 21 reps, 1.01x both rounds, IQR guard) attributes it. Optionally print an MB=4 twin readout in the same gate to separate pass fusion from batch depth.

### Open questions
- Is merge_range unrolled/load-batched in SASS?
- Register count / local bytes at launch_bounds(256,2) with MB=8; instantiate MB=4 twins up front if the compile budget allows, or pick MB from compile-time attrs.
- Compile/prepare budget: master's import clock ends ~295 s; print [sk] compiled-in time vs master; keep the gate behind the clock guard.
- Print (do not route) twin-10 / twin-9 readouts one grid step up.
- Run the whole phase after _sw_setup (after _ps_setup, _skf_setup and _sd_setup).

