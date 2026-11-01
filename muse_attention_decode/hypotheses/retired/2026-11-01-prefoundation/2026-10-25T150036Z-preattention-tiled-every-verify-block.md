---
problem: muse_attention_decode
slug: preattention-tiled-every-verify-block
created: 2026-10-25T150036Z
base: master
area: pre-attention norm+rope
change: >
  Built on current master eb1b68b (no depends_on). Restack the group-tiled pre-attention kernel
  that was measured twice as a sliding-verify win (preattention-group-tiled-programs, job
  877731a3; preattention-tiled-gate-large-tokens, job c2ebf6ed05824692: _pre_attention_kernel
  4.1-4.2 -> 2.7 us at sliding verify_32, shape 28.70 -> 27.55 us, 1.0385 paired; 2.8 -> 2.0 us at
  sliding verify_16), and widen its gate from tokens >= 64 to every verify block so the 16
  pre-tests exercise it. (1) Copy _pre_attention_tiled_kernel verbatim from
  muse_attention_decode/hypotheses/done/neutral/preattention-tiled-gate-large-tokens.patch into
  submission.py right after master's _pre_attention_kernel; master's kernel stays untouched (its
  signature is unchanged since that patch: qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, k_scale_ptr,
  v_scale_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr, qkv_stride, HQ, HKV, D, ROPE, EPS, PAIRED, G).
  The tiled kernel runs grid (tokens, Hkv), num_warps 4; program (t, g) RMS-normalizes the G =
  Hq/Hkv query heads of group g as one [G, 64] tile plus key head g, applies interleaved RoPE when
  ROPE (same tl.cos/tl.sin and rounding points as master), writes the FP8 E4M3 query in the PAIRED
  layout (own half, other half zeroed, every byte written) or the unpaired layout exactly as
  master, and stores the FP8 key / value rows divided by k_scale / v_scale at cache_loc[t]. (2) In
  kernel_fn, at the single existing _pre_attention_kernel launch (the one before the splitkv /
  pseudo / FA3 branches), launch _pre_attention_tiled_kernel[(tokens, hkv)] with the same
  arguments when cfg q_len == 8 (config-only predicate: every verify block, sliding and full,
  paired and unpaired, including the sliding verify B=1 pseudo-decode route, which consumes the
  same unpaired FP8 q and stored rows), and master's _pre_attention_kernel[(tokens, hq + hkv)]
  num_warps 1 otherwise (decode at every batch keeps master byte-for-byte). Nothing else changes:
  the self-check helpers (_sk_prepare_q, _ps_prepare_q, _skf references) keep calling master's
  kernel, and every prepare-time gate, route and CUDA module is untouched. Rules: no host sync, no
  allocation or branch on input values (q_len is config), nothing cached from inputs, KV stays
  E4M3 and q is E4M3, every key is attended (attention untouched).
expected:
  - shapes: "*layer=sliding load=verify_32 *"
    direction: faster
    why: "measured twice: pre-attention 4.2 -> 2.7 us, shape 28.70 -> 27.55 us (1.0385 paired) on c2ebf6ed; master still shows 4.2 us on this row (job d672f69e). 46-weight row, ~0.12%/us of headline"
  - shapes: "*layer=sliding load=verify_16 *"
    direction: faster
    why: "parent run 877731a3: 2.8 -> 2.0 us on the kernel row; master shows 2.8 us. ~0.21%/us of headline"
  - shapes: "*load=verify_1 *"
    direction: flat
    why: "8 tokens: parent profile measured the tiled kernel 0.1 us faster (sliding 1.7 -> 1.6, full 1.5 -> 1.4); within noise but on the 0.7%/us sliding verify_1 row it cannot hurt"
  - shapes: "*layer=full load=verify_16 * *layer=full load=verify_32 *"
    direction: flat
    why: "PAIRED layout: parent saw only 2.8 -> 2.7 us at full verify_16; ~1 us out of 163/301 us"
  - shapes: "*load=decode_*"
    direction: flat
    why: "q_len 1: master's kernel (or the cluster / Triton paths that bypass it) runs unchanged"
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-25T203045Z
review_companions: null
hold_reason: "adversarial review: Sixth scored run of the same _pre_attention_tiled_kernel; its parent preattention-tiled-gate-large-tokens is on Do-not-try. The only new part (gate widened to S"
---

# preattention-tiled-every-verify-block

## Change

Built on current master eb1b68b (no depends_on). Restack the group-tiled pre-attention kernel that was measured twice as a sliding-verify win (preattention-group-tiled-programs, job 877731a3; preattention-tiled-gate-large-tokens, job c2ebf6ed05824692: _pre_attention_kernel 4.1-4.2 -> 2.7 us at sliding verify_32, shape 28.70 -> 27.55 us, 1.0385 paired; 2.8 -> 2.0 us at sliding verify_16), and widen its gate from tokens >= 64 to every verify block so the 16 pre-tests exercise it. (1) Copy _pre_attention_tiled_kernel verbatim from muse_attention_decode/hypotheses/done/neutral/preattention-tiled-gate-large-tokens.patch into submission.py right after master's _pre_attention_kernel; master's kernel stays untouched (its signature is unchanged since that patch: qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, k_scale_ptr, v_scale_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr, qkv_stride, HQ, HKV, D, ROPE, EPS, PAIRED, G). The tiled kernel runs grid (tokens, Hkv), num_warps 4; program (t, g) RMS-normalizes the G = Hq/Hkv query heads of group g as one [G, 64] tile plus key head g, applies interleaved RoPE when ROPE (same tl.cos/tl.sin and rounding points as master), writes the FP8 E4M3 query in the PAIRED layout (own half, other half zeroed, every byte written) or the unpaired layout exactly as master, and stores the FP8 key / value rows divided by k_scale / v_scale at cache_loc[t]. (2) In kernel_fn, at the single existing _pre_attention_kernel launch (the one before the splitkv / pseudo / FA3 branches), launch _pre_attention_tiled_kernel[(tokens, hkv)] with the same arguments when cfg q_len == 8 (config-only predicate: every verify block, sliding and full, paired and unpaired, including the sliding verify B=1 pseudo-decode route, which consumes the same unpaired FP8 q and stored rows), and master's _pre_attention_kernel[(tokens, hq + hkv)] num_warps 1 otherwise (decode at every batch keeps master byte-for-byte). Nothing else changes: the self-check helpers (_sk_prepare_q, _ps_prepare_q, _skf references) keep calling master's kernel, and every prepare-time gate, route and CUDA module is untouched. Rules: no host sync, no allocation or branch on input values (q_len is config), nothing cached from inputs, KV stays E4M3 and q is E4M3, every key is attended (attention untouched).

## Expected

- `*layer=sliding load=verify_32 *` faster: measured twice: pre-attention 4.2 -> 2.7 us, shape 28.70 -> 27.55 us (1.0385 paired) on c2ebf6ed; master still shows 4.2 us on this row (job d672f69e). 46-weight row, ~0.12%/us of headline
- `*layer=sliding load=verify_16 *` faster: parent run 877731a3: 2.8 -> 2.0 us on the kernel row; master shows 2.8 us. ~0.21%/us of headline
- `*load=verify_1 *` flat: 8 tokens: parent profile measured the tiled kernel 0.1 us faster (sliding 1.7 -> 1.6, full 1.5 -> 1.4); within noise but on the 0.7%/us sliding verify_1 row it cannot hurt
- `*layer=full load=verify_16 * *layer=full load=verify_32 *` flat: PAIRED layout: parent saw only 2.8 -> 2.7 us at full verify_16; ~1 us out of 163/301 us
- `*load=decode_*` flat: q_len 1: master's kernel (or the cluster / Triton paths that bypass it) runs unchanged

## Risk

Low risk: the kernel is copied from a patch that passed every test and every timed-call check twice (877731a3, c2ebf6ed), including the PAIRED full-verify layout. Widening the gate to q_len == 8 means 7 of the 16 pre-tests (q_len 8, page 1 and 64, tp8 and dp, kv_scale 1/64) now exercise it before timing, which removes the previous run's residual risk that only the scored calls covered it; the auditor should confirm G = Hq/Hkv is a power of two under dp as well as tp8 (the parent audit said G = 8 for both). Expected headline effect ~+0.3-0.4%, inside the 1.5% CI by construction; judge it on the _pre_attention_tiled_kernel row and on sliding verify_16/32 paired ratios, not the headline. Stacks with restack-sliding-decode-sk-window-on-cluster-master and sliding-decode-sk-s3-fused-preattn-rope (decode only, q_len 1) without overlap.

## Adversarial review

Reviewed 2026-10-25T203045Z by an adversarial subagent (model opus).

**Verdict: hold.** Sixth scored run of the same _pre_attention_tiled_kernel; its parent preattention-tiled-gate-large-tokens is on Do-not-try. The only new part (gate widened to SV1/SV16/FV1/FV16/FV32) was measured flat or slower per class in combo-naive-c320-fd1-tiled-preattn-verify's whole-step gate (log b9d062aaa94c4cbf l.158-163); the one winning class (SV32) is already carried by merge-perkvhead-c320-sv32-tiled (1.5347x).

### Justification
Tiled cuts pre-attention at SV32 (4.1-4.2 -> 2.7 us; rows 1.0385/1.0456/1.0416/1.0425). Weakest link: 'SV16 faster' -- four judged runs 0.9955/1.0007/0.9955/1.0107. Gate step ratios tiled/master: SV1 1.013, SV16 1.008, FV1 1.009, FV16 1.005, FV32 0.999, SV32 0.964. No prior field although five earlier runs used this kernel; the change only removes the timing gate. Base stale (eb1b68b vs 5d528bf). Expected ~1.51-1.52, below best 1.5295.

### Confounders on the current kernel
- restack-sk-pseudo-cluster-plus-v3-fused-preattn: SV1 runs the PSEUDO stream-K route; tiled q into it measured 1.013x slower.
- fa3-hdim128-head-pairing / paired-heads-launch-cleanup: FV16/FV32 PAIRED layout, tiled gains ~0.1 us (noise).
- fused-preattention-triton: incumbent 1-warp grid fills better at low token counts, where tiled loses.
- merge-perkvhead-c320-sv32-tiled (not landed): already routes tiled at SV32.

### Interference and A/B needs
No companion: every touched row already has a recent per-class whole-step measurement. The worthwhile follow-up is the confirm of merge-perkvhead-c320-sv32-tiled.

### Open questions
- Why does the SV16 kernel saving (2.8 -> 2.0 us) not reach the step? Profile start-to-start gaps first.
- Small-token tiled variant for SV1, or fold pre-attention into the PSEUDO kernel?

