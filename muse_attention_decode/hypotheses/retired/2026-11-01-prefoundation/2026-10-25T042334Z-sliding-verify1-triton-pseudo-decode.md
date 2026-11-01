---
problem: muse_attention_decode
slug: sliding-verify1-triton-pseudo-decode
created: 2026-10-25T042334Z
base: master
area: sliding-layer attention (batch-1 verify, Triton fused decode kernel)
change: >
  Built on master. At sliding verify batch 1 only (window != 0, q_len 8, batch 1, page 1;
  config-only predicate) replace the FA3 path (prepare_varlen + FA3 main + combine +
  rms_norm_gate, ~14.3 us of the 16.7 us shape) with two launches: master _pre_attention_kernel
  (unchanged, ROPE on, PAIRED=False; writes the 8 new K/V rows and an FP8 q we ignore) then master
  _sliding_fused_kernel over 8 pseudo-requests, one per verify token, with q_len 1 decode math.
  Kernel edit: new constexpr PSEUDO (0 = master behaviour, byte-identical). With PSEUDO=8, program
  (p, g, s) maps to request b = p // 8, token j = p % 8: seq = seq_lens[b] - 7 + j, q and gate
  read from qkv row b*8 + j (tok = p), position from positions[b*8 + j], page_table row b, output
  row b*8+j; the new-key build/store is skipped (the row is already in cache, written by the
  pre-attention kernel) and that key is read from cache like any other, so no CTA reads a row
  another CTA of the same launch writes. Window mask and split coverage are the decode ones for
  seq (key t valid iff seq - window <= t < seq). Grid (8, 2, 16) = 256 programs, splits 16,
  counters _counters(8, hkv) keyed so they do not collide with decode batch 8 (use a distinct key
  tag). Batch > 1 verify keeps FA3 (B=16 would be 128 pseudo-requests, ~4x decode_32 work).
expected:
  - shapes: "*layer=sliding load=verify_1 *"
    direction: faster
    why: pre-attn ~1.3 us + fused kernel at 8 requests (between decode_1 10.4 and decode_16 12.9 kernel) ~13 us vs 16.7 shape; window re-reads (8 x 0.5 MB) hit L2. Sliding is 62% of the verify_1 load (weakest loads are verify at ~1.27-1.30x)
  - shapes: "*layer=sliding load=verify_16* *layer=sliding load=verify_32* *layer=sliding load=decode_* *layer=full*"
    direction: flat
    why: outside the predicate; PSEUDO=0 instantiation is master
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-25T043143Z
review_companions: null
hold_reason: "adversarial review: HOLD for two reasons. (1) The key range is wrong: seq = seq_lens[b]-7+j treats seq_lens as the post-block length, but seq_lens is the prefix length (positions ="
---

# sliding-verify1-triton-pseudo-decode

## Change

Built on master. At sliding verify batch 1 only (window != 0, q_len 8, batch 1, page 1; config-only predicate) replace the FA3 path (prepare_varlen + FA3 main + combine + rms_norm_gate, ~14.3 us of the 16.7 us shape) with two launches: master _pre_attention_kernel (unchanged, ROPE on, PAIRED=False; writes the 8 new K/V rows and an FP8 q we ignore) then master _sliding_fused_kernel over 8 pseudo-requests, one per verify token, with q_len 1 decode math. Kernel edit: new constexpr PSEUDO (0 = master behaviour, byte-identical). With PSEUDO=8, program (p, g, s) maps to request b = p // 8, token j = p % 8: seq = seq_lens[b] - 7 + j, q and gate read from qkv row b*8 + j (tok = p), position from positions[b*8 + j], page_table row b, output row b*8+j; the new-key build/store is skipped (the row is already in cache, written by the pre-attention kernel) and that key is read from cache like any other, so no CTA reads a row another CTA of the same launch writes. Window mask and split coverage are the decode ones for seq (key t valid iff seq - window <= t < seq). Grid (8, 2, 16) = 256 programs, splits 16, counters _counters(8, hkv) keyed so they do not collide with decode batch 8 (use a distinct key tag). Batch > 1 verify keeps FA3 (B=16 would be 128 pseudo-requests, ~4x decode_32 work).

## Expected

- `*layer=sliding load=verify_1 *` faster: pre-attn ~1.3 us + fused kernel at 8 requests (between decode_1 10.4 and decode_16 12.9 kernel) ~13 us vs 16.7 shape; window re-reads (8 x 0.5 MB) hit L2. Sliding is 62% of the verify_1 load (weakest loads are verify at ~1.27-1.30x)
- `*layer=sliding load=verify_16* *layer=sliding load=verify_32* *layer=sliding load=decode_* *layer=full*` flat: outside the predicate; PSEUDO=0 instantiation is master

## Risk

Kill: sliding verify_1 shape not below 16.0 us (kernel rows pre-attn + fused >= FA3 sum). Risk: two launches instead of one gap; position/rope must use each token own position.

## Adversarial review

Reviewed 2026-10-25T043143Z by an adversarial subagent (model opus).

**Verdict: hold.** HOLD for two reasons. (1) The key range is wrong: seq = seq_lens[b]-7+j treats seq_lens as the post-block length, but seq_lens is the prefix length (positions = seq_lens+j, cache_seqlens = seq_lens+8). Each pseudo-query would attend a window shifted 8 keys into the past, missing its own and the preceding block keys: an every-key/causal rule break. (2) It duplicates the B=1 band of sliding-verify-sk-pseudo-decode-s1 on the same row. Unhold, with the fix below, only if that run's B=1 band falls back to FA3 or lands above ~12.5 us shape.

### Justification
Mechanism (once fixed): 8 pseudo-requests x 2 kv heads x 16 splits = 256 programs. Each does about the same work per program as master's decode (~128 keys), so the kernel should sit between decode_1 (10.4 us, 32 programs) and decode_16 (12.9 us, 512 programs), call it ~11 us. Add pre-attn ~1.3 us and one launch gap, ~13-13.5 us, against the 16.7 us shape. That is a plausible ~20% on sliding verify_1 (weight 46, 62% of the verify_1 load). WEAKEST LINK: the arithmetic error above, plus the fact that the 16 tests never exercise this route (no sliding q_len 8 test at batch 1). The first correctness check would be the judge's timed-call check, after a full run is spent.

### Confounders on the current kernel
- sliding-fused-triton-decode1-gate / sliding-deep-splits-on-cuda-splitkv (landed): this edits _sliding_fused_kernel, which serves every sliding decode row. PSEUDO=0 must be the master specialization (constexpr-gated, so the Triton cache key differs and the decode code is unchanged). Expect no decode movement.
- fused-preattention-triton (landed): it writes the 8 new K/V rows before this kernel. Correctness now depends on that store (the S-1 new-key branch is skipped), so its cache rows are the attended keys. That matches what the judge checks ('reference attending to your own new entries').
- sliding-verify-sk-pseudo-decode-s1 (queued, p1): same row, same idea (verify token = decode query), different kernel. Running both now buys a second implementation of one row with no new information unless the CUDA one loses at B=1.

### Interference and A/B needs
No companion while held. If unheld, it is attributable on its own: only sliding verify_1 changes, against an FA3 base row, and every other row is master code.

### Open questions
- UNHOLD CONDITION: sliding-verify-sk-pseudo-decode-s1 prints B=1 band FA3 (fallback/timing) or its routed sliding verify_1 shape is > ~12.5 us. Then run this with the corrected range and an import-time fp32 windowed-causal self-check at B=1 (lens 5, 2000, 2050, 80000; kv_scale 1 and 0.25), failure keeping FA3.
- If unheld, consider S=8 vs 16 splits per pseudo-request as a prepare-timed choice (256 vs 128 programs). Decode evidence (sliding-decode32-splits-8) says fewer splits can win when programs exceed ~256.

