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
review: required
status: queued
---

# sliding-verify1-triton-pseudo-decode

## Change

Built on master. At sliding verify batch 1 only (window != 0, q_len 8, batch 1, page 1; config-only predicate) replace the FA3 path (prepare_varlen + FA3 main + combine + rms_norm_gate, ~14.3 us of the 16.7 us shape) with two launches: master _pre_attention_kernel (unchanged, ROPE on, PAIRED=False; writes the 8 new K/V rows and an FP8 q we ignore) then master _sliding_fused_kernel over 8 pseudo-requests, one per verify token, with q_len 1 decode math. Kernel edit: new constexpr PSEUDO (0 = master behaviour, byte-identical). With PSEUDO=8, program (p, g, s) maps to request b = p // 8, token j = p % 8: seq = seq_lens[b] - 7 + j, q and gate read from qkv row b*8 + j (tok = p), position from positions[b*8 + j], page_table row b, output row b*8+j; the new-key build/store is skipped (the row is already in cache, written by the pre-attention kernel) and that key is read from cache like any other, so no CTA reads a row another CTA of the same launch writes. Window mask and split coverage are the decode ones for seq (key t valid iff seq - window <= t < seq). Grid (8, 2, 16) = 256 programs, splits 16, counters _counters(8, hkv) keyed so they do not collide with decode batch 8 (use a distinct key tag). Batch > 1 verify keeps FA3 (B=16 would be 128 pseudo-requests, ~4x decode_32 work).

## Expected

- `*layer=sliding load=verify_1 *` faster: pre-attn ~1.3 us + fused kernel at 8 requests (between decode_1 10.4 and decode_16 12.9 kernel) ~13 us vs 16.7 shape; window re-reads (8 x 0.5 MB) hit L2. Sliding is 62% of the verify_1 load (weakest loads are verify at ~1.27-1.30x)
- `*layer=sliding load=verify_16* *layer=sliding load=verify_32* *layer=sliding load=decode_* *layer=full*` flat: outside the predicate; PSEUDO=0 instantiation is master

## Risk

Kill: sliding verify_1 shape not below 16.0 us (kernel rows pre-attn + fused >= FA3 sum). Risk: two launches instead of one gap; position/rope must use each token own position.
