---
problem: muse_attention_decode
slug: fa3-batched-q-verify
created: 2026-10-23T225858Z
base: master
area: FA3 call (q layout)
change: >
  On verify blocks only (config q_len == 8), call flash_attn_with_kvcache with q as the batched
  4-D tensor q.view(B, 8, Hq, 64) and cu_seqlens_q=None, max_seqlen_q=None, instead of the varlen
  form q.view(B*8, Hq, 64) + cu_seqlens_q + max_seqlen_q=8 that the parity pipeline (and SGLang)
  uses. Every request has exactly 8 queries, so the two forms describe the same batch; the varlen
  form, though, makes FA3 take its varlen-q path (cu_seqlens reads per tile, per-request offset
  arithmetic, non-TMA Q loads, the varlen tile scheduler) while the batched form lets FA3 load the
  [8 x Hq] query block per request through its TMA path and use the static batched scheduler.
  page_table, cache_seqlens, softmax_scale, causal=True, window_size, scheduler_metadata (already
  None on verify blocks) and descale stay as they are. The output is [B, 8, Hq, 64]; reshape it to
  (B*8, Hq*64) for rms_norm_gate as before. Decode steps (q_len 1) are untouched; this is the
  verify-block sibling of the queued fa3-drop-cu-seqlens-decode and never shares a shape with it.
expected:
  - shapes: "*layer=sliding load=verify_1 page*"
    direction: faster
    why: "23 us launch-bound shape: varlen bookkeeping is a per-call fixed cost; small, possibly inside the CI"
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: same, 16 requests x 8 queries
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: same
  - shapes: "*layer=full load=verify_16*"
    direction: flat
    why: 205 us FA3-bound over 16 x 80K tokens; the Q-load path is not the bottleneck, so a change here would attribute to the scheduler difference, not the Q load
  - shapes: "*layer=full load=verify_32*"
    direction: flat
    why: same, 390 us
  - shapes: "*layer=full load=verify_1 page*"
    direction: faster
    why: "35.6 us shape with 2 tiles: the same fixed cost as the sliding verify_1"
  - shapes: "*load=decode_*"
    direction: flat
    why: untouched; control
risk: low
depends_on: sglang-path-parity
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-batched-q-verify

## Change

On verify blocks only (config q_len == 8), call flash_attn_with_kvcache with q as the batched 4-D tensor q.view(B, 8, Hq, 64) and cu_seqlens_q=None, max_seqlen_q=None, instead of the varlen form q.view(B*8, Hq, 64) + cu_seqlens_q + max_seqlen_q=8 that the parity pipeline (and SGLang) uses. Every request has exactly 8 queries, so the two forms describe the same batch; the varlen form, though, makes FA3 take its varlen-q path (cu_seqlens reads per tile, per-request offset arithmetic, non-TMA Q loads, the varlen tile scheduler) while the batched form lets FA3 load the [8 x Hq] query block per request through its TMA path and use the static batched scheduler. page_table, cache_seqlens, softmax_scale, causal=True, window_size, scheduler_metadata (already None on verify blocks) and descale stay as they are. The output is [B, 8, Hq, 64]; reshape it to (B*8, Hq*64) for rms_norm_gate as before. Decode steps (q_len 1) are untouched; this is the verify-block sibling of the queued fa3-drop-cu-seqlens-decode and never shares a shape with it.

## Expected

- `*layer=sliding load=verify_1 page*` faster: 23 us launch-bound shape: varlen bookkeeping is a per-call fixed cost; small, possibly inside the CI
- `*layer=sliding load=verify_16*` faster: same, 16 requests x 8 queries
- `*layer=sliding load=verify_32*` faster: same
- `*layer=full load=verify_16*` flat: 205 us FA3-bound over 16 x 80K tokens; the Q-load path is not the bottleneck, so a change here would attribute to the scheduler difference, not the Q load
- `*layer=full load=verify_32*` flat: same, 390 us
- `*layer=full load=verify_1 page*` faster: 35.6 us shape with 2 tiles: the same fixed cost as the sliding verify_1
- `*load=decode_*` flat: untouched; control

## Risk

Low risk, low expected magnitude: a one-line change that establishes whether the varlen-q form costs anything at L = 8. Correctness unaffected (same math); the batched form requires FA3 to accept paged KV with a 4-D q and cache_seqlens, which is its documented decode API. If FA3 rejects the batched q together with page_size 1 or the dp head counts, the tests fail and the hypothesis is failed, which is itself the answer. Companion to fa3-drop-cu-seqlens-decode: if both land, every shape uses the batched form.
