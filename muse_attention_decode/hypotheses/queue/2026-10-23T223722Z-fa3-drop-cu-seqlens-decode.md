---
problem: muse_attention_decode
slug: fa3-drop-cu-seqlens-decode
created: 2026-10-23T223722Z
base: master
area: FA3 call
change: >
  On decode steps (config q_len == 1), call flash_attn_with_kvcache with q as [B, 1, Hq, 64] and
  cu_seqlens_q=None (max_seqlen_q unset) instead of the varlen form q [B*L, Hq, 64] + cu_seqlens_q
  + max_seqlen_q that the parity pipeline (and SGLang) uses; since L = 1 the two describe the same
  batch, but the varlen form makes FA3 take its varlen-q code path (cu_seqlens reads, per-request
  offset arithmetic, varlen tile scheduler). Everything else in the call (page_table,
  cache_seqlens, causal, window_size, softmax_scale, scheduler_metadata, descale) stays. Verify
  blocks (q_len 8) are untouched. The output is reshaped to [B*1, Hq*64] for rms_norm_gate as
  before.
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: "21 us launch-bound shape: the varlen bookkeeping is a fixed cost per call; small, inside the CI if nothing else"
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: same, 16 requests
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: same
  - shapes: "*layer=full load=decode_*"
    direction: flat
    why: FA3-bound at batch 16/32; decode_1 may gain the same fixed cost. Also a check that the judge-built scheduler_metadata (made with cu_seqlens_q) is still accepted on the non-varlen call
  - shapes: "*load=verify_*"
    direction: flat
    why: untouched
risk: low
depends_on: sglang-path-parity
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-drop-cu-seqlens-decode

## Change

On decode steps (config q_len == 1), call flash_attn_with_kvcache with q as [B, 1, Hq, 64] and cu_seqlens_q=None (max_seqlen_q unset) instead of the varlen form q [B*L, Hq, 64] + cu_seqlens_q + max_seqlen_q that the parity pipeline (and SGLang) uses; since L = 1 the two describe the same batch, but the varlen form makes FA3 take its varlen-q code path (cu_seqlens reads, per-request offset arithmetic, varlen tile scheduler). Everything else in the call (page_table, cache_seqlens, causal, window_size, softmax_scale, scheduler_metadata, descale) stays. Verify blocks (q_len 8) are untouched. The output is reshaped to [B*1, Hq*64] for rms_norm_gate as before.

## Expected

- `*layer=sliding load=decode_1*` faster: 21 us launch-bound shape: the varlen bookkeeping is a fixed cost per call; small, inside the CI if nothing else
- `*layer=sliding load=decode_16*` faster: same, 16 requests
- `*layer=sliding load=decode_32*` faster: same
- `*layer=full load=decode_*` flat: FA3-bound at batch 16/32; decode_1 may gain the same fixed cost. Also a check that the judge-built scheduler_metadata (made with cu_seqlens_q) is still accepted on the non-varlen call
- `*load=verify_*` flat: untouched

## Risk

Low risk, low expected magnitude: a one-line change meant to establish whether the varlen-q form costs anything at L=1. The scheduler_metadata was built by the judge with cu_seqlens_q set; if FA3 rejects or mis-schedules the non-varlen call with that metadata on full decode layers, the tests fail and the hypothesis is 'failed', which is itself the answer (then try with scheduler_metadata=None on the sliding layers only, which never had metadata). Likely outcome: inside the CI on every shape.
