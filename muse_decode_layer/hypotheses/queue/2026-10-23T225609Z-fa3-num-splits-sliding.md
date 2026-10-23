---
problem: muse_decode_layer
slug: fa3-num-splits-sliding
created: 2026-10-23T225609Z
base: master
area: FA3 call
change: >
  On sliding-window layers only (cfg['window'] == 2048, where the judge passes
  scheduler_metadata=None and SGLang lets FA3 choose), pass an explicit num_splits to
  flash_attn_with_kvcache instead of leaving it at 0. FA3's host-side split heuristic sees
  seqlen_k as the page table's full extent (max_pages * page_size, i.e. the whole ~80K context)
  because cache_seqlens is a device tensor it cannot read, and sizes the split-KV decomposition
  for 600+ key tiles, while the window only ever touches 16 tiles of 128 keys per (batch,
  kv_head): the extra splits produce empty partials plus a larger combine kernel. Choose
  num_splits by shape on the host, from batch B and kv_heads=2: num_splits = clamp(ceil(132 / (B *
  2)), 1, 16) so that B=1 gets 16 (2 CTAs x 16 splits over 16 tiles, one tile each), B=16 gets 4
  and B=32 gets 2-3; q_len=8 verify blocks use the same formula. The full-attention branch keeps
  scheduler_metadata and num_splits=0 untouched. Nothing else changes. Attribution: sliding shapes
  only (weight 46 of 62 per load); full shapes are the control.
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: "B=1: 2 CTAs with a 16-tile window; a sized split (16) replaces a heuristic sized for 625 tiles and its oversized combine"
  - shapes: "*layer=sliding load=verify_1*"
    direction: faster
    why: same, with 8 queries per request
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "B=16: 32 CTAs; fewer, non-empty splits and a smaller combine"
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "B=32: 64 CTAs; num_splits 2-3 instead of the heuristic pick"
  - shapes: "*layer=full*"
    direction: flat
    why: full branch untouched (metadata passed through); control
risk: medium
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-num-splits-sliding

## Change

On sliding-window layers only (cfg['window'] == 2048, where the judge passes scheduler_metadata=None and SGLang lets FA3 choose), pass an explicit num_splits to flash_attn_with_kvcache instead of leaving it at 0. FA3's host-side split heuristic sees seqlen_k as the page table's full extent (max_pages * page_size, i.e. the whole ~80K context) because cache_seqlens is a device tensor it cannot read, and sizes the split-KV decomposition for 600+ key tiles, while the window only ever touches 16 tiles of 128 keys per (batch, kv_head): the extra splits produce empty partials plus a larger combine kernel. Choose num_splits by shape on the host, from batch B and kv_heads=2: num_splits = clamp(ceil(132 / (B * 2)), 1, 16) so that B=1 gets 16 (2 CTAs x 16 splits over 16 tiles, one tile each), B=16 gets 4 and B=32 gets 2-3; q_len=8 verify blocks use the same formula. The full-attention branch keeps scheduler_metadata and num_splits=0 untouched. Nothing else changes. Attribution: sliding shapes only (weight 46 of 62 per load); full shapes are the control.

## Expected

- `*layer=sliding load=decode_1*` faster: B=1: 2 CTAs with a 16-tile window; a sized split (16) replaces a heuristic sized for 625 tiles and its oversized combine
- `*layer=sliding load=verify_1*` faster: same, with 8 queries per request
- `*layer=sliding load=decode_16*` faster: B=16: 32 CTAs; fewer, non-empty splits and a smaller combine
- `*layer=sliding load=decode_32*` faster: B=32: 64 CTAs; num_splits 2-3 instead of the heuristic pick
- `*layer=full*` flat: full branch untouched (metadata passed through); control

## Risk

Risk: the mechanism assumes FA3's heuristic ignores the window when it picks num_splits on this build (hopper flash_api get_num_splits uses seqlen_k and may or may not clamp by window_size); if it already clamps, this is flat on every shape, which is an informative absence. Over-splitting at B=32 (num_splits > tiles per CTA) wastes a combine pass, so the formula caps at 16 and never exceeds 16 tiles. Correctness: num_splits only changes the reduction order of the split-KV combine (fp32 partials), well within the 3-15% bound. CUDA-graph safe: num_splits is computed from config/batch on the host, never from tensor values. Do not pass num_splits together with scheduler_metadata on full decode; FA3 asserts they agree.
