---
problem: muse_attention_decode
slug: fa3-single-split-sliding
created: 2026-10-23T225858Z
base: master
area: FA3 call
change: >
  On sliding-window layers only (config window != 0) pass num_splits=1 to flash_attn_with_kvcache;
  everything else in the call (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q,
  softmax_scale, causal, window_size=(window-1, 0), scheduler_metadata=None, descale) stays as in
  the parity pipeline, and full layers are untouched. Mechanism (launch count, not balance): with
  num_splits left to FA3 on a sliding layer, FA3 runs its dynamic split-KV path, which is up to
  three launches (prepare_varlen_num_blocks, the split main kernel writing fp32 partials and LSE,
  flash_fwd_combine); with num_splits=1 the main kernel writes the bf16 output directly and the
  combine is skipped, and with an explicit split count the dynamic prepare pass has nothing to
  decide. On a 2,048-token window each (request, kv head) tile streams at most 16 blocks of 128
  rows x 128 B = 256 KB, so a single CTA per tile finishes in a few microseconds; the trade is
  that single-CTA streaming time against two launches of ~2.3 us each on shapes that are 21-35 us
  and launch-bound. This differs from fa3-num-splits-sliding, which keeps several splits per tile
  (16/9/5) and tests balance; this tests whether any split at all is worth its combine on sliding
  layers.
expected:
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: 32 tiles x 256 KB on 32 SMs is ~3-4 us of streaming; the combine and prepare launches (~4-5 us) go
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: 64 tiles, same mechanism
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: same tiles at q_len 8; the FA3 per-tile compute at 64 packed query rows is still memory-trivial
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: same
  - shapes: "*layer=sliding load=decode_1 page*"
    direction: flat
    why: "only 2 CTAs stream 512 KB in total: each pulls 256 KB alone, ~3-5 us, about what the two launches cost; a loss here attributes to single-CTA bandwidth"
  - shapes: "*layer=sliding load=verify_1 page*"
    direction: flat
    why: same as decode_1
  - shapes: "*layer=full*"
    direction: flat
    why: untouched; control
risk: low
depends_on: fused-preattention-triton
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-single-split-sliding

## Change

On sliding-window layers only (config window != 0) pass num_splits=1 to flash_attn_with_kvcache; everything else in the call (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q, softmax_scale, causal, window_size=(window-1, 0), scheduler_metadata=None, descale) stays as in the parity pipeline, and full layers are untouched. Mechanism (launch count, not balance): with num_splits left to FA3 on a sliding layer, FA3 runs its dynamic split-KV path, which is up to three launches (prepare_varlen_num_blocks, the split main kernel writing fp32 partials and LSE, flash_fwd_combine); with num_splits=1 the main kernel writes the bf16 output directly and the combine is skipped, and with an explicit split count the dynamic prepare pass has nothing to decide. On a 2,048-token window each (request, kv head) tile streams at most 16 blocks of 128 rows x 128 B = 256 KB, so a single CTA per tile finishes in a few microseconds; the trade is that single-CTA streaming time against two launches of ~2.3 us each on shapes that are 21-35 us and launch-bound. This differs from fa3-num-splits-sliding, which keeps several splits per tile (16/9/5) and tests balance; this tests whether any split at all is worth its combine on sliding layers.

## Expected

- `*layer=sliding load=decode_16*` faster: 32 tiles x 256 KB on 32 SMs is ~3-4 us of streaming; the combine and prepare launches (~4-5 us) go
- `*layer=sliding load=decode_32*` faster: 64 tiles, same mechanism
- `*layer=sliding load=verify_16*` faster: same tiles at q_len 8; the FA3 per-tile compute at 64 packed query rows is still memory-trivial
- `*layer=sliding load=verify_32*` faster: same
- `*layer=sliding load=decode_1 page*` flat: only 2 CTAs stream 512 KB in total: each pulls 256 KB alone, ~3-5 us, about what the two launches cost; a loss here attributes to single-CTA bandwidth
- `*layer=sliding load=verify_1 page*` flat: same as decode_1
- `*layer=full*` flat: untouched; control

## Risk

Low risk, one keyword argument; correctness unaffected (reduction order only). If FA3 still launches the prepare kernel with an explicit num_splits (it may, for the varlen tile scheduler's semaphore), only the combine launch is saved and the gain halves. If FA3's dynamic path already picked 1 split on sliding layers at batch 32, decode_32/verify_32 come out flat and the gain is confined to the smaller batches. Interacts with fa3-window-trimmed-page-table and fa3-num-splits-sliding (all three touch the sliding split schedule); each is a one-area change on the parity base so they attribute separately, and the reviewer may want them run in sequence rather than stacked.
