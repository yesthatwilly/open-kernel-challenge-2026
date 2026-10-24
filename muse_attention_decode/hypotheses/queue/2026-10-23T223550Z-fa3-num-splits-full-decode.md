---
problem: muse_attention_decode
slug: fa3-num-splits-full-decode
created: 2026-10-23T223550Z
base: master
area: FA3 call
change: >
  On full-attention decode steps (config window == 0 and q_len == 1, where the judge supplies
  FA3's scheduler_metadata built with num_splits=0, i.e. the heuristic), stop passing
  scheduler_metadata (pass None) and pass an explicit num_splits chosen from the shape so every SM
  gets at least two KV-split work items: num_splits = clamp(ceil(2*132 / (batch*num_kv_heads)), 1,
  128) -> 128 at batch 1 (2 tiles over 625 blocks of 128 tokens), 9 at batch 16, 5 at batch 32.
  Everything else in the FA3 call (page_table, cache_seqlens, cu_seqlens_q, causal, window
  (-1,-1), softmax_scale, descale) stays as in the parity pipeline; sliding layers and verify
  blocks are untouched. The mechanism under test: at batch 16/32 FA3 runs at ~1.7 TB/s effective
  (201 us for 16 x 80K x 2 heads x 128 B k+v, 388 us at 32) against a 4.8 TB/s floor; if the
  heuristic under-splits at these batch sizes (32 or 64 tiles on 132 SMs, each tile streaming 20
  MB of scattered 64-byte page rows), more splits raise memory-level parallelism; if it
  over-splits, fewer splits shrink the combine over 16 x 80K-long partial results.
expected:
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 201 us vs 63 us floor; 32 tiles cannot saturate HBM without splits; explicit splits give 288 work items
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: 388 us vs 126 us floor; 64 tiles, 320 work items
  - shapes: "*layer=full load=decode_1*"
    direction: flat
    why: "2 tiles: the heuristic already splits heavily; 128 vs its pick should be close. Also measures the cost of dropping the precomputed schedule (FA3 then computes the schedule in-kernel)"
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched
  - shapes: "*load=verify_*"
    direction: flat
    why: untouched (scheduler_metadata is already None there)
risk: medium
depends_on: fused-preattention-triton
priority: 4
kind: primary
controls_for: null
review: required
status: hold
hold_reason: num_splits is only a cap under FA3's on-device split sizing (LESSONS; the decode_layer equivalents were held for the same reason), and full layers are now paired, which changes FA3's split decision anyway
---

# fa3-num-splits-full-decode

## Change

On full-attention decode steps (config window == 0 and q_len == 1, where the judge supplies FA3's scheduler_metadata built with num_splits=0, i.e. the heuristic), stop passing scheduler_metadata (pass None) and pass an explicit num_splits chosen from the shape so every SM gets at least two KV-split work items: num_splits = clamp(ceil(2*132 / (batch*num_kv_heads)), 1, 128) -> 128 at batch 1 (2 tiles over 625 blocks of 128 tokens), 9 at batch 16, 5 at batch 32. Everything else in the FA3 call (page_table, cache_seqlens, cu_seqlens_q, causal, window (-1,-1), softmax_scale, descale) stays as in the parity pipeline; sliding layers and verify blocks are untouched. The mechanism under test: at batch 16/32 FA3 runs at ~1.7 TB/s effective (201 us for 16 x 80K x 2 heads x 128 B k+v, 388 us at 32) against a 4.8 TB/s floor; if the heuristic under-splits at these batch sizes (32 or 64 tiles on 132 SMs, each tile streaming 20 MB of scattered 64-byte page rows), more splits raise memory-level parallelism; if it over-splits, fewer splits shrink the combine over 16 x 80K-long partial results.

## Expected

- `*layer=full load=decode_16*` faster: 201 us vs 63 us floor; 32 tiles cannot saturate HBM without splits; explicit splits give 288 work items
- `*layer=full load=decode_32*` faster: 388 us vs 126 us floor; 64 tiles, 320 work items
- `*layer=full load=decode_1*` flat: 2 tiles: the heuristic already splits heavily; 128 vs its pick should be close. Also measures the cost of dropping the precomputed schedule (FA3 then computes the schedule in-kernel)
- `*layer=sliding*` flat: untouched
- `*load=verify_*` flat: untouched (scheduler_metadata is already None there)

## Risk

Medium risk. Dropping the judge's scheduler_metadata removes FA3's precomputed dynamic tile schedule; if the heuristic was already right, decode_1 full gets slightly slower (an extra in-kernel scheduling pass), which this hypothesis would show as a regression on that shape alone. The split count formula is a first guess; the reviewer may add an ab-tune companion at 2x and 0.5x the split counts. Correctness unaffected: reduction order only. Note the full-decode shapes carry weight 16 but at batch 16/32 they are 2.6-4.3x the sliding shapes' time, so they dominate four of the six loads.
