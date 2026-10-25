---
problem: muse_attention_decode
slug: fa3-num-splits-full-verify
created: 2026-10-23T225858Z
base: master
area: FA3 call
change: >
  On full-attention verify blocks only (config window == 0 and q_len == 8, where the judge passes
  scheduler_metadata=None and FA3's own heuristic picks the split count from max_seqlen_k =
  page_table width), pass an explicit num_splits = clamp(ceil(2*132 / (batch*num_kv_heads)), 1,
  128) to flash_attn_with_kvcache: 128 at batch 1, 9 at batch 16, 5 at batch 32, so every SM holds
  at least two (request, kv head, split) work items. Everything else in the call (page_table,
  cache_seqlens, cu_seqlens_q, max_seqlen_q=8, softmax_scale, causal=True, window_size=(-1,-1),
  scheduler_metadata=None, descale) stays as in the parity pipeline; decode steps keep the
  judge-built metadata untouched, sliding layers untouched. Mechanism: at q_len 8 FA3's heuristic
  sees 64 packed query rows per tile (8 tokens x 8 heads) and may weigh the extra compute per
  split against the combine, choosing fewer splits than at q_len 1 even though the shape is still
  memory-bound (205 us for 328 MB at batch 16, 390 us for 655 MB at batch 32, against a 4.8 TB/s
  floor); an explicit count forces the same SM fill the queued decode variant tests. This is the
  verify-block sibling of fa3-num-splits-full-decode and the explicit-split alternative to
  fa3-scheduler-metadata-full-verify; it never shares a shape with the first and the reviewer may
  run it against the second.
expected:
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: 205 us vs 63 us floor; 32 tiles on 132 SMs need splits to fill the GPU; 288 work items with 9 splits
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: 390 us vs 127 us floor; 64 tiles, 320 work items
  - shapes: "*layer=full load=verify_1 page*"
    direction: flat
    why: "2 tiles: the heuristic already splits heavily; 128 vs its pick should be close"
  - shapes: "*load=decode_*"
    direction: flat
    why: untouched; control
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched; control
risk: medium
depends_on: fused-preattention-triton
priority: 5
kind: primary
controls_for: null
review: required
status: done
hold_reason: "moot under master 1.269x (sweep by the attention minter): num_splits is only a cap under FA3's on-device split sizing, and full verify at batch>1 is now paired with scheduler_metadata=None"
outcome: expired
verdict_detail: "hold expired after 33 h: moot under master 1.269x (sweep by the attention minter): num_splits is only a cap under FA3's on-device split sizing, a"
expired_at: 2026-10-25T211327Z
---

# fa3-num-splits-full-verify

## Change

On full-attention verify blocks only (config window == 0 and q_len == 8, where the judge passes scheduler_metadata=None and FA3's own heuristic picks the split count from max_seqlen_k = page_table width), pass an explicit num_splits = clamp(ceil(2*132 / (batch*num_kv_heads)), 1, 128) to flash_attn_with_kvcache: 128 at batch 1, 9 at batch 16, 5 at batch 32, so every SM holds at least two (request, kv head, split) work items. Everything else in the call (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q=8, softmax_scale, causal=True, window_size=(-1,-1), scheduler_metadata=None, descale) stays as in the parity pipeline; decode steps keep the judge-built metadata untouched, sliding layers untouched. Mechanism: at q_len 8 FA3's heuristic sees 64 packed query rows per tile (8 tokens x 8 heads) and may weigh the extra compute per split against the combine, choosing fewer splits than at q_len 1 even though the shape is still memory-bound (205 us for 328 MB at batch 16, 390 us for 655 MB at batch 32, against a 4.8 TB/s floor); an explicit count forces the same SM fill the queued decode variant tests. This is the verify-block sibling of fa3-num-splits-full-decode and the explicit-split alternative to fa3-scheduler-metadata-full-verify; it never shares a shape with the first and the reviewer may run it against the second.

## Expected

- `*layer=full load=verify_16*` faster: 205 us vs 63 us floor; 32 tiles on 132 SMs need splits to fill the GPU; 288 work items with 9 splits
- `*layer=full load=verify_32*` faster: 390 us vs 127 us floor; 64 tiles, 320 work items
- `*layer=full load=verify_1 page*` flat: 2 tiles: the heuristic already splits heavily; 128 vs its pick should be close
- `*load=decode_*` flat: untouched; control
- `*layer=sliding*` flat: untouched; control

## Risk

Medium risk, one keyword argument; correctness unaffected (reduction order only). If FA3's heuristic already chose these counts the result is flat and the hypothesis is rejected cleanly; if it over-split, verify_16/32 get slower by the combine over 8x more partial rows (8 tokens x 16 heads x 64 fp32 per split), which this would show on those two shapes alone. The split count formula is the same first guess as the decode variant so the two runs can be compared; the reviewer may add a tune companion at 2x and 0.5x.
