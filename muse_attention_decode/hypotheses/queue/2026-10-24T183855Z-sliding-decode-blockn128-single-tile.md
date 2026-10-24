---
problem: muse_attention_decode
slug: sliding-decode-blockn128-single-tile
created: 2026-10-24T183855Z
base: master
area: sliding-layer attention (batched decode, tile size)
change: >
  Config-only change in _sliding_forward (master sliding decode at every batch, 16 splits): pass
  BLOCK_N=128 instead of 64 when length == 1. With S = 16 a split covers ceil(2047 / 16) = 128 old
  keys rounded up to BLOCK_N, so each program now runs its old-key loop once, issuing the
  page-index load and the whole chunk's K and V loads in one step instead of two dependent
  iterations. num_warps stays 4, num_stages 2, the new-key block, the masks, the ticket combine
  and verify (length 8, which keeps BLOCK_N 64) are unchanged.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "_sliding_fused_kernel is 10.4 us for 0.5 MB: a latency chain (q norm, idx load, KV load, dot, partial store, ticket, reload). One gather round trip instead of two per program should save ~0.5-1.5 us"
  - shapes: "*layer=sliding load=decode_16 * *layer=sliding load=decode_32 *"
    direction: faster
    why: same chain, 12.9 / 22.1 us kernels; smaller relative gain
  - shapes: "*layer=full* *load=verify_*"
    direction: flat
    why: not on the sliding decode path
risk: low
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-decode-blockn128-single-tile

## Change

Config-only change in _sliding_forward (master sliding decode at every batch, 16 splits): pass BLOCK_N=128 instead of 64 when length == 1. With S = 16 a split covers ceil(2047 / 16) = 128 old keys rounded up to BLOCK_N, so each program now runs its old-key loop once, issuing the page-index load and the whole chunk's K and V loads in one step instead of two dependent iterations. num_warps stays 4, num_stages 2, the new-key block, the masks, the ticket combine and verify (length 8, which keeps BLOCK_N 64) are unchanged.

## Expected

- `*layer=sliding load=decode_1 *` faster: _sliding_fused_kernel is 10.4 us for 0.5 MB: a latency chain (q norm, idx load, KV load, dot, partial store, ticket, reload). One gather round trip instead of two per program should save ~0.5-1.5 us
- `*layer=sliding load=decode_16 * *layer=sliding load=decode_32 *` faster: same chain, 12.9 / 22.1 us kernels; smaller relative gain
- `*layer=full* *load=verify_*` flat: not on the sliding decode path

## Risk

Touches master's _sliding_fused_kernel, which sliding-cuda-cluster-dsmem-onelaunch (in the judge now) replaces on every sliding shape at tp8/page 1; if that rewrite lands this is moot and should be held. Useful only as a tuning of the incumbent if the rewrite falls back. Judge on the _sliding_fused_kernel row, not the headline (sliding decode_1 drifted 2-3.5% between run batches). Sliding decode_1 carries weight 46 and is 58% of the decode_1 load, so ~1 us there is ~+4% on that load.
