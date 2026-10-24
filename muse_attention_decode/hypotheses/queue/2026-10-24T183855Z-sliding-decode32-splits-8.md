---
problem: muse_attention_decode
slug: sliding-decode32-splits-8
created: 2026-10-24T183855Z
base: master
area: sliding-layer attention (batched decode, split depth)
change: >
  Config-only change in _sliding_forward: at length == 1 use splits = 16 if batch <= 16 else 8
  (today 16 at every batch). At batch 32 that is 512 programs of 256 old keys (4 BLOCK_N tiles)
  instead of 1024 programs of 128 keys, and half the partials and ticket merges. The workspace
  sizing and counters follow `splits` as now; nothing else changes.
expected:
  - shapes: "*layer=sliding load=decode_32 *"
    direction: faster
    why: _sliding_fused_kernel takes 22.1 us at decode_32 for 16.8 MB (0.76 TB/s) with 1024 four-warp programs, more than one resident wave plus 1024 partial stores and 64 ticket merges of 16 partials; 8 splits was never measured at B = 32 (S = 2 lost to S = 16 there). Expect 5-10% on the kernel
  - shapes: "*layer=sliding load=decode_1 * *layer=sliding load=decode_16 *"
    direction: flat
    why: batch <= 16 keeps 16 splits
  - shapes: "*layer=full* *load=verify_*"
    direction: flat
    why: not on this path
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-decode32-splits-8

## Change

Config-only change in _sliding_forward: at length == 1 use splits = 16 if batch <= 16 else 8 (today 16 at every batch). At batch 32 that is 512 programs of 256 old keys (4 BLOCK_N tiles) instead of 1024 programs of 128 keys, and half the partials and ticket merges. The workspace sizing and counters follow `splits` as now; nothing else changes.

## Expected

- `*layer=sliding load=decode_32 *` faster: _sliding_fused_kernel takes 22.1 us at decode_32 for 16.8 MB (0.76 TB/s) with 1024 four-warp programs, more than one resident wave plus 1024 partial stores and 64 ticket merges of 16 partials; 8 splits was never measured at B = 32 (S = 2 lost to S = 16 there). Expect 5-10% on the kernel
- `*layer=sliding load=decode_1 * *layer=sliding load=decode_16 *` flat: batch <= 16 keeps 16 splits
- `*layer=full* *load=verify_*` flat: not on this path

## Risk

Touches master's _sliding_fused_kernel, which sliding-cuda-cluster-dsmem-onelaunch (in the judge now) replaces on every sliding shape at tp8/page 1; if that rewrite lands this is moot and should be held. Useful only as a tuning of the incumbent if the rewrite falls back. Judge on the _sliding_fused_kernel row, not the headline (sliding decode_1 drifted 2-3.5% between run batches).
