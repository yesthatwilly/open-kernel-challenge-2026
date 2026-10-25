---
problem: muse_attention_decode
slug: sliding-decode-paired-kvhead-rows-control-unpaired-s8
created: 2026-10-25T004021Z
base: master
area: sliding-layer attention (batched decode, split depth)
change: >
  From master (the parent's base), change only the split count in _sliding_forward: splits = 8 if
  (length == 1 and batch > 1 and hkv == 2) else (16 if length == 1 else _num_splits(batch, hkv)).
  Keep _sliding_fused_kernel, the unpaired gather (program (b, g, s), 64-byte rows), M=16 with its
  padding rows, BLOCK_N 64, num_warps 4, num_stages 2, the ticket counters _counters(batch, hkv),
  and the workspace sizing. Batch 1 decode stays at S=16; verify and full layers unchanged byte
  for byte. Grid (B, 2, 8): 256 programs at B=16, 512 at B=32, each 256 keys x 64 B = 16 KB: the
  same grid and bytes per program as the paired hypothesis, without the pairing.
expected:
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: same as sliding-decode32-splits-8 (kernel 22.1 -> 17.3 us)
  - shapes: "*layer=sliding load=decode_16*"
    direction: flat
    why: 256 programs vs 512; the unmeasured point
  - shapes: "*layer=sliding load=decode_1* *layer=sliding load=verify_* *layer=full*"
    direction: flat
    why: outside the predicate
risk: low
depends_on: sliding-decode-paired-kvhead-rows
priority: 2
kind: control
controls_for: sliding-decode-paired-kvhead-rows
review: required
status: queued
---

# sliding-decode-paired-kvhead-rows-control-unpaired-s8

## Change

From master (the parent's base), change only the split count in _sliding_forward: splits = 8 if (length == 1 and batch > 1 and hkv == 2) else (16 if length == 1 else _num_splits(batch, hkv)). Keep _sliding_fused_kernel, the unpaired gather (program (b, g, s), 64-byte rows), M=16 with its padding rows, BLOCK_N 64, num_warps 4, num_stages 2, the ticket counters _counters(batch, hkv), and the workspace sizing. Batch 1 decode stays at S=16; verify and full layers unchanged byte for byte. Grid (B, 2, 8): 256 programs at B=16, 512 at B=32, each 256 keys x 64 B = 16 KB: the same grid and bytes per program as the paired hypothesis, without the pairing.

## Expected

- `*layer=sliding load=decode_32*` faster: same as sliding-decode32-splits-8 (kernel 22.1 -> 17.3 us)
- `*layer=sliding load=decode_16*` flat: 256 programs vs 512; the unmeasured point
- `*layer=sliding load=decode_1* *layer=sliding load=verify_* *layer=full*` flat: outside the predicate

## Risk

Separates pairing from the grid halving that pairing also brings; pairing's effect = paired minus control kernel time per shape.
