---
problem: muse_attention_decode
slug: sliding-deep-splits-on-cuda-splitkv
created: 2026-10-24T145005Z
base: master
area: sliding attention, batched decode, split depth
change: >
  Stack sliding-fused-triton-decode-batched-deep-splits (1.309x on the pre-CUDA master; sliding
  decode_16 gained 33%, decode_32 +3%, every other shape flat) onto master
  full-decode-cuda-gather-splitkv (1.347x). The result is a clean 3-way merge (git merge-file, no
  conflicts): the fused sliding kernel's gate widens from batch == 1 to every batch at q_len 1,
  and _sliding_forward uses 16 splits for decode (length 1). The CUDA split-KV path (full layers,
  1 < batch <= 64, q_len 1) and everything else are master's. Shapes are disjoint: the merge
  touches only sliding decode.
expected:
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: +33% measured on the previous master
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: +3% measured; may read flat
  - shapes: "*layer=full*"
    direction: flat
    why: master CUDA/FA3 paths unchanged
  - shapes: "*load=verify*"
    direction: flat
    why: FA3, unchanged
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-deep-splits-on-cuda-splitkv

## Change

Stack sliding-fused-triton-decode-batched-deep-splits (1.309x on the pre-CUDA master; sliding decode_16 gained 33%, decode_32 +3%, every other shape flat) onto master full-decode-cuda-gather-splitkv (1.347x). The result is a clean 3-way merge (git merge-file, no conflicts): the fused sliding kernel's gate widens from batch == 1 to every batch at q_len 1, and _sliding_forward uses 16 splits for decode (length 1). The CUDA split-KV path (full layers, 1 < batch <= 64, q_len 1) and everything else are master's. Shapes are disjoint: the merge touches only sliding decode.

## Expected

- `*layer=sliding load=decode_16*` faster: +33% measured on the previous master
- `*layer=sliding load=decode_32*` faster: +3% measured; may read flat
- `*layer=full*` flat: master CUDA/FA3 paths unchanged
- `*load=verify*` flat: FA3, unchanged

## Risk

Minted by churnsubs to restack a harvested per-shape win onto the new master; merged file at /home/user/.claude/jobs/7156a0da/tmp/merge/a_merged.py.
