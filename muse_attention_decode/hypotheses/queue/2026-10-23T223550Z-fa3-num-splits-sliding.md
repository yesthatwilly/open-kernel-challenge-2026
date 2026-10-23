---
problem: muse_attention_decode
slug: fa3-num-splits-sliding
created: 2026-10-23T223550Z
base: master
area: FA3 call
change: >
  On sliding-window layers only (config window != 0, where SGLang passes scheduler_metadata=None
  and FA3 picks num_splits itself from max_seqlen_k), pass an explicit num_splits to
  flash_attn_with_kvcache instead of letting FA3's heuristic run. FA3's heuristic sizes the
  split-KV from the full cache length (cache_seqlens up to 86K tokens), not from the 2,048-token
  window it will actually attend, so for a window of 2048 tokens and n_block 128 at most 16 KV
  blocks exist per (request, kv-head) tile while the heuristic may choose far more splits (each
  split past the window runs empty and the combine kernel still reduces over all of them). Set
  num_splits = clamp(ceil(2*132 / (batch*num_kv_heads)), 1, 16): 16 at batch 1 (2 tiles, 16 blocks
  each), 9 at batch 16, 5 at batch 32. Leave scheduler_metadata, window_size, cu_seqlens_q and
  everything else as in the parity pipeline; full-attention layers are untouched (they keep the
  judge-built scheduler_metadata). Implement as one keyword argument gated on cfg['window'].
expected:
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "64 tiles x 16 window blocks: fewer, non-empty splits and a smaller combine than a heuristic fed an 86K-token length"
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: 32 tiles, same mechanism
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: same tiles at q_len 8
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: same
  - shapes: "*layer=sliding load=decode_1*"
    direction: flat
    why: "2 tiles: 16 splits is probably what the heuristic already picks for 2 tiles on 132 SMs; a change here attributes to the heuristic choosing more than 16"
  - shapes: "*layer=full*"
    direction: flat
    why: "untouched: control that the gate on window is right"
risk: medium
depends_on: sglang-path-parity
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-num-splits-sliding

## Change

On sliding-window layers only (config window != 0, where SGLang passes scheduler_metadata=None and FA3 picks num_splits itself from max_seqlen_k), pass an explicit num_splits to flash_attn_with_kvcache instead of letting FA3's heuristic run. FA3's heuristic sizes the split-KV from the full cache length (cache_seqlens up to 86K tokens), not from the 2,048-token window it will actually attend, so for a window of 2048 tokens and n_block 128 at most 16 KV blocks exist per (request, kv-head) tile while the heuristic may choose far more splits (each split past the window runs empty and the combine kernel still reduces over all of them). Set num_splits = clamp(ceil(2*132 / (batch*num_kv_heads)), 1, 16): 16 at batch 1 (2 tiles, 16 blocks each), 9 at batch 16, 5 at batch 32. Leave scheduler_metadata, window_size, cu_seqlens_q and everything else as in the parity pipeline; full-attention layers are untouched (they keep the judge-built scheduler_metadata). Implement as one keyword argument gated on cfg['window'].

## Expected

- `*layer=sliding load=decode_32*` faster: 64 tiles x 16 window blocks: fewer, non-empty splits and a smaller combine than a heuristic fed an 86K-token length
- `*layer=sliding load=decode_16*` faster: 32 tiles, same mechanism
- `*layer=sliding load=verify_16*` faster: same tiles at q_len 8
- `*layer=sliding load=verify_32*` faster: same
- `*layer=sliding load=decode_1*` flat: 2 tiles: 16 splits is probably what the heuristic already picks for 2 tiles on 132 SMs; a change here attributes to the heuristic choosing more than 16
- `*layer=full*` flat: untouched: control that the gate on window is right

## Risk

Medium risk: if FA3's heuristic already clamps num_n_blocks by the window (hopper flash_api.cpp may do this for local attention), the result is flat everywhere and the hypothesis is cleanly rejected. If an explicit num_splits > 1 forces the split path where the heuristic chose 1 (batch 32: 64 tiles may already fill the SMs without splitting), decode_32 could get slower by one combine kernel; the reviewer may want an ab-tune companion with num_splits=1 on sliding layers to separate 'no split, no combine' from 'fewer splits'. Correctness is unaffected: num_splits changes reduction order only.
