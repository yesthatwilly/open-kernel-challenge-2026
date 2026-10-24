---
problem: muse_attention_decode
slug: fa3-window-trimmed-page-table
created: 2026-10-23T225858Z
base: master
area: page table (sliding)
change: >
  On sliding-window layers only (config window != 0) hand FA3 a window-trimmed view of each
  request instead of the whole 80K-token context. Let W = window - 1 + L (2047 + L: the union of
  the L query tokens' windows), P = page_size, NP = ceil(W / P) + 1 pages. One Triton kernel, grid
  (B,), one program per request: len = cache_seqlens[b]; first_tok = max(0, len - W); first_page =
  first_tok // P; writes pt_win[b, i] = page_table[b, first_page + i] for i < NP (reading past the
  table's width masked to -1) and len_win[b] = len - first_page * P as int32. pt_win is a fresh
  int32 [B, NP] and len_win a fresh int32 [B] each call (fixed shapes). Then call
  flash_attn_with_kvcache exactly as the parity pipeline does but with page_table=pt_win and
  cache_seqlens=len_win (q, k_cache, v_cache, cu_seqlens_q, max_seqlen_q=L, softmax_scale,
  causal=True, window_size=(window-1, 0), scheduler_metadata=None, descale when kv_scale != 1 all
  unchanged). The attended set is identical: query j sits at local position len_win - L + j, its
  window [local - window + 1, local] lies inside the trimmed table because first_tok <= len - W,
  and FA3's window_size mask still cuts the window when len < W (short-context tests) and inside
  the verify block. The new tokens' rows are already written by store_cache before this call and
  are reached through the same page_table entries. Full layers are untouched. Mechanism: FA3
  derives max_seqlen_k, its split-KV heuristic, the dynamic-split prepare kernel's work and the
  combine's split count from the page table width (86K+ tokens, 672 blocks of 128) although at
  most 17 blocks per tile are inside the window; the trimmed view makes FA3 see a 2,055-token
  problem.
expected:
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "64 tiles: the FA3 split heuristic and combine sized for 2,055 tokens instead of 86K; one extra 64-program launch (~2.5 us) against the ~8-10 us FA3 itself spends on a 31.5 us shape"
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: 32 tiles, same mechanism, 27 us shape
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: same at q_len 8 (W = 2055), 29 us shape
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: same, 34.5 us shape
  - shapes: "*layer=sliding load=decode_1 page*"
    direction: flat
    why: 2 tiles; the trim launch costs about what a right-sized split schedule saves; a loss here attributes to the extra launch
  - shapes: "*layer=sliding load=verify_1 page*"
    direction: flat
    why: same as decode_1
  - shapes: "*layer=full*"
    direction: flat
    why: untouched; control
risk: medium
depends_on: fused-preattention-triton
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-window-trimmed-page-table

## Change

On sliding-window layers only (config window != 0) hand FA3 a window-trimmed view of each request instead of the whole 80K-token context. Let W = window - 1 + L (2047 + L: the union of the L query tokens' windows), P = page_size, NP = ceil(W / P) + 1 pages. One Triton kernel, grid (B,), one program per request: len = cache_seqlens[b]; first_tok = max(0, len - W); first_page = first_tok // P; writes pt_win[b, i] = page_table[b, first_page + i] for i < NP (reading past the table's width masked to -1) and len_win[b] = len - first_page * P as int32. pt_win is a fresh int32 [B, NP] and len_win a fresh int32 [B] each call (fixed shapes). Then call flash_attn_with_kvcache exactly as the parity pipeline does but with page_table=pt_win and cache_seqlens=len_win (q, k_cache, v_cache, cu_seqlens_q, max_seqlen_q=L, softmax_scale, causal=True, window_size=(window-1, 0), scheduler_metadata=None, descale when kv_scale != 1 all unchanged). The attended set is identical: query j sits at local position len_win - L + j, its window [local - window + 1, local] lies inside the trimmed table because first_tok <= len - W, and FA3's window_size mask still cuts the window when len < W (short-context tests) and inside the verify block. The new tokens' rows are already written by store_cache before this call and are reached through the same page_table entries. Full layers are untouched. Mechanism: FA3 derives max_seqlen_k, its split-KV heuristic, the dynamic-split prepare kernel's work and the combine's split count from the page table width (86K+ tokens, 672 blocks of 128) although at most 17 blocks per tile are inside the window; the trimmed view makes FA3 see a 2,055-token problem.

## Expected

- `*layer=sliding load=decode_32*` faster: 64 tiles: the FA3 split heuristic and combine sized for 2,055 tokens instead of 86K; one extra 64-program launch (~2.5 us) against the ~8-10 us FA3 itself spends on a 31.5 us shape
- `*layer=sliding load=decode_16*` faster: 32 tiles, same mechanism, 27 us shape
- `*layer=sliding load=verify_16*` faster: same at q_len 8 (W = 2055), 29 us shape
- `*layer=sliding load=verify_32*` faster: same, 34.5 us shape
- `*layer=sliding load=decode_1 page*` flat: 2 tiles; the trim launch costs about what a right-sized split schedule saves; a loss here attributes to the extra launch
- `*layer=sliding load=verify_1 page*` flat: same as decode_1
- `*layer=full*` flat: untouched; control

## Risk

Medium risk. Correctness edges: page_size 64 (tests) needs first_page alignment, so the trimmed table starts at a page boundary and len_win is measured from that boundary (len - first_page*64, up to W + 63); the edges tests plant keys at exactly len-L-2047 and len-L-2048 relative to the block's first token, and the trimmed range [first_tok, len) contains the first and excludes the second only through FA3's own window mask, which is unchanged. Short contexts (ctx 10, 2047) give first_page = 0 and pt_win = page_table[:, :NP], identical to today. The cache_seqlens the kernel reads is the input tensor; nothing is derived on the host (no .item, no max). Shares a symptom with fa3-num-splits-sliding (both fix an oversized split schedule) by a different mechanism: this changes what FA3 sees, that one overrides what it chooses; the reviewer may ask which lands first, and they can stack (the trimmed table makes the explicit num_splits redundant). Risk: if FA3 already bounds its sliding-window schedule by the window, this is flat minus one launch, and should go to do-not-retry together with the num_splits variant.
