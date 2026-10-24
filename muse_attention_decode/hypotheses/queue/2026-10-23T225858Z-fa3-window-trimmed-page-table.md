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
review: proceed
status: queued
review_at: 2026-10-24T111856Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T111856Z by an adversarial subagent (model opus).

**Verdict: proceed.** The change is correct, keeps every key in the window, and runs one sliding-only change with full layers as an in-run control, so a single run is readable from the per-kernel profile. The likely result is a small gain or flat: FA3's main loop and its host split heuristic probably already bound local attention by the window, so the realistic win is limited to a smaller combine/split schedule, minus the extra trim launch.

### Justification
Correctness holds. FA3 aligns causal masking to the bottom-right, so query j at local position len_win-L+j keeps the same relative window. RoPE is already applied before FA3, so absolute positions are not needed. first_page*P <= first_tok <= len-W keeps every window key inside the trimmed table. At page_size 64, len_win <= W+63 needs ceil((W+63)/64) <= ceil(W/64)+1 pages, which NP provides. Short contexts give first_page=0 and the same table as today. On the mechanism, the profile limits the upside. Sliding FA3 main time is 13.2 us at decode_32 for about 16.8 MB of window KV. That is about 1.3 TB/s, the page-size-1 gather rate LESSONS reports for full layers, which suggests the main kernel already iterates only window blocks (FA3's n_block_min/max are window-bounded per tile). The host get_num_splits also uses a window-clamped seqlen_k_loaded when is_local. So the only plausible lever is the device-side dynamic-split path: prepare_varlen_num_blocks_kernel (1.8 us) plus FlashAttnFwdCombine (2.5-4.1 us on every sliding shape). If prepare sizes per-batch splits from the full ~80K seqlen_k rather than the window, most splits are empty and combine reduces over too many partials, and trimming fixes that. If prepare already honours the window, this is flat minus one launch. The prepare kernel itself will not get cheaper: it is one launch of B threads whatever the table width, so the hypothesis's claim that the prepare kernel's work shrinks is wrong, and it stays in the graph because scheduler_metadata is None on sliding. Expected magnitudes are optimistic. The 'FA3 spends ~8-10 us' figure understates the 13.2 us main kernel but overstates what can move: at best about 1-3 us of combine per sliding shape against a ~2 us + ~0.3 us node trim launch. The batch-16/32 'faster' expectations are therefore plausible but small (low single digits per shape), and decode_1/verify_1 may lose by the launch. Sliding shapes carry weight 46 against 16 for full, so even 1-2 us per sliding shape moves the geomean by about 1-3%, near the ~1.5% drift. Weakest link: whether FA3's dynamic-split prepare already clamps per-batch n_blocks to the local window. The hypothesis says so itself and does not resolve it from FA3 source.

### Confounders on the current kernel
- fused-preattention-triton: no confound. It writes the new k/v rows and q before FA3, and the trim kernel reads only cache_seqlens and page_table (inputs). Their order on the stream is fixed and the cache contents are the same either way. Its own time (_pre_attention_kernel row) is separate in the profile.
- fa3-hdim128-head-pairing: no confound. It is gated on window == 0, so sliding layers never take the paired path. Full-layer rows double as this run's drift control (they should be flat within ~1.5-2.5%). The in-flight paired-heads-launch-cleanup runs touch only the full path and will not land on this branch's base.

### Interference and A/B needs
No companion is warranted. The change has one effect (what FA3 sees on sliding layers) plus one cost (the trim launch), and the profile separates the two without another run. A gain shows as a smaller FlashAttnFwdCombine and/or main-kernel row on sliding shapes. A loss with unchanged FA3/combine/prepare rows is the trim launch alone, and the hypothesis then goes to do-not-retry together with fa3-num-splits-sliding (already held). Neither landed hypothesis touches the sliding FA3 call, and sliding-single-launch-fused-attention regressed and did not land, so no ab-revert or ab-tune can change how this result reads. When harvesting, read the per-shape rows and the combine/prepare/main kernel times, not the headline, because the expected effect is near the drift floor. If FA3 does shrink but the launch eats the gain, the follow-up is to fold the trim into _pre_attention_kernel's grid (B extra programs) as a separate hypothesis, not a companion now.

### Open questions
- Does FA3 (the sgl_kernel build at this commit) clamp per-batch n_blocks to the local window in prepare_varlen_num_blocks / the dynamic split count? Reading flash_prepare_scheduler.cu would predict the result before spending a check. If it clamps, expect flat-minus-launch.
- Does the combine kernel's cost scale with the number of splits FA3 picks, or is it ~2.5-4 us fixed launch+latency? If fixed, only a 1-split schedule (fa3-single-split-sliding) removes it, and trimming helps only insofar as it makes FA3 choose num_splits=1.
- Order relative to fa3-single-split-sliding (queued p4): both target the same combine/split overhead. Run this one first. If FA3's split count does not drop, single-split is the remaining probe of the same lever. If it does drop, single-split becomes largely redundant.

