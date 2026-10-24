# muse_attention_decode · 2026-10-24T112038Z · check · fa3-window-trimmed-page-table

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-23T225858Z-fa3-window-trimmed-page-table.md` (branch `hyp/muse_attention_decode/fa3-window-trimmed-page-table`).

- **Job:** `1f63fcbbc0394bed` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `4c6a0d6`

## Goal
On sliding-window layers only (config window != 0) hand FA3 a window-trimmed view of each request instead of the whole 80K-token context. Let W = window - 1 + L (2047 + L: the union of the L query tokens' windows), P = page_size, NP = ceil(W / P) + 1 pages. One Triton kernel, grid (B,), one program per request: len = cache_seqlens[b]; first_tok = max(0, len - W); first_page = first_tok // P; writes pt_win[b, i] = page_table[b, first_page + i] for i < NP (reading past the table's width masked to -1) and len_win[b] = len - first_page * P as int32. pt_win is a fresh int32 [B, NP] and len_win a fresh int32 [B] each call (fixed shapes). Then call flash_attn_with_kvcache exactly as the parity pipeline does but with page_table=pt_win and cache_seqlens=len_win (q, k_cache, v_cache, cu_seqlens_q, max_seqlen_q=L, softmax_scale, causal=True, window_size=(window-1, 0), scheduler_metadata=None, descale when kv_scale != 1 all unchanged). The attended set is identical: query j sits at local position len_win - L + j, its window [local - window + 1, local] lies inside the trimmed table because first_tok <= len - W, and FA3's window_size mask still cuts the window when len < W (short-context tests) and inside the verify block. The new tokens' rows are already written by store_cache before this call and are reached through the same page_table entries. Full layers are untouched. Mechanism: FA3 derives max_seqlen_k, its split-KV heuristic, the dynamic-split prepare kernel's work and the combine's split count from the page table width (86K+ tokens, 672 blocks of 128) although at most 17 blocks per tile are inside the window; the trimmed view makes FA3 see a 2,055-token problem.

## What changed, and where
- Added `_window_trim_kernel` (Triton, grid `(B,)`, `num_warps=4`) after `_pre_attention_kernel` in `muse_attention_decode/submission.py`. Per request: `n = cache_seqlens[b]`, `first_page = max(n - W, 0) // P`, `pt_win[b, i] = page_table[b, first_page + i]` for `i < NP` (masked to -1 past the table width), `len_win[b] = n - first_page * P`. `tl.arange` needs a power of two, so it runs over `BLOCK = next_power_of_2(NP)` lanes with an `i < NP` mask on load and store (NP = 2049/2056 at page 1, 33 at page 64).
- In `kernel_fn`, only when `window != 0`: `W = window - 1 + L`, `NP = ceil(W / P) + 1` with `P = cfg["page_size"]`; fresh `pt_win` int32 `[B, NP]` and `len_win` int32 `[B]` from `torch.empty` (shapes depend only on config), then the FA3 call takes `page_table=pt_win, cache_seqlens=len_win`. Everything else in the call (q, caches, cu_seqlens_q, max_seqlen_q=L, softmax scale, causal, window_size, scheduler_metadata (None on sliding), descale) is unchanged. Full layers (window == 0) keep the input page_table/cache_seqlens and the paired-head path, untouched.
- The trim launch is enqueued after `_pre_attention_kernel` and reads only inputs (`page_table`, `cache_seqlens`), so its order relative to the cache store does not matter; the new rows are reached through the same page entries.
- No deviation from the spec. Not done (review open questions, out of scope for the implementer): reading flash_prepare_scheduler.cu to predict whether prepare already clamps to the window; folding the trim into `_pre_attention_kernel` (a separate follow-up if the launch eats the gain).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/fa3-window-trimmed-page-table diff` (1 file, +40 -2).

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints; the trim reads `cache_seqlens` on device; `pt_win`/`len_win` shapes and the `if window:` branch depend only on `config` (window, q_len, page_size, batch) and `page_table.shape[1]`, not on values.
- [x] Nothing derived from `data` is cached across calls: `pt_win`/`len_win` are fresh each call; no new module-level state (only the existing `_INV_FREQ`, keyed on config theta).
- [x] Weights: n/a: this problem has no weights; the diff touches none.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries FP8 E4M3: unchanged by the diff (only the page_table/cache_seqlens handed to FA3 differ).
- [x] Attention sees every key: the trimmed table starts at `first_page*P <= max(len - W, 0)`, the first key of query 0 window, and ends at `len`, so every key of every query window (2,048 tokens) is in it; FA3 own window mask (unchanged `window_size=(window-1, 0)`, bottom-right aligned causal) is what excludes keys outside the window, exactly as before. Only keys the window already excludes are dropped from the table. No top-k/sparse/weight-based skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds one Triton kernel and its launch; no new imports.
- [x] All work happens inside the timed call; no output caching or replay: the trim kernel runs inside `kernel_fn` each call.
- [x] First-line marker `# okc: problem=muse_attention_decode` kept; no `triton.autotune`.

Lint (`okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode`):

```
CLEAN: muse_attention_decode/submission.py
```

`python3 -I -m py_compile`: OK; no `__pycache__` left in the worktree.

## Risks / what could fail
medium
Medium risk. Correctness edges: page_size 64 (tests) needs first_page alignment, so the trimmed table starts at a page boundary and len_win is measured from that boundary (len - first_page*64, up to W + 63); the edges tests plant keys at exactly len-L-2047 and len-L-2048 relative to the block's first token, and the trimmed range [first_tok, len) contains the first and excludes the second only through FA3's own window mask, which is unchanged. Short contexts (ctx 10, 2047) give first_page = 0 and pt_win = page_table[:, :NP], identical to today. The cache_seqlens the kernel reads is the input tensor; nothing is derived on the host (no .item, no max). Shares a symptom with fa3-num-splits-sliding (both fix an oversized split schedule) by a different mechanism: this changes what FA3 sees, that one overrides what it chooses; the reviewer may ask which lands first, and they can stack (the trimmed table makes the explicit num_splits redundant). Risk: if FA3 already bounds its sliding-window schedule by the window, this is flat minus one launch, and should go to do-not-retry together with the num_splits variant.

## Result (after the run)
- job `1f63fcbbc0394bed` (check), status done, passed True, tests 16/16
- speedup 1.1411x, 95% CI [1.1240, 1.1582], score 3300.6006651806383 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.2686474414890054, "ci": [1.2496177298666704, 1.2876771531113405], "job": "bd64bf9f9e904f02"}; base result: muse_attention_decode/hypotheses/done/gain/fa3-hdim128-head-pairing.result.json
- **outcome: regression** (regression)
- files: `muse_attention_decode/hypotheses/done/regression/fa3-window-trimmed-page-table.result.json`, `muse_attention_decode/hypotheses/done/regression/fa3-window-trimmed-page-table.patch`; branch `hyp/muse_attention_decode/fa3-window-trimmed-page-table`
