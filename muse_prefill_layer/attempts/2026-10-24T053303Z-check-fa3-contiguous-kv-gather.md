# muse_prefill_layer · 2026-10-24T053303Z · check · fa3-contiguous-kv-gather

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-23T230250Z-fa3-contiguous-kv-gather.md` (branch `hyp/muse_prefill_layer/fa3-contiguous-kv-gather`).

- **Job:** `75bbf278dd9e430e` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `bf34ed2`

## Goal
Replace the paged FA3 call with a contiguous one; everything before and after stays the parity path. After store_cache (so the new tokens' rows are in the cache), gather this step's K and V rows into per-request contiguous FP8 buffers and call flash_attn_with_kvcache with page_table=None and scheduler_metadata=None. Only when config page_size == 1 (every scored case); for page_size != 1 fall back to the parity paged call. Full layers (window == 0): idx = data['page_table'].clamp(min=0) [B, max_pages]; k_buf = k_cache.view(-1, hkv, d)[idx] and v_buf likewise (two index_select launches), giving [B, max_pages, 2, 64] FP8; pass k_cache=k_buf, v_cache=v_buf, cache_seqlens=data['cache_seqlens'] unchanged, the same FP8 q with cu_seqlens_q/max_seqlen_q, causal=True, window_size=(-1, -1), the same descale kwargs. Sliding layers (window 2048): gather a fixed trailing window of W = window - 1 + q_len positions per request: start = (data['seq_lens'] - (window - 1)).clamp(min=0) [B]; pos = start[:, None] + arange(W)[None, :]; idx = page_table.gather(1, pos.clamp(max=max_pages - 1)).clamp(min=0); k_buf/v_buf [B, W, 2, 64]; cache_seqlens_w = (data['cache_seqlens'] - start).to(int32), which equals min(cache_seqlens, W); call with window_size=(window - 1, 0). This is exact: buffer key j is absolute position start + j, so the causal and window masks over the trimmed buffer select the same keys as over the full cache; positions beyond cache_seqlens_w are clamped -1 pages (row 0) and masked by cache_seqlens. All index tensors are fixed-shape functions of the inputs: graph-safe, no host sync, nothing cached across calls.

## What changed, and where
Implemented 2026-10-24 in the worktree's `muse_prefill_layer/submission.py`, `kernel_fn` only (the FA3 call), plus the module docstring's FA3 line. Diff: 1 file, +58 / -15.

- **Branch on config.** `if cfg["page_size"] == 1:` takes the contiguous path; `else:` is master's paged call verbatim (page_table, cache_seqlens, `scheduler_metadata=data["scheduler_metadata"]`). The page-size-64 test takes the parity path.
- **Ordering.** The gather sits after `store_cache` on the same (current) stream, so the new tokens' rows are in the buffers.
- **Gather on integer lanes (review requirement).** `k_cache.view(-1, hkv*d).view(torch.int64)` ([rows, 16] int64, 8-byte lanes) `.index_select(0, rows)`, then `.view(FP8).view(batch, span, hkv, d)`; same for v. No FP8 advanced indexing; the bytes are copied unchanged (the cache stays FP8). `rows` is int32 (index_select accepts int32 indices); it is `page_table` (or the gathered window slice), `.clamp(min=0).reshape(-1)`.
- **Full layers (window == 0).** `span = max_pages`, `rows = page_table`, `cache_seqlens = data["cache_seqlens"]` unchanged, `window_size=(-1, -1)`. Buffer key j is page_table[b, j], i.e. absolute position j; the whole page table extent is in the buffer, so every key in [0, cache_seqlens) is present. Positions >= cache_seqlens (any -1 pages) read row 0 and are masked by cache_seqlens.
- **Sliding layers (window = 2048).** `span = W = window - 1 + L`; `start = (seq_lens - (window-1)).clamp(min=0)` (int32 [B]); `pos = start.int64[:, None] + arange(W, int64)[None, :]` (int64 because torch.gather requires an int64 index); `rows = page_table.gather(1, pos.clamp(max=max_pages-1))`; `kv_seqlens = (cache_seqlens - start).to(int32)`; `window_size=(window-1, 0)`.
- **Mask mapping over the trimmed buffer (verified).** Buffer key j is absolute position start + j. FA3 aligns causal masks bottom-right: query i of a request (absolute position seq_lens + i) maps to buffer index i + kv_seqlens - L = i + seq_lens + L - start - L = seq_lens + i - start, which is absolute seq_lens + i: the diagonal is the same key. Causal right edge (offset 0) and window left edge (offset window-1 = 2047) are offsets from that diagonal, so they select buffer keys [seq_lens+i-2047-start, seq_lens+i-start], i.e. absolute [seq_lens+i-2047, seq_lens+i], the same keys as the paged call; the lower bound clamps at buffer index 0 exactly when the absolute bound clamps at 0 (only if start = 0). Coverage: the smallest left edge any query needs is max(0, seq_lens-2047) = start (query 0) -> buffer index 0; the largest right edge is seq_lens + L - 1 (query L-1) -> buffer index kv_seqlens - 1. If start > 0, kv_seqlens = 2047 + L = W exactly and the largest gathered position is start + W - 1 = cache_seqlens - 1 <= max_pages - 1 (the clamp never fires); if start = 0, kv_seqlens = cache_seqlens <= W and positions beyond cache_seqlens are clamped/row-0 and masked by kv_seqlens. So every key the reference attends is in the buffer, and nothing the reference excludes is unmasked.
- **Graph safety / no caching.** All index tensors are fixed-shape (B x max_pages or B x W, from config and tensor shapes); `torch.arange`, gather, clamp, index_select are device kernels; no `.item()`/`.cpu()`/sync, no branch on values. Buffers are allocated per call and nothing is put in `state`.
- **scheduler_metadata.** The contiguous path passes `scheduler_metadata=None` (as the change specifies; a paged tile schedule would not apply, and the input is None on every prefill case). The fallback passes master's `data["scheduler_metadata"]`.
- **Descale.** Unchanged (`k_descale`/`v_descale` [batch, hkv], only when kv_scale != 1).
- **FA3 API assumption (unverified locally; no GPU here).** sgl_kernel `flash_attn_with_kvcache` with `page_table=None`, `cu_seqlens_q` set and no `cu_seqlens_k`/`k`/`v` new is FA3's `is_varlen_q && !is_varlen_k` mode: `k_cache` is [batch_k, seqlen_k, hkv, d] with `k_cache.size(0)` required to equal batch (B = len(cu_seqlens_q) - 1), and `cache_seqlens` (int32 [B]) acts as seqused_k. Our buffers are [B, max_pages or W, 2, 64] FP8, contiguous. If the wrapper rejects an unpaged FP8 cache or this combination, the first test fails fast.
- **Deviations from the md:** none in substance. The md's `k_cache.view(-1, hkv, d)[idx]` FP8 advanced indexing is replaced by the int64-lane index_select the review asked for (same bytes).

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/fa3-contiguous-kv-gather diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints added in kernel_fn; allocations (arange, gather, index_select outputs) are fixed-shape from config/tensor shapes; the only new branches are on `cfg["page_size"]` and `cfg["window"]` (config).
- [x] Nothing derived from `data` is cached across calls: the buffers and index tensors are locals of kernel_fn; `state` is not written.
- [x] Weights: n/a: no weight handling touched; prepare() unchanged.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: the KV buffers are byte copies of the FP8 cache (int64 lanes, viewed back to float8_e4m3fn), q is still `.to(FP8)`, output path unchanged.
- [x] Attention sees every key: full layers gather the whole page table extent; sliding layers gather the 2047 + L positions covering every query's window (mask mapping verified in Implementation notes); no top-k/sparse/skip.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff only touches the FA3 call and a docstring line.
- [x] All work happens inside the timed call; no output caching or replay: the gather runs inside kernel_fn each call.
- [x] First-line marker `# okc: problem=muse_prefill_layer` kept; no triton.autotune added.

Lint (`okc_lint.py ... --problem muse_prefill_layer`):

```
warn  host-sync    L263: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_prefill_layer/submission.py
```

(The warning is the pre-existing synchronize in prepare(), unchanged by this diff.) `py_compile`: OK; `__pycache__` removed.

## Risks / what could fail
medium
Medium risk. FA3's unpaged kvcache path with an FP8 cache and headdim 64 must accept [B, S, Hk, D] buffers with cache_seqlens int32; if sgl_kernel's wrapper demands a page_table for FP8, the run fails fast on the first test. Any scheduler_metadata the inputs carry is dropped because it encodes the paged tile schedule; for prefill it is expected to be None anyway. Correctness: every key the reference attends is in the buffer (full: whole page table; sliding: the 2047 + L trailing positions), the KV bytes are copied not converted, so the FP8 cache rule and the every-key rule hold; the page-size-64 test takes the parity path. Attribution: only the FA3 call and its gather change, so a loss means the row-gather copy costs more than the TMA path saves at that shape.

## Result (after the run)
- job `75bbf278dd9e430e` (check), status done, passed True, tests 5/5
- speedup 0.9879x, 95% CI [0.9731, 1.0027], score 392156.0856527929 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.result.json
- **outcome: regression** (regression)
- files: `muse_prefill_layer/hypotheses/done/regression/fa3-contiguous-kv-gather.result.json`, `muse_prefill_layer/hypotheses/done/regression/fa3-contiguous-kv-gather.patch`; branch `hyp/muse_prefill_layer/fa3-contiguous-kv-gather`
