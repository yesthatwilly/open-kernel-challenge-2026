---
problem: muse_prefill_layer
slug: fa3-contiguous-kv-gather
created: 2026-10-23T230250Z
base: master
area: FA3 call
change: >
  Replace the paged FA3 call with a contiguous one; everything before and after stays the parity
  path. After store_cache (so the new tokens' rows are in the cache), gather this step's K and V
  rows into per-request contiguous FP8 buffers and call flash_attn_with_kvcache with
  page_table=None and scheduler_metadata=None. Only when config page_size == 1 (every scored
  case); for page_size != 1 fall back to the parity paged call. Full layers (window == 0): idx =
  data['page_table'].clamp(min=0) [B, max_pages]; k_buf = k_cache.view(-1, hkv, d)[idx] and v_buf
  likewise (two index_select launches), giving [B, max_pages, 2, 64] FP8; pass k_cache=k_buf,
  v_cache=v_buf, cache_seqlens=data['cache_seqlens'] unchanged, the same FP8 q with
  cu_seqlens_q/max_seqlen_q, causal=True, window_size=(-1, -1), the same descale kwargs. Sliding
  layers (window 2048): gather a fixed trailing window of W = window - 1 + q_len positions per
  request: start = (data['seq_lens'] - (window - 1)).clamp(min=0) [B]; pos = start[:, None] +
  arange(W)[None, :]; idx = page_table.gather(1, pos.clamp(max=max_pages - 1)).clamp(min=0);
  k_buf/v_buf [B, W, 2, 64]; cache_seqlens_w = (data['cache_seqlens'] - start).to(int32), which
  equals min(cache_seqlens, W); call with window_size=(window - 1, 0). This is exact: buffer key j
  is absolute position start + j, so the causal and window masks over the trimmed buffer select
  the same keys as over the full cache; positions beyond cache_seqlens_w are clamped -1 pages (row
  0) and masked by cache_seqlens. All index tensors are fixed-shape functions of the inputs:
  graph-safe, no host sync, nothing cached across calls.
expected:
  - shapes: batch=1 ctx=32768 * layer=full load=first_turn*
    direction: faster
    why: "FA3 on a contiguous [B, S, 2, 64] cache uses TMA KV loads; at page size 1 the paged path gathers KV one 128-byte row at a time; the gather copy is ~13 MB (~5 us) against ~3 ms of attention"
  - shapes: batch=1 ctx=65536 * layer=full load=first_turn*
    direction: faster
    why: same mechanism against ~7 ms of attention; ~21 MB gather
  - shapes: "*layer=full load=turn_4*"
    direction: faster
    why: 83 MB gather (~35 us) against ~1.4 ms of paged attention over 4 x 80K keys
  - shapes: "*layer=full load=turn_1*"
    direction: flat
    why: ~21 MB gather (~10 us) against ~350 us of attention; the gain may not clear the interval
  - shapes: "*ctx=0*"
    direction: flat
    why: only the 16384 new tokens are in the cache; small gather, attention is a small share of the shape
  - shapes: "*layer=sliding*"
    direction: flat
    why: the trailing-window gather is < 3 MB and sliding attention is ~2% of these shapes
risk: medium
depends_on: sglang-pipeline-parity
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T053108Z
review_companions: null
stage: implemented
claimed_at: 2026-10-24T053109Z
base_sha: 85d607280348efbd3eef09246edcdfa7132f09fe
base_blob: e4931f050c7fb99835fe116d25bb454a5ea731e9
branch: hyp/muse_prefill_layer/fa3-contiguous-kv-gather
worktree: .worktrees/muse_prefill_layer/fa3-contiguous-kv-gather
mode: check
implemented_at: 2026-10-24T053303Z
impl_commit: bf34ed281f7cb6c4d9425fcbabb532e6321f566e
submission_sha256: fb3ee1bcd938c2c092d1909c93c139a120a5604a2b749a7529ed3a8ad304e0fb
norm_sha256: 027ee9a916288967957fc1037dd7fb13d6b42e28bb1cd9474c5a352c0a05ddaa
submission_blob: 01e9a2100d168ca5955d1465bb3308871c7d8847
lint_warnings: 1
attempt_file: 2026-10-24T053303Z-check-fa3-contiguous-kv-gather.md
prepared_at: 2026-10-24T053303Z
---

# fa3-contiguous-kv-gather

## Change

Replace the paged FA3 call with a contiguous one; everything before and after stays the parity path. After store_cache (so the new tokens' rows are in the cache), gather this step's K and V rows into per-request contiguous FP8 buffers and call flash_attn_with_kvcache with page_table=None and scheduler_metadata=None. Only when config page_size == 1 (every scored case); for page_size != 1 fall back to the parity paged call. Full layers (window == 0): idx = data['page_table'].clamp(min=0) [B, max_pages]; k_buf = k_cache.view(-1, hkv, d)[idx] and v_buf likewise (two index_select launches), giving [B, max_pages, 2, 64] FP8; pass k_cache=k_buf, v_cache=v_buf, cache_seqlens=data['cache_seqlens'] unchanged, the same FP8 q with cu_seqlens_q/max_seqlen_q, causal=True, window_size=(-1, -1), the same descale kwargs. Sliding layers (window 2048): gather a fixed trailing window of W = window - 1 + q_len positions per request: start = (data['seq_lens'] - (window - 1)).clamp(min=0) [B]; pos = start[:, None] + arange(W)[None, :]; idx = page_table.gather(1, pos.clamp(max=max_pages - 1)).clamp(min=0); k_buf/v_buf [B, W, 2, 64]; cache_seqlens_w = (data['cache_seqlens'] - start).to(int32), which equals min(cache_seqlens, W); call with window_size=(window - 1, 0). This is exact: buffer key j is absolute position start + j, so the causal and window masks over the trimmed buffer select the same keys as over the full cache; positions beyond cache_seqlens_w are clamped -1 pages (row 0) and masked by cache_seqlens. All index tensors are fixed-shape functions of the inputs: graph-safe, no host sync, nothing cached across calls.

## Expected

- `batch=1 ctx=32768 * layer=full load=first_turn*` faster: FA3 on a contiguous [B, S, 2, 64] cache uses TMA KV loads; at page size 1 the paged path gathers KV one 128-byte row at a time; the gather copy is ~13 MB (~5 us) against ~3 ms of attention
- `batch=1 ctx=65536 * layer=full load=first_turn*` faster: same mechanism against ~7 ms of attention; ~21 MB gather
- `*layer=full load=turn_4*` faster: 83 MB gather (~35 us) against ~1.4 ms of paged attention over 4 x 80K keys
- `*layer=full load=turn_1*` flat: ~21 MB gather (~10 us) against ~350 us of attention; the gain may not clear the interval
- `*ctx=0*` flat: only the 16384 new tokens are in the cache; small gather, attention is a small share of the shape
- `*layer=sliding*` flat: the trailing-window gather is < 3 MB and sliding attention is ~2% of these shapes

## Risk

Medium risk. FA3's unpaged kvcache path with an FP8 cache and headdim 64 must accept [B, S, Hk, D] buffers with cache_seqlens int32; if sgl_kernel's wrapper demands a page_table for FP8, the run fails fast on the first test. Any scheduler_metadata the inputs carry is dropped because it encodes the paged tile schedule; for prefill it is expected to be None anyway. Correctness: every key the reference attends is in the buffer (full: whole page table; sliding: the 2047 + L trailing positions), the KV bytes are copied not converted, so the FP8 cache rule and the every-key rule hold; the page-size-64 test takes the parity path. Attribution: only the FA3 call and its gather change, so a loss means the row-gather copy costs more than the TMA path saves at that shape.

## Adversarial review

Reviewed 2026-10-24T053108Z by an adversarial Fable subagent (model fable).

**Verdict: proceed.** Run it as is: the change is exact, rules-clean, and self-attributing (the FA3 kernel and the two gather kernels appear as separate rows in the profile), so one run answers the only real question, which is whether FA3's hdim-64 FP8 prefill kernel is limited by the page-size-1 cp.async gather path or by its own softmax/V-transpose ceiling. No landed item interacts with the FA3 call, so no A/B or control is warranted. Expect the per-shape rows to be the signal; the load-level upside is about 1% even if the mechanism fully works.

### Justification
A. Mechanism: real in FA3. Paged KV uses the TMA path only when page_size is a multiple of kBlockN; at page size 1 (every scored case) the producer warpgroup walks the page table and issues per-row cp.async copies (two 64-byte head rows per token), the path the decode review measured at 1.33-1.73 TB/s. A [B, S, 2, 64] contiguous cache takes the TMA path with no dependent page-table lookup. Profile data (job 62025c368c3c4e8e): the FA3 kernel is 419.9 us (28.0%) on turn_1 full, 1656.1 us (32.7%) on turn_4 full, 4545.7 us (22.0%) on first_turn ctx=32768 full, 8164.1 us (32.8%) on ctx=65536 full, 907 us (4.9%) on ctx=0 full; sliding FA3 is 25.5 / 62.0 / 309.9 us, so the 'flat' sliding rows are right. Converting those to throughput: 2.49e11 FLOP / 419.9 us = 593 TFLOP/s; 9.96e11 / 1656 us = 601; 2.75e12 / 4546 us = 605; 4.95e12 / 8164 us = 606. The kernel sits at a flat ~600 TFLOP/s (30% of FP8 dense peak, 61% of BF16) whether it launches ~96 CTAs plus splits (turn_1, the FlashAttnFwdCombine row proves dynamic split is on) or 2048 m-tiles (first turn). A per-CTA limiter independent of occupancy is consistent with EITHER (i) a latency-bound paged producer (page-table read then row cp.async, two dependent round trips per KV tile, ~1.3 us observed per 176-row tile against ~0.39 us of MMA at peak) OR (ii) the intrinsic FA3 hdim-64 FP8 ceiling (softmax exp per FLOP is 2x hdim 128's, plus the in-smem V transpose FP8 needs); FA3's own hdim-64 numbers are its weakest column and are not far above 600 TFLOP/s. The weakest link is exactly that: the decode evidence (q_len=1, each KV byte used once, bandwidth-bound) does not transfer to q_len=753-16384 where every KV tile is reused across 128 queries and the load path has ~100x more time per byte to hide behind; if (ii) holds, the TMA swap buys nothing and the run only pays the gather. Shapes/directions vs TASK.md: max_pages = ctx + q_len for spread=fixed (reference.py:259), so the gather sizes (21 MB per request at 80K, 83 MB at turn_4) are as stated; but the byte estimates assume near-peak copy bandwidth, and the change as written (fp8 advanced indexing, k_cache.view(-1,hkv,d)[idx]) is a 1-byte-element index kernel over 2x the bytes (read+write): expect 10-60 us at turn_1/first-turn and plausibly 100-250 us at turn_4, not 35 us, i.e. break-even needs a 3-15% FA3 gain on turn_4 full and 2.5-15% on turn_1 full. Weights: full shapes are ~31-34% of every load (turn_1: 16x1435 vs 46x1079; turn_4: 16x4950 vs 46x3367; first turn: the two ctx>0 full shapes are 22% of 3.39 s), and FA3 is ~28-33% of those shapes, so a 10% FA3 gain is ~3% on the full rows and ~1% per load, right at the +-1.5% interval. The run is informative per shape, marginal on the score.

### Confounders on the current kernel
- sglang-pipeline-parity (only landed item; patch muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.patch): it made the FA3 call byte-identical to baseline.py (paged, varlen q, scheduler_metadata=None on prefill) and feeds the cache rows the gather reads through store_cache; nothing it did changes the FA3 kernel's own time, so it cannot cause or mask a gain here. The one thing to keep is ordering: the gather must run after store_cache on the same stream (the hypothesis says so). Note its own ctx=32768 full row was 0.987x (21536 vs SGLang 21246 us) while every other row was >=1.000x; a 'faster' reading there must be judged against SGLang's absolute row, not against the parity's noisy one, or regression to the mean looks like a mechanism.
- in flight, not landed: moe-align-side-stream moves MoE align work after attention onto a side stream; no shared launch, traffic or numerical path with the FA3 call, but if it lands first the base rows shift by a few us and the reading must use the merged base's profile.

### Interference and A/B needs
No ab-revert: the only landed hypothesis is the parity port itself, which is the base, and reverting it means the plain PyTorch kernel, not a meaningful A/B. No ab-tune: nothing landed has a knob that touches attention. No control: the two sub-changes (pay the gather; call FA3 on a contiguous buffer) are separated by the profile for free, because the index kernels and the flash::enable_sm90_or_later kernel appear as separate rows with their own us, and the parity profile already holds the paged FA3 time per shape (419.9 / 1656.1 / 907 / 4545.7 / 8164.1 us). Every outcome reads unambiguously from one run: FA3 row drops by more than the gather rows add = mechanism (i) confirmed, keep and then cheapen the gather; FA3 row unchanged = ceiling (ii), the paged gather is not the prefill limiter and the idea is closed (and it also tells fa3-batched-mode and any custom-kernel idea where the ceiling is); FA3 row drops but the gather rows eat it = re-mint with a wider-lane or Triton gather. The ambiguity the rubric asks about (both a gain and a loss uninterpretable) does not arise, so spending a second run on a companion is wasteful.

### Open questions
- Implementation detail that decides whether the gather is 10 us or 200 us: torch advanced indexing on a float8_e4m3fn tensor may not be dispatched at all (older index kernels lack FLOAT8 types) and if it is, it moves one byte per thread. View the pool as k_cache.view(-1, hkv*d).view(torch.int64) ([rows, 16] int64, 8-byte lanes; or torch.int32 [rows, 32]) before index_select, exactly as the parity already does with .view(torch.uint8) for store_cache, then view the result back to FP8 [B, S, 2, 64]. Same bytes, same rule standing, several times cheaper. If the first run shows the index rows above ~100 us at turn_4, that is the re-mint, not a new idea.
- API check that fails fast on test 1 if wrong: sgl_kernel's flash_attn_with_kvcache with page_table=None, cu_seqlens_q set and cu_seqlens_k_new None is FA3's is_varlen_q && !is_varlen_k mode, which requires k_cache.size(0) == batch (B, max_pages, 2, 64) and uses cache_seqlens as seqused_k; the sliding buffer's S = W = 2047 + L and cache_seqlens_w <= W must satisfy that. If the wrapper insists on page_table for FP8 the run dies on the first test at no scoring cost, as the hypothesis already says.
- Is 600 TFLOP/s FA3's hdim-64 FP8 ceiling with contiguous KV? The judge's page-size-64 test does not answer it (64 is not a multiple of kBlockN for this config, so it is also the non-TMA path). This run is the cheapest way to find out and the answer is worth recording in FINDINGS either way, including for fa3-batched-mode (if the ceiling is intrinsic, no launch-mode change can move the first-turn rows either).
- The sliding first-turn gather is 2 x 18431 x 128 B = 4.7 MB per direction, not < 3 MB as the md says; immaterial against 17 ms, but the row should still read flat and a visible loss there would point at the index-kernel cost rather than attention.
- If this lands, fa3-batched-mode stacked on it becomes pure non-varlen FA3 (static scheduler, static num_splits heuristic) rather than varlen-with-dynamic-split; that is a different regime from the one the batched-mode md argues, so re-review it against the merged base rather than running it as written.


## Implementation notes

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

## Rules check

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
