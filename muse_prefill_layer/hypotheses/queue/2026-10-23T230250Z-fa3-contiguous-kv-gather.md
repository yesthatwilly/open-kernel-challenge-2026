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
review: required
status: queued
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
