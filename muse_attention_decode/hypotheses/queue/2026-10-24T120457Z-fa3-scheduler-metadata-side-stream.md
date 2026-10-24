---
problem: muse_attention_decode
slug: fa3-scheduler-metadata-side-stream
created: 2026-10-24T120457Z
base: master
area: FA3 call (schedule, off critical path)
change: >
  Wherever master calls flash_attn_with_kvcache with scheduler_metadata=None (every sliding shape,
  full paired shapes at batch > 1, full verify_1), FA3 runs
  flash::prepare_varlen_num_blocks_kernel inside the call, serially after _pre_attention_kernel
  (1.7-1.8 us on sliding, 2.8-3.6 us on full; see the batch-gate profile bd64bf9f9e904f02). That
  kernel reads only cache_seqlens / cu_seqlens_q and shapes, not anything the pre-attention kernel
  writes, so build the same schedule concurrently. In kernel_fn, before the _pre_attention_kernel
  launch: cur = torch.cuda.current_stream(); _SIDE (one module-level torch.cuda.Stream() created
  lazily per device, never derived from data).wait_stream(cur); under torch.cuda.stream(_SIDE):
  meta = sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=L,
  max_seqlen_k=page_table.shape[1]*page_size, num_heads=Hq, num_heads_k=attn_heads (Hkv unpaired,
  Hkv/2 paired), headdim=2*d if paired else d, headdim_v same, cache_seqlens=cache_seqlens,
  qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True,
  window_size=(window-1, 0) if window else (-1, -1), num_splits=0, pack_gqa=None, sm_margin=0),
  i.e. exactly the arguments flash_attn_with_kvcache passes to its internal scheduler for this
  call (check the installed sgl_kernel flash_attn.py signature and mirror its defaults; if any
  argument such as max_seqlen_k or has_softcap differs, use what the in-call path derives). Then
  launch _pre_attention_kernel on cur as now, cur.wait_stream(_SIDE), and pass
  scheduler_metadata=meta to FA3. Also record meta on cur (meta.record_stream(cur)) so the
  allocator does not reuse it. Full decode_1 keeps the judge-built metadata unchanged (it already
  has no prepare kernel in its profile). Nothing else changes: query layout, pairing gate,
  epilogues, descale. This differs from the held fa3-scheduler-metadata-full-verify: that one
  launched the same scheduler serially (moving cost, not removing it) and used the unpaired
  geometry; this one runs it on a forked graph branch in parallel with the pre-attention kernel
  and uses the geometry of the actual call.
expected:
  - shapes: "*layer=sliding*"
    direction: faster
    why: "the 1.7-1.8 us prepare kernel overlaps the 1.7-4.2 us pre-attention kernel instead of following it: ~1.5 us of 16.6-28.9 us, 5-9%"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 3.1 us prepare off the critical path, ~2%
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: 3.5 us prepare off the critical path, ~1%
  - shapes: "*layer=full load=verify_*"
    direction: faster
    why: 2.8-3.6 us prepare off the critical path, 1-9% (verify_1 most)
  - shapes: "*layer=full load=decode_1 *"
    direction: same
    why: keeps the judge metadata; untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-scheduler-metadata-side-stream

## Change

Wherever master calls flash_attn_with_kvcache with scheduler_metadata=None (every sliding shape, full paired shapes at batch > 1, full verify_1), FA3 runs flash::prepare_varlen_num_blocks_kernel inside the call, serially after _pre_attention_kernel (1.7-1.8 us on sliding, 2.8-3.6 us on full; see the batch-gate profile bd64bf9f9e904f02). That kernel reads only cache_seqlens / cu_seqlens_q and shapes, not anything the pre-attention kernel writes, so build the same schedule concurrently. In kernel_fn, before the _pre_attention_kernel launch: cur = torch.cuda.current_stream(); _SIDE (one module-level torch.cuda.Stream() created lazily per device, never derived from data).wait_stream(cur); under torch.cuda.stream(_SIDE): meta = sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=L, max_seqlen_k=page_table.shape[1]*page_size, num_heads=Hq, num_heads_k=attn_heads (Hkv unpaired, Hkv/2 paired), headdim=2*d if paired else d, headdim_v same, cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True, window_size=(window-1, 0) if window else (-1, -1), num_splits=0, pack_gqa=None, sm_margin=0), i.e. exactly the arguments flash_attn_with_kvcache passes to its internal scheduler for this call (check the installed sgl_kernel flash_attn.py signature and mirror its defaults; if any argument such as max_seqlen_k or has_softcap differs, use what the in-call path derives). Then launch _pre_attention_kernel on cur as now, cur.wait_stream(_SIDE), and pass scheduler_metadata=meta to FA3. Also record meta on cur (meta.record_stream(cur)) so the allocator does not reuse it. Full decode_1 keeps the judge-built metadata unchanged (it already has no prepare kernel in its profile). Nothing else changes: query layout, pairing gate, epilogues, descale. This differs from the held fa3-scheduler-metadata-full-verify: that one launched the same scheduler serially (moving cost, not removing it) and used the unpaired geometry; this one runs it on a forked graph branch in parallel with the pre-attention kernel and uses the geometry of the actual call.

## Expected

- `*layer=sliding*` faster: the 1.7-1.8 us prepare kernel overlaps the 1.7-4.2 us pre-attention kernel instead of following it: ~1.5 us of 16.6-28.9 us, 5-9%
- `*layer=full load=decode_16*` faster: 3.1 us prepare off the critical path, ~2%
- `*layer=full load=decode_32*` faster: 3.5 us prepare off the critical path, ~1%
- `*layer=full load=verify_*` faster: 2.8-3.6 us prepare off the critical path, 1-9% (verify_1 most)
- `*layer=full load=decode_1 *` same: keeps the judge metadata; untouched

## Risk

Correctness risk: a schedule built for a different geometry than the call (headdim, num_heads_k, window, page_size, max_seqlen_k) can mis-size splits; the reviewer should confirm against sgl_kernel flash_api mha_fwd how it derives max_seqlen_k and pack_gqa for paged KV and that a passed metadata tensor fully replaces the in-call prepare (full decode_1 profile shows no prepare kernel when metadata is passed). Perf risk: a graph fork/join adds edges, not events, but if FA3 PDL overlap of prepare with pre-attention already hides part of it (GPU busy rows are ~1 us under the kernel sums) the gain shrinks to under 1 us. Not input-derived caching: the schedule is rebuilt every call; only the stream object is module-level. If the running fa3-window-trimmed-page-table lands, the sliding metadata must use its trimmed page table width and lengths.
