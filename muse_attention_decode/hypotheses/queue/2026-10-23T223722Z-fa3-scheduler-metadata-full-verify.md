---
problem: muse_attention_decode
slug: fa3-scheduler-metadata-full-verify
created: 2026-10-23T223722Z
base: master
area: FA3 call
change: >
  On full-attention verify blocks (config window == 0 and q_len == 8), where the judge passes
  scheduler_metadata=None (SGLang leaves the schedule to each call), build FA3's tile schedule
  inside kernel_fn and pass it: call sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B,
  max_seqlen_q=8, max_seqlen_k=page_table.shape[1]*page_size (a shape, not a value; the pool is
  sized for the longest possible length), num_heads=Hq, num_heads_k=Hkv, headdim=64,
  cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q,
  page_size=page_size, causal=True, num_splits=0) right before flash_attn_with_kvcache and pass
  its result as scheduler_metadata. It is a device-side kernel over cache_seqlens (no host sync;
  SGLang captures it in its own CUDA graphs), so it adds one small launch. Sliding layers and
  decode steps (which already carry the judge-built metadata) are untouched. The mechanism: with
  scheduler_metadata FA3 uses its precomputed dynamic split schedule balanced over the batch's
  actual lengths; without it, the verify path relies on the static heuristic, and at batch 16/32
  with 80K contexts (spread agentic, lengths 74K-86K) a balanced schedule may finish the longest
  tiles sooner.
expected:
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: 205 us vs 63 us floor; the decode_16 shape, which has the metadata, runs 201 us for the same bytes at q_len 1, so the metadata is worth testing at q_len 8
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: 390 us vs 127 us floor, same mechanism
  - shapes: "*layer=full load=verify_1*"
    direction: slower
    why: "one extra launch (~2-3 us) on a 35.6 us shape with 2 tiles: little to balance"
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched
  - shapes: "*load=decode_*"
    direction: flat
    why: untouched
risk: medium
depends_on: fused-preattention-triton
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-scheduler-metadata-full-verify

## Change

On full-attention verify blocks (config window == 0 and q_len == 8), where the judge passes scheduler_metadata=None (SGLang leaves the schedule to each call), build FA3's tile schedule inside kernel_fn and pass it: call sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=8, max_seqlen_k=page_table.shape[1]*page_size (a shape, not a value; the pool is sized for the longest possible length), num_heads=Hq, num_heads_k=Hkv, headdim=64, cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True, num_splits=0) right before flash_attn_with_kvcache and pass its result as scheduler_metadata. It is a device-side kernel over cache_seqlens (no host sync; SGLang captures it in its own CUDA graphs), so it adds one small launch. Sliding layers and decode steps (which already carry the judge-built metadata) are untouched. The mechanism: with scheduler_metadata FA3 uses its precomputed dynamic split schedule balanced over the batch's actual lengths; without it, the verify path relies on the static heuristic, and at batch 16/32 with 80K contexts (spread agentic, lengths 74K-86K) a balanced schedule may finish the longest tiles sooner.

## Expected

- `*layer=full load=verify_16*` faster: 205 us vs 63 us floor; the decode_16 shape, which has the metadata, runs 201 us for the same bytes at q_len 1, so the metadata is worth testing at q_len 8
- `*layer=full load=verify_32*` faster: 390 us vs 127 us floor, same mechanism
- `*layer=full load=verify_1*` slower: one extra launch (~2-3 us) on a 35.6 us shape with 2 tiles: little to balance
- `*layer=sliding*` flat: untouched
- `*load=decode_*` flat: untouched

## Risk

Medium risk: max_seqlen_k must be a Python int derived from shapes, never cache_seqlens.max() (a host sync, which breaks graph capture). The verify_1 full shape will likely lose one launch's worth; the load-level outcome is decided by whether verify_16/32 full shapes (2.4-3.4x the sliding shapes' load time) gain more than that. If get_scheduler_metadata's result makes no difference at q_len 8 (FA3 may ignore it when max_seqlen_q > 1 or when it is not a decode), the outcome is 'slower by one launch everywhere it is called' and the hypothesis is rejected cleanly.
