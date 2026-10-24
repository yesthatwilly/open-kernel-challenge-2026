---
problem: muse_attention_decode
slug: splitkv-full-line-gather-mapping
created: 2026-10-24T183847Z
base: master
area: FA3 call (full-layer decode gather)
change: >
  One change inside splitkv_attn_kernel's issue_tile lambda (master full decode, 1 < batch <= 64):
  remap threads so that each warp-wide cp.async instruction covers whole 128-byte cache rows
  instead of half rows. Today thread tid copies token tt = tid >> 2, 16-byte chunks c = 2 * (tid &
  3) + {0, 1}, so one warp instruction touches 8 tokens x 64 bytes (two sectors of each line) and
  the other half of every line arrives from the next instruction. New mapping: for j in {0, 1}: tt
  = (tid >> 3) + 32 * j, c = tid & 7; the destination stays dst = tt * ROW + ((c ^ (tt & 7)) << 4)
  for K and TILE * ROW + dst for V, the validity test and the page index read idx_s[(k & 1) *
  CHUNK + t * TILE + tt] are computed per tt as now, and the source is ks + c * 16 / vs + c * 16
  with src_bytes ok ? 16 : 0. Each thread still issues 4 cp.async16 per tile (2 tokens x K and V),
  the ring layout, swizzle, commit/wait counts and all compute are unchanged, so the smem contents
  per stage are byte-identical to master's. Rename the load_inline module to
  muse_full_decode_splitkv_v2 (no stale build). The import self-check runs unchanged.
expected:
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: faster
    why: kernel A gathers random page-1 rows at 2.94 TB/s (decode_16, 111.8 us) and 2.67 TB/s (decode_32, 245.9 us) against 4.8 TB/s; a full 128-byte line per L2 request should let L2 issue one 128 B DRAM fetch per row instead of two 64 B ones arriving apart, worth an estimated 3-8% on kernel A
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: these shapes do not run kernel A
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# splitkv-full-line-gather-mapping

## Change

One change inside splitkv_attn_kernel's issue_tile lambda (master full decode, 1 < batch <= 64): remap threads so that each warp-wide cp.async instruction covers whole 128-byte cache rows instead of half rows. Today thread tid copies token tt = tid >> 2, 16-byte chunks c = 2 * (tid & 3) + {0, 1}, so one warp instruction touches 8 tokens x 64 bytes (two sectors of each line) and the other half of every line arrives from the next instruction. New mapping: for j in {0, 1}: tt = (tid >> 3) + 32 * j, c = tid & 7; the destination stays dst = tt * ROW + ((c ^ (tt & 7)) << 4) for K and TILE * ROW + dst for V, the validity test and the page index read idx_s[(k & 1) * CHUNK + t * TILE + tt] are computed per tt as now, and the source is ks + c * 16 / vs + c * 16 with src_bytes ok ? 16 : 0. Each thread still issues 4 cp.async16 per tile (2 tokens x K and V), the ring layout, swizzle, commit/wait counts and all compute are unchanged, so the smem contents per stage are byte-identical to master's. Rename the load_inline module to muse_full_decode_splitkv_v2 (no stale build). The import self-check runs unchanged.

## Expected

- `*layer=full load=decode_16 * *layer=full load=decode_32 *` faster: kernel A gathers random page-1 rows at 2.94 TB/s (decode_16, 111.8 us) and 2.67 TB/s (decode_32, 245.9 us) against 4.8 TB/s; a full 128-byte line per L2 request should let L2 issue one 128 B DRAM fetch per row instead of two 64 B ones arriving apart, worth an estimated 3-8% on kernel A
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: these shapes do not run kernel A

## Risk

Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs). The rewrite's BULK=0 fallback uses this same full-line mapping, so a gain here also tells stage 2 of that rewrite which copy form to keep.
