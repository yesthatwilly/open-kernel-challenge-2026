---
problem: muse_attention_decode
slug: splitkv-l2-128b-prefetch-hint
created: 2026-10-24T183848Z
base: master
area: FA3 call (full-layer decode gather)
change: >
  One change in _SPLITKV_CUDA: the K/V cp_async16 helper used by issue_tile in splitkv_attn_kernel
  emits `cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;` instead of
  `cp.async.cg.shared.global [%0], [%1], 16, %2;`. Add it as a separate helper cp_async16_l2 and
  use it only for the four K/V copies in issue_tile; the q copy in issue_meta and the 4-byte
  page-index copies keep the existing helpers. The L2::128B qualifier is a prefetch-size hint (PTX
  ISA 7.4+, sm_80+): the first 16-byte request to a cache row brings the whole 128-byte line into
  L2, so the second half-row instruction hits. Nothing else changes; rename the module
  muse_full_decode_splitkv_v2.
expected:
  - shapes: "*layer=full load=decode_16 * *layer=full load=decode_32 *"
    direction: faster
    why: each 128-byte K or V row is fetched by two warp instructions 64 bytes apart; a whole-line L2 fill may cut DRAM transactions per row from two to one. Expect 0-5% on kernel A (111.8 / 245.9 us)
  - shapes: "*layer=sliding* *load=verify_* *load=decode_1 *"
    direction: flat
    why: these shapes do not run kernel A
risk: low
depends_on: null
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# splitkv-l2-128b-prefetch-hint

## Change

One change in _SPLITKV_CUDA: the K/V cp_async16 helper used by issue_tile in splitkv_attn_kernel emits `cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;` instead of `cp.async.cg.shared.global [%0], [%1], 16, %2;`. Add it as a separate helper cp_async16_l2 and use it only for the four K/V copies in issue_tile; the q copy in issue_meta and the 4-byte page-index copies keep the existing helpers. The L2::128B qualifier is a prefetch-size hint (PTX ISA 7.4+, sm_80+): the first 16-byte request to a cache row brings the whole 128-byte line into L2, so the second half-row instruction hits. Nothing else changes; rename the module muse_full_decode_splitkv_v2.

## Expected

- `*layer=full load=decode_16 * *layer=full load=decode_32 *` faster: each 128-byte K or V row is fetched by two warp instructions 64 bytes apart; a whole-line L2 fill may cut DRAM transactions per row from two to one. Expect 0-5% on kernel A (111.8 / 245.9 us)
- `*layer=sliding* *load=verify_* *load=decode_1 *` flat: these shapes do not run kernel A

## Risk

Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs). Overlaps in mechanism with splitkv-full-line-gather-mapping (both target half-line requests); if both run, they are an A/B on the same hypothesis, so the reviewer may hold one.
