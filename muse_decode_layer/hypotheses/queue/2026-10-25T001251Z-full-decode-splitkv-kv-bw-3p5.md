---
problem: muse_decode_layer
slug: full-decode-splitkv-kv-bw-3p5
created: 2026-10-25T001251Z
base: master
area: "FA3 call (full decode, batch > 1): splitkv_attn_kernel bytes in flight"
change: >
  Stage 2 of full-decode-cuda-splitkv-port (REWRITE_READ_2026-10-25 section 2, item 2), built on
  the restacked code from full-decode-splitkv-restack-on-pdl. Kernel A (splitkv_attn_kernel) is
  93-96% of the replaced time and reaches only 2.9 TB/s at B=16 (112.0 us) and 2.6 TB/s at B=32
  (249.5 us) vs a KV floor of ~63/126 us; it has STAGES=4 x 16 KB ring at 2 CTAs/SM (~96 KB in
  flight per SM) over a persistent 2xSM grid of CHUNK=1024-token items, and at B=32 the
  agentic-spread tail leaves SMs idle. Change only kernel A and its host grid/chunk constants: (a)
  deeper ring: STAGES 6 if per-CTA smem (ring + idx [2][CHUNK] + q + acc/ml) stays <= ~113 KB so 2
  CTAs/SM hold, else STAGES 5; keep the cp.async.cg 16-byte form and the wait_group<STAGES-2>
  logic; (b) smaller chunk where the tail dominates: make CHUNK a template parameter with
  instantiations 1024 and 512 and pick 512 when total KV tokens / (2 x SMs) is below ~3 chunks of
  1024 or batch >= 24 (host-side from config and cache_seqlens-free sizes only: decide from batch,
  no host sync), with the combine's split count sized accordingly (S per sequence doubles; combine
  reads ws by item exactly as now); (c) issue the next item's page-index prefetch two items ahead
  instead of one. Combine kernel, gate (_splitkv_gate) and everything else unchanged. Self-check:
  rerun the existing _splitkv_self_check cases (B=5 and B=33, kv_scale 1.0/0.25, device grid and
  grid 3) for every (STAGES, CHUNK) instantiation used; any failure falls back to the stage-1
  configuration (STAGES 4, CHUNK 1024), not to FA3. Print [splitkv-s2] stages/chunk/occupancy and
  kernel-A CUDA-event time + effective TB/s at B=16 and 32 for old and new configs.
expected:
  - shapes: "*layer=full load=decode_16 *"
    direction: faster
    why: "kernel A 112.0 us at 2.9 TB/s -> ~90-95 us at >=3.5 TB/s (M3 <=95 us, floor ~63): ~17-22 us off the restacked shape"
  - shapes: "*layer=full load=decode_32 *"
    direction: faster
    why: "kernel A 249.5 us at 2.6 TB/s -> ~185-190 us at >=3.5 TB/s with the 512 chunk draining the agentic tail (M3 <=189 us, floor ~126): ~55-60 us off"
  - shapes: "*layer=sliding *"
    direction: flat
    why: "outside gate: control"
  - shapes: "*load=verify_*"
    direction: flat
    why: "outside gate: control"
  - shapes: "*load=decode_1 *"
    direction: flat
    why: "batch 1 outside gate: control"
risk: medium
depends_on: full-decode-splitkv-restack-on-pdl
priority: 3
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: paired FA3 flash_fwd + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full decode, 1 < batch <= 64 (now splitkv_attn_kernel + splitkv_combine_kernel)
review: required
status: queued
---

# full-decode-splitkv-kv-bw-3p5

## Change

Stage 2 of full-decode-cuda-splitkv-port (REWRITE_READ_2026-10-25 section 2, item 2), built on the restacked code from full-decode-splitkv-restack-on-pdl. Kernel A (splitkv_attn_kernel) is 93-96% of the replaced time and reaches only 2.9 TB/s at B=16 (112.0 us) and 2.6 TB/s at B=32 (249.5 us) vs a KV floor of ~63/126 us; it has STAGES=4 x 16 KB ring at 2 CTAs/SM (~96 KB in flight per SM) over a persistent 2xSM grid of CHUNK=1024-token items, and at B=32 the agentic-spread tail leaves SMs idle. Change only kernel A and its host grid/chunk constants: (a) deeper ring: STAGES 6 if per-CTA smem (ring + idx [2][CHUNK] + q + acc/ml) stays <= ~113 KB so 2 CTAs/SM hold, else STAGES 5; keep the cp.async.cg 16-byte form and the wait_group<STAGES-2> logic; (b) smaller chunk where the tail dominates: make CHUNK a template parameter with instantiations 1024 and 512 and pick 512 when total KV tokens / (2 x SMs) is below ~3 chunks of 1024 or batch >= 24 (host-side from config and cache_seqlens-free sizes only: decide from batch, no host sync), with the combine's split count sized accordingly (S per sequence doubles; combine reads ws by item exactly as now); (c) issue the next item's page-index prefetch two items ahead instead of one. Combine kernel, gate (_splitkv_gate) and everything else unchanged. Self-check: rerun the existing _splitkv_self_check cases (B=5 and B=33, kv_scale 1.0/0.25, device grid and grid 3) for every (STAGES, CHUNK) instantiation used; any failure falls back to the stage-1 configuration (STAGES 4, CHUNK 1024), not to FA3. Print [splitkv-s2] stages/chunk/occupancy and kernel-A CUDA-event time + effective TB/s at B=16 and 32 for old and new configs.

## Expected

- `*layer=full load=decode_16 *` faster: kernel A 112.0 us at 2.9 TB/s -> ~90-95 us at >=3.5 TB/s (M3 <=95 us, floor ~63): ~17-22 us off the restacked shape
- `*layer=full load=decode_32 *` faster: kernel A 249.5 us at 2.6 TB/s -> ~185-190 us at >=3.5 TB/s with the 512 chunk draining the agentic tail (M3 <=189 us, floor ~126): ~55-60 us off
- `*layer=sliding *` flat: outside gate: control
- `*load=verify_*` flat: outside gate: control
- `*load=decode_1 *` flat: batch 1 outside gate: control

## Risk

Medium: 6 stages may drop residency to 1 CTA/SM (print occupancy; the fallback rule picks 5). A 512 chunk doubles combine work (8-9 us now); if combine grows past ~12 us, keep 1024 at B=16. Must not start until full-decode-splitkv-restack-on-pdl lands (it needs a hand merge); build on whatever that run measured.

## Milestones

Rewrite stage 2 of `paired FA3 flash_fwd + FlashAttnFwdCombine + pad/unpad copies + FusedRMSNormGate on full decode, 1 < batch <= 64 (now splitkv_attn_kernel + splitkv_combine_kernel)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.
