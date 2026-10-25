---
problem: muse_attention_decode
slug: full-decode1-sk-two-level-deep-ring
created: 2026-10-25T153027Z
base: master
area: full-layer attention (batch-1 decode, stream-K ring depth)
change: >
  Built on current master (no depends_on). Full decode_1 runs splitkv_attn_sk_kernel<4 stages,
  two-level ticket> at grid = SM count: 19.0 us kernel + pre-attn 1.4 in a 22.4 us shape, 0.9 TB/s
  on 20 MB (s5 log [sk3-diag]: 22.9 us eager two-level grid SM). At grid SM there is ONE CTA per
  SM holding ~9.5 tiles of 64 tokens, and the 4-stage ring keeps only 3 tiles (48 KB) in flight,
  so the CTA needs ~3 dependent DRAM round trips for its range. Grid 2xSM (more in flight) was
  slower (24.4 us) because it doubles the partials the two-level merge reads, so ring depth at
  grid SM is the untested lever that adds bytes in flight without adding partials. CHANGE:
  instantiate the existing kernel template at STAGES = 8 (and 12 if it fits) with TWO = true (the
  Lay<STAGES> layout grows by 16 KB per stage: ~160 KB at 8, ~225 KB at 12; 1 CTA/SM at grid SM is
  already the occupancy, so nothing is lost), add the variants to the B=1 [sk3-diag]-style prepare
  timing (interleaved cold-L2 graph replay of pre-attn + kernel, empty graph subtracted, median of
  31, lens 80000), and route B=1 to the fastest only at >= 1.03x vs master's <4, two-level> at
  grid SM, with rel L2 < 1e-2 vs master; else master. Self-check the routed variant on master's
  B=1 two-level cases (80000 / 4097 / 63 / 1 keys, kv 1 and 0.25, grids 3/7/SM) with tickets
  summing to 0, plus the two-state alternating replay. Print regs/local/smem per variant. Batch >
  1 untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no
  allocation sized by input values (shape/config only), no data-dependent Python control flow, no
  prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q
  at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child
  processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no
  module/static global may hold a device pointer into a state's tensors, a TMA map or a
  state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every
  launch, are fine).
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: kernel 19.0 us at 0.9 TB/s vs floor 4.3 us; 7 tiles in flight per CTA instead of 3 should remove 1-2 of ~3 round trips (-2 to -4 us)
  - shapes: "*layer=full load=decode_16 *"
    direction: flat
    why: B>1 untouched
  - shapes: "*layer=full load=decode_32 *"
    direction: flat
    why: B>1 untouched
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
  - shapes: "*load=verify_* *"
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# full-decode1-sk-two-level-deep-ring

## Change

Built on current master (no depends_on). Full decode_1 runs splitkv_attn_sk_kernel<4 stages, two-level ticket> at grid = SM count: 19.0 us kernel + pre-attn 1.4 in a 22.4 us shape, 0.9 TB/s on 20 MB (s5 log [sk3-diag]: 22.9 us eager two-level grid SM). At grid SM there is ONE CTA per SM holding ~9.5 tiles of 64 tokens, and the 4-stage ring keeps only 3 tiles (48 KB) in flight, so the CTA needs ~3 dependent DRAM round trips for its range. Grid 2xSM (more in flight) was slower (24.4 us) because it doubles the partials the two-level merge reads, so ring depth at grid SM is the untested lever that adds bytes in flight without adding partials. CHANGE: instantiate the existing kernel template at STAGES = 8 (and 12 if it fits) with TWO = true (the Lay<STAGES> layout grows by 16 KB per stage: ~160 KB at 8, ~225 KB at 12; 1 CTA/SM at grid SM is already the occupancy, so nothing is lost), add the variants to the B=1 [sk3-diag]-style prepare timing (interleaved cold-L2 graph replay of pre-attn + kernel, empty graph subtracted, median of 31, lens 80000), and route B=1 to the fastest only at >= 1.03x vs master's <4, two-level> at grid SM, with rel L2 < 1e-2 vs master; else master. Self-check the routed variant on master's B=1 two-level cases (80000 / 4097 / 63 / 1 keys, kv 1 and 0.25, grids 3/7/SM) with tickets summing to 0, plus the two-state alternating replay. Print regs/local/smem per variant. Batch > 1 untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## Expected

- `*layer=full load=decode_1 *` faster: kernel 19.0 us at 0.9 TB/s vs floor 4.3 us; 7 tiles in flight per CTA instead of 3 should remove 1-2 of ~3 round trips (-2 to -4 us)
- `*layer=full load=decode_16 *` flat: B>1 untouched
- `*layer=full load=decode_32 *` flat: B>1 untouched
- `*layer=sliding *` flat: untouched
- `*load=verify_* *` flat: untouched

## Risk

5 stages gave nothing at B=16 (bandwidth-bound there); B=1 is latency-bound, which is why depth may pay only here. Full decode_1 is 40% of the decode_1 load.
