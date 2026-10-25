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
review: proceed
status: queued
review_at: 2026-10-25T203307Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T203307Z by an adversarial subagent (model opus).

**Verdict: proceed.** Deepening the ring of the existing <STAGES, two-level> stream-K kernel at grid = SM count for full decode_1 is untried, rules-clean and self-measuring via in-prepare variant timing. Likely gain is small (~+0.5-1% headline, inside CI), and B=1 stays slower than the unlanded c320 path (17.4 us). Read it on the per-variant kernel times. 12 stages does not fit shared memory; use 8 and 10 (or 11).

### Justification
At grid 132, ~9.5 tiles of 16 KB per CTA; 4 stages = 3 in flight (~1 meta round trip + ~3 dependent waves); 8 stages ~1 + 1.4: 1-2 fewer round trips (1-2 us each). Weakest link: most of the 19.0 us is prologue, per-tile compute and the two-level merge/finalize tail, which depth does not touch; steady-state bytes in flight (6.3 MB) already exceed Little's law. Smem: Lay<8> 164,896 B fits; Lay<12> 234,528 B > 232,448 B limit; Lay<10> 199,712 B, Lay<11> 217,120 B. Full decode_1 weight 16 at 22.4 us; a 3 us cut ~+0.9% headline. Prior (missing field): full-decode1-sk4-cluster-dsmem-merge, -sk-two-level-fused-preattn, -sk-pdl-from-preattn, -cuda-splitkv-chunk320/640; none changed ring depth.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket: defines the kernel and merge tail; flat means the tail dominates; [sk3-diag] per-variant times separate it.
- restack-sk-pseudo-cluster-plus-v3-fused-preattn / restack-sliding-decode-sk-window-on-cluster-master: PSEUDO instantiations (variants 4-7) of the same template; do not renumber them or touch Lay<4,PS_MAXR,true> / sk_smem / sk_prepare for v>=4; extra instantiations add compile time.
- Unlanded c320 path (17.4 us full decode_1): confounds the value, not the measurement.

### Interference and A/B needs
No companion: prepare times <4,two-level> vs 8/10 stages under the same cold-L2 graph replay and routes at >= 1.03x; the log plus the profile's full decode_1 top kernel read it.

### Open questions
- Is attention's only scored run best spent here (best case ~1.52-1.53 vs board 1.5347)?
- Was STAGES=5 or deeper timed at B=1 grid SM in any [sk3-diag] log?
- Phase split of the 19.0 us (merge/tail vs streaming) via clock64 in prepare.

