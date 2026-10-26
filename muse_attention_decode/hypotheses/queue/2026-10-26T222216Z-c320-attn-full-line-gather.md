---
problem: muse_attention_decode
slug: c320-attn-full-line-gather
created: 2026-10-26T222216Z
base: master
area: full-layer attention (batch-1 decode, c320 kernel A K/V gather mapping)
change: >
  Stacked on merge-perkvhead-v3chain-sv32-tiled (claim restacks its patch onto master). ONE change
  in the c320 module's c320_attn_kernel (both FUSED instantiations), inside its issue_tile lambda:
  the same whole-line cp.async mapping as sk-full-line-gather. Today tt = tid >> 2, e = t * TILE +
  tt and chunks c = (tid & 3) * 2 + j (j in {0, 1}), so a warp instruction covers 8 tokens x half
  of each sector of the 128-byte K / V lines. New: c = tid & 7 per thread, and for j in {0, 1}: tt
  = (tid >> 3) + 32 * j; e = t * TILE + tt; ok = ichunk * CHUNK + e < iseq; row = ok ?
  (int64_t)idx_s[(k & 1) * CHUNK + e] : 0; dst = tt * ROW + ((c ^ (tt & 7)) << 4); the same two
  cp_async16 calls (K at st + dst, V at st + TILE * ROW + dst, src + c * 16, ok ? 16 : 0). 4
  copies per thread per tile as now; the stage image is byte-identical, so compute, the FUSED
  prologue, the new-row handling and the combine are unchanged. GATE A (stored-row and output
  checks, two-state replays) and the V0..V4 chain gate re-run unchanged and pick the route; read
  c320_attn_kernel's profile time (v3chain b4b20425: 11.4 us).
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: "kernel A 11.4 us at ~1.8 TB/s, 250 one-item CTAs: half the lines per instruction may trim ~0.3-0.8 us at 0.34%/us"
  - shapes: "*layer=full load=decode_16 *"
    direction: flat
    why: c320 routes batch 1 only
  - shapes: "*layer=sliding *"
    direction: flat
    why: untouched
risk: low
depends_on: merge-perkvhead-v3chain-sv32-tiled
priority: 3
kind: primary
controls_for: null
prior: "splitkv-full-line-gather-mapping: same remap measured on stage-1 kernel A at B=16/32 (-4.7% / -14% kernel time); the B=1 c320 kernel (one 320-token item per CTA) copied A's half-line mapping and was never tried with it"
review: required
status: queued
---

# c320-attn-full-line-gather

## Change

Stacked on merge-perkvhead-v3chain-sv32-tiled (claim restacks its patch onto master). ONE change in the c320 module's c320_attn_kernel (both FUSED instantiations), inside its issue_tile lambda: the same whole-line cp.async mapping as sk-full-line-gather. Today tt = tid >> 2, e = t * TILE + tt and chunks c = (tid & 3) * 2 + j (j in {0, 1}), so a warp instruction covers 8 tokens x half of each sector of the 128-byte K / V lines. New: c = tid & 7 per thread, and for j in {0, 1}: tt = (tid >> 3) + 32 * j; e = t * TILE + tt; ok = ichunk * CHUNK + e < iseq; row = ok ? (int64_t)idx_s[(k & 1) * CHUNK + e] : 0; dst = tt * ROW + ((c ^ (tt & 7)) << 4); the same two cp_async16 calls (K at st + dst, V at st + TILE * ROW + dst, src + c * 16, ok ? 16 : 0). 4 copies per thread per tile as now; the stage image is byte-identical, so compute, the FUSED prologue, the new-row handling and the combine are unchanged. GATE A (stored-row and output checks, two-state replays) and the V0..V4 chain gate re-run unchanged and pick the route; read c320_attn_kernel's profile time (v3chain b4b20425: 11.4 us).

## Expected

- `*layer=full load=decode_1 *` faster: kernel A 11.4 us at ~1.8 TB/s, 250 one-item CTAs: half the lines per instruction may trim ~0.3-0.8 us at 0.34%/us
- `*layer=full load=decode_16 *` flat: c320 routes batch 1 only
- `*layer=sliding *` flat: untouched

## Risk

Small (+0.1-0.3%) and below the margin alone: the readout is c320_attn_kernel's profile time and the chain gate's V3/V4 times. Kernel A at B=1 is latency-bound (3 dependent DRAM round trips, 5 tiles per CTA), so the whole-line requests may not move it; that answer decides whether the decode d1f port carries the mapping. Stacks with c320-combine-onepass and c320-pdl-early-trigger (separate kernels / mechanisms on the same row).
