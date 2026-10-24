---
problem: muse_attention_decode
slug: sliding-fused-hoist-first-tile
created: 2026-10-24T183855Z
base: master
area: sliding-layer attention (batched decode, gather prologue)
change: >
  In master _sliding_fused_kernel, issue the first old-key tile's gather before the query work so
  the KV round trip overlaps the q load, RMSNorm and RoPE. Move the computation of lo, count,
  chunk, start, end (it needs only seq = seq_lens[b], already loaded from seq_ptr) to the top of
  the kernel, right after `d`/`freq`, then load tile 0: t0 = start + n, tmask0 = t0 < end, page0 =
  tl.load(pt_ptr + b * pt_stride + t0 // PAGE, mask=tmask0, other=0), and kb0 / vb0 with the same
  kvoff formula as the loop. Then the existing q load + _norm_rope + the `if s == S - 1` new-key
  block run as now (its stores go to cache_loc rows, which are positions >= seq and never in [lo,
  seq - 1], so the hoisted loads cannot read them). After it, process kb0/vb0 with the same qk,
  valid mask (t0 >= p_row - (WINDOW - 1)) and _online_update call as the loop body, then run the
  loop from start + BLOCK_N. The partial store, ticket and combine are unchanged; verify (q_len 8,
  not routed here today) is unaffected.
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "every program today serializes q load -> norm/rope -> page-index load -> KV load; hoisting overlaps ~1-2 us of the 10.4 us kernel's dependency chain, and on split S-1 also the new-key store"
  - shapes: "*layer=sliding load=decode_16 * *layer=sliding load=decode_32 *"
    direction: faster
    why: same chain in 512 / 1024 programs; smaller relative gain
  - shapes: "*layer=full* *load=verify_*"
    direction: flat
    why: not on the sliding decode path
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-fused-hoist-first-tile

## Change

In master _sliding_fused_kernel, issue the first old-key tile's gather before the query work so the KV round trip overlaps the q load, RMSNorm and RoPE. Move the computation of lo, count, chunk, start, end (it needs only seq = seq_lens[b], already loaded from seq_ptr) to the top of the kernel, right after `d`/`freq`, then load tile 0: t0 = start + n, tmask0 = t0 < end, page0 = tl.load(pt_ptr + b * pt_stride + t0 // PAGE, mask=tmask0, other=0), and kb0 / vb0 with the same kvoff formula as the loop. Then the existing q load + _norm_rope + the `if s == S - 1` new-key block run as now (its stores go to cache_loc rows, which are positions >= seq and never in [lo, seq - 1], so the hoisted loads cannot read them). After it, process kb0/vb0 with the same qk, valid mask (t0 >= p_row - (WINDOW - 1)) and _online_update call as the loop body, then run the loop from start + BLOCK_N. The partial store, ticket and combine are unchanged; verify (q_len 8, not routed here today) is unaffected.

## Expected

- `*layer=sliding load=decode_1 *` faster: every program today serializes q load -> norm/rope -> page-index load -> KV load; hoisting overlaps ~1-2 us of the 10.4 us kernel's dependency chain, and on split S-1 also the new-key store
- `*layer=sliding load=decode_16 * *layer=sliding load=decode_32 *` faster: same chain in 512 / 1024 programs; smaller relative gain
- `*layer=full* *load=verify_*` flat: not on the sliding decode path

## Risk

Touches master's _sliding_fused_kernel, which sliding-cuda-cluster-dsmem-onelaunch (in the judge now) replaces on every sliding shape at tp8/page 1; if that rewrite lands this is moot and should be held. Useful only as a tuning of the incumbent if the rewrite falls back. Judge on the _sliding_fused_kernel row, not the headline (sliding decode_1 drifted 2-3.5% between run batches). Mechanism-overlapping with sliding-decode-blockn128-single-tile; if both run they combine trivially (at BLOCK_N 128 the hoisted tile is the whole chunk).
