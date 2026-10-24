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
review: proceed
status: queued
review_at: 2026-10-24T184058Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T184058Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a plausible, rules-clean latency-hiding change to the single kernel on the sliding decode shapes (S=16, BLOCK_N=64, so 2 tiles per program, and this hoists half of them); judge it on the _sliding_fused_kernel profile row rather than the headline, because the expected ~1.5% headline gain sits inside the CI.

### Justification
The mechanism holds up. Every sliding decode shape spends 100% of its time in _sliding_fused_kernel (10.4us at decode_1 with only 32 CTAs, 12.9us at decode_16, 22.1us at decode_32). At decode_1 the floor is 0.1us, so the kernel is pure latency: a serial chain of dependent DRAM round trips under a cold L2. That chain is q load -> norm/rope -> seq load -> page-index load -> KV load, x2 tiles, then partial store -> ticket -> .cg reloads -> gate load. With count = 2047 and S = 16, chunk = 128 = 2 BLOCK_N tiles. Hoisting tile 0's seq->page->KV loads ahead of the q work removes about one round trip plus the norm/rope from the critical path. On a ~10us kernel that is roughly 0.5-1.5us (5-12%) at decode_1, consistent with the 'faster' call. The gain shrinks at decode_16/32, where 512/1024 CTAs over ~132 SMs make occupancy and bandwidth hide more latency; 'faster but smaller' is right there, and flat is equally likely. 'Flat' for full and verify is correct: kernel_fn routes only window && q_len == 1 here. Headline arithmetic: on the decode_1 load, the sliding shape is 46*13.2 = 607us vs full 16*27.3 = 437us. A 1us kernel gain is about 4.4% of that load and about 0.7% of the 6-load geomean, plus small decode_16/32 contributions, so the headline lands inside the +/-1.5% CI and the reported 2-3.5% decode_1 batch drift. Weakest link: the argument assumes the compiled code really does serialize as the source does. ptxas can already float the independent seq load above the q math. Triton's num_stages=2 pipeliner already issues the loop's first-iteration loads in a prologue, which today sits after the q and new-key work. If ptxas already overlaps part of the chain, the gain is smaller than estimated. Also, holding kb0/vb0 (2 x 64x64 fp8, ~16 regs/thread at 4 warps) live across _norm_rope and the S-1 new-key dot/stores adds register pressure. Spills would eat the gain; check the compiled kernel's n_spills/n_regs.

### Confounders on the current kernel
- sliding-deep-splits-on-cuda-splitkv: set S = 16 for every decode batch, which fixes tiles-per-program at 2. The hoisted tile's relative value depends directly on this (with fewer splits, more tiles per program, smaller relative gain), so the result is specific to S = 16 / BLOCK_N = 64. It is not a confound of attribution, though, since the split count is unchanged by this hypothesis.
- sliding-fused-triton-decode1-gate: introduced _sliding_fused_kernel and its fixed source ordering (q work before seq/page loads), the very thing being changed. No interaction beyond being the base.
- full-decode-cuda-gather-splitkv: does not touch the sliding path, but its run recorded a sliding decode_1 'loss' with no code change on that path. That is direct evidence of a 2-3.5% between-batch drift on exactly the shape this targets, so a headline-level sliding decode_1 move of that size is uninterpretable without the per-kernel row.

### Interference and A/B needs
No companion is warranted. The change is a single mechanism (instruction ordering) on one kernel, with no landed hypothesis whose revert or retune would change its sign. The split count only scales its size. The real ambiguity is noise, not interference, and the in-run SGLang pairing plus the profile's _sliding_fused_kernel row (10.4 / 12.9 / 22.1us today) resolve it better than a second run would. A num_stages or tile-count retune would test a different hypothesis (and sliding-decode-blockn128-single-tile already covers the tile-count axis). Read a >= 0.5us drop on the decode_1 kernel row as a gain and an unchanged or higher row as a null, regardless of the headline.

### Open questions
- The change text says seq is 'already loaded from seq_ptr' at the top, but in master `seq = tl.load(seq_ptr + b)` comes after the q load/norm/rope. The implementer must move the seq load up as well (and compute n there), or the hoist only partly overlaps.
- Does the compiled kernel spill once kb0/vb0 are live across _norm_rope and the S-1 new-key block? Compare n_regs/n_spills before and after; if it spills, reduce what stays live (for example, issue only the page and K loads early and V after the q work).
- Should the hoist cover both tiles of the 128-key chunk, not just tile 0? Today tile 1's loads still wait behind tile 0's dot/softmax unless the pipeliner prefetches them. This overlaps sliding-decode-blockn128-single-tile; if that one wins, this is subsumed.
- Scheduling note, not a defect: if sliding-cuda-cluster-dsmem-onelaunch (running) lands and replaces this kernel on every sliding shape at tp8/page 1, this becomes moot. Gate dispatch on that outcome.

