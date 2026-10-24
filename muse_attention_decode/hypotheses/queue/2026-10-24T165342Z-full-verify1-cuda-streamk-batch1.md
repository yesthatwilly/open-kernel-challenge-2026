---
problem: muse_attention_decode
slug: full-verify1-cuda-streamk-batch1
created: 2026-10-24T165342Z
base: master
area: full-layer attention (batch-1 verify gate)
change: >
  Stacked on full-verify-cuda-streamk-splitkv (stage 2). Config-only widening of
  _splitkv_verify_gate from 1 < batch <= 64 to 1 <= batch <= 64, so FULL-layer verify at batch 1
  (q_len 8, tp8, page 1) runs master _pre_attention_kernel (PAIRED=False, as master batch-1
  already is) + A8 splitkv_verify_kernel + B8 splitkv_verify_combine instead of
  prepare_varlen_num_blocks + unpaired FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate.
  Stream-K needs no batch-specific code: one 80K-key request is ~1250 tiles over the grid. One
  extra config-only choice: at batch == 1 use grid = SM count (132, one CTA per SM, ~9.5 tiles
  each) instead of 2 x SM count, so the partial traffic (one 32 KB [8 tokens x 16 heads x 64] fp32
  partial per CTA, all of them merged by B8 for the single request) stays ~4 MB instead of ~8.6 MB
  against a 20 MB gather; batch > 1 keeps 2 x SM count. The grid is passed through the existing
  grid argument; the workspace is sized grid + B as in stage 2. Full decode_1 (q_len 1) and every
  other shape are unchanged.
expected:
  - shapes: "*layer=full load=verify_1*"
    direction: faster
    why: "incumbent 30.3 us shape (FA3 main 14.9 + combine 7.7 + prepare 2.8 + pre-attn 1.5 + norm-gate 1.1) against a 4.3 us floor (20.5 MB of KV); A8 at one CTA per SM streams ~155 KB per CTA with 4 tiles in flight (~7-9 us at 2.5-3 TB/s) and B8 merges 132 partials per (token, head) (~2-3 us): ~13-15 us shape, ~2x on the shape; full is 37% of the verify_1 load, ~1.2x on that load"
  - shapes: "*layer=full load=verify_16* *layer=full load=verify_32*"
    direction: flat
    why: batch > 1 keeps the stage 2 grid and code
  - shapes: "*load=decode_* *layer=sliding*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: full-verify-cuda-streamk-splitkv
priority: 2
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T170604Z
review_companions: null
---

# full-verify1-cuda-streamk-batch1

## Change

Stacked on full-verify-cuda-streamk-splitkv (stage 2). Config-only widening of _splitkv_verify_gate from 1 < batch <= 64 to 1 <= batch <= 64, so FULL-layer verify at batch 1 (q_len 8, tp8, page 1) runs master _pre_attention_kernel (PAIRED=False, as master batch-1 already is) + A8 splitkv_verify_kernel + B8 splitkv_verify_combine instead of prepare_varlen_num_blocks + unpaired FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate. Stream-K needs no batch-specific code: one 80K-key request is ~1250 tiles over the grid. One extra config-only choice: at batch == 1 use grid = SM count (132, one CTA per SM, ~9.5 tiles each) instead of 2 x SM count, so the partial traffic (one 32 KB [8 tokens x 16 heads x 64] fp32 partial per CTA, all of them merged by B8 for the single request) stays ~4 MB instead of ~8.6 MB against a 20 MB gather; batch > 1 keeps 2 x SM count. The grid is passed through the existing grid argument; the workspace is sized grid + B as in stage 2. Full decode_1 (q_len 1) and every other shape are unchanged.

## Expected

- `*layer=full load=verify_1*` faster: incumbent 30.3 us shape (FA3 main 14.9 + combine 7.7 + prepare 2.8 + pre-attn 1.5 + norm-gate 1.1) against a 4.3 us floor (20.5 MB of KV); A8 at one CTA per SM streams ~155 KB per CTA with 4 tiles in flight (~7-9 us at 2.5-3 TB/s) and B8 merges 132 partials per (token, head) (~2-3 us): ~13-15 us shape, ~2x on the shape; full is 37% of the verify_1 load, ~1.2x on that load
- `*layer=full load=verify_16* *layer=full load=verify_32*` flat: batch > 1 keeps the stage 2 grid and code
- `*load=decode_* *layer=sliding*` flat: outside the gate

## Risk

Cheap follow-up once stage 2 passes correctness: one predicate and one grid choice. Batch-1 pairing lost 27-35% because of extra pad/unpad launches (LESSONS); this path has 3 launches (pre-attn, A8, B8) vs master 5, and no padding. Risk: B8 at batch 1 is 128 blocks each merging 132 partials (68 KB per block, ~8.6 MB total at 2 x SM; ~4.3 MB at 1 x SM), so check its own row in top_kernels; if B8 > 4 us, a two-level merge is the fix. Tests reaching the path: none at batch 1 page 1 full verify (batch 1 ctx 1048576 full q_len 8 is page 64), so the self-check must include a batch-1 q_len 8 synthetic case (80K-ish length and one short length) with the 1 x SM grid. If full-fused-triton-decode1-gate (running) lands, decode_1 is handled there; this hypothesis does not touch decode_1.

## Adversarial review

Reviewed 2026-10-24T170604Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a config-only widening of the stage-2 stream-K verify gate to batch 1, plus a 1 x SM grid. It replaces 5 launches (28 us of kernels) on a 30.3 us shape whose floor is 4.3 us, so 'faster' on full verify_1 is robust and the per-kernel rows attribute it without an A/B. The magnitude rests on B8 merging about 132 partials per (token, head) in 2-3 us, which stage 1's own combine does not support.

### Justification
The mechanism is plausible. FINDINGS gives full verify_1 = 30.3 us against SGLang 36.5 us. Its kernels are FA3 main 14.9, FlashAttnFwdCombine 7.7, prepare_varlen_num_blocks<1,true> 2.8, pre-attention ~1.5 and sglang rms_norm_gate ~1.1. That is 28.0 us of kernels for 20.5 MB of KV (80000 x 256 B, floor 4.3 us), so FA3 main alone runs at only ~1.4 TB/s and about half the shape is fixed-cost launches and the combine. Stream-K over 64-token tiles needs no batch logic: Ttot = 1250 tiles over 132 CTAs is about 9.5 tiles (~155 KB) per CTA. A 4-stage x 16 KB ring keeps ~8.4 MB in flight device-wide, which is about what Little's law needs at 2.5-3 TB/s with ~1.5 us gather latency. So A8 at ~7-10 us is believable, even with one CTA per SM taking away the second CTA's latency hiding and the ~1000 clk/tile smem/MUFU work the parent's reviewer estimated (about 5 us per CTA serially). The expected directions match TASK.md. Full is 16 x 30.3 = 485 us of the 1294 us verify_1 load (37%). Shape 30.3 -> 14 us would give ~1.25x on that load and ~+3.8% on the 6-load geomean. Every other shape is gated out (batch > 1 keeps the stage-2 grid, decode and sliding are untouched), so 'flat' there is right. WEAKEST LINK: the B8 estimate of ~2-3 us. Stage 1's kernel B uses the same structure (one 128-thread block per (request, head), 4 warps striding the partials, two passes: max, then a weighted sum). It merges ~79 partials per block and measures 7.9 us at decode_16 and 8.6 us at decode_32, nearly constant with batch, so it is latency-bound per block, not bandwidth-bound. B8 at batch 1 has only 128 blocks (8 tokens x 16 heads), each merging 132 slots strided 32 KB apart. Built the same way, B8 should be ~8-13 us, about what FA3's combine costs (7.7 us). At 2 x SM (264 slots) it would be worse still. Realistic shape: ~1.4 + 8 + 10 = ~20 us, ~1.14x on the load and ~+2.2% headline. That is still a gain, but near the ~1.5-2.5% run drift, so read the full verify_1 row and the A8/B8 top_kernels rows, not the headline. Build note: give B8 one-pass online merging with more lanes across slots (e.g. 8 warps, or each warp owning a contiguous slot range with independent prefetched loads, then a smem merge), or the two-level merge the Risk section names. Do this up front rather than as a follow-up, because B8's time decides whether the 1 x SM grid choice was right.

### Confounders on the current kernel
- full-verify-cuda-streamk-splitkv (parent, stage 2, not yet built or harvested): this hypothesis measures nothing unless the parent compiles and its verify self-check passes. If load_inline fails (e.g. the m16n8k32 e4m3 mma) or the verify self-check fails, the verify path is disabled and full verify_1 silently runs the master FA3 path. That reads as 'flat' and would be misattributed to this change. Read the log tail ('[splitkv] load_inline failed' / 'SELF-CHECK FAILED') before judging any shape. If the parent's module rename rebuilds the stage-1 decode kernels in the same module, a build failure also reverts decode_16/32.
- full-decode-cuda-gather-splitkv (stage 1, landed): B8 inherits stage 1's combine structure, which is latency-bound per block (7.9/8.6 us at ~79 partials). Whether this change gains a lot or a little depends mostly on how the parent implemented B8, not on the gate widening itself, so attribute it with the B8 row.
- paired-heads-launch-cleanup-ab-tune-batch-gate (landed): it is why batch 1 is unpaired on master (PAIRED=False, sglang rms_norm_gate epilogue). This change keeps PAIRED=False at batch 1, so the pre-attention q layout is unchanged and its ~1.5 us row should be byte-identical. Any pre-attention delta on the shape is drift, not part of the change.
- fused-preattention-triton (landed): A8 attends to the 8 new keys the pre-attention kernel stores just before it in the same stream. The causal limit t <= cache_seqlens - 8 + j relies on cache_seqlens = seq_lens + q_len. At batch 1 this holds as at batch > 1, so there is no confound beyond stream order.

### Interference and A/B needs
No companion is warranted. The gate is shape-disjoint: only full verify at batch 1, tp8, page 1 changes, and every other shape runs the parent's code byte for byte. The profile's per-shape top_kernels rows (pre-attn, A8, B8 against the five incumbent kernels the same run does not profile, taken from FINDINGS) attribute a gain or a loss directly. A gain means the gated path beats unpaired FA3 at batch 1. A loss is located by whichever of A8 or B8 overshoots, and the Risk section already names the fix (a two-level merge). An ab-revert of a landed hypothesis would not help: pairing is not used at batch 1, and reverting stage 1 or 2 removes the path itself. The one sub-change whose effect a single run cannot isolate is the 1 x SM vs 2 x SM grid. That is a tuning knob, not part of the mechanism, and the A8/B8 split shows its trade-off (A8 bandwidth vs B8 partial count) well enough to decide a follow-up. Spend the check on a stage that fixes B8 if B8 > 4 us, rather than on a grid control now.

### Open questions
- B8 design at batch 1: 128 blocks each merging 132 slots strided 32 KB apart. Stage 1's two-pass, 4-warp combine is latency-bound (~8 us at 79 partials). Will B8 be one-pass with independent prefetched loads, or more warps per (token, head)? Report B8's own row. If B8 > ~4 us, the next step is a two-level merge or a wider block, not a grid change.
- Self-check coverage: no judge test reaches batch-1, page-1 full verify (the batch-1 1M-ctx case is page 64), although the timed verify_1 calls are reference-checked. Add batch-1 q_len-8 synthetic cases at the 1 x SM grid: an ~80K length, a length where most of the 132 CTAs get empty ranges (e.g. cache_seqlens 8 and 70: Ttot 1-2 tiles), and cache_seqlens % 64 in 1..7. These confirm that empty slots weigh 0 in B8 and that the c_lo..c_hi inverse is exact when a single request spans every CTA.
- Does the parent's B8 locate c_lo..c_hi by an exact inverse of floor(c*Ttot/grid)? At B = 1 every CTA overlaps the request, so an off-by-one would drop or double-count a whole partial.
- If A8 at 1 x SM runs well below the parent's verify_16 bandwidth (e.g. < 2 TB/s on 20.5 MB), the 9.5-tile-per-CTA prologue (q fetch, index prefetch, ring fill) dominates. Then a grid of 2 x SM with a two-level merge may beat 1 x SM. That is a follow-up, read from the A8/B8 rows.
- Dispatch order: run only after the parent harvests with its verify self-check passing; otherwise this run measures master FA3 at batch 1.

