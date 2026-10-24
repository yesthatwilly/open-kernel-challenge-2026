---
problem: muse_attention_decode
slug: preattention-group-tiled-programs
created: 2026-10-24T120513Z
base: master
area: pre-attention norm+rope
change: >
  Regrid master _pre_attention_kernel from one 1-warp program per (token, head) ((tokens, Hq+Hkv)
  = 18 programs per token, 4608 at verify_32) to one program per (token, kv group): grid (tokens,
  Hkv), num_warps=4. Program (t, g) loads the G = Hq/Hkv = 8 query heads of group g as one [G, D]
  tile (rows h = g*G + r, columns even/odd halves as now, or a [G, D] tile plus a pair-swapped
  copy for RoPE as _norm_rope in the sliding-single-launch patch does), plus key head g and value
  head g as [1, D] rows; it applies the same scaleless RMSNorm per row (sum over axis 1, eps 1e-5,
  fp32), the same interleaved RoPE from positions[t] and the cached inv_freq when ROPE, writes the
  8 FP8 query rows (in the PAIRED layout, query h writes its half (h // G) % 2 of its 128-byte
  padded row and FP8 zeros into the other half, every byte written each call, exactly as now), and
  stores the FP8 key and value rows divided by k_scale/v_scale at cache_loc[t]. Arithmetic,
  rounding points and output layouts are identical to master; only the decomposition into programs
  changes. kernel_fn, FA3 and both epilogues are unchanged. Mechanism: the kernel moves under 1.2
  MB even at verify_32 (0.25 us at 4.8 TB/s) yet takes 4.2 us there, 2.7-2.8 us at verify_16 and
  ~2 us at decode shapes: it is bound by issuing thousands of single-warp CTAs each doing a
  dependent load-reduce-store chain, so 9x fewer, 4x wider programs should cut it toward ~1-1.5
  us.
expected:
  - shapes: "*load=verify_32*"
    direction: faster
    why: "pre-attention 4.2 us -> ~1.5 us: ~9% on sliding verify_32 (28.9 us), ~1% full"
  - shapes: "*load=verify_16*"
    direction: faster
    why: "pre-attention 2.8 us -> ~1.3 us: ~6% on sliding verify_16"
  - shapes: "*load=decode_32*"
    direction: faster
    why: "pre-attention 2.2-2.3 us -> ~1.2 us: ~4% sliding"
  - shapes: "*load=decode_16*"
    direction: faster
    why: pre-attention 1.8-2.0 us -> ~1.2 us, small
  - shapes: "*load=*_1 *"
    direction: same
    why: "1-8 tokens: 18-144 programs already fit one wave; latency per program may rise slightly with 8 rows each, expected within drift"
risk: low
depends_on: null
priority: 3
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-24T120745Z
review_companions: null
stage: claimed
claimed_at: 2026-10-24T120745Z
base_sha: f51cac06481cef9d3ac77248b4fec8d18a7cfb47
base_blob: efc544103e6d303d1f8602caa9439628043a17a7
branch: hyp/muse_attention_decode/preattention-group-tiled-programs
worktree: .worktrees/muse_attention_decode/preattention-group-tiled-programs
mode: null
---

# preattention-group-tiled-programs

## Change

Regrid master _pre_attention_kernel from one 1-warp program per (token, head) ((tokens, Hq+Hkv) = 18 programs per token, 4608 at verify_32) to one program per (token, kv group): grid (tokens, Hkv), num_warps=4. Program (t, g) loads the G = Hq/Hkv = 8 query heads of group g as one [G, D] tile (rows h = g*G + r, columns even/odd halves as now, or a [G, D] tile plus a pair-swapped copy for RoPE as _norm_rope in the sliding-single-launch patch does), plus key head g and value head g as [1, D] rows; it applies the same scaleless RMSNorm per row (sum over axis 1, eps 1e-5, fp32), the same interleaved RoPE from positions[t] and the cached inv_freq when ROPE, writes the 8 FP8 query rows (in the PAIRED layout, query h writes its half (h // G) % 2 of its 128-byte padded row and FP8 zeros into the other half, every byte written each call, exactly as now), and stores the FP8 key and value rows divided by k_scale/v_scale at cache_loc[t]. Arithmetic, rounding points and output layouts are identical to master; only the decomposition into programs changes. kernel_fn, FA3 and both epilogues are unchanged. Mechanism: the kernel moves under 1.2 MB even at verify_32 (0.25 us at 4.8 TB/s) yet takes 4.2 us there, 2.7-2.8 us at verify_16 and ~2 us at decode shapes: it is bound by issuing thousands of single-warp CTAs each doing a dependent load-reduce-store chain, so 9x fewer, 4x wider programs should cut it toward ~1-1.5 us.

## Expected

- `*load=verify_32*` faster: pre-attention 4.2 us -> ~1.5 us: ~9% on sliding verify_32 (28.9 us), ~1% full
- `*load=verify_16*` faster: pre-attention 2.8 us -> ~1.3 us: ~6% on sliding verify_16
- `*load=decode_32*` faster: pre-attention 2.2-2.3 us -> ~1.2 us: ~4% sliding
- `*load=decode_16*` faster: pre-attention 1.8-2.0 us -> ~1.2 us, small
- `*load=*_1 *` same: 1-8 tokens: 18-144 programs already fit one wave; latency per program may rise slightly with 8 rows each, expected within drift

## Risk

Low correctness risk (same math, same stores); the reviewer should check that the paired zero-half store and the value store remain complete for every byte and that the key program per group still writes to cache_loc with the page_size-64 test layout ((loc * HKV + kh) * D is unchanged). Perf risk at batch-1 shapes: 2 programs of 4 warps per token instead of 18 of 1 may lengthen the critical path by a few hundred ns; if so, a companion with grid (tokens, Hkv, 2) (4 query heads per program) is the fallback. Independent of fa3-scheduler-metadata-side-stream except that a shorter pre-attention kernel leaves less time to hide the side-stream prepare kernel under; they should be measured separately.

## Adversarial review

Reviewed 2026-10-24T120745Z by an adversarial subagent (model opus).

**Verdict: proceed.** The mechanism is sound and the master profile supports it: _pre_attention_kernel time rises roughly linearly with CTAs per SM. But the floor is about 1.5-2 us per launch even at 18 CTAs, so only verify_16/verify_32 (and a little of decode_32) can gain. The headline effect is about 0.5-1% geomean, inside the ±1.5% CI, so judge the run on the kernel's own profiled time per shape, not on the headline.

### Justification
Profile of the current best (bd64bf9f9e904f02, archive/paired-heads-launch-cleanup-ab-tune-batch-gate/…profile.txt) gives _pre_attention_kernel at 2.1/1.4 us decode_1 sliding/full (18 CTAs), 1.7/1.5 verify_1 (144), 1.8/2.0 decode_16 (288), 2.2/2.3 decode_32 (576), 2.8/2.8 verify_16 (2304) and 4.2/4.2 verify_32 (4608). Above roughly 4 CTAs per SM (132 SMs), time grows about 70-90 ns per extra 1-warp CTA per SM: (2.2->2.8->4.2 us at 4.4->17.5->35 CTA/SM). That fits per-CTA dispatch/issue cost, not bandwidth (under 1.2 MB, 0.25 us at 4.8 TB/s), so cutting the grid 9x to (tokens, Hkv) = 512 CTAs at verify_32 (about 3.9/SM) should bring verify_32 to about 1.8-2.2 us and verify_16 to about 1.7-2.0 us. Weakest link: the expected targets (~1.2-1.5 us) sit below the intercept the profile shows. Even 18 programs take 1.4-2.1 us, because each launch has a dependent chain: a cold-L2 DRAM load, the reduce, sin/cos when ROPE is on (sliding decode_1 2.1 us vs full 1.4 us), then the store. Regridding does not shorten that chain. So decode_16 (1.8 us) and decode_32 (2.2 us) will gain at most about 0.0-0.4 us, not the 4% claimed for sliding decode_32 (more like 1-1.5%). The verify_16 and verify_32 sliding claims (about 4-8% per shape) are plausible. Under the load weights (sliding x46, full x16, geomean over 6 loads), a 2.2 us saving at verify_32 is about 2.2% of that load, about 1 us at verify_16 is about 1.7%, and decode_32 is about 0.3%. Geomean about +0.6-0.8%, which cannot be resolved by one run. Triton kernel times reproduce within about 1% (LESSONS), so the _pre_attention_kernel row in top_kernels is the readable outcome. A secondary risk is batch-1 latency: 2 CTAs of 4 warps, each owning 8 q rows plus k plus v, instead of 18 1-warp CTAs. The chain depth is the same, but per-thread work is about 4x, plus the RoPE sin/cos on a [8,32] tile. Expect flat to +0.2 us there; it shows in the decode_1/verify_1 kernel row.

### Confounders on the current kernel
- fused-preattention-triton: this is the kernel being regridded and the base of the comparison, not an independent confounder. Its result shows the launch was the lever; this change only changes its decomposition, so any delta in that kernel row comes from the regrid itself.
- paired-heads-launch-cleanup (+ fa3-hdim128-head-pairing): on full layers at batch > 1 the query output is the zero-padded [tokens, Hq, 128] layout. Every query in group g shares half = g % 2, so a group program writes 8 rows to the same half and 8 zero halves; the write pattern per program changes, not the bytes. Full-layer pre-attention is only 0.7-1.6% of those shapes, so the interaction cannot move a full-layer row beyond drift. The implementation must keep every padded byte written (zero half included) on every call, or FA3 reads stale bytes from torch.empty.
- paired-heads-launch-cleanup-ab-tune-batch-gate: batch-1 full layers use the unpaired layout, so the batch-1 shapes (the regression-risk ones) exercise the PAIRED=False store path. Both constexpr branches must be regridded and tested. No measurement confound: the gate is config-only and unchanged.

### Interference and A/B needs
No companion is warranted. The change is confined to one kernel whose own time is profiled per shape and reproduces within about 1%, so a gain or a loss can be attributed without a revert. No landed hypothesis changes that kernel's grid economics (pairing changes only which half of the q row is written). An ab-revert of pairing would only move about 1% of the full-layer rows. The one plausible split result is verify_16/32 faster but decode_1/verify_1 slower. That is already readable from the same run's per-shape kernel rows, and the fix is a shape gate (keep the per-head grid when tokens*(Hq+Hkv) is below about 4 CTAs per SM, i.e. tokens < ~32), which belongs in a follow-up hypothesis, not a pre-emptive companion. The spec's fallback, grid (tokens, Hkv, 2), is also a follow-up only if batch-1 rows lose. With 5 attention checks left and fa3-window-trimmed-page-table in the judge, spend one check, not two or three.

### Open questions
- Is the ~1.5-2 us intercept (18 CTAs taking 1.4-2.1 us) cold-L2 DRAM latency on the qkv/positions/cache_loc loads, or the RoPE sin/cos chain? If it is the latter, precomputing cos/sin is not allowed (it is derived from positions), but ordering the loads so pos/inv_freq are issued before the qkv tile could hide it. Worth a follow-up only if verify rows gain as predicted.
- Should the new grid be gated on token count, with the per-head grid kept for decode_1/verify_1, if the batch-1 kernel rows regress by more than ~0.2 us? Decide from this run's per-shape _pre_attention_kernel rows.
- num_warps=4 with a [8,32] tile is 2 elements per thread per half. Would num_warps=2 cut per-CTA overhead further at verify_32? That is a cheap later ab-tune if the main run gains.

