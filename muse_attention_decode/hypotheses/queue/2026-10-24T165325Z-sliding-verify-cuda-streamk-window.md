---
problem: muse_attention_decode
slug: sliding-verify-cuda-streamk-window
created: 2026-10-24T165325Z
base: master
area: sliding-layer attention (verify)
change: >
  Stacked on full-verify-cuda-streamk-splitkv (stage 2: kernels splitkv_verify_kernel A8 +
  splitkv_verify_combine B8, stream-K over 64-token tiles, q_len 8 on the MMA M dimension). Route
  SLIDING-window verify (cfg window != 0, q_len == 8, 1 <= batch <= 64, hq 16, hkv 2, head_dim 64,
  page_size 1; config-only predicate) through A8 + B8 instead of the unpaired FA3 path
  (prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate). Master
  _pre_attention_kernel is unchanged and runs first (ROPE on, PAIRED=False), so q and the new
  cache rows are already roped and stored. Kernel edits, one new int argument window (0 = full,
  the stage 2 behaviour, which must compile and run unchanged on full verify): (1) each request
  key range becomes [lo_b, seq_b) with seq_b = cache_seqlens[b] and lo_b = window ? max(0, seq_b -
  8 - (window - 1)) : 0, computed in-kernel from cache_seqlens; tiles_b = ceil((seq_b - lo_b) /
  64), tile k of request b covers keys lo_b + 64k .. +63 (not 64-aligned; page 1 means every key
  has its own page_table entry so alignment does not matter), and B8 recomputes the same TP
  prefix; (2) the per-row mask adds the window lower edge: key t is valid for block token j iff t
  <= seq_b - 8 + j and (window == 0 or t >= seq_b - 8 + j - (window - 1)) and t >= lo_b; it is
  applied on the request first and last tiles (a 2055-key range is 33 tiles). The window is
  exactly FA3 window_size=(window - 1, 0) as master passes it, so every key in each query window
  is attended. Grid stays 2 x SM count from the device: at sliding verify_1 Ttot = 33 tiles, so 33
  CTAs each own one tile and the rest exit on an empty range (no data-dependent grid); workspace
  slots grid + B as in stage 2. Self-check: add sliding q_len 8 synthetic cases (window 2048, lens
  including one < window and one < 8 + window, kv_scale 0.25) against an FP32 windowed causal
  reference; on failure disable only the sliding route. Sliding decode (q_len 1, the fused Triton
  kernel / FA3) and all full shapes keep their master code.
expected:
  - shapes: "*layer=sliding load=verify_1*"
    direction: faster
    why: "incumbent 17.6 us shape (FA3 main 7.4 + combine 3.8 + prepare 1.8 + pre-attn 1.6 + norm-gate 1.1) against a 0.1 us floor (0.5 MB): pure latency of 5 launches; A8 spreads the 33 tiles over 33 CTAs (one dependent index->gather->mma per CTA, ~3-4 us) and B8 merges 33 partials (~2 us): ~9-11 us shape. Sliding is 62% of the verify_1 load at weight 46, so ~1.25x on that load"
  - shapes: "*layer=sliding load=verify_16*"
    direction: faster
    why: incumbent 22.8 us (FA3 10.7 + combine 4.1 + prepare 1.9 + pre 2.7 + norm 1.3); 528 tiles over 264 CTAs = 2 tiles each, 8.4 MB; target ~13-15 us; sliding is 29% of the verify_16 load
  - shapes: "*layer=sliding load=verify_32*"
    direction: faster
    why: incumbent 29.3 us (FA3 14.7 + combine 4.2 + prepare 1.8 + pre 4.2 + norm 1.5); 1056 tiles, 4 per CTA, 17 MB; target ~16-19 us; sliding is 22% of the verify_32 load
  - shapes: "*layer=full*"
    direction: flat
    why: window = 0 instantiation is the stage 2 code path
  - shapes: "*layer=sliding load=decode_*"
    direction: flat
    why: q_len 1 outside the gate
risk: high
depends_on: full-verify-cuda-streamk-splitkv
priority: 2
kind: primary
controls_for: null
review: proceed
status: queued
review_at: 2026-10-24T170604Z
review_companions: null
---

# sliding-verify-cuda-streamk-window

## Change

Stacked on full-verify-cuda-streamk-splitkv (stage 2: kernels splitkv_verify_kernel A8 + splitkv_verify_combine B8, stream-K over 64-token tiles, q_len 8 on the MMA M dimension). Route SLIDING-window verify (cfg window != 0, q_len == 8, 1 <= batch <= 64, hq 16, hkv 2, head_dim 64, page_size 1; config-only predicate) through A8 + B8 instead of the unpaired FA3 path (prepare_varlen_num_blocks + FA3 main + FlashAttnFwdCombine + sglang rms_norm_gate). Master _pre_attention_kernel is unchanged and runs first (ROPE on, PAIRED=False), so q and the new cache rows are already roped and stored. Kernel edits, one new int argument window (0 = full, the stage 2 behaviour, which must compile and run unchanged on full verify): (1) each request key range becomes [lo_b, seq_b) with seq_b = cache_seqlens[b] and lo_b = window ? max(0, seq_b - 8 - (window - 1)) : 0, computed in-kernel from cache_seqlens; tiles_b = ceil((seq_b - lo_b) / 64), tile k of request b covers keys lo_b + 64k .. +63 (not 64-aligned; page 1 means every key has its own page_table entry so alignment does not matter), and B8 recomputes the same TP prefix; (2) the per-row mask adds the window lower edge: key t is valid for block token j iff t <= seq_b - 8 + j and (window == 0 or t >= seq_b - 8 + j - (window - 1)) and t >= lo_b; it is applied on the request first and last tiles (a 2055-key range is 33 tiles). The window is exactly FA3 window_size=(window - 1, 0) as master passes it, so every key in each query window is attended. Grid stays 2 x SM count from the device: at sliding verify_1 Ttot = 33 tiles, so 33 CTAs each own one tile and the rest exit on an empty range (no data-dependent grid); workspace slots grid + B as in stage 2. Self-check: add sliding q_len 8 synthetic cases (window 2048, lens including one < window and one < 8 + window, kv_scale 0.25) against an FP32 windowed causal reference; on failure disable only the sliding route. Sliding decode (q_len 1, the fused Triton kernel / FA3) and all full shapes keep their master code.

## Expected

- `*layer=sliding load=verify_1*` faster: incumbent 17.6 us shape (FA3 main 7.4 + combine 3.8 + prepare 1.8 + pre-attn 1.6 + norm-gate 1.1) against a 0.1 us floor (0.5 MB): pure latency of 5 launches; A8 spreads the 33 tiles over 33 CTAs (one dependent index->gather->mma per CTA, ~3-4 us) and B8 merges 33 partials (~2 us): ~9-11 us shape. Sliding is 62% of the verify_1 load at weight 46, so ~1.25x on that load
- `*layer=sliding load=verify_16*` faster: incumbent 22.8 us (FA3 10.7 + combine 4.1 + prepare 1.9 + pre 2.7 + norm 1.3); 528 tiles over 264 CTAs = 2 tiles each, 8.4 MB; target ~13-15 us; sliding is 29% of the verify_16 load
- `*layer=sliding load=verify_32*` faster: incumbent 29.3 us (FA3 14.7 + combine 4.2 + prepare 1.8 + pre 4.2 + norm 1.5); 1056 tiles, 4 per CTA, 17 MB; target ~16-19 us; sliding is 22% of the verify_32 load
- `*layer=full*` flat: window = 0 instantiation is the stage 2 code path
- `*layer=sliding load=decode_*` flat: q_len 1 outside the gate

## Risk

Why this should not repeat sliding-single-launch-fused-attention (0.945x, lost 2-2.6x on verify): that Triton kernel ran one program per (request, kv head) over the whole 33-tile window serially (~1.5-1.8 us per dependent tile); A8 parallelises the window over tiles (stream-K), keeps page indices prefetched and the 64 query rows per kv head on the tensor-core M dimension, so the verify work per CTA is 1-4 tiles. Tests that reach the new route: batch 5 ctx 5000 sliding q_len 8 spread mixed page 1; batch 3 ctx 4000 sliding q_len 8 page 1 kv_scale 1/64; batch 4 ctx 6000 sliding q_len 8 dist soft page 1 (keys at the window edges are covered by the edge mask on the first tile). Risk: the B8 merge at batch 1 is a single request over 33 slots (fine); launch latency floor of 3 kernels ~6-7 us bounds verify_1. Depends on stage 2 landing (or at least passing correctness); if stage 2 only reaches M1, this is still worth running because sliding verify is latency-bound, not bandwidth-bound.

## Adversarial review

Reviewed 2026-10-24T170604Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound and shape-disjoint. It reuses the parent's stream-K A8/B8 for sliding verify, where all three shapes are latency-bound (1-4 tiles per CTA) and the FA3 path pays for 5 launches. Dispatch only after full-verify-cuda-streamk-splitkv passes its self-check and the 16 tests. Fix the edge-mask predicate and the build/self-check coupling below. The verify_1 target (9-11 us) is optimistic; ~12 us is more likely, which is still a clear gain.

### Justification
The mechanism is plausible. FINDINGS gives the incumbent sliding verify shapes as 17.6 / 22.8 / 29.3 us against floors of 0.1 / 1.9 / 3.7 us. They are pure latency plus launch count: FA3 main is 7.4 / 10.7 / 14.7 us, and prepare_varlen (1.8-1.9), FlashAttnFwdCombine (3.8-4.2) and rms_norm_gate (1.1-1.5) together are ~7 us of serial small kernels plus graph-node gaps. Replacing FA3 main + prepare + combine + norm-gate with A8 + B8 (norm-gate fused into B8, no prepare) removes two launches and the separate combine/norm pass. The windowed range is right. lo_b = max(0, seq_b - 8 - (window - 1)) gives 2055 keys = 33 tiles at ctx 80K, and the mask t >= seq_b - 8 + j - (window - 1) is exactly FA3 window_size=(window-1, 0) as master passes it, so every key of each query's 2048-key window is attended. The expected directions match TASK. Sliding is ~62% / 29% / 22% of the verify_1/16/32 load times (46 x 17.6 vs 16 x 30.3, and so on), and decode and full shapes are outside the gate. Weakest link: the verify_1 estimate (A8 ~3-4 us, B8 ~2 us). The incumbent FA3 main already runs the same dependent chain (cache_seqlens -> page index -> K/V gather -> mma) over 33-ish splits in 7.4 us. A8 at one tile per CTA runs that same chain from a cold L2, with three dependent DRAM round trips plus the q-block fetch, so it is unlikely to be much under ~4-5 us. B8 has its own chain (seqlens -> partials -> gate). The landed fused Triton sliding kernel, one launch at decode_1, takes 10.3 us. Realistically verify_1 lands at ~11-13 us, not 9-11: ~1.35-1.6x on the shape and ~1.2-1.3x on the verify_1 load, still a clear signal well outside the CI. verify_16 (2 tiles per CTA) and verify_32 (4 per CTA, ~17 MB) should be 13-17 and 17-21 us. The math is trivial here, so the parent's MUFU and shared-memory ceilings do not bind. Design notes: (a) The spec applies the per-row mask only on the request's first and last tiles. That is correct at ctx 80K, because (seq_b - lo_b) = 2055 and 2055 % 64 = 7, so the 7-key causal band sits exactly in the last tile. It is wrong whenever (seq_b - lo_b) % 64 is in 1..6, which happens for short requests with lo_b = 0: the causal band [seq_b-7, seq_b-1] then straddles the last two tiles. Use 'tile overlaps [lo_b, lo_b+7) or [seq_b-7, seq_b)' as the mask predicate, or just mask every tile on the sliding route (33 tiles, negligible ALU). The same straddle exists in the parent's full path when cache_seqlens % 64 is in 1..7. (b) The grid can be config-only min(2 x SM, B x ceil((window + 7) / 64)), which gives 33 at batch 1 instead of 264 with 231 empty CTAs. It is legal (shape-derived) and slightly cuts launch and drain. (c) Make window a template parameter (or a constexpr instantiation), so the full-verify window = 0 SASS is byte-identical to stage 2 and the 'full flat' expectation holds by construction.

### Confounders on the current kernel
- full-verify-cuda-streamk-splitkv (parent, running, not landed): this run carries the parent's full verify_16/32 change too, so the headline mixes both. Attribute only on the sliding verify rows and top_kernels. If the parent ends up not promoted, the headline is not a measure of this change. If the parent's A8 has spills or is smem-bound, that matters little here, because sliding works on 1-4 tiles per CTA (latency, not throughput).
- full-decode-cuda-gather-splitkv (landed, same load_inline module): adding the windowed instantiation to the shared module means a build failure or a shared self-check failure sets _EXT=None. That silently reverts full decode_16/32 (and the parent's verify) to FA3. It would show as regressions this change did not cause, and the sliding rows would read flat. Read the log tail ('[splitkv] load_inline failed' / 'SELF-CHECK FAILED') before attributing anything, and give the sliding route its own self-check flag.
- fused-preattention-triton (landed): runs unchanged on both sides with ROPE on and PAIRED=False and stores the 8 new K/V rows that A8 reads in stream order. Its time (1.6/2.7/4.2 us) is common to incumbent and new, so there is no confound beyond requiring that A8 not be reordered before it.
- sliding-fused-triton-decode1-gate (landed): its early return is q_len == 1 only, so it is disjoint from the sliding verify route. No interaction, except that the new sliding-verify predicate must sit after that early return and not catch q_len 1.

### Interference and A/B needs
No companion is warranted. The gate is config-only and shape-disjoint: only sliding q_len 8, tp8, page 1 shapes change relative to the parent, and the pre-attention kernel is unchanged there (sliding was never paired). So the three sliding verify rows and their top_kernels (A8 + B8 vs FA3 main + prepare + combine + rms_norm_gate) attribute the effect directly. A gain or loss there cannot come from a landed hypothesis acting differently. An ab-revert of fused-preattention-triton would only change a cost common to both sides. A control splitting 'fewer launches' from 'A8 itself' (for example FA3 main + B8-style fused combine) is not implementable without FA3 internals and would not change the decision. The one interpretive hazard is the shared-module build and self-check coupling and the parent's headline contribution. Both are a log check and a per-shape reading, not an extra run.

### Open questions
- Dispatch ordering: run only after the parent harvests with M1 (all 16 tests pass and its self-check is OK in the log). If the parent's verify path fails, fix it first, because the sliding route shares A8/B8.
- Self-check coverage: include batch 1 (the verify_1 target). Include lens with (seq_b - lo_b) % 64 in 1..6 (short lo_b = 0 requests, to catch the causal band straddling two tiles), seq_b in [window, window+7], seq_b < 8 + window, and full 2055-key windows. Use kv_scale 1 and 0.25 and several grids (3, 7, device), so that segments start and end mid-request and at the window lower edge.
- Share one device helper for lo_b / tiles_b / TP between A8 and B8 (the window formula must match bit for bit), and keep the c_lo..c_hi inverse exact as the parent reviewer asked.
- Merge hygiene with full-verify1-cuda-streamk-batch1 (also stacked on the parent; it widens _splitkv_verify_gate and sets grid = SM count at batch 1): put the sliding route in its own predicate (e.g. _splitkv_sliding_verify_gate) and key any batch-1 grid choice on window, so a later merge does not silently change sliding verify_1's grid.
- Report from the profile: A8 and B8 times per sliding verify shape, and the shape time minus kernel sum (graph-node gaps). If A8 at verify_1 is > 5 us, the next lever is fusing pre-attention into A8's prologue for the sliding route (the queued fused-sliding-preattn-into-window-kernel idea, now on CUDA).

