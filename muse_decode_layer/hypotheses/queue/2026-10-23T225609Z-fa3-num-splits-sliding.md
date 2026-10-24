---
problem: muse_decode_layer
slug: fa3-num-splits-sliding
created: 2026-10-23T225609Z
base: master
area: FA3 call
change: >
  On sliding-window layers only (cfg['window'] == 2048, where the judge passes
  scheduler_metadata=None and SGLang lets FA3 choose), pass an explicit num_splits to
  flash_attn_with_kvcache instead of leaving it at 0. FA3's host-side split heuristic sees
  seqlen_k as the page table's full extent (max_pages * page_size, i.e. the whole ~80K context)
  because cache_seqlens is a device tensor it cannot read, and sizes the split-KV decomposition
  for 600+ key tiles, while the window only ever touches 16 tiles of 128 keys per (batch,
  kv_head): the extra splits produce empty partials plus a larger combine kernel. Choose
  num_splits by shape on the host, from batch B and kv_heads=2: num_splits = clamp(ceil(132 / (B *
  2)), 1, 16) so that B=1 gets 16 (2 CTAs x 16 splits over 16 tiles, one tile each), B=16 gets 4
  and B=32 gets 2-3; q_len=8 verify blocks use the same formula. The full-attention branch keeps
  scheduler_metadata and num_splits=0 untouched. Nothing else changes. Attribution: sliding shapes
  only (weight 46 of 62 per load); full shapes are the control.
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: "B=1: 2 CTAs with a 16-tile window; a sized split (16) replaces a heuristic sized for 625 tiles and its oversized combine"
  - shapes: "*layer=sliding load=verify_1*"
    direction: faster
    why: same, with 8 queries per request
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "B=16: 32 CTAs; fewer, non-empty splits and a smaller combine"
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: "B=32: 64 CTAs; num_splits 2-3 instead of the heuristic pick"
  - shapes: "*layer=full*"
    direction: flat
    why: full branch untouched (metadata passed through); control
risk: medium
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 2
kind: primary
controls_for: null
review: hold
status: hold
review_at: 2026-10-24T040612Z
review_companions: null
hold_reason: "adversarial review: The premise (FA3 sizes the sliding split-KV for the 80K page-table extent) is contradicted by both FA3's own get_num_splits, which clamps seqlen_k by the window"
---

# fa3-num-splits-sliding

## Change

On sliding-window layers only (cfg['window'] == 2048, where the judge passes scheduler_metadata=None and SGLang lets FA3 choose), pass an explicit num_splits to flash_attn_with_kvcache instead of leaving it at 0. FA3's host-side split heuristic sees seqlen_k as the page table's full extent (max_pages * page_size, i.e. the whole ~80K context) because cache_seqlens is a device tensor it cannot read, and sizes the split-KV decomposition for 600+ key tiles, while the window only ever touches 16 tiles of 128 keys per (batch, kv_head): the extra splits produce empty partials plus a larger combine kernel. Choose num_splits by shape on the host, from batch B and kv_heads=2: num_splits = clamp(ceil(132 / (B * 2)), 1, 16) so that B=1 gets 16 (2 CTAs x 16 splits over 16 tiles, one tile each), B=16 gets 4 and B=32 gets 2-3; q_len=8 verify blocks use the same formula. The full-attention branch keeps scheduler_metadata and num_splits=0 untouched. Nothing else changes. Attribution: sliding shapes only (weight 46 of 62 per load); full shapes are the control.

## Expected

- `*layer=sliding load=decode_1*` faster: B=1: 2 CTAs with a 16-tile window; a sized split (16) replaces a heuristic sized for 625 tiles and its oversized combine
- `*layer=sliding load=verify_1*` faster: same, with 8 queries per request
- `*layer=sliding load=decode_16*` faster: B=16: 32 CTAs; fewer, non-empty splits and a smaller combine
- `*layer=sliding load=decode_32*` faster: B=32: 64 CTAs; num_splits 2-3 instead of the heuristic pick
- `*layer=full*` flat: full branch untouched (metadata passed through); control

## Risk

Risk: the mechanism assumes FA3's heuristic ignores the window when it picks num_splits on this build (hopper flash_api get_num_splits uses seqlen_k and may or may not clamp by window_size); if it already clamps, this is flat on every shape, which is an informative absence. Over-splitting at B=32 (num_splits > tiles per CTA) wastes a combine pass, so the formula caps at 16 and never exceeds 16 tiles. Correctness: num_splits only changes the reduction order of the split-KV combine (fp32 partials), well within the 3-15% bound. CUDA-graph safe: num_splits is computed from config/batch on the host, never from tensor values. Do not pass num_splits together with scheduler_metadata on full decode; FA3 asserts they agree.

## Adversarial review

Reviewed 2026-10-24T040612Z by an adversarial Fable subagent (model fable).

**Verdict: hold.** The premise (FA3 sizes the sliding split-KV for the 80K page-table extent) is contradicted by both FA3's own get_num_splits, which clamps seqlen_k by the window for local attention, and by the landed profile, where the sliding B=1 combine kernel is smaller than the full B=1 one (below the 4.7 us cutoff vs 6.9 us) and the FA3 kernel is 9.7 us vs 18.0 us. Even if the premise held, FA3 plus combine is 10-24 us of 172-695 us on sliding shapes, so the best case is ~2% on decode_1 and <1% elsewhere, at the per-shape noise floor; the chosen num_splits coincide with or slightly overshoot the heuristic's own one-wave picks at B=16/32.

### Justification
A. Mechanism. The hypothesis assumes FA3's host heuristic sees seqlen_k = max_pages*page_size (~80K, 625 tiles) and splits accordingly. Upstream FA3 hopper flash_api.cpp get_num_splits computes seqlen_k_loaded = min(seqlen_k, window_left + window_right + 1 + kBlockM) when is_local (comment: 'If is_local, we're not going to load all the KV'), giving ~17 tiles of 128 for a 2047-key window, and then num_splits_heuristic(total_mblocks = B*h_k*1, 132, 17, 128). Worked through: B=1 -> total_mblocks 2, max_splits 17, picks ~14; B=16 -> 32 blocks, picks 4 (128 CTAs, one wave); B=32 -> 64 blocks, picks 2 (128 CTAs). Those are exactly, or within one of, the hypothesis's own targets, so the change is a near no-op by construction at B=16/32. Moreover, because cu_seqlens_q and cache_seqlens are passed, FA3 is on the varlen path and uses dynamic split (prepare_varlen_num_blocks on device, b <= 992): the static num_splits, heuristic or explicit, is only a cap and the per-batch count is derived on device from the real cache_seqlens; an explicit value therefore cannot remove 'empty partials' the way the text claims. Profile evidence (profiles/submission.5755a412dda244a1.txt): full decode_1 (B=1, 80K, metadata) shows FA3 18.0 us + FlashAttnFwdCombine 6.9 us; sliding decode_1 shows FA3 9.7 us and no combine among the rows >= 4.7 us; sliding verify_1 10.3 us with no combine >= 4.4 us; sliding decode_16 13.8 + 4.2 us combine; sliding verify_16/32 14.7 + 4.3 and 20.0 + 4.4 us; full B=16/32 combines are 4.1-4.2 us. The combine kernel costs ~4.1-4.4 us wherever it appears at B>=16 regardless of layer, i.e. it is a fixed launch/latency floor as long as num_splits > 1, not a function of split count; and at B=1 the sliding combine is already smaller than the full one, so the sliding split count is already smaller than the full one. Floors: sliding FA3 moves 2048 x 2 x 128 B = 0.5 MB per sequence, trivial against 4.8 TB/s; the 9.7 us at B=1 is launch + page-table gather + pipeline latency over 2 CTAs, which split count barely touches. Expected table vs weights: sliding is 46/62 of each load, so a real decode_1 gain would carry, but the ceiling is ~3-4 us of 172 us (~2%) on decode_1, ~1% on decode_16/32, <0.6% on verify_16/32, versus ~1.5% run-to-run per-shape drift; the geomean effect would be <0.5%, inside the CI. Arithmetic: the stated formula clamp(ceil(132/(B*2)),1,16) gives 5 at B=16 (text says 4) and 3 at B=32 (text says 2-3); 5 and 3 put 160 and 192 CTAs on 132 SMs, a second partial wave, which the heuristic's efficiency metric rates worse than its one-wave 4 and 2; floor() would reproduce the heuristic exactly. Weakest link: the premise that FA3 ignores the window when sizing splits on this build, which the hypothesis itself concedes may be false, and which the profile and the upstream source both argue against.

### Confounders on the current kernel
- none identified

### Interference and A/B needs
Only one hypothesis has landed (sglang-pipeline-parity-router-hilo-m8, patch muse_decode_layer/hypotheses/done/gain/sglang-pipeline-parity-router-hilo-m8.patch). Its FA3 call is byte-identical to baseline.py (same q cast, page_table, cache_seqlens, cu_seqlens_q, window_size, scheduler_metadata pass-through) and its one structural change (ROUTER_FP32_MAX_TOKENS 8 -> 1) acts on the router side stream after the o-projection, with no shared memory traffic, launch ordering or numerical path with the attention call; the main stream is joined before local_ids. A gain or loss here could not be caused by the router change, so no ab-revert or ab-tune is warranted. A control isolating a sub-change is moot: the hypothesis has one sub-change (the num_splits value on the sliding branch) and the full shapes are already its control. No companions. The real interpretability problem is not interference but power: a flat result cannot distinguish 'FA3 already clamps by window' from 'the effect is real but under the 1.5% per-shape noise', and both are the likely outcomes. If the queue wants one run on explicit num_splits, run the full-layer twin (fa3-num-splits-full-no-metadata) first: its verify_1/16/32 full shapes already have metadata=None, so its per-shape rows show whether an explicit num_splits moves anything at all under the dynamic-split varlen path; if those are flat, this hypothesis is moot and should be closed without a run.

### Open questions
- Does the judge image's sgl_kernel FA3 build clamp seqlen_k by the window in get_num_splits (upstream does, via seqlen_k_loaded), and does its prepare_varlen_num_blocks kernel (dynamic split, active here because cu_seqlens_q and cache_seqlens are passed) also clamp? If both do, every shape is flat by construction.
- Does an explicit num_splits disable the dynamic split or only cap it? If it only caps, the hypothesis cannot change the per-batch split count at B=16/32 at all (heuristic cap 4 and 2 vs proposed 4-5 and 2-3).
- What is the FlashAttnFwdCombine time on sliding decode_1/verify_1/decode_32? It is below the profile's 12-row cutoff (4.7/4.4/3.8 us); the full per-shape kernel list (if okc can emit it) would show whether sliding B=1 is split at all today.
- Which rounding does the implementer use for 132/(2B)? ceil (as written) gives 5 and 3 and a second wave at B=16/32; floor gives the heuristic's 4 and 2 and a guaranteed no-op there.
- Cheap diagnostic that needs no run of this hypothesis: print sgl_kernel.__version__ (and the FA3 commit if exposed) from prepare() in any queued run; the log tail returns it, and the source can then be read to settle the first two questions offline.

