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
review: required
status: queued
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
