---
problem: muse_attention_decode
slug: triton-flash-decode-full-both-heads
created: 2026-10-23T223722Z
base: master
area: FA3 call
change: >
  On full-attention decode steps (config window == 0, q_len == 1) replace flash_attn_with_kvcache
  with a two-launch Triton flash-decoding pair that exploits the page_size-1 cache layout: k_cache
  is [pages, 1, Hkv, 64] FP8, so one token's two KV heads are one contiguous 128-byte row; FA3
  gathers each head's 64 bytes separately per (request, kv-head) tile, touching every row twice.
  Kernel 1, grid (B, num_splits): each program takes a contiguous range of positions of request b
  (split = ceil(cache_seqlens_max_shape / num_splits) with num_splits = clamp(ceil(2*132/B), 1,
  128) from shapes only, masked by t < cache_seqlens[b]), gathers rows page_table[b, t] (page 1;
  general formula with // and % for page 64) as whole 128-byte rows for K and for V in blocks of
  64 tokens, and computes the online softmax for all 16 query heads at once (both groups of 8
  against their own kv head from the same loaded row), writing fp32 partial (m, l, acc[16, 64]) to
  a [B, num_splits, 16, 64+2] scratch. Kernel 2, grid (B, 16): combines the splits into bf16 [B,
  16, 64] in the buffer rms_norm_gate consumes. Queries come from the FP8 query buffer (or bf16 if
  the pre-attention stage changes) times sm_scale*qk_scale_factor; K/V dequantized to fp32 with
  k_scale/v_scale. Sliding layers and verify blocks keep FA3.
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: 388 us vs 126 us floor; halving the number of distinct 64-byte gathers (one 128-byte row per token instead of two 64-byte rows) is the mechanism; 32 x 80K x 256 B = 655 MB at 1.7 TB/s now
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 201 us vs 63 us, same mechanism
  - shapes: "*layer=full load=decode_1*"
    direction: flat
    why: 20 MB over 128 splits; FA3 is already near its launch-bound time here; a loss would attribute to the extra combine launch
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched
  - shapes: "*load=verify_*"
    direction: flat
    why: untouched
risk: high
depends_on: sglang-path-parity
priority: 5
kind: primary
controls_for: null
review: required
status: queued
---

# triton-flash-decode-full-both-heads

## Change

On full-attention decode steps (config window == 0, q_len == 1) replace flash_attn_with_kvcache with a two-launch Triton flash-decoding pair that exploits the page_size-1 cache layout: k_cache is [pages, 1, Hkv, 64] FP8, so one token's two KV heads are one contiguous 128-byte row; FA3 gathers each head's 64 bytes separately per (request, kv-head) tile, touching every row twice. Kernel 1, grid (B, num_splits): each program takes a contiguous range of positions of request b (split = ceil(cache_seqlens_max_shape / num_splits) with num_splits = clamp(ceil(2*132/B), 1, 128) from shapes only, masked by t < cache_seqlens[b]), gathers rows page_table[b, t] (page 1; general formula with // and % for page 64) as whole 128-byte rows for K and for V in blocks of 64 tokens, and computes the online softmax for all 16 query heads at once (both groups of 8 against their own kv head from the same loaded row), writing fp32 partial (m, l, acc[16, 64]) to a [B, num_splits, 16, 64+2] scratch. Kernel 2, grid (B, 16): combines the splits into bf16 [B, 16, 64] in the buffer rms_norm_gate consumes. Queries come from the FP8 query buffer (or bf16 if the pre-attention stage changes) times sm_scale*qk_scale_factor; K/V dequantized to fp32 with k_scale/v_scale. Sliding layers and verify blocks keep FA3.

## Expected

- `*layer=full load=decode_32*` faster: 388 us vs 126 us floor; halving the number of distinct 64-byte gathers (one 128-byte row per token instead of two 64-byte rows) is the mechanism; 32 x 80K x 256 B = 655 MB at 1.7 TB/s now
- `*layer=full load=decode_16*` faster: 201 us vs 63 us, same mechanism
- `*layer=full load=decode_1*` flat: 20 MB over 128 splits; FA3 is already near its launch-bound time here; a loss would attribute to the extra combine launch
- `*layer=sliding*` flat: untouched
- `*load=verify_*` flat: untouched

## Risk

High risk: beating FA3 on the long-context full-attention shapes is the hardest target in this problem, but these shapes dominate four of the six loads (full decode_16 is 2.6x the sliding shape's load time, decode_32 is 4.3x), so even a 20% gain there moves the geometric mean more than any sliding-layer win. Must pass the 1,048,576-token full test (page 64 path and int64 row arithmetic), the dp-heads tests (Hq=128, Hkv=16: both-heads-per-row no longer holds, so the kernel must take Hkv as a constexpr and loop groups), and the kv_scale tests. The coherent dist puts a shared direction on low-attention values, so fp32 accumulation over all 80K keys matters; do not reduce precision. If the gather is the bottleneck regardless of row width (DRAM page locality, not sector count), this comes out flat and should be marked do-not-retry in this form.
