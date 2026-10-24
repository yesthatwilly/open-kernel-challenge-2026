---
problem: muse_decode_layer
slug: triton-flash-decode-full-paired-kv
created: 2026-10-24T075400Z
base: master
area: FA3 call
change: >
  On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1) replace
  flash_attn_with_kvcache with a two-launch Triton flash-decoding pair built for the page-size-1
  layout: k_cache/v_cache are [pages, page_size, 2, 64] FP8, so one token's two KV heads are ONE
  contiguous 128-byte row (view as [pages*page_size, 128]), while FA3 gathers each head's 64 bytes
  per (request, kv-head) tile. Kernel 1, grid (B, S) with S = clamp(ceil(2*132/B), 1, 128) from
  shapes only (B=1: 128, B=16: 17, B=32: 9; ~2 waves of 132 SMs), each program owns request b and
  a contiguous token range of length ceil(page_table.shape[1]*page_size / S) rounded up to
  BLOCK_N=64, masked by t < cache_seqlens[b]. Per BLOCK_N tile it loads the int32 rows
  page_table[b, t // P] * P + t % P (P = page_size, so page 64 tests work too), then K and V as
  [BLOCK_N, 128] fp8 tiles (contiguous 128-byte segments), converts them to bf16 (exact) and runs
  the online softmax for all 16 query heads at once with a block-diagonal trick: Q2 is a [16, 128]
  bf16 tile whose rows 0-7 hold heads 0-7 in columns 0-63 (zeros in 64-127) and rows 8-15 hold
  heads 8-15 in columns 64-127 (zeros in 0-63), so S = Q2 @ K^T gives each head's scores against
  its own kv head; acc[16, 128] += P @ V and only each row-group's own 64 columns are kept.
  Queries are master's FP8 q (q.view(T,16,64).to(FP8), as FA3 receives) up-converted to bf16
  in-kernel; scores scaled by sm_scale*qk_scale_factor, K/V times k_scale/v_scale (scalars), fp32
  m/l/acc, num_warps=4, num_stages=3. It writes fp32 partials acc [B, S, 16, 64] and (m, l) [B, S,
  16] to fresh scratch. Kernel 2, grid (B, 16), combines the S partials (log-sum-exp) into a bf16
  [B, 16, 64] tensor that goes into the unchanged state['o_norm_gate'](out.reshape(T, 1024),
  gate). scheduler_metadata is ignored on this path. Sliding layers, verify blocks (q_len=8,
  including full verify) and everything else keep master's code. Same design as
  muse_attention_decode's queued triton-flash-decode-full-both-heads, applied here where full
  decode FA3 is 189.1 us of 442 us (decode_16 full, 1.73 TB/s) and 491.6 us of 623 us (decode_32
  full, 1.33 TB/s against a 137 us KV floor).
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "655 MB of FP8 KV read as 128-byte rows by 288 CTAs; at ~3 TB/s FA3's 491.6 us becomes ~220 us, ~40% of the shape and ~20% of the decode_32 load"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 328 MB of KV, FA3 189.1 us at 1.73 TB/s; 272 CTAs at ~3 TB/s is ~110 us
  - shapes: "*layer=full load=decode_1*"
    direction: flat
    why: B=1 is latency-bound (FA3 18 us + 6.9 us combine against a 4.3 us byte floor); 128 CTAs of ~5 tiles plus a combine is about the same
  - shapes: "*layer=sliding*"
    direction: flat
    why: FA3 unchanged; control
  - shapes: "*load=verify_*"
    direction: flat
    why: q_len=8 keeps FA3 (full verify included); control
risk: high
depends_on: null
priority: 3
kind: rewrite
controls_for: null
review: required
status: queued
stage: 1
replaces: "FA3 on full-attention decode (q_len=1): flash_fwd main + FlashAttnFwdCombine"
---

# triton-flash-decode-full-paired-kv

## Change

On full-attention decode steps only (cfg['window'] == 0 and cfg['q_len'] == 1) replace flash_attn_with_kvcache with a two-launch Triton flash-decoding pair built for the page-size-1 layout: k_cache/v_cache are [pages, page_size, 2, 64] FP8, so one token's two KV heads are ONE contiguous 128-byte row (view as [pages*page_size, 128]), while FA3 gathers each head's 64 bytes per (request, kv-head) tile. Kernel 1, grid (B, S) with S = clamp(ceil(2*132/B), 1, 128) from shapes only (B=1: 128, B=16: 17, B=32: 9; ~2 waves of 132 SMs), each program owns request b and a contiguous token range of length ceil(page_table.shape[1]*page_size / S) rounded up to BLOCK_N=64, masked by t < cache_seqlens[b]. Per BLOCK_N tile it loads the int32 rows page_table[b, t // P] * P + t % P (P = page_size, so page 64 tests work too), then K and V as [BLOCK_N, 128] fp8 tiles (contiguous 128-byte segments), converts them to bf16 (exact) and runs the online softmax for all 16 query heads at once with a block-diagonal trick: Q2 is a [16, 128] bf16 tile whose rows 0-7 hold heads 0-7 in columns 0-63 (zeros in 64-127) and rows 8-15 hold heads 8-15 in columns 64-127 (zeros in 0-63), so S = Q2 @ K^T gives each head's scores against its own kv head; acc[16, 128] += P @ V and only each row-group's own 64 columns are kept. Queries are master's FP8 q (q.view(T,16,64).to(FP8), as FA3 receives) up-converted to bf16 in-kernel; scores scaled by sm_scale*qk_scale_factor, K/V times k_scale/v_scale (scalars), fp32 m/l/acc, num_warps=4, num_stages=3. It writes fp32 partials acc [B, S, 16, 64] and (m, l) [B, S, 16] to fresh scratch. Kernel 2, grid (B, 16), combines the S partials (log-sum-exp) into a bf16 [B, 16, 64] tensor that goes into the unchanged state['o_norm_gate'](out.reshape(T, 1024), gate). scheduler_metadata is ignored on this path. Sliding layers, verify blocks (q_len=8, including full verify) and everything else keep master's code. Same design as muse_attention_decode's queued triton-flash-decode-full-both-heads, applied here where full decode FA3 is 189.1 us of 442 us (decode_16 full, 1.73 TB/s) and 491.6 us of 623 us (decode_32 full, 1.33 TB/s against a 137 us KV floor).

## Expected

- `*layer=full load=decode_32*` faster: 655 MB of FP8 KV read as 128-byte rows by 288 CTAs; at ~3 TB/s FA3's 491.6 us becomes ~220 us, ~40% of the shape and ~20% of the decode_32 load
- `*layer=full load=decode_16*` faster: 328 MB of KV, FA3 189.1 us at 1.73 TB/s; 272 CTAs at ~3 TB/s is ~110 us
- `*layer=full load=decode_1*` flat: B=1 is latency-bound (FA3 18 us + 6.9 us combine against a 4.3 us byte floor); 128 CTAs of ~5 tiles plus a combine is about the same
- `*layer=sliding*` flat: FA3 unchanged; control
- `*load=verify_*` flat: q_len=8 keeps FA3 (full verify included); control

## Risk

Risk: Triton's gather loads through a page table may not reach 3 TB/s; check fa3-full-single-split-probe's result first if it has landed in done/ (if FA3 is gather-pattern-bound the 128-byte-row access is the lever, if MLP-bound the 2-wave grid is). Competes with trtllm-decode-full-attention (queued, same shapes, same call site): whichever lands first is the other's base, so read this one against the base it was built on. Correctness: the block-diagonal Q2 makes the off-group dot products exactly zero contributions only because the zero columns multiply real K values; keep the zero padding explicit, never masked loads of q only. Tests through this path: batch 1 ctx 3000 full, batch 33 ctx 20000 full spread mixed page 1, batch 1 ctx 80000 full page 1, batch 16 ctx 20000 full page 1 (plus any page-64 full decode). Every key in the context is attended; no block skipping. Graph-safe: S and the split length come from B and page_table.shape only; scratch is allocated per call.
