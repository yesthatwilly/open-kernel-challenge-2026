---
problem: muse_attention_decode
slug: triton-window-attention-sliding
created: 2026-10-23T223550Z
base: master
area: FA3 call
change: >
  On sliding-window layers (config window != 0) replace the flash_attn_with_kvcache call with a
  Triton sliding-window attention kernel written for page_size-1 paged FP8 caches; full-attention
  layers keep FA3 untouched. Grid (B, Hkv, L): one program per (request, kv head, query token j).
  The program reads the 8 query heads of its group for token j from the FP8 query buffer the
  parity pipeline already produces (or from the bf16 normalized q if the pre-attention stage is
  changed later; either is allowed, bf16 is finer than FP8), computes p = seq_lens[b] + j, start =
  max(0, p - window + 1), and loops over the window positions [start, p] in blocks of 128: for
  each block it gathers the rows row = page_table[b, t // page_size] * page_size + t % page_size
  (page_size from config; page 1 in the scored cases) and loads k_cache.view(-1, Hkv, 64)[row,
  kvh] and v_cache likewise as FP8, converts to fp32 (times k_scale/v_scale), does an online
  softmax over the 8 query heads x 128 keys with scale sm_scale*qk_scale_factor, masks t > p and t
  < start, and accumulates P@V. The new tokens' rows are already in the cache (the cache store
  happens before this kernel, as in the parity pipeline), so the block's own keys are read from
  the cache like any other; causal-in-block is just t <= p. It writes bf16 [B*L, Hq, 64] to the
  buffer rms_norm_gate consumes, exactly where FA3's output went. Window 2048 means at most 16
  blocks per program, so no split-KV and no combine kernel. Softmax, exp and accumulation in fp32.
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: FA3 on this shape is a general paged kernel fed an 86K cache length; a 2048-token window kernel with no split/combine reads 512 KB per request and should finish in a few microseconds
  - shapes: "*layer=sliding load=verify_1*"
    direction: faster
    why: 8 programs per (b, kvh), each 16 blocks; still tiny
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: 32 programs x 16 blocks of 128-byte gathered rows; 8 MB total
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: 64 programs, 16 MB
  - shapes: "*layer=sliding load=verify_*"
    direction: faster
    why: same mechanism at q_len 8
  - shapes: "*layer=full*"
    direction: flat
    why: untouched; control
risk: high
depends_on: fused-preattention-triton
priority: 5
kind: primary
controls_for: null
review: required
status: hold
hold_reason: "moot under master 1.269x (sweep by the attention minter): its chain ends in the single fused sliding launch, measured as sliding-single-launch-fused-attention (0.945x); its one winning shape is covered by sliding-fused-triton-decode1-gate"
---

# triton-window-attention-sliding

## Change

On sliding-window layers (config window != 0) replace the flash_attn_with_kvcache call with a Triton sliding-window attention kernel written for page_size-1 paged FP8 caches; full-attention layers keep FA3 untouched. Grid (B, Hkv, L): one program per (request, kv head, query token j). The program reads the 8 query heads of its group for token j from the FP8 query buffer the parity pipeline already produces (or from the bf16 normalized q if the pre-attention stage is changed later; either is allowed, bf16 is finer than FP8), computes p = seq_lens[b] + j, start = max(0, p - window + 1), and loops over the window positions [start, p] in blocks of 128: for each block it gathers the rows row = page_table[b, t // page_size] * page_size + t % page_size (page_size from config; page 1 in the scored cases) and loads k_cache.view(-1, Hkv, 64)[row, kvh] and v_cache likewise as FP8, converts to fp32 (times k_scale/v_scale), does an online softmax over the 8 query heads x 128 keys with scale sm_scale*qk_scale_factor, masks t > p and t < start, and accumulates P@V. The new tokens' rows are already in the cache (the cache store happens before this kernel, as in the parity pipeline), so the block's own keys are read from the cache like any other; causal-in-block is just t <= p. It writes bf16 [B*L, Hq, 64] to the buffer rms_norm_gate consumes, exactly where FA3's output went. Window 2048 means at most 16 blocks per program, so no split-KV and no combine kernel. Softmax, exp and accumulation in fp32.

## Expected

- `*layer=sliding load=decode_1*` faster: FA3 on this shape is a general paged kernel fed an 86K cache length; a 2048-token window kernel with no split/combine reads 512 KB per request and should finish in a few microseconds
- `*layer=sliding load=verify_1*` faster: 8 programs per (b, kvh), each 16 blocks; still tiny
- `*layer=sliding load=decode_16*` faster: 32 programs x 16 blocks of 128-byte gathered rows; 8 MB total
- `*layer=sliding load=decode_32*` faster: 64 programs, 16 MB
- `*layer=sliding load=verify_*` faster: same mechanism at q_len 8
- `*layer=full*` flat: untouched; control

## Risk

High risk: a new attention kernel must pass the window-edge tests (dist edges plants keys at p-2047 and p-2048: the mask must keep t >= p-window+1 and drop t = p-window) and the page_size-64 tests, where row = page_table[b, t//64]*64 + t%64. Correctness: queries at bf16 or FP8 E4M3 are both allowed; the output bound is 2.5x the reference's own FP8-query error + 1.5%, so fp32 softmax over 2048 keys is well inside it. The gather of 128-byte rows at page 1 may not beat FA3's cp.async path at batch 32 (16 MB of scattered rows); if decode_1 gains and decode_32 does not, the kernel is gather-bound and a split-KV variant is the follow-up. Watch register pressure: 8 heads x 64 dims x 128 keys in fp32 per program; use BLOCK 64 if Triton spills. This is the base for fusing the pre-attention and norm-gate stages into one sliding-layer launch (two stacked hypotheses).
