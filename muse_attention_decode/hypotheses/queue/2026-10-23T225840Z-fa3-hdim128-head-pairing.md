---
problem: muse_attention_decode
slug: fa3-hdim128-head-pairing
created: 2026-10-23T225840Z
base: master
area: FA3 call (head packing)
change: >
  On full-attention layers only (config window == 0, decode and verify) call
  flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV heads are
  fetched as ONE 128-byte row instead of two 64-byte rows. With G = Hq // Hkv and S = Hkv // 2
  (Hkv is even: 2 under tp8, 16 under dp; if odd, fall back to the unpaired call):
  k_cache.view(pages, page_size, S, 128) and v_cache likewise (contiguous views: kv heads 2s and
  2s+1 of a token are adjacent 64-byte halves of the same row). Queries: q_pad =
  torch.zeros((tokens, Hq, 128), dtype=float8_e4m3fn) each call (fixed shape, no value
  dependence); view q_pad as [tokens, S, 2, G, 128] and the parity pipeline's FP8 q as [tokens, S,
  2, G, 64]; q_pad[:, :, 0, :, :64] = q[:, :, 0] and q_pad[:, :, 1, :, 64:] = q[:, :, 1]. Because
  FA3 maps query head h to kv head h // (2G), query heads s*2G .. s*2G+2G-1 land on super-head s,
  the first G in the low half (original kv head 2s) and the next G in the high half (2s+1), which
  is exactly the original grouping; the zero half makes q_pad . [k_2s | k_2s+1] equal q . k_g
  exactly, so the softmax over keys is unchanged and FA3's output out[tokens, Hq, 128] holds the
  correct 64-wide result in the same half as the query (the other half is the other head's P@V and
  is discarded). Un-pad: attn = torch.empty((tokens, Hq, 64), bf16); a = attn.view(tokens, S, 2,
  G, 64); o = out.view(tokens, S, 2, G, 128); a[:, :, 0] = o[:, :, 0, :, :64]; a[:, :, 1] = o[:,
  :, 1, :, 64:]; then rms_norm_gate on attn.reshape(tokens, Hq*64) exactly as before. FA3
  arguments otherwise identical (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q=L,
  softmax_scale=sm_scale*qk_scale_factor, causal=True, window_size=(-1,-1)); descale when kv_scale
  != 1 becomes scale.reshape(1,1).expand(B, S). The judge's scheduler_metadata was built for (Hq,
  Hkv=2, headdim 64) and must NOT be passed with the paired geometry: on full decode steps rebuild
  it in kernel_fn with sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=1,
  max_seqlen_k=page_table.shape[1]*page_size (a shape, never cache_seqlens.max()), num_heads=Hq,
  num_heads_k=S, headdim=128, cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn,
  cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True, num_splits=0) (device side, one
  small launch); on verify blocks pass None as before. Sliding layers keep the unpaired FA3 call
  untouched. The pre-attention stages and the cache store are untouched.
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: "388 us vs 126 us floor at ~1.7 TB/s effective: FA3 with page_size 1 gathers each (token, kv head) as a separate 64-byte row; paired rows halve the number of random gathers, make every fetch a full 128-byte line, and double the bytes per pipeline stage of the hdim-128 FP8 kernel (16 KB K blocks instead of 8 KB) so each SM keeps twice as many bytes in flight"
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: 201 us vs 63 us floor, same mechanism; 16 tiles of (request, super-head) instead of 32
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: 390 us vs 127 us floor, same bytes; pack_gqa packs 2G x 8 = 128 query rows per super-head tile, one M tile
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: 205 us vs 63 us, same mechanism
  - shapes: "*layer=full load=decode_1 page*"
    direction: slower
    why: six small extra launches (zeros, two pad copies, two un-pad copies, the metadata rebuild) on a 30.8 us shape with only 20 MB of KV; the gather gain cannot pay for ~12 us of launches; the cleanup child removes them
  - shapes: "*layer=full load=verify_1 page*"
    direction: slower
    why: five extra launches on a 35.6 us shape, same reason
  - shapes: "*layer=sliding*"
    direction: flat
    why: untouched; control that the window gate is right
risk: medium
depends_on: fused-preattention-triton
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-hdim128-head-pairing

## Change

On full-attention layers only (config window == 0, decode and verify) call flash_attn_with_kvcache on a head-paired view of the cache so each token's two KV heads are fetched as ONE 128-byte row instead of two 64-byte rows. With G = Hq // Hkv and S = Hkv // 2 (Hkv is even: 2 under tp8, 16 under dp; if odd, fall back to the unpaired call): k_cache.view(pages, page_size, S, 128) and v_cache likewise (contiguous views: kv heads 2s and 2s+1 of a token are adjacent 64-byte halves of the same row). Queries: q_pad = torch.zeros((tokens, Hq, 128), dtype=float8_e4m3fn) each call (fixed shape, no value dependence); view q_pad as [tokens, S, 2, G, 128] and the parity pipeline's FP8 q as [tokens, S, 2, G, 64]; q_pad[:, :, 0, :, :64] = q[:, :, 0] and q_pad[:, :, 1, :, 64:] = q[:, :, 1]. Because FA3 maps query head h to kv head h // (2G), query heads s*2G .. s*2G+2G-1 land on super-head s, the first G in the low half (original kv head 2s) and the next G in the high half (2s+1), which is exactly the original grouping; the zero half makes q_pad . [k_2s | k_2s+1] equal q . k_g exactly, so the softmax over keys is unchanged and FA3's output out[tokens, Hq, 128] holds the correct 64-wide result in the same half as the query (the other half is the other head's P@V and is discarded). Un-pad: attn = torch.empty((tokens, Hq, 64), bf16); a = attn.view(tokens, S, 2, G, 64); o = out.view(tokens, S, 2, G, 128); a[:, :, 0] = o[:, :, 0, :, :64]; a[:, :, 1] = o[:, :, 1, :, 64:]; then rms_norm_gate on attn.reshape(tokens, Hq*64) exactly as before. FA3 arguments otherwise identical (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q=L, softmax_scale=sm_scale*qk_scale_factor, causal=True, window_size=(-1,-1)); descale when kv_scale != 1 becomes scale.reshape(1,1).expand(B, S). The judge's scheduler_metadata was built for (Hq, Hkv=2, headdim 64) and must NOT be passed with the paired geometry: on full decode steps rebuild it in kernel_fn with sgl_kernel.flash_attn.get_scheduler_metadata(batch_size=B, max_seqlen_q=1, max_seqlen_k=page_table.shape[1]*page_size (a shape, never cache_seqlens.max()), num_heads=Hq, num_heads_k=S, headdim=128, cache_seqlens=cache_seqlens, qkv_dtype=float8_e4m3fn, cu_seqlens_q=cu_seqlens_q, page_size=page_size, causal=True, num_splits=0) (device side, one small launch); on verify blocks pass None as before. Sliding layers keep the unpaired FA3 call untouched. The pre-attention stages and the cache store are untouched.

## Expected

- `*layer=full load=decode_32*` faster: 388 us vs 126 us floor at ~1.7 TB/s effective: FA3 with page_size 1 gathers each (token, kv head) as a separate 64-byte row; paired rows halve the number of random gathers, make every fetch a full 128-byte line, and double the bytes per pipeline stage of the hdim-128 FP8 kernel (16 KB K blocks instead of 8 KB) so each SM keeps twice as many bytes in flight
- `*layer=full load=decode_16*` faster: 201 us vs 63 us floor, same mechanism; 16 tiles of (request, super-head) instead of 32
- `*layer=full load=verify_32*` faster: 390 us vs 127 us floor, same bytes; pack_gqa packs 2G x 8 = 128 query rows per super-head tile, one M tile
- `*layer=full load=verify_16*` faster: 205 us vs 63 us, same mechanism
- `*layer=full load=decode_1 page*` slower: six small extra launches (zeros, two pad copies, two un-pad copies, the metadata rebuild) on a 30.8 us shape with only 20 MB of KV; the gather gain cannot pay for ~12 us of launches; the cleanup child removes them
- `*layer=full load=verify_1 page*` slower: five extra launches on a 35.6 us shape, same reason
- `*layer=sliding*` flat: untouched; control that the window gate is right

## Risk

Medium risk. Correctness: the zero half contributes exactly 0 to every logit, the FP8 rounding of q is the same values in the same E4M3 format, the descale is per super-head with the same scalar, so the output is bit-for-bit the same math as FA3 unpaired up to accumulation order; nothing is coarser than FP8 queries. Rule edge: every key is still attended; queries stay E4M3; the cache is read in place. Hazards: FA3 must accept FP8 hdim 128 with page_size 1 (the non-TMA paged path) and Hq / S = 2G = 16 query heads per kv head under tp8 (32 under dp: 128 heads / 8 super-heads is 16 too); if FA3 caps qhead_per_khead for pack_gqa, the dp tests fail and the hypothesis is failed cleanly. The judge metadata must not be reused (built for headdim 64 / Hkv 2): rebuilding it is one launch and also changes the split schedule, which confounds decode_16/32 with fa3-num-splits-full-decode; the reviewer may want an A/B that passes None on decode instead of rebuilding. Extra traffic: FA3 writes a [tokens, Hq, 128] bf16 output (double) and reads a [tokens, Hq, 128] q: negligible against 328-655 MB of KV. The batch-1 full shapes lose ~10-12 us from launches: with weight 16 against the sliding shape's 46 x 21 us, decode_1 load gets ~10% slower while decode_16/32 and verify_16/32 loads, dominated by the full shapes, decide the geometric mean.
