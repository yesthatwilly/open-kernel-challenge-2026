---
problem: muse_decode_layer
slug: fa3-hdim128-head-pairing-batched
created: 2026-10-24T103909Z
base: master
area: FA3 call
change: >
  Port muse_attention_decode's landed fa3-hdim128-head-pairing (1.171x) to the
  flash_attn_with_kvcache call in muse_decode_layer kernel_fn, gated to full-attention layers with
  batch > 1. Gate, config only: paired = cfg["window"] == 0 and cfg["batch"] > 1 and
  cfg["num_kv_heads"] % 2 == 0 (never a tensor read; hkv is 2 under tp8). When not paired (all
  sliding layers, decode_1 and verify_1 full, odd hkv) the existing call stays byte for byte,
  including q.view(tokens, hq, d).to(FP8) and the judge's data["scheduler_metadata"]. Paired path,
  with G = hq // hkv and S = hkv // 2: k_attn = k_cache.view(pages, page_size, S, 2*d) and v_attn
  = v_cache.view(...) (contiguous in-place views of the FP8 cache AFTER store_cache has written
  the step's rows; works for any page_size, the tests use page sizes other than 1). Queries: in
  decode_layer q comes out of state["qk_norm"] as BF16 [tokens, hq*d] and is cast to FP8 inline in
  the FA3 call, so fold that cast into the pad: q_attn = torch.zeros((tokens, hq, 2*d), dtype=FP8,
  device); qp = q_attn.view(tokens, S, 2, G, 2*d); qv = q.view(tokens, S, 2, G, d); qp[:, :, 0, :,
  :d] = qv[:, :, 0]; qp[:, :, 1, :, d:] = qv[:, :, 1] (copy_ does the BF16->E4M3 cast with the
  same round-to-nearest as .to(FP8), so the FP8 values are identical; the separate cast launch
  disappears). FA3 then maps query head h to super-head h // (2G): heads 0..G-1 sit in the low
  half (kv head 2s) and G..2G-1 in the high half (2s+1), the zero half adds exactly 0 to every
  logit. descale when scaled: expand(batch, S) instead of (batch, hkv). scheduler_metadata=None on
  the paired call (the judge's metadata was built for hdim 64 / Hkv 2 and must not be reused;
  attention_decode's run passed None and FA3 built its own schedule). softmax_scale passed
  explicitly as today, cfg["sm_scale"] * cfg["qk_scale_factor"] (it must not default to
  1/sqrt(128)). All other FA3 args unchanged (page_table, cache_seqlens, cu_seqlens_q,
  max_seqlen_q=length, causal=True, window_size=(-1,-1)). Un-pad: attn = torch.empty((tokens, hq,
  d), bf16); a = attn.view(tokens, S, 2, G, d); o = out.view(tokens, S, 2, G, 2*d); a[:, :, 0] =
  o[:, :, 0, :, :d]; a[:, :, 1] = o[:, :, 1, :, d:]; then
  state["o_norm_gate"](attn.reshape(tokens, hq*d), gate) exactly as before. QKV GEMM,
  qk_norm/rope, store_cache, o GEMM and the whole MoE block are untouched. No autotune, no host
  sync, no new imports.
expected:
  - shapes: "*layer=full load=decode_32*"
    direction: faster
    why: FA3 is 374.4 of 601.7 us here (floor 208.1 us for the whole layer); attention_decode measured the same kernel on the same cache geometry go 378 -> 307 us with pairing, so ~60-70 us (10-12% of the shape)
  - shapes: "*layer=full load=decode_16*"
    direction: faster
    why: FA3 188.8 of 433.0 us; attention_decode took it 194 -> 165 us, ~25-30 us (6-7% of the shape)
  - shapes: "*layer=full load=verify_32*"
    direction: faster
    why: FA3 374.0 of 969.9 us; attention_decode verify_32 383 -> 309 us, ~65-75 us (7% of the shape)
  - shapes: "*layer=full load=verify_16*"
    direction: faster
    why: FA3 191.9 of 618.9 us; attention_decode verify_16 198 -> 173 us, ~20-25 us (3-4% of the shape)
  - shapes: "*layer=full load=decode_1 *"
    direction: flat
    why: "gated off by cfg batch == 1: master path byte for byte; control that the gate avoids attention_decode's 27-35% batch-1 loss"
  - shapes: "*layer=full load=verify_1 *"
    direction: flat
    why: gated off by batch == 1, same control
  - shapes: "*layer=sliding*"
    direction: flat
    why: "window != 0: untouched unpaired call"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# fa3-hdim128-head-pairing-batched

## Change

Port muse_attention_decode's landed fa3-hdim128-head-pairing (1.171x) to the flash_attn_with_kvcache call in muse_decode_layer kernel_fn, gated to full-attention layers with batch > 1. Gate, config only: paired = cfg["window"] == 0 and cfg["batch"] > 1 and cfg["num_kv_heads"] % 2 == 0 (never a tensor read; hkv is 2 under tp8). When not paired (all sliding layers, decode_1 and verify_1 full, odd hkv) the existing call stays byte for byte, including q.view(tokens, hq, d).to(FP8) and the judge's data["scheduler_metadata"]. Paired path, with G = hq // hkv and S = hkv // 2: k_attn = k_cache.view(pages, page_size, S, 2*d) and v_attn = v_cache.view(...) (contiguous in-place views of the FP8 cache AFTER store_cache has written the step's rows; works for any page_size, the tests use page sizes other than 1). Queries: in decode_layer q comes out of state["qk_norm"] as BF16 [tokens, hq*d] and is cast to FP8 inline in the FA3 call, so fold that cast into the pad: q_attn = torch.zeros((tokens, hq, 2*d), dtype=FP8, device); qp = q_attn.view(tokens, S, 2, G, 2*d); qv = q.view(tokens, S, 2, G, d); qp[:, :, 0, :, :d] = qv[:, :, 0]; qp[:, :, 1, :, d:] = qv[:, :, 1] (copy_ does the BF16->E4M3 cast with the same round-to-nearest as .to(FP8), so the FP8 values are identical; the separate cast launch disappears). FA3 then maps query head h to super-head h // (2G): heads 0..G-1 sit in the low half (kv head 2s) and G..2G-1 in the high half (2s+1), the zero half adds exactly 0 to every logit. descale when scaled: expand(batch, S) instead of (batch, hkv). scheduler_metadata=None on the paired call (the judge's metadata was built for hdim 64 / Hkv 2 and must not be reused; attention_decode's run passed None and FA3 built its own schedule). softmax_scale passed explicitly as today, cfg["sm_scale"] * cfg["qk_scale_factor"] (it must not default to 1/sqrt(128)). All other FA3 args unchanged (page_table, cache_seqlens, cu_seqlens_q, max_seqlen_q=length, causal=True, window_size=(-1,-1)). Un-pad: attn = torch.empty((tokens, hq, d), bf16); a = attn.view(tokens, S, 2, G, d); o = out.view(tokens, S, 2, G, 2*d); a[:, :, 0] = o[:, :, 0, :, :d]; a[:, :, 1] = o[:, :, 1, :, d:]; then state["o_norm_gate"](attn.reshape(tokens, hq*d), gate) exactly as before. QKV GEMM, qk_norm/rope, store_cache, o GEMM and the whole MoE block are untouched. No autotune, no host sync, no new imports.

## Expected

- `*layer=full load=decode_32*` faster: FA3 is 374.4 of 601.7 us here (floor 208.1 us for the whole layer); attention_decode measured the same kernel on the same cache geometry go 378 -> 307 us with pairing, so ~60-70 us (10-12% of the shape)
- `*layer=full load=decode_16*` faster: FA3 188.8 of 433.0 us; attention_decode took it 194 -> 165 us, ~25-30 us (6-7% of the shape)
- `*layer=full load=verify_32*` faster: FA3 374.0 of 969.9 us; attention_decode verify_32 383 -> 309 us, ~65-75 us (7% of the shape)
- `*layer=full load=verify_16*` faster: FA3 191.9 of 618.9 us; attention_decode verify_16 198 -> 173 us, ~20-25 us (3-4% of the shape)
- `*layer=full load=decode_1 *` flat: gated off by cfg batch == 1: master path byte for byte; control that the gate avoids attention_decode's 27-35% batch-1 loss
- `*layer=full load=verify_1 *` flat: gated off by batch == 1, same control
- `*layer=sliding*` flat: window != 0: untouched unpaired call

## Risk

Medium risk, but the mechanism is already proven on this exact FA3 build and cache layout (attention_decode, job facc479992e24a35, 16/16 tests incl. kv_scale and odd page sizes): same hq=16, hkv=2, d=64 FP8 e4m3 cache, so the hdim-128 e4m3 paged instantiation and pack_gqa at 16 q heads per super-head are known to work. Differences from attention_decode: (1) q arrives as BF16 from FusedQKNorm rather than FP8 from a fused pre-attention kernel, so the pad copies also do the cast (net extra launches: zeros + 2 un-pad copies, ~1 us against 30-75 us saved); (2) the full layer here is dist=soft rather than coherent, which changes softmax values but not the gather pattern; (3) FA3 is a smaller fraction of the shape (36-60%), so the per-load gain is diluted: estimated ~1.5-3% per batch-16/32 load on the geomean (~+2% headline, 4 of 6 loads), small enough that the harvester must read the FA3 kernel row and per-shape full rows, not the headline. The batch gate costs decode_1/verify_1 nothing. Confounders: queued triton-flash-decode-full-* and fa3-num-splits-full-no-metadata touch the same call; dropping the judge metadata on paired calls also changes the split schedule (LESSONS: num_splits is only a cap under dynamic split, so expected small). Rules: every key attended (causal, full window), FP8 KV read in place, queries E4M3 with identical values, no host syncs, no autotune, no caching across calls.
