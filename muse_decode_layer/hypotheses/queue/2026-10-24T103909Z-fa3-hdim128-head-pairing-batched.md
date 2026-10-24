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
review: proceed
status: queued
review_at: 2026-10-24T104215Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-24T104215Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a direct port of a measured 14-23% FA3 gain on the same kernel (hq=16, hkv=2, d=64, FP8 page-1 cache). The batch>1 gate drops the shapes where attention_decode lost. Because the full layer has weight 16 against the sliding layer's 46, the expected headline gain is only about +1.8%, inside the drift. So the run must be judged on the FA3 kernel row and the per-shape full-layer rows, and no companion is needed.

### Justification
The mechanism is plausible, and it has been measured on this exact kernel already. In FINDINGS, the full-layer FA3 kernel takes 188.8 us at decode_16, 374.4 at decode_32, 191.9 at verify_16 and 374.0 at verify_32. In attention_decode's base the same kernel took 194/378/198/383 us, so the geometry and the kernel are the same, and pairing took those to 165/307/173/309 us there (job facc479992e24a35, 16/16 tests). Every load here is 46 sliding steps plus 16 full steps. Taking the brief's savings (25-30, 60-70, 20-25 and 65-75 us per full shape) gives these load gains: decode_32 ~4.9% (about 1040 us of 21306), decode_16 ~2.3%, verify_32 ~2.5%, verify_16 ~1.2%, and 0 for decode_1/verify_1. The geomean comes to about +1.8%, which is below the ±1.5% CI resolution plus the 1.5-2.5% per-shape drift in LESSONS. The brief's per-shape expectations (faster on full B=16/32, flat on sliding and B=1) agree with the TASK weights. The brief's '~1.5-3% per load' undersells decode_32 and oversells verify_16, but the directions are right. The full-shape floors (137.5-267.2 us for the whole layer) leave ample room. In decode_layer, FA3 runs alone on the main stream between store_cache and o_norm_gate, with nothing overlapping it. Weakest link: the net per-call overhead. The brief counts zeros, two pad copies, empty and two un-pad copies, minus the removed cast, as ~1 us. Inside a captured graph each tiny elementwise node costs roughly 1.5-3 us (LESSONS: graph-node residuals of ~0.3-4 us), and dropping the judge's scheduler_metadata makes FA3 run its own prepare_varlen_num_blocks launch on decode. A realistic overhead is therefore 5-10 us per full B=16/32 call. That is still small against 25-70 us saved, but at verify_16 the per-load gain could shrink to ~0.8%. Second link: the BF16->E4M3 cast now happens in a strided copy_ into a non-contiguous destination. The values are identical (copy_ and .to share the same conversion), but these are two strided TensorIterator kernels rather than one contiguous cast, which is negligible at tokens x 16 x 64.

### Confounders on the current kernel
- marlin-block-m-cap-16: no confound. It changes only the MoE Marlin GEMMs, which run after o GEMM with no stream overlap against FA3 and no shared memory traffic within the call; its effect is in the baseline equally for both sides of the A/B against best.
- sglang-pipeline-parity-router-hilo-m8: no separable confound. It is the base pipeline that produces q as BF16 from FusedQKNorm and casts inline; the hypothesis already folds that cast into the pad. The router side-stream overlaps the pooler, not FA3. Reverting it would remove the whole pipeline the change is written against, so an ab-revert is meaningless.

### Interference and A/B needs
No companion is needed. If the run gains, the FA3 kernel row (cutlass flash_fwd) on the four full B=16/32 shapes drops toward ~165/307/173/309 us while sliding and B=1 stay flat, and only the paired geometry can cause that. If it loses or comes out flat, the profile splits the causes three ways without another run. First, the paired FA3 kernel is itself slower (its own row). Second, the split/combine schedule changed because the judge's metadata was dropped (the FlashAttnFwdCombine row, and the extra prepare launch on decode only). Third, pad/un-pad overhead (the new small copy rows plus the shape-minus-kernel-sum residual). A 'keep the judge metadata' control cannot be built, because that metadata was made for hdim 64 / Hkv 2. A num_splits control would duplicate the queued fa3-num-splits-full-no-metadata and fa3-full-single-split-probe. The B=1 gate is self-controlling: decode_1/verify_1 full run master's code byte for byte, so any movement there is drift and tells you the run's noise level. Harvest instruction: classify on FA3's own us and the per-shape full-layer ratios, not the headline, since about +1.8% expected can land as 'within-ci' while the mechanism is fully confirmed. Note that triton-flash-decode-full-oversub-kv-prefetch (queued) replaces this same call on full decode; whichever lands second must be measured against the other.

### Open questions
- What is the real net overhead of the zeros + 2 pad copies + 2 un-pad copies + FA3's internal metadata prepare, against the one removed cast, at full B=16/32? Read the shape time minus kernel sum before and after. If it exceeds ~10 us, a follow-up should fuse the pad into FusedQKNorm's output or write q_pad from a tiny Triton kernel, as attention_decode's paired-heads-launch-cleanup does.
- Does dist=soft (diffuse softmax) change FA3's split schedule or combine cost, compared with attention_decode's dist=coherent? The gather pattern is the same, but the combine row should be checked.
- Which tests exercise page_size 64 with the paired path? batch 3 ctx 6000 full q_len 8 and batch 1 ctx 3000 full use the default page size; batch 1 ctx 3000 full is gated off. Confirm that the paired hdim-128 FP8 kernel with page_size>1 (possibly the TMA paged path) passes here, as it did in attention_decode.
- Will the harvester's verdict logic mark a confirmed FA3 gain of ~1.8% on the headline as within-ci/neutral? If so, promotion should be decided on the per-shape full rows.

