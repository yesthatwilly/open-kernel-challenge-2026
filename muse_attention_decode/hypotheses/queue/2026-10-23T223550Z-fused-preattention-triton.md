---
problem: muse_attention_decode
slug: fused-preattention-triton
created: 2026-10-23T223550Z
base: master
area: pre-attention norm+rope
change: >
  Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8),
  v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, grid (B*L, Hq+Hkv), one program per
  (token, head), num_warps=1. Programs h < Hq load the head's 64 bf16 entries from qkv row t at
  offset h*64 as even/odd halves, apply scaleless RMSNorm (rsqrt(mean(x^2)+1e-5)), apply
  interleaved RoPE with pos=positions[t] and inv_freq[i] = theta^(-2i/64) (a per-(theta, device)
  module-level table, built from config, not data) when rope_theta != 0, and store the result as
  float8e4nv into a fresh [B*L, Hq, 64] FP8 query buffer (the 5.473 factor stays folded into FA3's
  softmax_scale). Programs h >= Hq do the same norm+RoPE on key head h-Hq, divide by k_scale, and
  store FP8 straight into k_cache.view(-1, Hkv*64) at row cache_loc[t]; the same program loads the
  matching value head (offset (Hq+Hkv+kh)*64), divides by v_scale and stores it FP8 into v_cache
  at the same row. FA3 (with the FP8 query buffer) and rms_norm_gate stay exactly as in the parity
  pipeline. The exact kernel is already written in the uncommitted working-tree submission.py
  (check job dfee6f2939974bad, queued, no result yet); churnsubs should take that code verbatim
  onto the parity base.
expected:
  - shapes: "*layer=sliding load=decode_1*"
    direction: faster
    why: "launch-bound: SGLang spends ~21 us on a shape whose floor is 0.1 us; four of its seven launches disappear (about 8-10 us of ~2.5 us launches)"
  - shapes: "*layer=sliding load=verify_1*"
    direction: faster
    why: same launch count, 8 tokens; still launch-bound (23 us vs 0.1 us floor)
  - shapes: "*layer=sliding load=decode_16*"
    direction: faster
    why: "27 us vs 1.6 us floor: still dominated by fixed launch cost, same four launches removed"
  - shapes: "*layer=sliding load=decode_32*"
    direction: faster
    why: 31.5 us vs 3.3 us floor, same mechanism
  - shapes: "*layer=sliding load=verify_*"
    direction: faster
    why: same mechanism at 8 tokens
  - shapes: "*layer=full load=decode_1*"
    direction: faster
    why: 30.8 us vs 4.3 us floor; the same four launches go, FA3 unchanged
  - shapes: "*layer=full load=decode_16*"
    direction: flat
    why: 201 us is FA3 over 16 x 80K tokens; the saved launches are under 5% of it
  - shapes: "*layer=full load=decode_32*"
    direction: flat
    why: 388 us, FA3-bound
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# fused-preattention-triton

## Change

Replace the five pre-attention launches of the parity pipeline (FusedQKNorm (+RoPE), k .to(FP8), v .to(FP8), store_cache, q .to(FP8)) with one Triton kernel, grid (B*L, Hq+Hkv), one program per (token, head), num_warps=1. Programs h < Hq load the head's 64 bf16 entries from qkv row t at offset h*64 as even/odd halves, apply scaleless RMSNorm (rsqrt(mean(x^2)+1e-5)), apply interleaved RoPE with pos=positions[t] and inv_freq[i] = theta^(-2i/64) (a per-(theta, device) module-level table, built from config, not data) when rope_theta != 0, and store the result as float8e4nv into a fresh [B*L, Hq, 64] FP8 query buffer (the 5.473 factor stays folded into FA3's softmax_scale). Programs h >= Hq do the same norm+RoPE on key head h-Hq, divide by k_scale, and store FP8 straight into k_cache.view(-1, Hkv*64) at row cache_loc[t]; the same program loads the matching value head (offset (Hq+Hkv+kh)*64), divides by v_scale and stores it FP8 into v_cache at the same row. FA3 (with the FP8 query buffer) and rms_norm_gate stay exactly as in the parity pipeline. The exact kernel is already written in the uncommitted working-tree submission.py (check job dfee6f2939974bad, queued, no result yet); churnsubs should take that code verbatim onto the parity base.

## Expected

- `*layer=sliding load=decode_1*` faster: launch-bound: SGLang spends ~21 us on a shape whose floor is 0.1 us; four of its seven launches disappear (about 8-10 us of ~2.5 us launches)
- `*layer=sliding load=verify_1*` faster: same launch count, 8 tokens; still launch-bound (23 us vs 0.1 us floor)
- `*layer=sliding load=decode_16*` faster: 27 us vs 1.6 us floor: still dominated by fixed launch cost, same four launches removed
- `*layer=sliding load=decode_32*` faster: 31.5 us vs 3.3 us floor, same mechanism
- `*layer=sliding load=verify_*` faster: same mechanism at 8 tokens
- `*layer=full load=decode_1*` faster: 30.8 us vs 4.3 us floor; the same four launches go, FA3 unchanged
- `*layer=full load=decode_16*` flat: 201 us is FA3 over 16 x 80K tokens; the saved launches are under 5% of it
- `*layer=full load=decode_32*` flat: 388 us, FA3-bound

## Risk

Risks: the per-(theta, device) inv_freq table is built lazily on the first sliding call; build it at import or in the first eager call so nothing allocates under graph capture. num_warps=1 with 32 lanes per half-head is sized for head_dim 64 only. New FP8 cache entries must stay within 8% of the reference: norm and RoPE are computed in fp32 exactly as reference._rms/_rope, so the only difference is the fp32-vs-bf16 rounding before the FP8 cast, which SGLang's own path shares. The rms check for the output is against the submission's own cache entries, so the k/v write path is checked separately.
