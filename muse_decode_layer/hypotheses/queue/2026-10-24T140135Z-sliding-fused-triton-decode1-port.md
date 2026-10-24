---
problem: muse_decode_layer
slug: sliding-fused-triton-decode1-port
created: 2026-10-24T140135Z
base: master
area: FA3 call
change: >
  Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, sliding
  decode_1 16.56 -> 13.12 us) to muse_decode_layer kernel_fn. Copy verbatim from
  muse_attention_decode/submission.py on master: _sliding_fused_kernel and its helpers
  (_norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _num_splits, _counters with its
  _COUNTERS cache, _inv_freq with its _INV_FREQ cache) and the body of _sliding_forward. In
  kernel_fn, right after qkv = _gemm(attn_input, qkv_gate), add a config-only gate: if
  cfg['window'] and batch == 1 and length == 1, compute o with one _sliding_fused_kernel launch
  (grid (1, hkv, splits=16), BLOCK_N 64, num_warps 4, num_stages 2, ROPE =
  bool(cfg['rope_theta']), EPS RMS_EPS) on qkv (the GEMM output, [1, 2304], stride(0) passed),
  data['positions'], data['cache_loc'], data['seq_lens'], data['page_table'], data['k_scale'],
  data['v_scale'], k_cache/v_cache viewed uint8, writing the [1, 1024] bf16 normed-and-gated
  attention output, then continue at attn_output = _gemm(o, dense['o']) exactly as master. That
  one launch does the qk RMSNorm + interleaved RoPE, the FP8 K/V store at cache_loc, the 16-way
  split 2048-token window attention with in-kernel ticket combine, and the per-head output norm x
  sigmoid(gate), so it replaces master's FusedQKNorm, the k and v FP8 casts, store_cache, the q
  FP8 cast, FA3's prepare/main/combine and FusedRMSNormGate at that shape. Every other shape
  (sliding batch 16/32, every verify block, every full layer, paired or not) keeps master's path
  byte for byte. The _inv_freq table is built from cfg['rope_theta'] with the reference's formula,
  as in the attention problem (decode's cos_sin table stays for the other shapes).
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: attention measured 16.6 -> 13.1 us for the same replaced launches; here the master path has 4-5 more pre-attention launches (separate casts and store_cache), so expect 4-8 us off 172.8 us, +2.5-4.5% on the shape and ~+2-3% on the decode_1 load
  - shapes: "*layer=full*"
    direction: flat
    why: untouched
  - shapes: "*layer=sliding load=decode_16*"
    direction: flat
    why: gated out
  - shapes: "*layer=sliding load=decode_32*"
    direction: flat
    why: gated out
  - shapes: "*layer=sliding load=verify_*"
    direction: flat
    why: gated out
risk: medium
depends_on: null
priority: 3
kind: primary
controls_for: null
review: required
status: queued
---

# sliding-fused-triton-decode1-port

## Change

Port muse_attention_decode's landed sliding-fused-triton-decode1-gate (1.293x there, sliding decode_1 16.56 -> 13.12 us) to muse_decode_layer kernel_fn. Copy verbatim from muse_attention_decode/submission.py on master: _sliding_fused_kernel and its helpers (_norm_rope, _gpu_fence, _online_update, _gate_epilogue, LOG2E, _num_splits, _counters with its _COUNTERS cache, _inv_freq with its _INV_FREQ cache) and the body of _sliding_forward. In kernel_fn, right after qkv = _gemm(attn_input, qkv_gate), add a config-only gate: if cfg['window'] and batch == 1 and length == 1, compute o with one _sliding_fused_kernel launch (grid (1, hkv, splits=16), BLOCK_N 64, num_warps 4, num_stages 2, ROPE = bool(cfg['rope_theta']), EPS RMS_EPS) on qkv (the GEMM output, [1, 2304], stride(0) passed), data['positions'], data['cache_loc'], data['seq_lens'], data['page_table'], data['k_scale'], data['v_scale'], k_cache/v_cache viewed uint8, writing the [1, 1024] bf16 normed-and-gated attention output, then continue at attn_output = _gemm(o, dense['o']) exactly as master. That one launch does the qk RMSNorm + interleaved RoPE, the FP8 K/V store at cache_loc, the 16-way split 2048-token window attention with in-kernel ticket combine, and the per-head output norm x sigmoid(gate), so it replaces master's FusedQKNorm, the k and v FP8 casts, store_cache, the q FP8 cast, FA3's prepare/main/combine and FusedRMSNormGate at that shape. Every other shape (sliding batch 16/32, every verify block, every full layer, paired or not) keeps master's path byte for byte. The _inv_freq table is built from cfg['rope_theta'] with the reference's formula, as in the attention problem (decode's cos_sin table stays for the other shapes).

## Expected

- `*layer=sliding load=decode_1 *` faster: attention measured 16.6 -> 13.1 us for the same replaced launches; here the master path has 4-5 more pre-attention launches (separate casts and store_cache), so expect 4-8 us off 172.8 us, +2.5-4.5% on the shape and ~+2-3% on the decode_1 load
- `*layer=full*` flat: untouched
- `*layer=sliding load=decode_16*` flat: gated out
- `*layer=sliding load=decode_32*` flat: gated out
- `*layer=sliding load=verify_*` flat: gated out

## Risk

Single-shape gain (~+0.4-0.6% geomean) but on the heaviest-weighted shape of a launch-bound load, and the kernel is proven correct and faster on identical inputs in the sibling. Caveat: decode's fused-preattention-triton (neutral) showed that fusing pre-attention launches alone did not move sliding decode_1 here, so the gain may be only the attention+norm-gate part (~3.5 us, +2%). Correctness risk is in the I/O mapping: decode's qkv is [T, 2304] contiguous from the GEMM (attention's was [B, L, 2304]); the fused kernel's RoPE must match FusedQKNorm's interleaved convention (it passed the attention problem's tests with the same SGLang reference). The 13 tests include batch 1 ctx 10 sliding (q_len 1), which takes this path at small ctx and its page size. Composes with post-expert-norm-triton-small-t and the bf16 GEMM port (disjoint kernels on the same shape).
