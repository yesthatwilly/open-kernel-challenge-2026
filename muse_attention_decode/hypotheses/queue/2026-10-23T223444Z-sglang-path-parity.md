---
problem: muse_attention_decode
slug: sglang-path-parity
created: 2026-10-23T223444Z
base: master
area: SGLang path parity
change: >
  Replace the starter's plain-PyTorch kernel_fn with SGLang's own pipeline exactly as baseline.py
  runs it, so the submission is the same five-stage sequence SGLang is timed on:
  FusedQKNorm(head_dim, num_q_heads, num_kv_heads, eps=1e-5) from
  sglang.kernels.ops.muse_spark_v12.fused_qk_norm applied to qkv viewed as [B*L, 2304] (with
  positions and the cached [2^21, 64] cos|sin table when rope_theta != 0, else without), then
  k.reshape(tokens, Hkv*64) and v = qkv[:, (Hq+Hkv)*64:(Hq+2*Hkv)*64], both divided by
  k_scale/v_scale only when config kv_scale != 1, cast .to(float8_e4m3fn).view(uint8) and written
  by sglang.kernels.ops.kvcache.kvcache.store_cache into k_cache/v_cache viewed as [-1, Hkv*64]
  uint8 at cache_loc with reserved_skip_index=-1; then
  sgl_kernel.flash_attn.flash_attn_with_kvcache(q.view(tokens, Hq, 64).to(float8_e4m3fn), k_cache,
  v_cache, page_table=page_table, cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q,
  max_seqlen_q=L, softmax_scale=sm_scale*qk_scale_factor, causal=True, window_size=(window-1, 0)
  if window else (-1, -1), scheduler_metadata=data['scheduler_metadata'],
  k_descale/v_descale=scale.reshape(1,1).expand(B, Hkv) only when kv_scale != 1); take out[0] if a
  tuple; then rms_norm_gate(out.reshape(tokens, Hq*64).to(bfloat16), gate=qkv[:, (Hq+2*Hkv)*64:],
  None, head_dim=64, eps=1e-5, has_norm=True) and return .view(B, L, Hq*64). The module-level
  _NORMS and _ROPE caches are keyed on config values (theta, head counts), never on data. Nothing
  else changes. This is the base every other attention hypothesis stacks on; the starter's 0.0023x
  is an fp32 cuBLAS GEMM over every page-table slot and carries no information about any area.
expected:
  - shapes: "*"
    direction: faster
    why: "from 0.0023x to about 1.0x: the starter attends over every slot the page table can hold with an fp32 gemmSN_NN GEMM (85-88% of every shape) plus a gather; SGLang runs FA3 over the FP8 cache"
  - shapes: "*layer=sliding*"
    direction: flat
    why: "relative to SGLang: identical launches (FusedQKNorm, two FP8 casts, store_cache, q cast, FA3, rms_norm_gate), so the paired-ratio median should sit at 1.0x within the CI"
  - shapes: "*layer=full*"
    direction: flat
    why: same FA3 call with the same scheduler_metadata as SGLang; 1.0x within the CI
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# sglang-path-parity

## Change

Replace the starter's plain-PyTorch kernel_fn with SGLang's own pipeline exactly as baseline.py runs it, so the submission is the same five-stage sequence SGLang is timed on: FusedQKNorm(head_dim, num_q_heads, num_kv_heads, eps=1e-5) from sglang.kernels.ops.muse_spark_v12.fused_qk_norm applied to qkv viewed as [B*L, 2304] (with positions and the cached [2^21, 64] cos|sin table when rope_theta != 0, else without), then k.reshape(tokens, Hkv*64) and v = qkv[:, (Hq+Hkv)*64:(Hq+2*Hkv)*64], both divided by k_scale/v_scale only when config kv_scale != 1, cast .to(float8_e4m3fn).view(uint8) and written by sglang.kernels.ops.kvcache.kvcache.store_cache into k_cache/v_cache viewed as [-1, Hkv*64] uint8 at cache_loc with reserved_skip_index=-1; then sgl_kernel.flash_attn.flash_attn_with_kvcache(q.view(tokens, Hq, 64).to(float8_e4m3fn), k_cache, v_cache, page_table=page_table, cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=L, softmax_scale=sm_scale*qk_scale_factor, causal=True, window_size=(window-1, 0) if window else (-1, -1), scheduler_metadata=data['scheduler_metadata'], k_descale/v_descale=scale.reshape(1,1).expand(B, Hkv) only when kv_scale != 1); take out[0] if a tuple; then rms_norm_gate(out.reshape(tokens, Hq*64).to(bfloat16), gate=qkv[:, (Hq+2*Hkv)*64:], None, head_dim=64, eps=1e-5, has_norm=True) and return .view(B, L, Hq*64). The module-level _NORMS and _ROPE caches are keyed on config values (theta, head counts), never on data. Nothing else changes. This is the base every other attention hypothesis stacks on; the starter's 0.0023x is an fp32 cuBLAS GEMM over every page-table slot and carries no information about any area.

## Expected

- `*` faster: from 0.0023x to about 1.0x: the starter attends over every slot the page table can hold with an fp32 gemmSN_NN GEMM (85-88% of every shape) plus a gather; SGLang runs FA3 over the FP8 cache
- `*layer=sliding*` flat: relative to SGLang: identical launches (FusedQKNorm, two FP8 casts, store_cache, q cast, FA3, rms_norm_gate), so the paired-ratio median should sit at 1.0x within the CI
- `*layer=full*` flat: same FA3 call with the same scheduler_metadata as SGLang; 1.0x within the CI

## Risk

Low risk: it is SGLang's code path, imported from the same image baseline.py imports from. Only risk: the _ROPE table (2^21 x 64 x fp32 = 512 MiB) is built lazily on the first sliding call; if the judge captures the graph on the very first call that allocation happens under capture. Build it at import time for theta=500000 (config is fixed per problem) or on the warm-up call to remove the doubt. Expected gate to pass: all 16 tests, since the baseline passes them by construction.
