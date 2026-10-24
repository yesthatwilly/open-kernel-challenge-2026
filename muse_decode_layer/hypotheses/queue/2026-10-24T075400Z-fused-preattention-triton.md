---
problem: muse_decode_layer
slug: fused-preattention-triton
created: 2026-10-24T075400Z
base: master
area: pre-attention norm+rope
change: >
  Port muse_attention_decode's fused pre-attention Triton kernel (main tree
  muse_attention_decode/submission.py, _pre_attention_kernel: passed 16/16 at 1.127x there) into
  kernel_fn, replacing on every shape the five launches between the qkv GEMM and
  flash_attn_with_kvcache: state['qk_norm'](...) (FusedQKNorm, with RoPE on sliding layers), the
  optional k/v divide by the cache scales, k.to(FP8), v.to(FP8), store_cache(...) and
  q.view(T,16,64).to(FP8). One launch, grid (T, 16+2), num_warps=1, one program per (token, head):
  heads 0-15 RMS-normalize (scaleless, eps 1e-5) the q head read from the qkv row (stride
  qkv.stride(0)), apply interleaved RoPE when cfg['rope_theta'] is set and store FP8 into a fresh
  q buffer [T, 16, 64] float8_e4m3fn; heads 16-17 normalize (+RoPE) the k head, store (k /
  k_scale) as FP8 into k_cache.view(-1, 128) row cache_loc[t] at column (h-16)*64, and store the
  matching v head (v / v_scale, no norm or RoPE) into v_cache at the same row. One deliberate
  difference from the sibling: RoPE cos/sin are read from master's state['cos_sin'] table (row
  positions[t], cos in columns 0-31, sin in 32-63) instead of recomputed in-kernel, so the
  rotation is the table SGLang's FusedQKNorm uses. The FP8 q goes to flash_attn_with_kvcache
  unchanged otherwise; gate and v are still slices of qkv. FA3, the output norm-gate and
  everything downstream stay as on master.
expected:
  - shapes: "*load=decode_1*"
    direction: faster
    why: 5 serialized launches -> 1 on a latency-bound 169/181 us shape; the sibling saved 3.5-6.4 us for this exact 5->1 fusion (~2-3.5%)
  - shapes: "*load=verify_1*"
    direction: faster
    why: T=8, same 5->1 fusion on 229-251 us shapes (~1.5-2.5%)
  - shapes: "*load=decode_16*"
    direction: flat
    why: a few us on 265-440 us shapes, under the drift
  - shapes: "*load=decode_32*"
    direction: flat
    why: as decode_16
  - shapes: "*load=verify_16*"
    direction: flat
    why: 520-714 us shapes
  - shapes: "*load=verify_32*"
    direction: flat
    why: 713-1050 us shapes
risk: medium
depends_on: null
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# fused-preattention-triton

## Change

Port muse_attention_decode's fused pre-attention Triton kernel (main tree muse_attention_decode/submission.py, _pre_attention_kernel: passed 16/16 at 1.127x there) into kernel_fn, replacing on every shape the five launches between the qkv GEMM and flash_attn_with_kvcache: state['qk_norm'](...) (FusedQKNorm, with RoPE on sliding layers), the optional k/v divide by the cache scales, k.to(FP8), v.to(FP8), store_cache(...) and q.view(T,16,64).to(FP8). One launch, grid (T, 16+2), num_warps=1, one program per (token, head): heads 0-15 RMS-normalize (scaleless, eps 1e-5) the q head read from the qkv row (stride qkv.stride(0)), apply interleaved RoPE when cfg['rope_theta'] is set and store FP8 into a fresh q buffer [T, 16, 64] float8_e4m3fn; heads 16-17 normalize (+RoPE) the k head, store (k / k_scale) as FP8 into k_cache.view(-1, 128) row cache_loc[t] at column (h-16)*64, and store the matching v head (v / v_scale, no norm or RoPE) into v_cache at the same row. One deliberate difference from the sibling: RoPE cos/sin are read from master's state['cos_sin'] table (row positions[t], cos in columns 0-31, sin in 32-63) instead of recomputed in-kernel, so the rotation is the table SGLang's FusedQKNorm uses. The FP8 q goes to flash_attn_with_kvcache unchanged otherwise; gate and v are still slices of qkv. FA3, the output norm-gate and everything downstream stay as on master.

## Expected

- `*load=decode_1*` faster: 5 serialized launches -> 1 on a latency-bound 169/181 us shape; the sibling saved 3.5-6.4 us for this exact 5->1 fusion (~2-3.5%)
- `*load=verify_1*` faster: T=8, same 5->1 fusion on 229-251 us shapes (~1.5-2.5%)
- `*load=decode_16*` flat: a few us on 265-440 us shapes, under the drift
- `*load=decode_32*` flat: as decode_16
- `*load=verify_16*` flat: 520-714 us shapes
- `*load=verify_32*` flat: 713-1050 us shapes

## Risk

This is the superset of cache-store-fused-fp8-cast (3 of 4 cast/store launches removed, came back flat at 0.996/1.000 on decode_1): it additionally folds FusedQKNorm, the launch the earlier run left in, which is what the muse_attention_decode sibling fused to get its 3.5-6.4 us. Built on master, not on that neutral branch (the fused kernel replaces its code entirely). Expected per-shape gains (~2-3%) sit near the 1.5% drift, so read the T=1 and T=8 rows individually. Correctness: q is rounded fp32 -> FP8 directly instead of fp32 -> bf16 -> FP8 (allowed: queries may be rounded to FP8); k likewise; the kv_scale 0.015625 test and page-64 tests exercise the scale divide and the flat-row addressing (loc * 2 + kh) * 64. If triton-flash-decode-full-paired-kv lands first, its q input is this kernel's FP8 q buffer.
