---
problem: muse_decode_layer
slug: cache-store-fused-fp8-cast
created: 2026-10-23T225338Z
base: master
area: cache store
change: >
  In kernel_fn, replace the four separate launches between FusedQKNorm and
  flash_attn_with_kvcache, namely k.to(FP8), v.to(FP8), store_cache(...) and q.view(tokens, hq,
  d).to(FP8), with ONE Triton kernel, one program per token row: it reads the normed+roped k row
  (bf16 [T, 128]) and the v slice of qkv (bf16 [T, 128], strided view of the [T, 2304] GEMM
  output, read with explicit strides, no .contiguous()), divides both by k_scale / v_scale only
  when cfg['kv_scale'] != 1.0 (a shape/config branch, decided on the host), rounds to
  float8_e4m3fn with round-to-nearest-even and writes them to k_cache.view(-1, 128)[cache_loc[t]]
  and v_cache.view(-1, 128)[cache_loc[t]]; the same program also reads the 1024-wide q row and
  writes it to a preallocated fp8 [T, 16, 64] tensor that is then passed to FA3 as q. k/v stay
  bf16 before the cast exactly as SGLang keeps them. reserved_skip_index=-1 semantics are
  preserved by never skipping any row. Everything else in the attention path (FusedQKNorm, FA3
  call, FusedRMSNormGate, o GEMM) is unchanged. Attribution: 4 launches become 1 on every shape;
  it is only visible on the launch-bound T<=8 shapes.
expected:
  - shapes: "*load=decode_1*"
    direction: faster
    why: "T=1 is launch-bound (SGLang ~172 us against a 40 us floor, ~30 launches): three fewer ~2-3 us launches is ~4-5% of the shape"
  - shapes: "*load=verify_1*"
    direction: faster
    why: "T=8, same launch-bound regime: ~2-3% of ~268 us"
  - shapes: "*load=decode_16*"
    direction: flat
    why: same 3 launches saved but the shape is dominated by GEMM/Marlin/FA3 bytes; at or inside the CI, control
  - shapes: "*load=verify_32*"
    direction: flat
    why: "T=256: the saved launches are below 1% of ~700-1050 us; control"
risk: low
depends_on: sglang-pipeline-parity-router-hilo-m8
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# cache-store-fused-fp8-cast

## Change

In kernel_fn, replace the four separate launches between FusedQKNorm and flash_attn_with_kvcache, namely k.to(FP8), v.to(FP8), store_cache(...) and q.view(tokens, hq, d).to(FP8), with ONE Triton kernel, one program per token row: it reads the normed+roped k row (bf16 [T, 128]) and the v slice of qkv (bf16 [T, 128], strided view of the [T, 2304] GEMM output, read with explicit strides, no .contiguous()), divides both by k_scale / v_scale only when cfg['kv_scale'] != 1.0 (a shape/config branch, decided on the host), rounds to float8_e4m3fn with round-to-nearest-even and writes them to k_cache.view(-1, 128)[cache_loc[t]] and v_cache.view(-1, 128)[cache_loc[t]]; the same program also reads the 1024-wide q row and writes it to a preallocated fp8 [T, 16, 64] tensor that is then passed to FA3 as q. k/v stay bf16 before the cast exactly as SGLang keeps them. reserved_skip_index=-1 semantics are preserved by never skipping any row. Everything else in the attention path (FusedQKNorm, FA3 call, FusedRMSNormGate, o GEMM) is unchanged. Attribution: 4 launches become 1 on every shape; it is only visible on the launch-bound T<=8 shapes.

## Expected

- `*load=decode_1*` faster: T=1 is launch-bound (SGLang ~172 us against a 40 us floor, ~30 launches): three fewer ~2-3 us launches is ~4-5% of the shape
- `*load=verify_1*` faster: T=8, same launch-bound regime: ~2-3% of ~268 us
- `*load=decode_16*` flat: same 3 launches saved but the shape is dominated by GEMM/Marlin/FA3 bytes; at or inside the CI, control
- `*load=verify_32*` flat: T=256: the saved launches are below 1% of ~700-1050 us; control

## Risk

Risk: the FP8 rounding must match torch's .to(float8_e4m3fn) (RNE, non-saturating: values above 448 become NaN in torch; Triton's fp8e4nv conversion saturates or NaNs depending on version). The cache check tolerance is 8% relative L2 and no-NaN, so a saturation difference on an outlier value could trip the NaN check in the dist=outliers test (batch 3, ctx 6000, q_len 8) or the kv_scale=1/64 test: implement the scaled path and test both. The v slice is a strided view of the qkv GEMM output; passing strides to Triton avoids a hidden copy kernel that would give back one of the launches. The q fp8 output must be contiguous [T,16,64] so FA3 takes it without a cast. Confounder: FA3 may itself cast q if given bf16; after this change q is already fp8, so any FA3-side cast kernel disappears too, which belongs to the same mechanism. Also note the second cache-related kernel SGLang runs, store_cache, is a JIT op that may already be a single fused scatter; the saving is then 3 launches, not 4.
