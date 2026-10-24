---
problem: muse_prefill_layer
slug: qk-norm-fp8-cache-store-fusion
created: 2026-10-23T230250Z
base: master
area: pre-attention norm+rope + cache store
change: >
  One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8),
  v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per token row (grid T; split q
  and k/v into two programs only if register pressure forces it), reading the row of the qkv GEMM
  output [T, 2304] bf16 once: per q head (16) and k head (2) an RMSNorm over 64 lanes in fp32 with
  eps 1e-5 and no weight (FusedQKNorm is built without a weight; reference.py _rms confirms); on
  sliding layers (config rope_theta != 0) the interleaved RoPE in fp32 from the parity cos|sin
  table row at positions[t] (cos lanes 0..31, sin lanes 32..63), rotating lane pairs (2i, 2i+1)
  exactly as reference.py _rope; q rounded to bf16 then to FP8 E4M3 (two roundings, matching
  parity's bf16 FusedQKNorm output followed by .to(FP8)) and written to q_fp8 [T, 16, 64]; k
  rounded the same way, divided by k_scale after the bf16 rounding when config kv_scale != 1
  (parity order), stored as FP8 bytes at k_cache.view(-1, 2, 64)[cache_loc[t]]; v (bf16 columns
  1152..1279) divided by v_scale when scaled, cast to FP8 and stored at the same cache row of
  v_cache. The gate slice stays a view of qkv; FA3 takes q_fp8 directly. Everything after the FA3
  call stays the parity path. Before submitting, compare against the parity kernels on random T=64
  inputs for both layer types: q_fp8 and the cache rows bitwise equal, allowing at most one-ULP
  FP8 differences from fp32 summation order in the RMS.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "four fewer launches (~10-15 us) and three fewer passes over the [753, 1152] bf16 q/k slices; ~1-1.5% of 1120-1465 us, the shapes where launch count still matters"
  - shapes: "*load=turn_4*"
    direction: flat
    why: same launches saved, ~0.3% of 3.4-4.9 ms, inside the interval
  - shapes: "*load=first_turn*"
    direction: flat
    why: ~0.1% of 17-26 ms
risk: medium
depends_on: sglang-pipeline-parity
priority: 4
kind: primary
controls_for: null
review: required
status: queued
---

# qk-norm-fp8-cache-store-fusion

## Change

One Triton kernel replaces five launches on the attention front: FusedQKNorm, k.to(FP8), v.to(FP8), store_cache and q.view(T, 16, 64).to(FP8). One program per token row (grid T; split q and k/v into two programs only if register pressure forces it), reading the row of the qkv GEMM output [T, 2304] bf16 once: per q head (16) and k head (2) an RMSNorm over 64 lanes in fp32 with eps 1e-5 and no weight (FusedQKNorm is built without a weight; reference.py _rms confirms); on sliding layers (config rope_theta != 0) the interleaved RoPE in fp32 from the parity cos|sin table row at positions[t] (cos lanes 0..31, sin lanes 32..63), rotating lane pairs (2i, 2i+1) exactly as reference.py _rope; q rounded to bf16 then to FP8 E4M3 (two roundings, matching parity's bf16 FusedQKNorm output followed by .to(FP8)) and written to q_fp8 [T, 16, 64]; k rounded the same way, divided by k_scale after the bf16 rounding when config kv_scale != 1 (parity order), stored as FP8 bytes at k_cache.view(-1, 2, 64)[cache_loc[t]]; v (bf16 columns 1152..1279) divided by v_scale when scaled, cast to FP8 and stored at the same cache row of v_cache. The gate slice stays a view of qkv; FA3 takes q_fp8 directly. Everything after the FA3 call stays the parity path. Before submitting, compare against the parity kernels on random T=64 inputs for both layer types: q_fp8 and the cache rows bitwise equal, allowing at most one-ULP FP8 differences from fp32 summation order in the RMS.

## Expected

- `*load=turn_1*` faster: four fewer launches (~10-15 us) and three fewer passes over the [753, 1152] bf16 q/k slices; ~1-1.5% of 1120-1465 us, the shapes where launch count still matters
- `*load=turn_4*` flat: same launches saved, ~0.3% of 3.4-4.9 ms, inside the interval
- `*load=first_turn*` flat: ~0.1% of 17-26 ms

## Risk

Medium risk on exactness, low on performance. The cache-entry bound is 8% relative and the new entries are FP8-rounded in both paths, so a one-ULP FP8 flip on a few lanes is harmless; a RoPE pairing or cos/sin layout mistake would fail the first sliding test. Torch's .to(float8_e4m3fn) and Triton's .to(tl.float8e4nv) both round to nearest even without saturation; q and k are RMS-normalized so no lane can overflow, v comes from the same bf16 GEMM output in both paths. Rules: queries FP8 as FA3 does, cache FP8, nothing cached, no host sync. The gain is small by construction; a flat result on turn_1 is the honest outcome and attributes to launch overhead not mattering at M=753.
