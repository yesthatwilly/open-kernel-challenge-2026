---
problem: muse_bf16_proj
slug: router-triton-hilo-fused
created: 2026-10-23T225128Z
base: master
area: router GEMM
change: >
  Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the two
  torch.mm hi/lo GEMMs + add above 8) with one Triton kernel launch at every m. prepare() keeps
  the hi/lo split but stores router_hi and router_lo as [256, 8192] bf16 row-major (N x K, not
  transposed). The kernel: grid = (N/BLOCK_N, SPLIT_K); each program loads an x tile [BLOCK_M,
  BLOCK_K] of bf16 (BLOCK_M=16 for m<=16 with masked rows, else 32/64/128 chosen from m at launch;
  M is looped in BLOCK_M chunks for m=256), loads the hi and lo weight tiles [BLOCK_N, BLOCK_K]
  for its K-slice, and accumulates acc += tl.dot(x, hi.T) + tl.dot(x, lo.T) in fp32 over its K
  range (BLOCK_K=128, num_stages=3); with SPLIT_K>1 it tl.atomic_add's the fp32 partial into an
  output that kernel_fn zeroes with torch.zeros((m, 256), fp32) just before the launch (SPLIT_K=8
  and BLOCK_N=64 for m<=32 gives 32 programs x 8 = 256 CTAs; SPLIT_K=1 with a plain store for
  m>=128). The output is the fp32 [m, 256] tensor, dtype and shape as before. Kernels moved: at
  m<=8 the dot_kernel + x.float() cast + reduce_1Block (m=1) or the two xmma f32 SIMT GEMMs + cast
  (m=8) become one launch (plus one memset); at m>8 the 2 nvjet + 2 splitKreduce + 1 add become
  one launch (plus one memset). The weight read stays 8.4 MB (hi + lo bf16) which at 4.8 TB/s is
  the 1.8 us floor.
expected:
  - shapes: layer=router load=verify_1 *
    direction: faster
    why: "m=8: 26.6 us fp32 SIMT path becomes one tensor-core kernel over 8.4 MB; target under 8 us"
  - shapes: layer=router load=decode_1 *
    direction: faster
    why: "m=1: 3 launches (gemv dot + cast + reduce, 10.4 us) become 1 kernel + memset"
  - shapes: layer=router load=decode_16 *
    direction: faster
    why: 5 launches (2 nvjet splitK + 2 splitKreduce + add, 14.8 us) become 1 kernel + memset
  - shapes: layer=router load=decode_32 *
    direction: faster
    why: same 5-to-1 launch reduction at m=32
  - shapes: layer=router load=verify_16 *
    direction: faster
    why: "m=128: same, SPLIT_K=1, 2x 128x256x8192 bf16 dots are ~1 us of tensor-core work"
  - shapes: layer=router load=verify_32 *
    direction: faster
    why: "m=256: 17.2 us today; one kernel reading 8.4 MB weights + 4 MB x"
  - shapes: "*layer=qkv_gate*"
    direction: flat
    why: bf16 layers untouched; control
risk: medium
depends_on: sglang-gemm-dispatch-parity
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# router-triton-hilo-fused

## Change

Replace the whole router branch of kernel_fn (both the F.linear fp32 path at m<=8 and the two torch.mm hi/lo GEMMs + add above 8) with one Triton kernel launch at every m. prepare() keeps the hi/lo split but stores router_hi and router_lo as [256, 8192] bf16 row-major (N x K, not transposed). The kernel: grid = (N/BLOCK_N, SPLIT_K); each program loads an x tile [BLOCK_M, BLOCK_K] of bf16 (BLOCK_M=16 for m<=16 with masked rows, else 32/64/128 chosen from m at launch; M is looped in BLOCK_M chunks for m=256), loads the hi and lo weight tiles [BLOCK_N, BLOCK_K] for its K-slice, and accumulates acc += tl.dot(x, hi.T) + tl.dot(x, lo.T) in fp32 over its K range (BLOCK_K=128, num_stages=3); with SPLIT_K>1 it tl.atomic_add's the fp32 partial into an output that kernel_fn zeroes with torch.zeros((m, 256), fp32) just before the launch (SPLIT_K=8 and BLOCK_N=64 for m<=32 gives 32 programs x 8 = 256 CTAs; SPLIT_K=1 with a plain store for m>=128). The output is the fp32 [m, 256] tensor, dtype and shape as before. Kernels moved: at m<=8 the dot_kernel + x.float() cast + reduce_1Block (m=1) or the two xmma f32 SIMT GEMMs + cast (m=8) become one launch (plus one memset); at m>8 the 2 nvjet + 2 splitKreduce + 1 add become one launch (plus one memset). The weight read stays 8.4 MB (hi + lo bf16) which at 4.8 TB/s is the 1.8 us floor.

## Expected

- `layer=router load=verify_1 *` faster: m=8: 26.6 us fp32 SIMT path becomes one tensor-core kernel over 8.4 MB; target under 8 us
- `layer=router load=decode_1 *` faster: m=1: 3 launches (gemv dot + cast + reduce, 10.4 us) become 1 kernel + memset
- `layer=router load=decode_16 *` faster: 5 launches (2 nvjet splitK + 2 splitKreduce + add, 14.8 us) become 1 kernel + memset
- `layer=router load=decode_32 *` faster: same 5-to-1 launch reduction at m=32
- `layer=router load=verify_16 *` faster: m=128: same, SPLIT_K=1, 2x 128x256x8192 bf16 dots are ~1 us of tensor-core work
- `layer=router load=verify_32 *` faster: m=256: 17.2 us today; one kernel reading 8.4 MB weights + 4 MB x
- `*layer=qkv_gate*` flat: bf16 layers untouched; control

## Risk

Risks: (1) probe rows at m=1 now use hi/lo (allowed by the rules at any m) where the residual ~1.5e-5 relative must stay under the 1e-3 probe bound and the 1e-4 L2 bound; it should by ~10x, but m=1 has no averaging. (2) tl.atomic_add makes the fp32 sum order nondeterministic across replays; fp32 rounding ~1e-7 is far inside 1e-4. (3) Triton compile of 4-6 (m-bucket) specializations must fit the 10-minute run limit; well under. (4) Graph capture: torch.zeros inside kernel_fn is a fixed-shape allocation plus a memset node, allowed (shapes and config are fixed); no value-dependent branching. (5) The memset + atomics could be replaced by a SPLIT_K=1 full-K loop if the atomic path is slow at m<=32; note it as the ab-tune if the result is only partly faster. (6) If tl.dot with BLOCK_M=16 masked rows is slow at m=1, an FMA fallback (x[:, None, :] * w broadcast + tl.sum) is the alternative; keep that out of this run.
