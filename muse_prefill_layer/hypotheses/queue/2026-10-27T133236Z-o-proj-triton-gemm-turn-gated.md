---
problem: muse_prefill_layer
slug: o-proj-triton-gemm-turn-gated
created: 2026-10-27T133236Z
base: master
area: o GEMM (attention output projection at 200 < T <= 4096)
change: >
  Built on master (blob 42ded471). Only the call attn_output = _gemm(o, dense['o']) in kernel_fn
  changes, and only at 200 < T <= 4096. MECHANISM: the o projection is [T, 1024] x [8192, 1024]^T
  (K = 1024). At turn_1 (T = 753) cuBLAS runs it as nvjet_sm90_tst_112x128_64x7_1x2_h_bz_TNN in
  39.7 us (sliding) / 40.1 us (full) against a 12.8 us math floor (12.6 GFLOP) and ~6.4 us of
  bytes: 3.1x, the largest gap to the floor among the dense GEMMs (the others run at ~80-85% of
  peak). With K = 1024 each 112x128 tile has only 16 k-steps, so 448 tiles in 3.4 waves are
  dominated by per-tile prologue/epilogue and the partial last wave. CHANGE: a Triton bf16 GEMM
  _o_proj_kernel (C = A @ W^T, tl.dot with fp32 accumulation, one bf16 rounding at the store, K =
  1024 constexpr, unmasked K loop, masked M edge, grouped tile order GROUP_M = 8 for L2 reuse of
  W, plain pointer loads with num_stages pipelining: no TMA descriptors, no allocator hook) with
  three candidate configs (BM, BN, BK, stages, warps) = (128, 128, 64, 4, 8), (128, 256, 64, 3,
  8), (64, 256, 64, 4, 4). GATE in prepare() per class, low (T = 753) and high (T = 3012), after
  master's setup and before the rega-lo / fsilu phases (start deadline 150 s, skipped otherwise):
  correctness per config on random A (and A from a real o_norm_gate output) against an fp64
  reference: finite, rel L2 <= 1.05x cuBLAS's own rel L2, worst row <= 2x cuBLAS's; then one-GEMM
  CUDA graphs (cuBLAS _gemm and each passing config) plus an empty graph, 2 rounds x 21
  interleaved alternations, cold L2 (256 MB read flush before each replay, as the judge),
  empty-graph baseline subtracted; ON with the fastest config only when it is <= 0.97x cuBLAS in
  BOTH rounds (the interleaved selector with margin is the form that carried over in
  muse_bf16_proj; a single prepare timing did not). One torch.profiler call with the path ON
  asserts the launched kernel name is _o_proj_kernel. state['o_proj_cfg'][cls] = config tuple or
  None; kernel_fn branches on tokens (a shape) only. T > 4096 and T <= 200 keep _gemm. Print per
  class cuBLAS us, each config's us and rel L2, and '[o-proj] PATH CHOSEN'.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: "o row 39.7 us (sliding) / 40.1 us (full) -> ~15-25 us (128x128 tiles: 384 tiles, 2.9 waves of 16 k-steps; 128x256: 192 tiles): -15 to -25 us per shape, 2-3.5% of sliding turn_1 and 1.3-2.2% of full turn_1; about +0.7-1.0% headline (turn_1 is the weakest load)"
  - shapes: "*load=turn_4*"
    direction: flat
    why: At T = 3012 cuBLAS has 12+ waves and runs near its peak share; the gate should keep cuBLAS unless a config is >= 3% faster
  - shapes: "*load=first_turn*"
    direction: flat
    why: T = 16384 is untouched (master _gemm)
risk: low
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "dense-gemm-cublaslt-algo-pick-prefill-t: that (expired, never run) picked cuBLASLt algos for all dense GEMMs, and bf16's in-prepare algo pick did not carry to the judge (LESSONS: qkv m=256 6.8% faster in prepare, 14% slower judged); this replaces only the 3x-off-floor o projection at T <= 4096 with a Triton kernel behind an interleaved cold-L2 graph selector with a 3% margin and a launched-kernel-name assert"
review: required
status: queued
---

# o-proj-triton-gemm-turn-gated

## Change

Built on master (blob 42ded471). Only the call attn_output = _gemm(o, dense['o']) in kernel_fn changes, and only at 200 < T <= 4096. MECHANISM: the o projection is [T, 1024] x [8192, 1024]^T (K = 1024). At turn_1 (T = 753) cuBLAS runs it as nvjet_sm90_tst_112x128_64x7_1x2_h_bz_TNN in 39.7 us (sliding) / 40.1 us (full) against a 12.8 us math floor (12.6 GFLOP) and ~6.4 us of bytes: 3.1x, the largest gap to the floor among the dense GEMMs (the others run at ~80-85% of peak). With K = 1024 each 112x128 tile has only 16 k-steps, so 448 tiles in 3.4 waves are dominated by per-tile prologue/epilogue and the partial last wave. CHANGE: a Triton bf16 GEMM _o_proj_kernel (C = A @ W^T, tl.dot with fp32 accumulation, one bf16 rounding at the store, K = 1024 constexpr, unmasked K loop, masked M edge, grouped tile order GROUP_M = 8 for L2 reuse of W, plain pointer loads with num_stages pipelining: no TMA descriptors, no allocator hook) with three candidate configs (BM, BN, BK, stages, warps) = (128, 128, 64, 4, 8), (128, 256, 64, 3, 8), (64, 256, 64, 4, 4). GATE in prepare() per class, low (T = 753) and high (T = 3012), after master's setup and before the rega-lo / fsilu phases (start deadline 150 s, skipped otherwise): correctness per config on random A (and A from a real o_norm_gate output) against an fp64 reference: finite, rel L2 <= 1.05x cuBLAS's own rel L2, worst row <= 2x cuBLAS's; then one-GEMM CUDA graphs (cuBLAS _gemm and each passing config) plus an empty graph, 2 rounds x 21 interleaved alternations, cold L2 (256 MB read flush before each replay, as the judge), empty-graph baseline subtracted; ON with the fastest config only when it is <= 0.97x cuBLAS in BOTH rounds (the interleaved selector with margin is the form that carried over in muse_bf16_proj; a single prepare timing did not). One torch.profiler call with the path ON asserts the launched kernel name is _o_proj_kernel. state['o_proj_cfg'][cls] = config tuple or None; kernel_fn branches on tokens (a shape) only. T > 4096 and T <= 200 keep _gemm. Print per class cuBLAS us, each config's us and rel L2, and '[o-proj] PATH CHOSEN'.

## Expected

- `*load=turn_1*` faster: o row 39.7 us (sliding) / 40.1 us (full) -> ~15-25 us (128x128 tiles: 384 tiles, 2.9 waves of 16 k-steps; 128x256: 192 tiles): -15 to -25 us per shape, 2-3.5% of sliding turn_1 and 1.3-2.2% of full turn_1; about +0.7-1.0% headline (turn_1 is the weakest load)
- `*load=turn_4*` flat: At T = 3012 cuBLAS has 12+ waves and runs near its peak share; the gate should keep cuBLAS unless a config is >= 3% faster
- `*load=first_turn*` flat: T = 16384 is untouched (master _gemm)

## Risk

Likely just under the 1.2% judge margin alone (turn_1 load -2 to -3%); a later stage can fuse o_norm_gate into the A loads (head_dim 64 = one BK chunk, so the per-head RMSNorm and sigmoid gate fit the prologue) to remove that kernel and the o round trip. Risks: Triton at small M may not beat nvjet (LESSONS: a 128x128x64 split-K Triton kernel lost at m=128 in bf16_proj; this is no-split-K at M = 753 with a short K, where cuBLAS is visibly off its peak); Triton JIT for three configs costs ~10-15 s of prepare (prepare ~216 s today). Read the scored turn_1 profile: the nvjet 112x128 row should be gone and an _o_proj_kernel row in its place; a PATH CHOSEN line alone proves only the selector (LESSONS 03:41Z).
