---
problem: muse_bf16_proj
slug: bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole
created: 2026-10-25T211857Z
base: master
area: bf16 GEMM dispatch (m=128/256, per-cell cuBLASLt plan selector)
change: >
  DIAGNOSIS Action 2, narrowed by today's evidence. Built on master 36c26199. Generalise master's
  cell-B machinery (_LT_CPP candidates/run with MuseLtDesc(n, k, m), _lt_prepare, _cell_gate,
  _cell_kernels) from qkv_gate m=128 to five more cells, each with its own flag and state entries:
  o m=128 (row 7.68, nvjet 128x64 TNT 5.7 us kernel, 2 us of row-minus-kernel, floor 4.0), o m=256
  (9.44, nvjet 128x128 8.3, floor 4.5), qkv_gate m=256 (19.46, nvjet 96x64 2x4 TNN 20.0, floor
  9.8), unpooler m=128 (18.88, nvjet 128x64 NNT 19.3 on the NN repack, floor 14.6), unpooler m=256
  (24.18, nvjet 128x128 TNT 24.8, floor 17.4). Pooler is left to
  pooler-k-halves-fp32-partials-restack-m128-m256 and qkv m=128 keeps cell B, so attribution stays
  clean. Candidates per cell: master; the top 8 cublasLtMatmulAlgoGetHeuristic plans with split-K
  1 (as cell B); plus up to 4 split-K > 1 plans with a per-cell fp32 workspace allocated once per
  state in prepare (query with a 32 MB workspace cap; the reduce stays inside cuBLASLt); for o and
  unpooler both the TN weight and an NN view (unpooler: master's existing unpooler_t; o: a new
  lossless o_t = W_o.t().contiguous() [1024, 8192], 16.8 MB, built in prepare), added to
  MuseLtDesc as a transa flag; for unpooler m=256 only, also the K-halves bmm with fp32 partials
  (S=2, as in the pooler restack). EXCLUDED as measured dead: qkv K-split bmm (S=4 T judged 0.89
  at m=256, S=2/4 0.87-0.99 at m=128, 9d93c3d0) and unpooler K-halves at m=128 (0.888/0.865 in
  prepare, e82feba3). Pre-screen each cell's candidates with the existing LT_PICK_REPS cold-L2
  timing, send the fastest two to the gate. Gate (hardened from today's evidence): copy master's
  _cell_gate model selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush
  before each replay, 41 interleaved reps, empty-graph baseline subtracted) but route only if BOTH
  of two independent interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is
  <= 1.5x master's IQR. Today two picks that passed a single 1.02 round flipped in the judge:
  qkv-k-split S=4 T at m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us;
  router-hilo-tall at m=256 judged 9.72->9.99. The pick that carried (pooler K-halves N) predicted
  1.069/1.037. Then one torch.profiler call of the chosen path asserts the launched kernel names,
  and one PATH CHOSEN line per cell prints both medians, both IQRs and the kernels. No per-call
  preferred_blas_library toggles (Do-not-try); plans live in state; descriptors per call on the
  host as cell B does. kernel_fn gets one cfg/flag-only branch per cell before the
  _bf16_gemm_dispatch_impl fallback.
expected:
  - shapes: layer=o load=verify_* m=*
    direction: faster
    why: SGLang-parity nvjet rows never targeted; a single-node Lt plan or NN layout can cut the 2 us row-minus-kernel at m=128
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: 497 TFLOP/s nvjet 96x64; the Lt split-K-with-workspace plans were never offered under an interleaved gate
  - shapes: layer=unpooler load=verify_* m=*
    direction: faster
    why: per-cell plan; backend toggle showed Lt changes the kernel per cell
  - shapes: layer=pooler load=* m=*
    direction: flat
    why: not touched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
prior: "bf16-cublaslt-algo-pick-in-prepare-m128-m256: that pick was timed without interleaving against the incumbent, with no empty-graph baseline and no margin, and its qkv m=256 choice flipped (+14%) in the judge; this reuses master's cell-B Lt plan machinery per cell with the hardened two-round 1.03 gate, a profiler kernel assert, and excludes the dead K-split candidates"
review: required
status: queued
---

# bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole

## Change

DIAGNOSIS Action 2, narrowed by today's evidence. Built on master 36c26199. Generalise master's cell-B machinery (_LT_CPP candidates/run with MuseLtDesc(n, k, m), _lt_prepare, _cell_gate, _cell_kernels) from qkv_gate m=128 to five more cells, each with its own flag and state entries: o m=128 (row 7.68, nvjet 128x64 TNT 5.7 us kernel, 2 us of row-minus-kernel, floor 4.0), o m=256 (9.44, nvjet 128x128 8.3, floor 4.5), qkv_gate m=256 (19.46, nvjet 96x64 2x4 TNN 20.0, floor 9.8), unpooler m=128 (18.88, nvjet 128x64 NNT 19.3 on the NN repack, floor 14.6), unpooler m=256 (24.18, nvjet 128x128 TNT 24.8, floor 17.4). Pooler is left to pooler-k-halves-fp32-partials-restack-m128-m256 and qkv m=128 keeps cell B, so attribution stays clean. Candidates per cell: master; the top 8 cublasLtMatmulAlgoGetHeuristic plans with split-K 1 (as cell B); plus up to 4 split-K > 1 plans with a per-cell fp32 workspace allocated once per state in prepare (query with a 32 MB workspace cap; the reduce stays inside cuBLASLt); for o and unpooler both the TN weight and an NN view (unpooler: master's existing unpooler_t; o: a new lossless o_t = W_o.t().contiguous() [1024, 8192], 16.8 MB, built in prepare), added to MuseLtDesc as a transa flag; for unpooler m=256 only, also the K-halves bmm with fp32 partials (S=2, as in the pooler restack). EXCLUDED as measured dead: qkv K-split bmm (S=4 T judged 0.89 at m=256, S=2/4 0.87-0.99 at m=128, 9d93c3d0) and unpooler K-halves at m=128 (0.888/0.865 in prepare, e82feba3). Pre-screen each cell's candidates with the existing LT_PICK_REPS cold-L2 timing, send the fastest two to the gate. Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both medians, both IQRs and the kernels. No per-call preferred_blas_library toggles (Do-not-try); plans live in state; descriptors per call on the host as cell B does. kernel_fn gets one cfg/flag-only branch per cell before the _bf16_gemm_dispatch_impl fallback.

## Expected

- `layer=o load=verify_* m=*` faster: SGLang-parity nvjet rows never targeted; a single-node Lt plan or NN layout can cut the 2 us row-minus-kernel at m=128
- `layer=qkv_gate load=verify_32 m=256` faster: 497 TFLOP/s nvjet 96x64; the Lt split-K-with-workspace plans were never offered under an interleaved gate
- `layer=unpooler load=verify_* m=*` faster: per-cell plan; backend toggle showed Lt changes the kernel per cell
- `layer=pooler load=* m=*` flat: not touched

## Risk

Effect: realistic -0.5 to -2 us on each of the five cells (0.230 %/us verify_16, 0.184 %/us verify_32), about +0.4-1.0% headline: likely below the margin alone, must stack (combo). The gate may route none; then the run reads as master and its PATH CHOSEN lines are the result. New cuBLASLt workspace split-K path: first measurement is an okc check. Supersedes the held combo-cublaslt-single-node-m128-m256-plus-naive (retire it).
