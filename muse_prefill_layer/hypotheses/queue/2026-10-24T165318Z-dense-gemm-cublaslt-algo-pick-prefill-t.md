---
problem: muse_prefill_layer
slug: dense-gemm-cublaslt-algo-pick-prefill-t
created: 2026-10-24T165318Z
base: master
area: dense bf16 GEMM dispatch (qkv, o, pooler, unpooler; cuBLASLt algo selection)
change: >
  Built on master (635cfee submission blob, experts-hybrid-marlin-turn-triton-first-turn). For the
  four dense bf16 projections that _gemm() issues (qkv_gate [2304, 8192], o [8192, 1024], pooler
  [4096, 8192], unpooler [8192, 4096]) at exactly tokens T in {753, 3012, 16384} (turn_1, turn_4,
  first_turn), replace _bf16_gemm_dispatch_impl's cuBLAS call (cuBLASLt heuristic result 0) with a
  cuBLASLt matmul whose algorithm is picked ONCE in prepare() by timing the heuristic's candidates
  on synthetic inputs, then replayed unchanged in kernel_fn. Port the machinery verbatim from
  muse_bf16_proj's bf16-cublaslt-algo-pick-in-prepare-m128-m256 (branch
  hyp/muse_bf16_proj/bf16-cublaslt-algo-pick-in-prepare-m128-m256, impl commit a77cdec): the
  load_inline C++ module _LT_CPP (candidates(n, k, m, max_algos=32, ws_bytes) via
  cublasLtMatmulAlgoGetHeuristic for D[n, m] = W^T-op x, TRANSA=T, TRANSB=N, bf16 in/out,
  CUBLAS_COMPUTE_32F, alpha 1 beta 0, 32 MiB workspace; run(x, W, out, algo_row, ws) on the
  current stream, graph-capturable, descriptors cached by (n, k, m)), and its
  _lt_prepare/_lt_pick/_lt_graph_median_us/_lt_route_ok/_lt_forward helpers with all its
  exclusions (split-K reduction scheme must be COMPUTE_TYPE, rel L2 vs torch <= 1e-3, fp64-error
  <= 1.10x the incumbent's, workspace <= 32 MiB, raises/non-finite dropped). Keys are (layer, T)
  for the 4 layers x 3 T values = 12 keys; each candidate and the incumbent are timed as a
  captured CUDA graph, 30 replays each preceded by a 256 MiB flush write (cold L2, as the judge),
  median; keep a candidate only if it beats the incumbent by >= 3%; print the per-key candidate
  table (tile, stages, splitK, waves, us) and the summary (layer, T, cands, incumbent_us, best_us,
  chosen). kernel_fn: _gemm(x, w) gains the layer name; when (layer, x.shape[0]) is in
  state['lt_algo'] and _lt_route_ok (contiguous bf16, full weight shape, 256-B aligned pointers)
  it returns _lt_forward(...) (torch.empty((T, n), bf16) + ext.run), else master's
  _bf16_gemm_dispatch_impl unchanged. The router (torch.mm with out_dtype fp32 on the side
  stream), every attention/expert kernel, and every T outside the three exact values (all 5 tests)
  stay byte-identical to master. Call _lt_prepare last in prepare() so the expert banks and their
  checks are unchanged. Mechanism, from master's profile (fb1e52d2119d4175): the dense GEMMs sit
  at ~75-80% of the 989 TFLOP/s floor overall but two shapes are clearly off: at turn_1 qkv_gate
  (M=753, N=2304) runs nvjet 128x192 tiles = 6 x 12 = 72 CTAs on 132 SMs (half the GPU idle for
  the whole K=8192 loop), and the o projection (K=1024) is nvjet_sm90_tst_112x128 at 39 us vs a
  12.8 us floor (3x); at turn_4 the four GEMMs take ~725 us (nvjet 256x152 x2 322 us, 128x256 253
  us, 192x144 150 us) against a ~574 us floor. No one has asked cuBLAS for its non-default
  candidates on these shapes (the bf16 problem's NN-layout probe showed better nvjet configs exist
  for TN problems that heuristic 0 does not return).
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: qkv_gate 128x192 tile fills only 72 of 132 SMs at M=753 and o is 39 us vs 12.8 floor; a split-K / smaller-tile / stream-K candidate should save ~10-25 us of ~1080-1420 us (1-2%); read the nvjet rows and the prepare() table
  - shapes: "*load=turn_4*"
    direction: faster
    why: ~725 us of dense GEMM vs ~574 floor; a better tile on any of the four saves ~15-50 us of 3335-4938 (0.5-1.5%)
  - shapes: "*load=first_turn*"
    direction: flat
    why: dense GEMMs already ~81% of peak (192x192 x3 2499 us + 320x128 1366 us vs ~3100 us floor); the 3% keep rule likely keeps the incumbent on most keys, so those rows are byte-identical
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: required
status: queued
---

# dense-gemm-cublaslt-algo-pick-prefill-t

## Change

Built on master (635cfee submission blob, experts-hybrid-marlin-turn-triton-first-turn). For the four dense bf16 projections that _gemm() issues (qkv_gate [2304, 8192], o [8192, 1024], pooler [4096, 8192], unpooler [8192, 4096]) at exactly tokens T in {753, 3012, 16384} (turn_1, turn_4, first_turn), replace _bf16_gemm_dispatch_impl's cuBLAS call (cuBLASLt heuristic result 0) with a cuBLASLt matmul whose algorithm is picked ONCE in prepare() by timing the heuristic's candidates on synthetic inputs, then replayed unchanged in kernel_fn. Port the machinery verbatim from muse_bf16_proj's bf16-cublaslt-algo-pick-in-prepare-m128-m256 (branch hyp/muse_bf16_proj/bf16-cublaslt-algo-pick-in-prepare-m128-m256, impl commit a77cdec): the load_inline C++ module _LT_CPP (candidates(n, k, m, max_algos=32, ws_bytes) via cublasLtMatmulAlgoGetHeuristic for D[n, m] = W^T-op x, TRANSA=T, TRANSB=N, bf16 in/out, CUBLAS_COMPUTE_32F, alpha 1 beta 0, 32 MiB workspace; run(x, W, out, algo_row, ws) on the current stream, graph-capturable, descriptors cached by (n, k, m)), and its _lt_prepare/_lt_pick/_lt_graph_median_us/_lt_route_ok/_lt_forward helpers with all its exclusions (split-K reduction scheme must be COMPUTE_TYPE, rel L2 vs torch <= 1e-3, fp64-error <= 1.10x the incumbent's, workspace <= 32 MiB, raises/non-finite dropped). Keys are (layer, T) for the 4 layers x 3 T values = 12 keys; each candidate and the incumbent are timed as a captured CUDA graph, 30 replays each preceded by a 256 MiB flush write (cold L2, as the judge), median; keep a candidate only if it beats the incumbent by >= 3%; print the per-key candidate table (tile, stages, splitK, waves, us) and the summary (layer, T, cands, incumbent_us, best_us, chosen). kernel_fn: _gemm(x, w) gains the layer name; when (layer, x.shape[0]) is in state['lt_algo'] and _lt_route_ok (contiguous bf16, full weight shape, 256-B aligned pointers) it returns _lt_forward(...) (torch.empty((T, n), bf16) + ext.run), else master's _bf16_gemm_dispatch_impl unchanged. The router (torch.mm with out_dtype fp32 on the side stream), every attention/expert kernel, and every T outside the three exact values (all 5 tests) stay byte-identical to master. Call _lt_prepare last in prepare() so the expert banks and their checks are unchanged. Mechanism, from master's profile (fb1e52d2119d4175): the dense GEMMs sit at ~75-80% of the 989 TFLOP/s floor overall but two shapes are clearly off: at turn_1 qkv_gate (M=753, N=2304) runs nvjet 128x192 tiles = 6 x 12 = 72 CTAs on 132 SMs (half the GPU idle for the whole K=8192 loop), and the o projection (K=1024) is nvjet_sm90_tst_112x128 at 39 us vs a 12.8 us floor (3x); at turn_4 the four GEMMs take ~725 us (nvjet 256x152 x2 322 us, 128x256 253 us, 192x144 150 us) against a ~574 us floor. No one has asked cuBLAS for its non-default candidates on these shapes (the bf16 problem's NN-layout probe showed better nvjet configs exist for TN problems that heuristic 0 does not return).

## Expected

- `*load=turn_1*` faster: qkv_gate 128x192 tile fills only 72 of 132 SMs at M=753 and o is 39 us vs 12.8 floor; a split-K / smaller-tile / stream-K candidate should save ~10-25 us of ~1080-1420 us (1-2%); read the nvjet rows and the prepare() table
- `*load=turn_4*` faster: ~725 us of dense GEMM vs ~574 floor; a better tile on any of the four saves ~15-50 us of 3335-4938 (0.5-1.5%)
- `*load=first_turn*` flat: dense GEMMs already ~81% of peak (192x192 x3 2499 us + 320x128 1366 us vs ~3100 us floor); the 3% keep rule likely keeps the incumbent on most keys, so those rows are byte-identical

## Risk

Rules posture is the bf16 precedent's (choice from weight shapes + synthetic data inside untimed prepare(), fixed replay in kernel_fn, same bf16-in/fp32-acc/bf16-out precision, nothing derived from data cached); the reviewer should check that hypothesis's harvest (job 26a8be34a5eb4426) before claiming: if cuBLASLt's TN heuristic returned only near-duplicates there, expect the same here at M=753 but not necessarily at 3012/16384. Budget: load_inline ~40-90 s compile + 12 keys x up to 32 candidates of graph timing (T=16384 GEMMs ~1.4 ms x 31 replays) inside the 300 s prepare and the 15-minute run, on top of master's nibble reorder, Marlin repack and expert checks; cap max_algos at 16 for the T=16384 keys if prepare() gets tight, and print prepare() wall time. Memory: one 32 MiB workspace per chosen key only. The judge's per-shape time includes graph-node cost; time whole graph replays so a split-K candidate's separate reduce node is counted. Attribution: every changed row should pair with a non-incumbent pick in log_tail; keys that keep the incumbent are byte-identical to master.
