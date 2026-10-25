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
review: proceed
status: done
review_at: 2026-10-24T165736Z
review_companions: null
hold_reason: "reviewer: claim only after the bf16 precedent (bf16-cublaslt-algo-pick-in-prepare-m128-m256, 26a8be34a5eb4426) harvests; ~0.5-1.1% headline; time pooler keys with the router running or require a >=5% margin"
outcome: expired
verdict_detail: "retired: on the prefill DIAGNOSIS Do-not-try list"
expired_at: 2026-10-25T211931Z
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

## Adversarial review

Reviewed 2026-10-24T165736Z by an adversarial subagent (model opus).

**Verdict: proceed.** The mechanism is plausible and the result can be read without companions: every changed row pairs with a non-incumbent pick in the prepare() table, and keys that keep the incumbent stay byte-identical. The expected headline gain (~+0.5-1%) is below run drift, so judge it on the per-shape nvjet rows. Claim it only after the bf16 precedent 26a8be34a5eb4426 harvests: it was dispatched at 16:48Z and has no result yet, so the 'check its harvest' gate in the Risk section cannot be met at review time.

### Justification
Plausible on H200 for these shapes, but the profile evidence is weaker than the body says. (1) The turn_1 premise is overstated. In master's profile (fb1e52d2119d4175, archive/experts-hybrid-marlin-turn-triton-first-turn/fb1e52d2119d4175.profile.txt), nvjet_sm90_tst_128x192 at 154.0 us is 3 calls: qkv_gate, pooler and unpooler, all with K large and all TNN. Their combined floor at T=753 is 28.7 + 51.1 + 51.1 = ~131 us, so the three together already run at ~85% of peak. Pooler and unpooler alone need >=51 us each, which leaves at most ~50 us for qkv_gate against its 28.7 us floor. A 72-CTA grid costs qkv_gate perhaps 10-20 us, and the GPU is not 'half idle for the whole kernel'. The o projection is the real outlier: 39 us for one call against a 12.8 us compute floor. With K=1024 there are only 16 k-steps of 64, so prologue and epilogue dominate, and 438 CTAs make 3.3 waves. Split-K and stream-K candidates could save 10-20 us there. Turn_1 headroom is therefore ~20-35 us across 4 GEMMs of a 1080-1420 us shape, so 1-2% per turn_1 shape is a fair upper estimate. (2) turn_4 checks out: the four nvjet rows sum to ~725 us against a 574 us floor (115 qkv + 51 o + 204 pooler + 204 unpooler at T=3012), ~79% of peak. A 5-10% better tile on the two 256x152 TNT calls or the 128x256 call is 15-50 us of 3335-4938 us. (3) first_turn: the dense GEMMs are 3865 us against a ~3127 us floor (~81%). 'Flat' is the right expectation, and the 3% keep rule should hold the incumbent there. (4) Headline arithmetic: each load counts once in the geomean. +1-2% on turn_1 and +0.5-1.5% on turn_4 gives about +0.5-1.1% on the headline, inside the ~1.5-2.5% per-run drift (LESSONS). One scored run cannot resolve the headline. It is judged by the prepare() table plus the per-shape nvjet rows, as the spec already says. Weakest link: whether cublasLtMatmulAlgoGetHeuristic returns anything for this TN descriptor at M=753/3012 that is materially different from heuristic 0. bf16-cublaslt-backend-toggle showed that the backend preference alone changes nothing. The only evidence that better configs exist is the NN-layout probe at m=128, a different M and layout. The precedent that would answer this (26a8be34a5eb4426, m=128/256) has not harvested. Second weakest link: prepare() times each GEMM alone with every operand cold. In the real graph the pooler shares the GPU with the side-stream router (tss_64x96 split-K, 30.6-79.9 us for 2 calls at turn_1, 129.8-137.0 us at turn_4). The o, pooler and unpooler inputs are also L2-warm outputs of the previous kernel. A pooler pick that wins by 3-5% alone, for example by adding CTAs or split-K, can lose that margin when it contends with the router for SMs. One factual error in the spec: 'every T outside the three exact values (all 5 tests)' is wrong. Test 2 (batch 1, ctx 3000, full, q_len 753) has T=753 and will take the Lt path when a turn_1 key is chosen. That is harmless, and it means the Lt path gets correctness-tested, but the implementer should not assume the tests avoid it.

### Confounders on the current kernel
- sglang-pipeline-parity: it puts the router (two torch.mm calls, fp32 out) on a side stream that overlaps the pooler GEMM. A pooler candidate picked by timing it alone can perform differently under that overlap, so a pooler-row change at turn_1 or turn_4 is partly an interaction with the side-stream layout, not a pure algorithm effect. Mitigation in the build, not a companion: time the pooler keys with the router's two mm calls replayed on the side stream inside the same captured graph, or require a larger margin (>=5%) for the pooler keys. qkv_gate, o and unpooler do not overlap anything, so they are unaffected.
- experts-hybrid-marlin-turn-triton-first-turn / experts-triton-w4a16-first-turn-blockt256: no confound on the dense GEMMs. The expert kernels run strictly between pooler and unpooler on the main stream, share no buffers with the dense weights, and the experts' gains/losses live in rows (Marlin at T<=4096, _moe_nvfp4_gemm_kernel above) that this change does not touch. The one indirect effect: the unpooler's input is the expert output, L2-warm in the graph but cold in prepare() timing. That is a small bias, the same for every candidate. Prepare memory: the second expert bank (+0.9 GiB) plus up to 12 x 32 MiB of workspaces is still far from the H200 limit.

### Interference and A/B needs
No companion is warranted. A gain or a loss is attributable on its own. Each changed (layer, T) key prints its chosen algorithm (tile, stages, splitK, waves) and the prepare() timing, and each changed shape row shows a different nvjet/cutlass kernel name in top_kernels. Keys that keep the incumbent run master's byte-identical _bf16_gemm_dispatch_impl. An ab-revert of a landed hypothesis does not make sense: sglang-pipeline-parity is the whole pipeline, and the expert stages are disjoint from the dense GEMMs. The pooler/router overlap is better handled in the build (time the pooler under overlap, or use a stricter threshold) than with an extra run. The one control I considered was the Lt wrapper replaying heuristic 0, to separate wrapper overhead from algorithm choice. It is moot: the wrapper only replaces a key after it beat the incumbent by >=3% in a graph-replay timing that includes the wrapper. Prefill checks are also spent, so every run is a scored submit, which argues further against an extra run.

### Open questions
- Did the bf16 precedent 26a8be34a5eb4426 compile, capture and pick any non-incumbent algorithm? Its log tail shows this. If load_inline or graph capture failed there, fix the port before claiming. If the TN heuristic returned only near-duplicates of result 0 at m=128/256, expect few picks at T=753, though maybe not at 3012/16384. Claim this hypothesis only after that harvest.
- How many candidates does cublasLtMatmulAlgoGetHeuristic return for TN at M=753, 3012 and 16384 with a 32 MiB workspace, and does any of them use stream-K or split-K for the o projection (K=1024)? The o key at turn_1 (39 us vs 12.8 floor) is where a win is most likely.
- Should the pooler keys be timed while the router's two mm calls run on the side stream, given that the real graph overlaps them? The answer decides whether a pooler pick is real.
- What is the prepare() wall time on top of master's Triton bank compile, Marlin repack and expert checks? Cap max_algos at 16 for the T=16384 keys if it approaches the 300 s limit, and print it.
- The first_turn keys are expected flat and cost the most prepare() time. Consider skipping them entirely (T in {753, 3012} only) to cut compile/timing risk, if the precedent shows no picks at large M.

