---
problem: muse_bf16_proj
slug: combo-cublaslt-single-node-m128-m256-plus-naive
created: 2026-10-25T161307Z
base: master
area: combination (naive stack + cuBLASLt single-node probe on pooler/unpooler/o m=128/256)
change: >
  Built on current master (commit 67b46a57, blob f919748b). Everything in
  combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt (sources A and B, same routing and gates), PLUS
  a conjectural widening of source B to the other m=128/256 cuBLAS cells, each behind its own
  gate: (B2) pooler m=128 and unpooler (n 8192) m=128: torch.mm(x, state[layer+'_t']) (master's NN
  path) under the cuBLASLt preference; (B3) o m=128: master's _bf16_gemm_dispatch_impl under Lt;
  (B4) pooler m=256 NN (master's nn_p256 path), unpooler m=256 and o m=256 under Lt. qkv_gate
  m=256 is EXCLUDED (Lt lost it 0.920x in the source). Sources: SOURCE A:
  pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, branch
  hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, impl commit b3e80b45
  (job 571ea532ad7a410d). Port _BULK_CPP/_BULK_CUDA (bulk_kernel<R,K,NT,MODE>, bulk_launch,
  bulk_run), _bulk_build (keep extra_cuda_cflags -O3 -lineinfo -std=c++20; build at module import
  like master's _build_ouns/_build_router_cluster, own extension name), _bulk_repack,
  _bulk_forward. Instantiate only MODE 0 (drop the MODE 1/2 pure-read probes and the 3.8 TB/s
  kill-gate, which are diagnostics). Routing: layer == 'pooler' and cfg m == 1 only ->
  bulk_kernel<32, 8192, 8, 0> on state['bulk_wt_pooler'] (lossless permutation repack, round trip
  verified bit-exact in prepare). Check it before master's swapab16 branch in kernel_fn. Evidence:
  pooler decode_1 paired ratio 1.226x vs SGLang (bulk kernel 17.0 us, row 17.28 us) against master
  1.135x (_bf16_swapab_kernel 18.7 us, row 18.93 us; check 85c91578): -8% on the largest decode_1
  cell. Its m=8 row lost (1.084 vs 1.118) and m>=16 lose badly, so m == 1 only. SOURCE B:
  bf16-cublaslt-backend-toggle, branch hyp/muse_bf16_proj/bf16-cublaslt-backend-toggle, impl
  commit 800f6067 (job 5ed9ace10d9a4f68). Port only its save/set/restore of
  torch.backends.cuda.preferred_blas_library('cublaslt') around the one _bf16_gemm_dispatch_impl
  call, plus its untimed Lt warm-up in prepare so the Lt workspace exists before capture.
  Evidence: qkv_gate verify_16 (m=128) ran as ONE nvjet 48x64 kernel (16.2 us) with no cublasLt
  splitKreduce node, row 16.24 us, paired 1.068x vs SGLang, against master's nvjet 96x64 splitK
  12.7 + splitKreduce 3.0 us, row 17.38 us, 0.998x. The same toggle LOST qkv_gate m=256 (0.920x),
  so it never applies there. Note for review: this toggle is a host-side library preference set
  and restored within the call (shape-gated, never value-gated); the judge's static check passed
  it once (5ed9ace10d9a4f68). The source run measured Lt only on the TN layout (pooler v16 0.999x
  TN 64x64, unpooler v16 0.998x TN, o v16 1.012x same kernel); the NN-layout cells are untested,
  which is the point. Routing: pooler m==1 -> A; qkv m==128 -> B; {pooler, unpooler} m==128 -> B2
  if its gate wins else master NN; o m==128 -> B3 else master; {pooler, unpooler, o} m==256 -> B4
  per cell else master; everything else master byte-identical. GATES (one per source per cell, all
  AFTER master's own _prepare_state work, master's code and constants untouched): copy master's
  _ouns_gate model selector (graph replay of the captured incumbent and candidate for the same
  synthetic x, 256 MB L2 flush before each replay, 41 interleaved reps, empty-graph baseline
  subtracted, both medians printed). Route the candidate only if candidate/incumbent <= 0.98 AND
  its correctness check passes (rel L2 vs an fp64 reference <= 3e-3 on random x, probe row (single
  power-of-two entry) rel <= 1e-4, finite); otherwise keep master for that cell. Print exactly one
  'PATH CHOSEN <source> <layer> m=<m>: <candidate|master> (cand X us vs master Y us)' line per
  source per cell, and profile one call of each routed cell with torch.profiler and assert the
  launched kernel name (LESSONS: PATH CHOSEN proves the selector, not the kernel). Each flag lives
  in state (two judge copies: per-state repacked weights, error words and flags; no module-global
  device pointers). The second state of master's A/B self-check inherits the first state's flags.
  kernel_fn branches only on cfg layer/m/n and the state flags; NO per-call state writes (the
  source's state['bulk_route_hit'] = (layer, m) line in _bulk_forward MUST be dropped: it is
  per-call mutable state), no syncs, no allocation sized by input values. Each Lt cell's gate
  captures its graph under the preference, so what is timed is what is replayed. PREPARE BUDGET: A
  as combo 1 (~37-40 s import build, <= 2 s/state, deadline 200 s); Lt warm-up + 7 cell gates at
  ~0.3-0.5 s each <= 4 s per state (deadline 250 s). Total prepare ~15 s plus ~115 s of
  import-time builds.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: source A, as combo 1
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: source B, as combo 1
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "conjecture: Lt picks a single-node NN algo, dropping the 3.1 us splitKreduce node; row 23.2 -> ~21 if so, else gate keeps master"
  - shapes: layer=unpooler load=verify_16 m=128
    direction: unchanged
    why: master NN is already one node (19.6 us); Lt may pick a fuller tile; gate decides
  - shapes: layer=* load=verify_32 m=256
    direction: unchanged
    why: "probing cells: pooler/unpooler/o m=256 under Lt, gate decides; qkv m=256 excluded"
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# combo-cublaslt-single-node-m128-m256-plus-naive

## Change

Built on current master (commit 67b46a57, blob f919748b). Everything in combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt (sources A and B, same routing and gates), PLUS a conjectural widening of source B to the other m=128/256 cuBLAS cells, each behind its own gate: (B2) pooler m=128 and unpooler (n 8192) m=128: torch.mm(x, state[layer+'_t']) (master's NN path) under the cuBLASLt preference; (B3) o m=128: master's _bf16_gemm_dispatch_impl under Lt; (B4) pooler m=256 NN (master's nn_p256 path), unpooler m=256 and o m=256 under Lt. qkv_gate m=256 is EXCLUDED (Lt lost it 0.920x in the source). Sources: SOURCE A: pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, branch hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, impl commit b3e80b45 (job 571ea532ad7a410d). Port _BULK_CPP/_BULK_CUDA (bulk_kernel<R,K,NT,MODE>, bulk_launch, bulk_run), _bulk_build (keep extra_cuda_cflags -O3 -lineinfo -std=c++20; build at module import like master's _build_ouns/_build_router_cluster, own extension name), _bulk_repack, _bulk_forward. Instantiate only MODE 0 (drop the MODE 1/2 pure-read probes and the 3.8 TB/s kill-gate, which are diagnostics). Routing: layer == 'pooler' and cfg m == 1 only -> bulk_kernel<32, 8192, 8, 0> on state['bulk_wt_pooler'] (lossless permutation repack, round trip verified bit-exact in prepare). Check it before master's swapab16 branch in kernel_fn. Evidence: pooler decode_1 paired ratio 1.226x vs SGLang (bulk kernel 17.0 us, row 17.28 us) against master 1.135x (_bf16_swapab_kernel 18.7 us, row 18.93 us; check 85c91578): -8% on the largest decode_1 cell. Its m=8 row lost (1.084 vs 1.118) and m>=16 lose badly, so m == 1 only. SOURCE B: bf16-cublaslt-backend-toggle, branch hyp/muse_bf16_proj/bf16-cublaslt-backend-toggle, impl commit 800f6067 (job 5ed9ace10d9a4f68). Port only its save/set/restore of torch.backends.cuda.preferred_blas_library('cublaslt') around the one _bf16_gemm_dispatch_impl call, plus its untimed Lt warm-up in prepare so the Lt workspace exists before capture. Evidence: qkv_gate verify_16 (m=128) ran as ONE nvjet 48x64 kernel (16.2 us) with no cublasLt splitKreduce node, row 16.24 us, paired 1.068x vs SGLang, against master's nvjet 96x64 splitK 12.7 + splitKreduce 3.0 us, row 17.38 us, 0.998x. The same toggle LOST qkv_gate m=256 (0.920x), so it never applies there. Note for review: this toggle is a host-side library preference set and restored within the call (shape-gated, never value-gated); the judge's static check passed it once (5ed9ace10d9a4f68). The source run measured Lt only on the TN layout (pooler v16 0.999x TN 64x64, unpooler v16 0.998x TN, o v16 1.012x same kernel); the NN-layout cells are untested, which is the point. Routing: pooler m==1 -> A; qkv m==128 -> B; {pooler, unpooler} m==128 -> B2 if its gate wins else master NN; o m==128 -> B3 else master; {pooler, unpooler, o} m==256 -> B4 per cell else master; everything else master byte-identical. GATES (one per source per cell, all AFTER master's own _prepare_state work, master's code and constants untouched): copy master's _ouns_gate model selector (graph replay of the captured incumbent and candidate for the same synthetic x, 256 MB L2 flush before each replay, 41 interleaved reps, empty-graph baseline subtracted, both medians printed). Route the candidate only if candidate/incumbent <= 0.98 AND its correctness check passes (rel L2 vs an fp64 reference <= 3e-3 on random x, probe row (single power-of-two entry) rel <= 1e-4, finite); otherwise keep master for that cell. Print exactly one 'PATH CHOSEN <source> <layer> m=<m>: <candidate|master> (cand X us vs master Y us)' line per source per cell, and profile one call of each routed cell with torch.profiler and assert the launched kernel name (LESSONS: PATH CHOSEN proves the selector, not the kernel). Each flag lives in state (two judge copies: per-state repacked weights, error words and flags; no module-global device pointers). The second state of master's A/B self-check inherits the first state's flags. kernel_fn branches only on cfg layer/m/n and the state flags; NO per-call state writes (the source's state['bulk_route_hit'] = (layer, m) line in _bulk_forward MUST be dropped: it is per-call mutable state), no syncs, no allocation sized by input values. Each Lt cell's gate captures its graph under the preference, so what is timed is what is replayed. PREPARE BUDGET: A as combo 1 (~37-40 s import build, <= 2 s/state, deadline 200 s); Lt warm-up + 7 cell gates at ~0.3-0.5 s each <= 4 s per state (deadline 250 s). Total prepare ~15 s plus ~115 s of import-time builds.

## Expected

- `layer=pooler load=decode_1 m=1` faster: source A, as combo 1
- `layer=qkv_gate load=verify_16 m=128` faster: source B, as combo 1
- `layer=pooler load=verify_16 m=128` faster: conjecture: Lt picks a single-node NN algo, dropping the 3.1 us splitKreduce node; row 23.2 -> ~21 if so, else gate keeps master
- `layer=unpooler load=verify_16 m=128` unchanged: master NN is already one node (19.6 us); Lt may pick a fuller tile; gate decides
- `layer=* load=verify_32 m=256` unchanged: probing cells: pooler/unpooler/o m=256 under Lt, gate decides; qkv m=256 excluded

## Risk

Expected: combo 1's ~+0.8% plus 0 to +0.6% if the NN cells take Lt single-node algos. Risk medium: pure dispatch for the B cells, so build risk is A's only. ## Conjecture: the verify_16 gap to the floor is graph-node count, not kernel speed. cuBLAS's heuristic picks split-K plus a separate cublasLt splitKreduce node at m=128 for both qkv (TN) and pooler (NN); the cuBLASLt heuristic chose a one-node 48x64 algo for qkv TN. The conjecture is that the preference, not the layout, drives that choice, so the NN pooler m=128 cell also loses its reduce node under Lt (as LESSONS 04:36Z found for memset nodes at m<=8, but here on a 23 us row). It also tests whether a selector re-decides once its neighbour changed: master's nn_ok A/B self-check now compares NN-under-Lt against TN-under-cuBLAS. Confirmed if the profile's pooler verify_16 row shows one nvjet kernel and no splitKreduce, and the B2 gate line prints cand < 0.98x master. Refuted if Lt keeps splitK+splitKreduce on NN (same kernel names as master), in which case the reduce node is a property of the K=8192 shape and the queued pooler-k-halves-batched-gemm-m65-m256 is the right lever. The m=256 B4 cells are probes: read their gate medians even where master is chosen. Overlaps the cells of the queued qkv-k-split-batched-gemm-m65-m256 and pooler-k-halves-batched-gemm-m65-m256; if either lands first, its path becomes the incumbent the gates compare against.
