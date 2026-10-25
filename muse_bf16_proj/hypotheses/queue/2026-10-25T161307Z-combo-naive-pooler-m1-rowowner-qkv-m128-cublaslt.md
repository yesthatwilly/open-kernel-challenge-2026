---
problem: muse_bf16_proj
slug: combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt
created: 2026-10-25T161307Z
base: master
area: combination (pooler m=1 bulk row-owner + qkv_gate m=128 cuBLASLt single-node)
change: >
  Built on current master (commit 67b46a57, blob f919748b: restack-three-cell-wins +
  pooler-nn-layout-widen-m256). NAIVE TOP-ROW STACK: the only two cells where an unharvested
  branch beat today's master on a paired, drift-corrected basis with a different launched kernel.
  Re-base finding: every other EXTRA-listed row either already runs master's code
  (restack-router/unpooler-slice m8, pooler-unpooler-m128-nn-layout-gated) or its 'gain' was drift
  on an untouched kernel (streamk decode qkv/o: master's _bf16_swapab_kernel;
  router-cluster-dsmem-s2 router d32: master's _router_swapab_kernel; qkv-swapab-w128 o v32:
  untouched nvjet). SOURCE A: pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, branch
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
  it once (5ed9ace10d9a4f68). Routing: decode_1 pooler (m == 1) -> A; verify_16 qkv_gate (cfg m ==
  128 exactly, n 2304) -> B; every other (layer, m) -> master byte-identical, including the
  m=130/512/1000 tests. GATES (one per source per cell, all AFTER master's own _prepare_state
  work, master's code and constants untouched): copy master's _ouns_gate model selector (graph
  replay of the captured incumbent and candidate for the same synthetic x, 256 MB L2 flush before
  each replay, 41 interleaved reps, empty-graph baseline subtracted, both medians printed). Route
  the candidate only if candidate/incumbent <= 0.98 AND its correctness check passes (rel L2 vs an
  fp64 reference <= 3e-3 on random x, probe row (single power-of-two entry) rel <= 1e-4, finite);
  otherwise keep master for that cell. Print exactly one 'PATH CHOSEN <source> <layer> m=<m>:
  <candidate|master> (cand X us vs master Y us)' line per source per cell, and profile one call of
  each routed cell with torch.profiler and assert the launched kernel name (LESSONS: PATH CHOSEN
  proves the selector, not the kernel). Each flag lives in state (two judge copies: per-state
  repacked weights, error words and flags; no module-global device pointers). The second state of
  master's A/B self-check inherits the first state's flags. kernel_fn branches only on cfg
  layer/m/n and the state flags; NO per-call state writes (the source's state['bulk_route_hit'] =
  (layer, m) line in _bulk_forward MUST be dropped: it is per-call mutable state), no syncs, no
  allocation sized by input values. PREPARE BUDGET (300 s limit; master's state build is ~4 s plus
  two ~37 s extension builds at import): bulk extension build ~37-40 s at import (one MODE-0
  instantiation), bulk repack + self-check + gate <= 2 s per state (deadline: skip and print PATH
  CHOSEN master if prepare elapsed > 200 s), Lt warm-up + gate <= 1 s (deadline 250 s). Expected
  total prepare ~10 s plus ~115 s of import-time builds, well inside 300 s and the 10-minute run.
expected:
  - shapes: layer=pooler load=decode_1 m=1
    direction: faster
    why: bulk row-owner 17.0 us kernel vs swapab 18.7; row 18.9 -> ~17.4 us (~ -8%)
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: one Lt nvjet node replaces nvjet splitK + splitKreduce; row 17.4 -> ~16.3 us (~ -6%)
  - shapes: layer=* load=* m=*
    direction: unchanged
    why: all other cells byte-identical master
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: required
status: queued
---

# combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt

## Change

Built on current master (commit 67b46a57, blob f919748b: restack-three-cell-wins + pooler-nn-layout-widen-m256). NAIVE TOP-ROW STACK: the only two cells where an unharvested branch beat today's master on a paired, drift-corrected basis with a different launched kernel. Re-base finding: every other EXTRA-listed row either already runs master's code (restack-router/unpooler-slice m8, pooler-unpooler-m128-nn-layout-gated) or its 'gain' was drift on an untouched kernel (streamk decode qkv/o: master's _bf16_swapab_kernel; router-cluster-dsmem-s2 router d32: master's _router_swapab_kernel; qkv-swapab-w128 o v32: untouched nvjet). SOURCE A: pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, branch hyp/muse_bf16_proj/pooler-unpooler-bulk-tma-row-owner-s2-cxx20-rebuild, impl commit b3e80b45 (job 571ea532ad7a410d). Port _BULK_CPP/_BULK_CUDA (bulk_kernel<R,K,NT,MODE>, bulk_launch, bulk_run), _bulk_build (keep extra_cuda_cflags -O3 -lineinfo -std=c++20; build at module import like master's _build_ouns/_build_router_cluster, own extension name), _bulk_repack, _bulk_forward. Instantiate only MODE 0 (drop the MODE 1/2 pure-read probes and the 3.8 TB/s kill-gate, which are diagnostics). Routing: layer == 'pooler' and cfg m == 1 only -> bulk_kernel<32, 8192, 8, 0> on state['bulk_wt_pooler'] (lossless permutation repack, round trip verified bit-exact in prepare). Check it before master's swapab16 branch in kernel_fn. Evidence: pooler decode_1 paired ratio 1.226x vs SGLang (bulk kernel 17.0 us, row 17.28 us) against master 1.135x (_bf16_swapab_kernel 18.7 us, row 18.93 us; check 85c91578): -8% on the largest decode_1 cell. Its m=8 row lost (1.084 vs 1.118) and m>=16 lose badly, so m == 1 only. SOURCE B: bf16-cublaslt-backend-toggle, branch hyp/muse_bf16_proj/bf16-cublaslt-backend-toggle, impl commit 800f6067 (job 5ed9ace10d9a4f68). Port only its save/set/restore of torch.backends.cuda.preferred_blas_library('cublaslt') around the one _bf16_gemm_dispatch_impl call, plus its untimed Lt warm-up in prepare so the Lt workspace exists before capture. Evidence: qkv_gate verify_16 (m=128) ran as ONE nvjet 48x64 kernel (16.2 us) with no cublasLt splitKreduce node, row 16.24 us, paired 1.068x vs SGLang, against master's nvjet 96x64 splitK 12.7 + splitKreduce 3.0 us, row 17.38 us, 0.998x. The same toggle LOST qkv_gate m=256 (0.920x), so it never applies there. Note for review: this toggle is a host-side library preference set and restored within the call (shape-gated, never value-gated); the judge's static check passed it once (5ed9ace10d9a4f68). Routing: decode_1 pooler (m == 1) -> A; verify_16 qkv_gate (cfg m == 128 exactly, n 2304) -> B; every other (layer, m) -> master byte-identical, including the m=130/512/1000 tests. GATES (one per source per cell, all AFTER master's own _prepare_state work, master's code and constants untouched): copy master's _ouns_gate model selector (graph replay of the captured incumbent and candidate for the same synthetic x, 256 MB L2 flush before each replay, 41 interleaved reps, empty-graph baseline subtracted, both medians printed). Route the candidate only if candidate/incumbent <= 0.98 AND its correctness check passes (rel L2 vs an fp64 reference <= 3e-3 on random x, probe row (single power-of-two entry) rel <= 1e-4, finite); otherwise keep master for that cell. Print exactly one 'PATH CHOSEN <source> <layer> m=<m>: <candidate|master> (cand X us vs master Y us)' line per source per cell, and profile one call of each routed cell with torch.profiler and assert the launched kernel name (LESSONS: PATH CHOSEN proves the selector, not the kernel). Each flag lives in state (two judge copies: per-state repacked weights, error words and flags; no module-global device pointers). The second state of master's A/B self-check inherits the first state's flags. kernel_fn branches only on cfg layer/m/n and the state flags; NO per-call state writes (the source's state['bulk_route_hit'] = (layer, m) line in _bulk_forward MUST be dropped: it is per-call mutable state), no syncs, no allocation sized by input values. PREPARE BUDGET (300 s limit; master's state build is ~4 s plus two ~37 s extension builds at import): bulk extension build ~37-40 s at import (one MODE-0 instantiation), bulk repack + self-check + gate <= 2 s per state (deadline: skip and print PATH CHOSEN master if prepare elapsed > 200 s), Lt warm-up + gate <= 1 s (deadline 250 s). Expected total prepare ~10 s plus ~115 s of import-time builds, well inside 300 s and the 10-minute run.

## Expected

- `layer=pooler load=decode_1 m=1` faster: bulk row-owner 17.0 us kernel vs swapab 18.7; row 18.9 -> ~17.4 us (~ -8%)
- `layer=qkv_gate load=verify_16 m=128` faster: one Lt nvjet node replaces nvjet splitK + splitKreduce; row 17.4 -> ~16.3 us (~ -6%)
- `layer=* load=* m=*` unchanged: all other cells byte-identical master

## Risk

Expected headline about +0.8% (decode_1 load -1.5 us of 47.5, verify_16 load -1.1 us of 74.4), so 1.24 -> ~1.25: a coin flip against the 1.2491 board, but it is the honest floor for every other combination, which all contain these two cells. Risks: bulk kernel uses cp.async.bulk + mbarrier (already ran clean in the judge, error word 0); the Lt toggle is unusual under the rules (see change). If either gate prints master, the run measures master on that cell. ## Conjecture: the two source cells sit in different loads (decode_1, verify_16) and different graphs, so the conjecture is that they are ADDITIVE with no interaction: their per-cell paired ratios should reproduce their source runs (pooler d1 ~1.22x, qkv v16 ~1.06x) within +-2%, and every master-identical cell should stay within drift. Refuted if the bulk extension's import-time build or its 67 MB per-state repack shifts any untouched row by more than drift (compare the profile rows of the untouched pooler m=8/16 swapab cells), or if the Lt preference leaks into the next captured cell (check that qkv verify_32 still launches nvjet 96x64 2x4 TNN, master's kernel). Read: PATH CHOSEN lines, the gate medians, top_kernels on pooler decode_1 (bulk_kernel<32, 8192, 8, 0>) and qkv verify_16 (single nvjet, no splitKreduce).
