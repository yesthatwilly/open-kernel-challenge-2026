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
review: proceed
status: queued
review_at: 2026-10-25T161627Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T161627Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed only after a required change: Source A (bulk row-owner kernel, pooler m==1) can be ported as specified once state['bulk_route_hit'] is dropped. Source B's per-call save/set/restore of torch.backends.cuda.preferred_blas_library is forbidden by standing policy and MUST be replaced by a prepare-time cuBLASLt plan held in state (reuse the bf16-cublaslt-algo-pick-in-prepare-m128-m256 load_inline extension), forced to the no-split-K 48x64 algo; if prepare cannot reproduce that algo, drop cell B and ship A alone.

### Justification
A. Source A is plausible and well evidenced. pooler decode_1 is a 67.1 MB pure weight stream with a 14.0 us floor. In job 571ea532 the bulk_kernel<32,8192,8,0> ran at 17.0 us (~3.95 TB/s), while master's _bf16_swapab_kernel runs at 18.7 us (FINDINGS, check 85c91578). Paired 1.226x vs 1.135x is about 8%, well outside the ~1.5% row noise. Restricting it to m==1 matches that run's own losses at m>=8/16/32. Source B's mechanism is real but fragile. With the toggle, qkv_gate verify_16 launched one nvjet_sm90_tst_48x64_64x15_4x2_h_bz_TNN (16.2 us) and dropped the 12.7 us split-K kernel plus the 3.0 us splitKreduce node, so the 1.068x vs 0.998x row is attributable. WEAKEST LINK: that algo came from torch's Lt path running under torch's small Lt workspace, which excludes the split-K candidates. The policy-compliant prepare-time route has already been tried. bf16-cublaslt-algo-pick-in-prepare-m128-m256 (job 26a8be34, log lines 6-14) asked the Lt heuristic with a 32 MiB workspace and got 8 qkv_gate m=128 candidates, ALL split-K (splitK 2-6, red=2). It picked cand 3 and the row came out flat (16.81 -> 16.49, 1.0095). A naive port of that extension therefore will NOT reproduce the 48x64 no-split-K kernel. The plan must set CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES to 0 or to torch's Lt workspace size, or enumerate AlgoGetIds/ConfigSetAttribute with SPLITK_NUM=1, and assert in prepare that the chosen algo has splitK=1. A second weakness is the expected magnitude. The master qkv verify_16 row has drifted between 16.8 and 17.4 us across runs, so the realistic gain is -3% to -6%, not a firm -6%. The headline expectation (+0.8%, ~1.25) is inside the CI and cannot by itself confirm or refute anything. CONJECTURE (additive, no interaction): justified. The two cells are in different loads and different captured graphs, and every other cell is byte-identical master. The run CAN test it per cell: PATH CHOSEN lines, the gate medians, the asserted kernel names, and the per-row paired ratios against the source runs (pooler d1 ~1.22x, qkv v16 ~1.06x, +-2%). The 'Lt preference leaks into the next captured cell' refutation clause becomes moot once the toggle is replaced. Keep the replacement check: qkv_gate verify_32 must still launch master's nvjet, and no other (layer, m) may show a cublasLt run.

### Confounders on the current kernel
- pooler-nn-layout-widen-m256 / restack-three-cell-wins (NN layout gated at m=65..256): they touch the pooler weights (state['pooler_t']) but not m==1. The bulk repack adds a third 67 MB copy of W_pooler per state, so watch HBM/L2 placement effects only as drift. No functional interaction.
- bf16-m32-swapab-wgmma-triton / bf16-restack-m16-swapab (master's _bf16_swapab_kernel at pooler m<=32): this is the incumbent that A replaces at m==1 only. The bulk branch must be checked BEFORE the swapab16 branch, as specified. If the branch order is wrong, the cell silently stays master and the PATH CHOSEN/kernel-name assert must catch it.
- sglang-gemm-dispatch-parity (_bf16_gemm_dispatch_impl at qkv_gate m=128): this is the incumbent that B replaces, and it acts as the control rather than a confounder. The replacement Lt plan must time against exactly this call in its gate.
- restack-three-cell-wins (ouns/router cluster extensions built at import): the third import-time extension build adds ~37-40 s of compile, plus ~15 s for the Lt load_inline. This affects only the time budget, not the timed rows.

### Interference and A/B needs
No companion is warranted, and none is worth a scored run. The two changed cells are in different loads and graphs, and every other cell runs byte-identical master code. Each changed cell carries its own attribution: a PATH CHOSEN line with candidate and master medians, plus a torch.profiler assertion of the launched kernel (bulk_kernel<32,8192,8,0>; a single nvjet 48x64 with no splitKreduce). That attribution makes both a gain and a loss interpretable per row without an ab-revert or a control. A per-source control would only re-measure what 571ea532 and 5ed9ace1 already measured. If B's prepare-time plan cannot find a splitK=1 algo, B prints PATH CHOSEN master and the run degenerates to A alone, which is still interpretable.

### Open questions
- Which workspace size does torch's Lt path pass on this image (CUBLASLT_WORKSPACE_SIZE / at::cuda::getCUDABlasLtWorkspaceSize)? Matching it should reproduce the 48x64 no-split-K heuristic result 0 exactly.
- Does the 48x64 no-split-K algo stay bit-identical in error to the incumbent (rel vs fp64 ~1.66e-3)? Keep the algo-pick extension's 1.10x-of-incumbent fp64 guard.
- Is the bulk extension built once per process (shared by both judge states) with per-state repacked weights, as the change says? Confirm that no module-global device pointer is captured into the graph.
- Simpler alternative if the extension is unwanted: torch.addmm(zero_bias_bf16_in_state, x, W.t()) takes torch's cublasLt gemm_and_bias path without touching the global preference. Its algo choice is unverified, so it would need the same gate and kernel-name assert.

