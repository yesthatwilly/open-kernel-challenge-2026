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
status: running
review_at: 2026-10-25T161627Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T161628Z
base_sha: 65440e974f1eccc4c5d39e2e004e0f64d2c9d806
base_blob: f919748b5af6e815f319e5d2d828ed68e5b28a71
branch: hyp/muse_bf16_proj/combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt
worktree: .worktrees/muse_bf16_proj/combo-naive-pooler-m1-rowowner-qkv-m128-cublaslt
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T163320Z
impl_commit: 81048abffa7ec82c3af58cf5fa7ccd8ba5c91718
submission_sha256: c3a86c47cf009e0cc389517303d4f4fa02287a45b93b28a3cea1dc497d162965
norm_sha256: 67ee24281e5a53be2bbc0663f3f01019ab3bcd0ca7b3279280844fc772bbde8e
submission_blob: 9008c43b0750f70feb5d559b0947cb067f00aac8
lint_warnings: 39
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

## Implementation notes

Built 2026-10-25 on base 65440e97 (blob f919748b). Master f77fd49a differs only by a claim commit: `git diff 65440e97 master -- muse_bf16_proj/submission.py` is empty, so no merge was needed. Only `muse_bf16_proj/submission.py` in the worktree was edited, and the diff is purely additive outside the new section (9). Diff stat: 1 file, +1054/-10.

**What changed**
- **New section (9)** sits before `_prepare_state`.
  - **Cell A:** `_BULK_CPP`/`_BULK_CUDA`, `_bulk_build` (built at import, own extension name `okc_bf16_bulk_rowown_p1_v1`), `_bulk_repack`, `_bulk_route_ok`, `_bulk_forward`, `_bulk_check` and `_bulk_cell_prepare`.
  - **Cell B:** `_LT_CPP` (with `lt_candidates`, `lt_force_splitk1` and `lt_run`), `_lt_build` (built at import as C++ only, `-lcublasLt`, name `muse_bf16_lt_qkv128_v1`), `_lt_route_ok`, `_lt_forward`, `_lt_find_plan` and `_lt_cell_prepare`.
  - **Shared helpers:** `_cell_gate`, `_cell_capture`, `_cell_kernel_names`, `_cells_prepare`, `_cells_flags` and `_cells_inherit`.
- **`_prepare_state`** gains `cells` and `t_prepare0`. `_cells_prepare` runs AFTER all of master's own work, including ouns and router_cluster. The A/B second state calls `_cells_inherit`, which gives it its own repacked weight, error word, Lt workspace, algo-row copy or zero bias.
- **`kernel_fn`**: the two new branches come right after the ouns branch and BEFORE master's swapab16 branch. Cell A is `layer == "pooler"` and `_bulk_route_ok` (flag, then cfg m == 1, n 4096, k 8192, x shape/dtype/stride/16 B alignment). Cell B is `layer == "qkv_gate"` and `_lt_route_ok` (flag, then cfg m == 128, n 2304, k 8192, x contiguous, 256 B aligned). In cell B, mode 1 runs `_lt_forward` and mode 2 runs `torch.addmm(state["lt_qkv128_bias"], x, W.t())`. kernel_fn has no prints, no try/except and no state writes.
- **A/B self-check**: AB_CASES adds `("qkv_gate", 2304, 8192, 128)`. The existing pooler m=1 case now exercises cell A. `_ab_component` maps pooler m==1 to `bulk_p1_ok` and qkv_gate m==128 to `lt_qkv128`, so a miss turns off only that cell's flag and never master's other components. Switch-off sets `lt_qkv128` to 0. If the second-state build fails, both cells are turned off too.
- **Prints**: each cell prints one PATH CHOSEN line in prepare, in the format `PATH CHOSEN A(bulk-row-owner) pooler m=1: <candidate|master> (cand X us vs master Y us)` and `PATH CHOSEN B(cublaslt-no-splitk) qkv_gate m=128: ...`. prepare() then prints a final post-A/B PATH CHOSEN summary and FINAL FLAGS with both new flags.

**Cell A (source impl b3e80b45), per the review and the caller**
- `bulk_kernel` device code is ported verbatim. Only `<32, 8192, 8, 0>` is instantiated. The MODE 1/2 branches are `if constexpr` and compile out, and `dummy` is passed as nullptr. The NT16/NT32, unpooler and MODE 1/2 instances, the pure-read diagnostics, the 3.8 TB/s kill gate and the BULK_INCUMBENT print table are all dropped.
- The C++ `bulk_run(wt, x, out, err, m)` launches pooler `<32,8192,8,0>` only. Its host-side TORCH_CHECKs cover m 1..8 and the shapes.
- **Required fix applied:** the source's `state["bulk_route_hit"] = (layer, m)` per-call write is DROPPED. The route probe is replaced by a torch.profiler assertion: kernel_fn pooler m=1 must launch exactly one kernel whose name contains `bulk_kernel`.
- The error word `state["bulk_err"]` is per state. It is read only in the prepare self-check (synced), zeroed once at the end of `_bulk_cell_prepare` (and after each self-check read), and never read in kernel_fn.
- The bit-exact repack round-trip check (`_bulk_repack`) is kept verbatim. The repacked weight is per state (`state["bulk_wt_pooler"]`) and is dropped from the state when the cell is off.
- **Self-check (m=1):** a random launch plus a bit-identical repeat, and a probe row (single 0.5 entry). Bounds: rel L2 vs fp64 <= 3e-3, probe rel <= 1e-4, finite, error words 0. Each launch has a flushed print before it and a synchronize plus print after.
- **Gate:** `_cell_gate`, copied from `_ouns_gate`, runs kernel_fn captured with flag off (master's swapab16 path) and flag on. Settings: 256 MB flush before every replay, 41 interleaved reps, empty-graph baseline subtracted, candidate/master <= 0.98 required. Both medians are printed.
- **Deadline:** if prepare elapsed is over 200 s, the cell prints PATH CHOSEN master.
- **Build:** TORCH_CUDA_ARCH_LIST is set to 9.0a around the build and restored after; the arch flags are printed. extra_cuda_cflags are `-O3 -lineinfo -std=c++20`. The source's `#if __CUDA_ARCH__ >= 900` / err=3 guard is kept. The kernel uses no wgmma or setmaxnreg; cp.async.bulk and mbarrier are baseline sm_90. Every mbarrier wait is bounded at 0.5 s and sets an error flag.
- **Deviation:** if torch.profiler itself is unavailable, cell A stays on, with a loud print, because the routed branch calls `bulk_run` directly. If the profiler records names and they are not exactly one bulk_kernel, the cell goes off.

**Cell B: the review's required change replaces the spec's per-call toggle**
- **Spec error overridden:** there is NO `torch.backends.cuda.preferred_blas_library` toggle anywhere. B is now a prepare-time cuBLASLt plan. The matmul desc and layouts are cached by shape inside the extension; they are host-only objects with no device pointer. The algo row (CPU int64[8]) and the workspace are held in state.
- **Getting splitK=1 (review WEAKEST LINK):**
  - `cublasLtMatmulAlgoGetHeuristic` is queried under workspace caps of 0, 1 MiB, 4 MiB and 32 MiB, smallest first.
  - Every split-K result is ALSO tried with `CUBLASLT_ALGO_CONFIG_SPLITK_NUM` forced to 1 and REDUCTION_SCHEME set to NONE, validated by `cublasLtMatmulAlgoCheck`.
  - Only candidates whose SPLITK_NUM reads exactly 1 are kept.
  - Each candidate must be finite, have rel L2 vs fp64 <= 3e-3 and <= 1.10x master's own error (the algo-pick extension's guard, from the review's open question), and have a probe row <= 1e-4. It is then screened by cold-L2 graph replay, and the fastest is taken.
- **Open question on torch's Lt workspace size:** it is not queried directly, because `at::cuda::getCUDABlasLtWorkspaceSize` is version-dependent. The 0/1/4/32 MiB sweep brackets it, and every heuristic result is printed.
- **No-plan case:** if no splitK=1 algo is obtained, `[lt] NO splitK=1 cuBLASLt algo obtained for qkv_gate m=128: mode 1 (Lt plan) dropped` is printed.
- **Review's fallback (mode 2):** `torch.addmm(zero bf16 bias in state, x, W.t())` is tried only if mode 1 is unavailable or fails, and only behind the SAME check, gate and kernel-name assertion. If both fail, PATH CHOSEN prints master and the run is A alone.
- **Gate:** the same `_cell_gate`, with kernel_fn flag off (master's `_bf16_gemm_dispatch_impl`, the review's control) vs on. It requires <= 0.98x on cold-L2 graph replays.
- **Kernel-name assertion (strict):** kernel_fn qkv_gate m=128 must launch exactly ONE kernel with no "splitk" in its name, so neither a split-K nvjet nor a splitKreduce node is allowed. If the profiler is unavailable, the cell goes off, because the name would be unproven. Master's m=128 kernel names are printed for the record.
- **verify_32 check (review "keep the replacement check"):** with the flag on, the profiler compares kernel_fn qkv_gate m=256 against `_bf16_gemm_dispatch_impl` directly. Any difference turns B off. Since no global preference exists, the leak clause is moot by construction.
- **Deadline:** if prepare elapsed is over 250 s, the cell prints PATH CHOSEN master.

**Two judge states**
- Nothing device-side is module-global. The extensions themselves are module globals, like master's `_OUNS_EXT`/`_RC_EXT`.
- The C++ globals are a static bool (cudaFuncSetAttribute done once) and the Lt descriptor cache (shape-keyed, host-only).
- Repacked weight, error word, Lt workspace, algo row and zero bias are all per state.
- Every prepare() decides its own flags. The A/B second state inherits them and builds its own buffers, and master's A/B harness captures and replays both states alternately with refilled x, now including both cells.

**Biggest risks**
1. No splitK=1 Lt algo is as fast as the 48x64 kernel the toggle got. Then B lands on the addmm fallback or on master, which the PATH CHOSEN line will show.
2. A third CUDA build (about 40 s) plus the Lt C++ build (about 15-25 s) at import adds to the 10-minute budget.
3. torch.profiler/CUPTI misbehaving in prepare would leave B off, which is safe.

## Rules check

- [x] CUDA-graph capturable: kernel_fn's new branches do torch.empty plus one extension launch, or one torch.addmm. There is no .item/.cpu/sync/print, and no allocation or branching on values: only cfg m/n/k, shapes, strides, data_ptr alignment and prepare() flags.
- [x] Nothing derived from `data` is cached: kernel_fn writes nothing to state (the bulk_route_hit write was dropped).
- [x] Weights: the pooler repack is a lossless bf16 permutation done in prepare() with a bit-exact round-trip check. qkv_gate is used as-is. The zero bias is a bf16 constant that does not change the result.
- [x] Activations BF16 where SGLang keeps BF16: x and out are bf16 with fp32 accumulation, as in cuBLAS.
- [x] n/a: attention rule (no attention in this problem).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. load_inline only, at import. No `preferred_blas_library` or other global library state is changed.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] prepare-only syncs, prints, try/except and profiler; each flag lives in state; a build or self-check miss lands on master's path and never raises.

okc_lint output (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`): `OK with warnings`. There are no errors. Every warning is `host-sync ... (outside kernel_fn's call graph; fine in prepare())` or `dtype ... torch.empty(..., dtype=torch.uint8)`; the uint8 ones are flush and Lt workspace buffers in prepare, not compute precision. `py_compile` OK, `__pycache__` removed. An AST undefined-name scan found 0 bare names.

## Pre-flight audit

Audited 2026-10-25 by reading the code (model opus). No GPU on this node, so nothing was compiled or run. **Verdict: dispatch-with-fixes.**

**Fixes** (worktree `muse_bf16_proj/submission.py`, uncommitted)
1. **`bulk_run` host bounds checks.** Added TORCH_CHECKs before the launch: cuda tensors on one device, bf16 wt/x/out and int32 err, rank 2, contiguous wt and out, x unit inner stride (row stride 8192 when m > 1), x.size(0) >= m, 16-byte aligned x and wt. Reason: the C++ checked only m and the sizes. A bad x (too few rows, strided, misaligned) would have reached cp.async.bulk and could fault the device in a prepare canary, which voids the whole run.
2. **cuBLASLt descriptors are now built per call, not cached.** The function-static `std::map` descriptor cache (module-global C++ state) is replaced by a RAII `MuseLtDesc(n, k, m)` that each of `lt_candidates`, `lt_force_splitk1` and `lt_run` creates and destroys, the same way ATen's own `gemm_and_bias` does. The algo row and workspace stay per state. The handle is torch's current Lt handle, the same one `torch.mm` uses; no handle of ours is held. The cost is host-only and happens at capture, never at replay.
3. **A failed inherit turns off only its own cell.** `_cells_inherit` (the A/B second state's repack and Lt buffers) ran unguarded inside `_prepare_state`. Any exception there failed the whole second-state build, and the except in `_ab_selfcheck` then turned off `nn_ok`, `nn_p256_ok`, `router_cluster_ok` and `ouns_cl`. Each cell's inherit is now wrapped and turns off only that cell. `_ab_selfcheck` then turns the same cell off in the kept state A.

**Verified, no change needed**
- A build failure sets `_BULK_EXT` or `_LT_EXT` to None, which disables only cell A, or Lt mode 1 (the addmm mode 2 is still gated).
- No `preferred_blas_library` or `torch.backends` write anywhere. The addmm fallback changes no global flag.
- kernel_fn branches only on cfg, shape, stride, alignment and the state flags. It has no state writes, syncs, prints or try/except.
- The verify_32 check is not an always-fail gate: master routes qkv_gate m=256 to `_bf16_gemm_dispatch_impl`, which is exactly the "want" side of the comparison.
- Judge x comes from a fresh `.to(bf16)` tensor, so the 16 B and 256 B alignment gates pass.
- `_ab_component` maps pooler m=1 and qkv_gate m=128 to their own flags, so an A/B miss on either cell never touches another flag.
- **Budget:** import builds are about 37.7 + 36.8 (master) + ~37 (bulk) + ~15 (Lt) = ~127 s. Prepare is ~4 s for master plus an estimated ~10-30 s for the cells. Master-lineage runs took ~150 s end to end with 2 builds, so the estimate is ~4-5 min against the 10-minute limit and prepare well under 300 s.

**Residual risk:** this is the first compile of the ported bulk TU, the new Lt C++ and the edits above (any failure lands on master for that cell only), and cell B may find no splitK=1 algo within 0.98x, in which case it prints PATH CHOSEN master.
