# muse_bf16_proj · 2026-10-26T012539Z · check · bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T211857Z-bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole.md` (branch `hyp/muse_bf16_proj/bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole`).

- **Job:** `37ed8722bc9b402d` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `f687dee0`

## Goal
DIAGNOSIS Action 2, narrowed by today's evidence. Built on master 36c26199. Generalise master's cell-B machinery (_LT_CPP candidates/run with MuseLtDesc(n, k, m), _lt_prepare, _cell_gate, _cell_kernels) from qkv_gate m=128 to five more cells, each with its own flag and state entries: o m=128 (row 7.68, nvjet 128x64 TNT 5.7 us kernel, 2 us of row-minus-kernel, floor 4.0), o m=256 (9.44, nvjet 128x128 8.3, floor 4.5), qkv_gate m=256 (19.46, nvjet 96x64 2x4 TNN 20.0, floor 9.8), unpooler m=128 (18.88, nvjet 128x64 NNT 19.3 on the NN repack, floor 14.6), unpooler m=256 (24.18, nvjet 128x128 TNT 24.8, floor 17.4). Pooler is left to pooler-k-halves-fp32-partials-restack-m128-m256 and qkv m=128 keeps cell B, so attribution stays clean. Candidates per cell: master; the top 8 cublasLtMatmulAlgoGetHeuristic plans with split-K 1 (as cell B); plus up to 4 split-K > 1 plans with a per-cell fp32 workspace allocated once per state in prepare (query with a 32 MB workspace cap; the reduce stays inside cuBLASLt); for o and unpooler both the TN weight and an NN view (unpooler: master's existing unpooler_t; o: a new lossless o_t = W_o.t().contiguous() [1024, 8192], 16.8 MB, built in prepare), added to MuseLtDesc as a transa flag; for unpooler m=256 only, also the K-halves bmm with fp32 partials (S=2, as in the pooler restack). EXCLUDED as measured dead: qkv K-split bmm (S=4 T judged 0.89 at m=256, S=2/4 0.87-0.99 at m=128, 9d93c3d0) and unpooler K-halves at m=128 (0.888/0.865 in prepare, e82feba3). Pre-screen each cell's candidates with the existing LT_PICK_REPS cold-L2 timing, send the fastest two to the gate. Gate (hardened from today's evidence): copy master's _cell_gate model selector (graph replay of kernel_fn with the flag on vs off, 256 MB flush before each replay, 41 interleaved reps, empty-graph baseline subtracted) but route only if BOTH of two independent interleaved rounds give master/candidate >= 1.03 AND the candidate's IQR is <= 1.5x master's IQR. Today two picks that passed a single 1.02 round flipped in the judge: qkv-k-split S=4 T at m=256 predicted 1.041 (IQR 2.05) and judged 19.46->22.17 us; router-hilo-tall at m=256 judged 9.72->9.99. The pick that carried (pooler K-halves N) predicted 1.069/1.037. Then one torch.profiler call of the chosen path asserts the launched kernel names, and one PATH CHOSEN line per cell prints both medians, both IQRs and the kernels. No per-call preferred_blas_library toggles (Do-not-try); plans live in state; descriptors per call on the host as cell B does. kernel_fn gets one cfg/flag-only branch per cell before the _bf16_gemm_dispatch_impl fallback.

## What changed, and where
Only `<worktree>/muse_bf16_proj/submission.py` changed (+498/-9). Cell B (`_LT_CPP`, `_lt_prepare`, `MuseLtDesc`) is untouched.

What changed:
- New C++ module `_LT2_CPP` (`muse_bf16_lt_percell_v1`; symbols `lt2_candidates`/`lt2_run`, struct `MuseLt2Desc`) with a `transa` flag: TN weight `[n,k]` or the NN view `[k,n]`. A separate module, so cell B's machinery cannot move (review: qkv m=128 is a machinery-defect canary). Built at import with build time printed; it holds no device kernels, so there are no regs/spills to report.
- `PC_CELLS`: o128, o256, qkv256, unp128, unp256, each with its own flag (`lt_o128_ok`, ...) in `NEW_CELL_FLAGS`, so the A/B check, FINAL FLAGS and the failure path see them. Plans live in `state["pc_plan"]`; one shared 32 MB fp32 workspace per state (`pc_ws`; cells run serially on one stream; answers the review's open question). `o_t = W_o.t().contiguous()` is built in prepare and dropped if unused. State B inherits flags and plans, with its own algo-row clones, workspace and `o_t`.
- `_pc_cell`: candidates are the heuristic plans on the TN weight and, for o and unpooler, the NN view, with split-K 1 (top 8 per layout) and split-K > 1 (top 4 per layout, 32 MB cap). unp256 also gets the K-halves bmm (S=2, fp32 partials via `torch.bmm(out_dtype=float32)`, add, cast). qkv K-split and unpooler m=128 K-halves are excluded as measured dead. Each candidate is correctness-checked (`_cell_check`, verbose off; its probe row is compared with the true weight column, which answers the review's NN-probe question) and cold-L2 graph-timed (LT_PICK_REPS). The master routed path is timed too. The top 2 go to the gate.
- `_pc_gate_round`: two independent interleaved rounds. Each round uses a fresh x, recaptured graphs, 41 reps, a 256 MB flush and the empty-graph baseline. Route only if BOTH rounds give master/cand >= 1.03 AND cand IQR <= 1.5x master IQR. Both rounds are always run and printed, with medians and IQRs. Then `torch.profiler` lists master's and the candidate's kernels. The cell is dropped if the kernel names are identical to master's, or the launch count is 0 or above 4.
- kernel_fn: one `_pc_route` branch (cfg/flag/shape only) placed ABOVE the bulk, NN and dispatch_impl branches. This is the required fix for unpooler m=128, whose incumbent is the NN torch.mm. The gate's flag-off side is kernel_fn with the flag off, i.e. the actually routed path. PATH CHOSEN prints which incumbent was gated (NN torch.mm vs dispatch_impl).
- AB_CASES gained o 128, o 256, qkv_gate 256 and unpooler 256. `_ab_component(layer, m, state)` maps a routed PC cell to its own flag, so a miss cannot switch off nn_ok.
- Deadlines: the phase does not start after 150 s into `_prepare_state`, and each cell not started inside 100 s of the phase prints `skipped: deadline`.
- The LAST lines of prepare are one `PC-SUMMARY PATH CHOSEN` line per cell with its FINAL flag (after the A/B check).

Review fixes applied:
- (1) The unp128 branch sits above the NN branch.
- (2) Lt candidates with OUTPUT_TYPE reduction (red=4) are excluded.
- Open question 1: master's profiled kernel names are printed next to the candidate's, and an identical kernel set is dropped.
- Open question 2: the master IQR is floored at 0.25 us in the bound, so a tiny master IQR cannot reject everything.
- Open question 3: rounds are independent (fresh x, recaptured graphs).
- Open question 4: the NN probe check is re-pointed (see above).
- Open question 5: one shared workspace per state.
- Open question 6 (retiring combo-cublaslt-single-node-m128-m256-plus-naive) is a churn action, not code.

Deviations and clarifications:
- Split-K > 1 plans are kept only with reduction scheme COMPUTE_TYPE (fp32 workspace partials). INPLACE (red=1) also accumulates in bf16, so it is excluded as well as OUTPUT_TYPE, for precision safety. Split-K-1 plans are not filtered by red.
- `_cell_check` gained `verbose=True` (default unchanged) so screening ~100 candidates does not flood the log. Existing callers are unchanged.
- The gate threshold is expressed as master/cand >= 1.03 (the hypothesis wording), not cell B's cand/master <= 0.98.

Biggest risks:
- The `torch.bmm(out_dtype=...)` support on the judge's torch is unknown. If it is missing, the K-halves candidate is excluded via the exception handler.
- The new C++ module's compile time at import adds to the run's wall clock.
- Nothing could be run on this node (no GPU or torch), so nothing is executed.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn's PC branch only calls `lt2_run` (host descriptors, no sync, no value-dependent allocation) or bmm/add/cast; the diff has no .item()/.cpu()/synchronize in kernel_fn. All syncs sit in prepare-only helpers.
- [x] Nothing derived from `data` is cached: plans and workspace come from weights and synthetic prepare inputs only. The workspace is overwritten each call.
- [x] Weights: `o_t` is a lossless transpose built in prepare; the bf16 weights are unchanged. No precision change (split-K only with fp32 COMPUTE_TYPE reduction; K-halves with fp32 partials).
- [x] Activations BF16 where SGLang keeps BF16: x and out are bf16; fp32 is used only internally for accumulation and partials.
- [x] Attention: n/a, bf16 projection problem.
- [x] No timing patches, harness frames, module edits, reference import, subprocesses or network. load_inline compiles in-process, as cell B already does.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] Any miss (build, check, gate, profile, deadline, A/B) lands on master's exact path per cell with its own flag; bounded waits n/a (no custom kernels, no spins).

okc_lint: `OK with warnings: muse_bf16_proj/submission.py` (warnings are only prepare-time host-sync and flush-buffer dtype notes, the same kinds as on master). `py_compile` passes. `__pycache__` removed.

## Risks / what could fail
medium
Effect: realistic -0.5 to -2 us on each of the five cells (0.230 %/us verify_16, 0.184 %/us verify_32), about +0.4-1.0% headline: likely below the margin alone, must stack (combo). The gate may route none; then the run reads as master and its PATH CHOSEN lines are the result. New cuBLASLt workspace split-K path: first measurement is an okc check. Supersedes the held combo-cublaslt-single-node-m128-m256-plus-naive (retire it).

## Result (after the run)
- job `37ed8722bc9b402d` (check), status done, passed True, tests 9/9
- speedup 1.2525x, 95% CI [1.2337, 1.2713], score 3856.947197330629 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole.patch`; branch `hyp/muse_bf16_proj/bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpoole`
