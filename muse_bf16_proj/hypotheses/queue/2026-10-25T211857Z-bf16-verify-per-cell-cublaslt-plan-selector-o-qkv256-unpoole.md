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
review: proceed
status: queued
review_at: 2026-10-25T212522Z
review_companions: null
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

## Adversarial review

Reviewed 2026-10-25T212522Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a legitimate and well-scoped follow-up (DIAGNOSIS Action 2). It reuses master's proven cell-B Lt machinery behind a stricter two-round 1.03 + IQR gate, and its per-cell flags plus PATH CHOSEN lines let each row explain itself, so no companion run is needed. Two things will quietly void the run if missed: unpooler m=128 must branch before master's NN branch, and Lt candidates with reduction scheme OUTPUT_TYPE must be excluded.

### Justification
The mechanism is plausible but modest. The five cells sit at SGLang parity, 1.6-2.1x their floors. o is 7.68/9.44 us against floors of 4.0/4.5, qkv m=256 is 19.46 against 9.8, and unpooler is 18.88/24.18 against 14.6/17.4. The m=128/256 band is where DIAGNOSIS says the absolute gaps are, and every m>=128 win so far was a cuBLAS reshape: NN layout, K-halves, and the cell-B single-node Lt plan (predicted 0.919, judged 0.895 at 741072dc). The machinery is cheap: master's prepare total is 4.5 s against a 300 s budget, so 5 cells x <=24 candidates x 2 rounds fit easily. Split-K>1 Lt plans do exist on this image: archived [lt] logs show splitK=2/3, red=2, ws 2.3-4.7 MB, so a 32 MB query cap is enough. The prior field is accurate. bf16-cublaslt-algo-pick-in-prepare-m128-m256 (do-not-try, within-ci) used a non-interleaved pick with no margin. Its qkv m=256 choice flipped +14%, and the LESSONS correction blames the gate design, not the knob. The hardening (two interleaved rounds >=1.03, IQR bound, profiler kernel-name assert) is a real difference and follows today's new lesson. The per-call preferred_blas_library toggle is avoided as Do-not-try requires. The excluded dead candidates (qkv K-split bmm, unpooler K-halves at m=128) match their done/regression and neutral files. Weakest links: (1) qkv m=256 is the cell with the worst prepare-vs-judge record. Three separate picks there flipped (backend toggle 22.08 vs 19.5, algo-pick +14%, K-split S=4 judged 22.17). The incumbent is already cuBLAS's own heuristic top pick, so a gain needs a non-top heuristic plan or a split-K plan to genuinely beat nvjet 96x64 cold. Expect 'master' here more often than not. (2) The o rationale that 'the 2 us row-minus-kernel can be cut' is not supported. o m=128 is already ONE nvjet node (no splitKreduce), so there is no second graph node to remove. The gap is most likely cold-start/launch effects that any single-kernel plan also pays. The only new lever for o is the NN layout (o_t), which helped pooler/unpooler at m=128 but has never been measured on o (K=1024, short mainloop). (3) The expected 'faster' on all five cells contradicts the hypothesis's own Risk section, which says the gate may route none. Realistically 0-3 cells route, for a headline of about +0-1%, which is inside the >=1.5% CI. Judge it on per-row PATH CHOSEN plus the paired row ratios, not the headline.

### Confounders on the current kernel
- restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (NN unpooler at 65<=m<=128 behind nn_ok): unpooler m=128's incumbent is torch.mm on unpooler_t, not dispatch_impl. kernel_fn's NN branch comes BEFORE the dispatch_impl fallback, so a new unpooler-m128 branch placed 'before the _bf16_gemm_dispatch_impl fallback' as the change says is unreachable. The cell would silently measure master. It must sit above the NN branch, and the gate's flag-off side must be the NN path. If nn_ok ever comes up False, the incumbent changes under the cell; PATH CHOSEN should print which incumbent was gated.
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (cell B qkv m=128 Lt plan, _cell_gate, _LT_CPP, per-state Lt workspace via flags inheritance): this is the machinery being generalized. A regression in the shared MuseLtDesc (new transa flag) or in the workspace handling could also move the qkv m=128 row, which is not a target. Treat any qkv verify_16 move as a machinery defect, not as signal.
- sglang-gemm-dispatch-parity: _bf16_gemm_dispatch_impl is the incumbent for o m=128/256, qkv m=256 and unpooler m=256. Its nvjet heuristic choice is very likely among the Lt top-8, so a candidate identical to master can win the pre-screen by noise. The 1.03 two-round gate should then reject it. Print the algo/tile of the candidate next to master's profiled kernel name so an identical kernel is visible.
- pooler-nn-layout-widen-m256 / pooler NN at m=128: not touched (pooler is left to pooler-k-halves-fp32-partials-restack-m128-m256), so no confounding. The two queued items edit the same kernel_fn/prepare region, which is a merge problem for a later combo, not a measurement confounder.

### Interference and A/B needs
No companion is warranted. Each cell has its own state flag decided by an in-run A/B against master's exact path (flag off = byte-identical master routing), and the judge times shapes in isolation with a cold L2, so there are no cross-cell or cross-landed interactions in the timing. A row that PATH CHOSEN says routed and that moves in the paired ratio is attributable to that cell. A row that says master and moves is drift (LESSONS: +/-3-4% row noise between master-equivalent runs). A loss on a routed cell is unambiguous too: the selector failed to predict the judge, the same failure class as today's lesson. An ab-revert of the NN restack would only change unpooler m=128's incumbent, which the gate already measures directly. A control isolating one sub-change is redundant because each cell's candidate set is itself selected and printed. The headline cannot be resolved by any single extra run at the expected +0-1% size, so companions would buy nothing the PATH CHOSEN and GATE lines do not already give.

### Open questions
- Does the cuBLASLt heuristic at m=256 qkv_gate return any plan that is not the nvjet 96x64 2x4 dispatch_impl already uses? If the pre-screen's top two include master's own algo, the cell is a no-op by construction. Print the master kernel name next to each finalist.
- IQR <= 1.5x master's IQR: is IQR computed from the 41 net replays per round? On ~7-9 us o rows, is the empty-graph-baseline jitter (~5.3 us base) large enough to make the IQR bound reject everything?
- Are the two rounds independent, i.e. recaptured graphs and a fresh x? Or the same graphs replayed twice? Same-graph rounds share capture-time layout and are less independent than the lesson intends.
- For o the NN layout (o_t [1024, 8192]) needs MuseLtDesc's transa flag and A layout (n, k, ld n). Is cell B's correctness check (_cell_check with the power-of-two probe column) re-pointed at the NN view so the probe row still compares against the true W_o column?
- Per-cell 32 MB workspace x 5 cells x state copies is about 160+ MB per state. That is fine on H200, but would a single shared per-state workspace (cells run serially on one stream) be simpler and equally safe?
- The hypothesis says it supersedes and retires the held combo-cublaslt-single-node-m128-m256-plus-naive. Confirm churn actually retires it so the two are not both dispatched.

