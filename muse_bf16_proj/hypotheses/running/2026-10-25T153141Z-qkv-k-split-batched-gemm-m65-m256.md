---
problem: muse_bf16_proj
slug: qkv-k-split-batched-gemm-m65-m256
created: 2026-10-25T153141Z
base: master
area: qkv GEMM (m=65..256, K-split as a batched cuBLAS GEMM)
change: >
  Built on current master (blob 353e3b19). For layer == 'qkv_gate' at 65 <= cfg m <= 256 only,
  which covers the scored rows verify_16 m=128 and verify_32 m=256. The m=512 test stays on master
  (m > 256). WHY: qkv_gate (n 2304, k 8192) is the cuBLAS cell furthest from its floor in ratio.
  At m=128 it is 16.58 us against a floor of 8.42, as nvjet 96x64 splitK 12.7 us plus cublasLt
  splitKreduce 3.1 us plus a node. At m=256 it is 19.71 against 9.77, as nvjet 96x64 2x4 TNN 19.9
  us, about 490 TFLOP/s. cuBLAS's own split-K at m=128 pays a separate 3.1 us reduce kernel over
  fp32 partials. CHANGE: the pooler-k-halves mechanism applied to qkv, with S in {2, 4} K-slices.
  xs = x.view(m, S, 8192 // S).permute(1, 0, 2) is a strided view with no copy. ws =
  state['qkv_gate'].view(2304, S, 8192 // S).permute(1, 0, 2) is used transposed, or a contiguous
  [S, 8192 // S, 2304] NN repack made once in prepare(); the repack is lossless and adds 37.7 MB
  per S. p = torch.bmm(xs, ws_t) gives [S, m, 2304] bf16; then out = torch.add(p[0], p[1]) for
  S=2, or p.sum(0) for S=4. Precision: S bf16 roundings, about 0.2-0.4% per element and L2 well
  under 1%. Single-nonzero rows are exact. Selector (copy
  restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture
  each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB
  L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both
  medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH
  CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert
  the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Candidates
  per m: S=2 TN, S=2 NN, S=4 TN and S=4 NN. Route the fastest only if it is >= 1.02x master at
  that m; the m=128 and m=256 decisions are separate and printed. The profiler assertion must show
  one GEMM kernel plus one add/reduce and no copy. Every other layer and m stays byte-identical.
expected:
  - shapes: layer=qkv_gate load=verify_16 m=128
    direction: faster
    why: "nvjet splitK 12.7 + splitKreduce 3.1 / row 16.58, floor 8.42; a bf16 add over 2x0.6 MB (~0.8 us) replaces the 3.1 us fp32 reduce: row ~14"
  - shapes: layer=qkv_gate load=verify_32 m=256
    direction: faster
    why: "nvjet 96x64 19.9 / row 19.71, floor 9.77; shorter K per batch lets cuBLAS pick a fuller tile grid: row ~16-17 if a candidate wins, else master"
  - shapes: layer=* load=decode_* m=*
    direction: unchanged
    why: m <= 32 untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T203645Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T203645Z
base_sha: 752b3dcbd27a45849093c5387a4716ce9fca68ae
base_blob: ad8d7254a64a52db2a8f56ec0ad262748153a2e8
branch: hyp/muse_bf16_proj/qkv-k-split-batched-gemm-m65-m256
worktree: .worktrees/muse_bf16_proj/qkv-k-split-batched-gemm-m65-m256
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T204055Z
impl_commit: dbd922b795fec313a8f3f0ea92420b6bb4920a74
submission_sha256: a8da8943c1e510c3eea66f7ae0d4027206774cde527ae908fa90e64fef2d31d7
norm_sha256: b5e58a8d3dab336830b7cd27d81818e927348944e6498a6c73b418789457f25c
submission_blob: 76386bd87b0c1d31910cc21b81598ddb2c1addba
lint_warnings: 39
attempt_file: 2026-10-25T204055Z-submit-qkv-k-split-batched-gemm-m65-m256.md
prepared_at: 2026-10-25T204055Z
job: 9d93c3d0cc184226
dispatched_at: 2026-10-25T204106Z
---

# qkv-k-split-batched-gemm-m65-m256

## Change

Built on current master (blob 353e3b19). For layer == 'qkv_gate' at 65 <= cfg m <= 256 only, which covers the scored rows verify_16 m=128 and verify_32 m=256. The m=512 test stays on master (m > 256). WHY: qkv_gate (n 2304, k 8192) is the cuBLAS cell furthest from its floor in ratio. At m=128 it is 16.58 us against a floor of 8.42, as nvjet 96x64 splitK 12.7 us plus cublasLt splitKreduce 3.1 us plus a node. At m=256 it is 19.71 against 9.77, as nvjet 96x64 2x4 TNN 19.9 us, about 490 TFLOP/s. cuBLAS's own split-K at m=128 pays a separate 3.1 us reduce kernel over fp32 partials. CHANGE: the pooler-k-halves mechanism applied to qkv, with S in {2, 4} K-slices. xs = x.view(m, S, 8192 // S).permute(1, 0, 2) is a strided view with no copy. ws = state['qkv_gate'].view(2304, S, 8192 // S).permute(1, 0, 2) is used transposed, or a contiguous [S, 8192 // S, 2304] NN repack made once in prepare(); the repack is lossless and adds 37.7 MB per S. p = torch.bmm(xs, ws_t) gives [S, m, 2304] bf16; then out = torch.add(p[0], p[1]) for S=2, or p.sum(0) for S=4. Precision: S bf16 roundings, about 0.2-0.4% per element and L2 well under 1%. Single-nonzero rows are exact. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Candidates per m: S=2 TN, S=2 NN, S=4 TN and S=4 NN. Route the fastest only if it is >= 1.02x master at that m; the m=128 and m=256 decisions are separate and printed. The profiler assertion must show one GEMM kernel plus one add/reduce and no copy. Every other layer and m stays byte-identical.

## Expected

- `layer=qkv_gate load=verify_16 m=128` faster: nvjet splitK 12.7 + splitKreduce 3.1 / row 16.58, floor 8.42; a bf16 add over 2x0.6 MB (~0.8 us) replaces the 3.1 us fp32 reduce: row ~14
- `layer=qkv_gate load=verify_32 m=256` faster: nvjet 96x64 19.9 / row 19.71, floor 9.77; shorter K per batch lets cuBLAS pick a fuller tile grid: row ~16-17 if a candidate wins, else master
- `layer=* load=decode_* m=*` unchanged: m <= 32 untouched

## Risk

Same mechanism as pooler-k-halves-batched-gemm-m65-m256, on a separate layer and rows, so each is attributed per row. The prepare-time cublasLt algo pick on qkv m=256 did not carry over to the judge (LESSONS). This selector times graph replays with a cold L2, as the ouns selector did, which predicted the judge.

## Adversarial review

Reviewed 2026-10-25T203645Z by an adversarial subagent (model opus).

**Verdict: proceed.** Fair transfer of pooler-k-halves (selector 1.069x/1.037x, judge 1.063x/1.123x on pooler verify_16/32) to qkv_gate; the m=128 rationale is stale: master 36c26199 already runs cuBLASLt cell B (nvjet 40x64 4x2 TNN ~15.5 us, no splitKreduce) at qkv m=128. Baseline against cell B, give the new path its own flag, expect a smaller m=128 gain; m=256 (~20 -> ~16-17 us) is the real bet.

### Justification
qkv_gate n 2304 k 8192 at m=256: nvjet 96x64 2x4 ~20.0 us vs 9.8 us floor, ~96 CTAs < 132 SMs over full K; K-split raises parallelism. pooler-k-halves (ce808e5a): strided bmm without copy, 128x128 tile at m=256, selector under-predicted judge. Weakest link: wave quantization for narrow N=2304 (72/144/192 CTAs depending on tile and S); no sibling layer predicts the tile cuBLAS picks; the selector contains it. Prior: pooler-k-halves (different layer); bf16-cublaslt-algo-pick-in-prepare-m128-m256 did not change problem shape.

### Confounders on the current kernel
- combo-rowowner-reduce-free-m1-qkv-o-plus-naive (landed): cell B (LT_FLAG) is the m=128 incumbent; time whatever master runs after _lt_prepare; own flag checked before _lt_route_ok; _ab_component returns it before LT_FLAG so a miss falls back to cell B.
- sglang-gemm-dispatch-parity: m=256 incumbent and m=128 fallback.
- pooler-nn-layout-widen-m256 / restack-three-cell-wins: keep the new flag separate from nn_ok/nn_p256_ok.
- bf16-m32-swapab-wgmma-triton / bf16-restack-m16: gate at m<=32 first; no interaction.

### Interference and A/B needs
No companion: the prepare-time selector times master's actual path and the four candidates (S=2/4, TN/NN) in-process with cold L2 and prints medians, IQR and tile names.

### Open questions
- Print the tile cuBLAS picks for all four candidates.
- At m=128 can any candidate beat cell B's 15.5 us by >= 1.02x after the add?
- Run after _lt_prepare; choose once on state A; pass to state B via _ab_flags.
- Credit per row against 20.0/15.5 us incumbents.
- Stack later with unlanded pooler-k-halves?


## Implementation notes

Edited only muse_bf16_proj/submission.py (+197/-2) in the worktree. Layer name in code is `qkv_gate`.
- `_qk_forward(x, state, (S, 'T'|'N'))`: strided K-slice bmm + torch.add (S=2) / p.sum(0) (S=4). One lossless NN repack `state['qkv_t']` [8192, 2304] made in `_prepare_state` (both states); S=2/4 are views of it.
- `_qk_select(state)`: runs once on state A at the end of the flags-None branch, AFTER `_lt_prepare`. Master baseline is `kernel_fn` itself with the new flag off, so it times exactly what master runs per m (cell B at 128 if LT_FLAG, dispatch at 256). Four candidates, cold-L2 (256 MB flush) interleaved 41 reps, empty-graph baseline subtracted, medians/IQR/kernel names printed for all, 1.02x threshold per m. Name check rejects only copy kernels and splitKreduce; prints 'name check skipped' on empty events. Per-candidate and global try/except keep master; never raises.
- Own flag `qkv_ksplit_ok` + dict `qkv_ksplit` {m: (S, layout)}; passed to B via `_ab_flags`; `_ab_component(layer, m, state)` returns it before LT_FLAG; AB_CASES gains qkv m=256. A miss only turns this flag off. nn_ok/nn_p256_ok/LT_FLAG untouched.
- kernel_fn: one gate (layer, cfg m in (128,256), n/k/x shape/dtype/contiguity, dict lookup) before `if layer != "router"`; no print/try/sync/mutation; outputs fresh.
- Deviation: none. Not run (no GPU).

## Rules check

- [x] No subprocess/fork/spawn added (diff checked).
- [x] No triton.autotune; no host sync, print, try/except or .item() in kernel_fn.
- [x] Nothing derived from inputs cached; weights repack lives in `state` only.
- [x] Both states: B inherits the choice via _ab_flags; A/B graph check covers qkv m=128 and 256.
- [x] Precision: S bf16 roundings; selector rejects rel L2 > 6e-3 vs fp32 reference.
- [x] Any selector failure keeps master and never raises.
- [x] No CUDA/PTX load_inline added: n/a.
- [x] First-line marker kept; not committed, not staged, okc not run.

okc_lint: OK with warnings (prepare-side host-sync / flush-dtype only); py_compile OK; __pycache__ removed.

## Pre-flight audit

Audited 2026-10-25 (model opus), by reading the code plus a numpy emulation of the view/permute K-slice math. No GPU was available. **Verdict: dispatch.** No fixes were made.
- Base blob ad8d7254 matches master. The first-line marker is intact. okc_lint reports OK with prepare-side warnings only, and py_compile passes.
- Routing: qkv_gate at m=128/256 is not caught by any earlier gate (_qkv_route_ok covers m<=16, swapab16 m<=16, swapab 17..32, bulk m=1). The new gate comes before the NN and Lt branches and checks only cfg, shapes and state.
- Selector: it runs after _lt_prepare, with qkv_ksplit_ok forced False. That makes "master" kernel_fn itself (cell B at m=128 when LT_FLAG is set, dispatch at m=256). Each candidate and master is wrapped in its own try. A global except keeps master, so the selector never raises.
- Name check: it rejects only copy/contiguous/memcpy kernels and cublasLt splitKreduce. The kernel names pooler-k-halves produced on the judge (ce808e5a: nvjet_*_TNT/NNT + CUDAFunctor_add) contain none of these, so the check cannot reject every candidate.
- A/B: state B inherits qkv_ksplit and builds its own qkv_t. The refill uses the same (m, k) shape and the same fp32-reference bound (6e-3) as state A, so this is not an always-fail gate. _ab_component returns qkv_ksplit_ok before LT_FLAG, so a miss turns off only the new flag. Build failure also clears it.
- Cleanup: the flush buffer is dropped in finally and the selector graphs go out of scope. prepare() calls empty_cache after the A/B check.
- Numerics: the slices index correctly for S=2/4 in both T and N layouts (emulated exactly). Two bf16 partials plus the add give about 2.5e-3 rel L2, the same as pooler-k-halves on the judge. Single-nonzero rows are exact. The m=512 test is not routed.
- Minor deviation from the spec (accepted): kernel counts are printed, not asserted. The S=4 sum shows up as reduce_kernel.

Residual risk: at m=128 the incumbent is cell B (~15.5 us, no splitKreduce), so the 1.02x gate may keep master there. The gain probably depends on m=256 alone.
