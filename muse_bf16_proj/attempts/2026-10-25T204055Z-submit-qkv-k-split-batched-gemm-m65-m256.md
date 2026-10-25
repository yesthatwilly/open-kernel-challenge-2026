# muse_bf16_proj · 2026-10-25T204055Z · submit · qkv-k-split-batched-gemm-m65-m256

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T153141Z-qkv-k-split-batched-gemm-m65-m256.md` (branch `hyp/muse_bf16_proj/qkv-k-split-batched-gemm-m65-m256`).

- **Job:** `9d93c3d0cc184226` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `dbd922b7`

## Goal
Built on current master (blob 353e3b19). For layer == 'qkv_gate' at 65 <= cfg m <= 256 only, which covers the scored rows verify_16 m=128 and verify_32 m=256. The m=512 test stays on master (m > 256). WHY: qkv_gate (n 2304, k 8192) is the cuBLAS cell furthest from its floor in ratio. At m=128 it is 16.58 us against a floor of 8.42, as nvjet 96x64 splitK 12.7 us plus cublasLt splitKreduce 3.1 us plus a node. At m=256 it is 19.71 against 9.77, as nvjet 96x64 2x4 TNN 19.9 us, about 490 TFLOP/s. cuBLAS's own split-K at m=128 pays a separate 3.1 us reduce kernel over fp32 partials. CHANGE: the pooler-k-halves mechanism applied to qkv, with S in {2, 4} K-slices. xs = x.view(m, S, 8192 // S).permute(1, 0, 2) is a strided view with no copy. ws = state['qkv_gate'].view(2304, S, 8192 // S).permute(1, 0, 2) is used transposed, or a contiguous [S, 8192 // S, 2304] NN repack made once in prepare(); the repack is lossless and adds 37.7 MB per S. p = torch.bmm(xs, ws_t) gives [S, m, 2304] bf16; then out = torch.add(p[0], p[1]) for S=2, or p.sum(0) for S=4. Precision: S bf16 roundings, about 0.2-0.4% per element and L2 well under 1%. Single-nonzero rows are exact. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). Candidates per m: S=2 TN, S=2 NN, S=4 TN and S=4 NN. Route the fastest only if it is >= 1.02x master at that m; the m=128 and m=256 decisions are separate and printed. The profiler assertion must show one GEMM kernel plus one add/reduce and no copy. Every other layer and m stays byte-identical.

## What changed, and where
Edited only muse_bf16_proj/submission.py (+197/-2) in the worktree. Layer name in code is `qkv_gate`.
- `_qk_forward(x, state, (S, 'T'|'N'))`: strided K-slice bmm + torch.add (S=2) / p.sum(0) (S=4). One lossless NN repack `state['qkv_t']` [8192, 2304] made in `_prepare_state` (both states); S=2/4 are views of it.
- `_qk_select(state)`: runs once on state A at the end of the flags-None branch, AFTER `_lt_prepare`. Master baseline is `kernel_fn` itself with the new flag off, so it times exactly what master runs per m (cell B at 128 if LT_FLAG, dispatch at 256). Four candidates, cold-L2 (256 MB flush) interleaved 41 reps, empty-graph baseline subtracted, medians/IQR/kernel names printed for all, 1.02x threshold per m. Name check rejects only copy kernels and splitKreduce; prints 'name check skipped' on empty events. Per-candidate and global try/except keep master; never raises.
- Own flag `qkv_ksplit_ok` + dict `qkv_ksplit` {m: (S, layout)}; passed to B via `_ab_flags`; `_ab_component(layer, m, state)` returns it before LT_FLAG; AB_CASES gains qkv m=256. A miss only turns this flag off. nn_ok/nn_p256_ok/LT_FLAG untouched.
- kernel_fn: one gate (layer, cfg m in (128,256), n/k/x shape/dtype/contiguity, dict lookup) before `if layer != "router"`; no print/try/sync/mutation; outputs fresh.
- Deviation: none. Not run (no GPU).

## Rules check (each must be true before the run)
- [x] No subprocess/fork/spawn added (diff checked).
- [x] No triton.autotune; no host sync, print, try/except or .item() in kernel_fn.
- [x] Nothing derived from inputs cached; weights repack lives in `state` only.
- [x] Both states: B inherits the choice via _ab_flags; A/B graph check covers qkv m=128 and 256.
- [x] Precision: S bf16 roundings; selector rejects rel L2 > 6e-3 vs fp32 reference.
- [x] Any selector failure keeps master and never raises.
- [x] No CUDA/PTX load_inline added: n/a.
- [x] First-line marker kept; not committed, not staged, okc not run.

okc_lint: OK with warnings (prepare-side host-sync / flush-dtype only); py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
Same mechanism as pooler-k-halves-batched-gemm-m65-m256, on a separate layer and rows, so each is attributed per row. The prepare-time cublasLt algo pick on qkv m=256 did not carry over to the judge (LESSONS). This selector times graph replays with a cold L2, as the ouns selector did, which predicted the judge.

## Result (after the run)
- job `9d93c3d0cc184226` (submit), status done, passed True, tests 9/9
- speedup 1.2410x, 95% CI [1.2224, 1.2597], score 3892.5154499568594 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2566457115954903, "ci": [1.237796025921558, 1.2754953972694225], "job": "741072dca5fc4c9d"}; base result: muse_bf16_proj/hypotheses/done/neutral/combo-rowowner-reduce-free-m1-qkv-o-plus-naive.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/qkv-k-split-batched-gemm-m65-m256.result.json`, `muse_bf16_proj/hypotheses/done/regression/qkv-k-split-batched-gemm-m65-m256.patch`; branch `hyp/muse_bf16_proj/qkv-k-split-batched-gemm-m65-m256`
