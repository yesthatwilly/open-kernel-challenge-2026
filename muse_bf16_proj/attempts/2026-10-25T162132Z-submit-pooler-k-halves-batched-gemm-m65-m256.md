# muse_bf16_proj · 2026-10-25T162132Z · submit · pooler-k-halves-batched-gemm-m65-m256

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T153141Z-pooler-k-halves-batched-gemm-m65-m256.md` (branch `hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256`).

- **Job:** `ce808e5a055f41b4` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `e5de059e`

## Goal
Built on current master (blob 353e3b19). For layer == 'pooler' at 65 <= cfg m <= 256 only (scored rows verify_16 m=128 and verify_32 m=256; no test uses this band), replace master's path with a user-level split-K of 2. Master's path is torch.mm on the NN weight at m=65..128 (nvjet 128x64 splitK 20.2 us + cublasLt splitKreduce 3.0 us) and _bf16_gemm_dispatch_impl at m=256 (nvjet 128x64 TNT 30.4 us). WHY: pooler (n 4096, k 8192) and unpooler (n 8192, k 4096) do the same FLOPs and bytes, but cuBLAS runs the long-K, narrow-N pooler worse. At m=256 the pooler takes 31.3 us (128x64 tiles, about 550 TFLOP/s) while the unpooler takes 24.5 us (128x128 tiles); at m=128 it is 22.9 vs 18.9 us. Two K-halves of the pooler have the unpooler's tile count: n 4096 x k 4096 x 2 batches gives the same 128 CTAs of 128x128 (m=256) or 128x64 (m=128) tiles, each over k=4096, as the unpooler's one GEMM. CHANGE: xh = x.view(m, 2, 4096).permute(1, 0, 2), a strided view [2, m, 4096] with strides (4096, 8192, 1) and no copy. For the weight, either candidate T, a TN view of state['pooler'] as [2, 4096, 4096] with strides (4096, 8192, 1) used transposed, or candidate N, the existing NN weight state['pooler_t'].view(2, 4096, 4096) (contiguous halves, no new memory). Then p = torch.bmm(xh, w_halves) into [2, m, 4096] bf16, and out = torch.add(p[0], p[1]): two graph nodes, with the add reading 4 MB and writing 2 MB at m=256. Precision: each half rounds once to bf16 (about 0.2% per element, L2 well under 1%). The single-nonzero rows stay exact, because the other half is exactly 0. Selector (copy restack-unpooler-slice-cluster-dsmem-m8's model selector, LESSONS 04:36Z): in prepare(), capture each candidate and master's path for the same synthetic x as CUDA graphs, replay with a 256 MB L2 flush before each replay, interleave 41 reps, subtract an empty-graph baseline, print both medians per (layer, m), and route the new path only if it WINS by >= 1.02x; otherwise print PATH CHOSEN master. Also profile one call of the chosen path with torch.profiler and print and assert the launched kernel names (LESSONS: PATH CHOSEN proves the selector, not the kernel). The profiler assertion must show exactly one nvjet/cutlass GEMM kernel plus one elementwise add, and no copy kernel; if torch inserts a contiguous() copy of xh, print it and keep master. Every other layer and m stays byte-identical.

## What changed, and where
Base: worktree submission.py blob equals master's f919748b, so no merge was needed. Edited only muse_bf16_proj/submission.py (+181/-5).
- New `_khalf_forward` (bmm of the two K halves of x, then torch.add) with candidates T (TN view of state['pooler']) and N (pooler_t halves, no new copy). Reuses pooler_t.
- `_khalf_select(state)` runs once, on state A only, at the end of `_prepare_state(flags=None)`. For m=128 and 256 it captures master's NN torch.mm and both candidates as graphs and checks rel L2 and the profiler kernel names (exactly one GEMM and one elementwise add, no copy, memset or splitK reduce). It replays interleaved with a 256 MB flush, 41 reps, empty-graph baseline subtracted, and routes a candidate only at master/cand >= 1.02. Prints both medians, IQR and kernel names even when master is kept, plus xh.stride() and a PATH CHOSEN line per m.
- Review fixes: (1) re-baselined: the selector compares against master's actual NN torch.mm at both m. (2) New flag `pooler_khalf_ok`, plus a per-m choice dict `pooler_khalf`. `_ab_component(layer, m, state)` returns it BEFORE the nn_ok and nn_p256_ok ranges, only for m in the chosen dict. A miss turns off only that flag, so master's NN path stays and unpooler m=128 NN is untouched. (3) The choice is made once on A and inherited by B via `_ab_flags`. (4) Final flags and a `FINAL PATH pooler m=..` line per m are printed in prepare(). (5) No new weight copy. The per-m 1.02x threshold is applied per m, and the IQR is printed.
- Deviation: the route is exact-m (128, 256) and not the whole 65..256 band, because only those are measured and self-checked. The kernel_fn gate adds one dict lookup keyed on cfg m; there are no prints, state mutation or try/except in kernel_fn.
- Note: if nn_ok or nn_p256_ok turn off later in the A/B check, the baseline shifts (the review flagged this); the attempt log must record all three flags.

## Rules check (each must be true before the run)
- [x] No subprocess/fork/spawn added.
- [x] No triton.autotune; no host sync, print or try/except in kernel_fn (diff checked).
- [x] Nothing derived from inputs cached; no device pointers or tensors in module globals (all in `state`).
- [x] Both states: B inherits the choice through _ab_flags; the A/B graph self-check covers pooler m=128 and 256 for both.
- [x] Precision: two bf16 roundings, below the 6e-3 self-check bound; the selector rejects a candidate above it.
- [x] Any selector failure keeps master's path and never raises.
- [x] No sm_90a PTX/CUDA load_inline added: n/a.
- [x] First-line marker kept; not committed, okc not run.

okc_lint: OK with warnings (only existing prepare()-side host-sync and flush-dtype warnings); py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
Overlaps the cells of the queued bf16-verify32-cutlass-coop-256x128-splitk-stage2 (pooler m=256) and of bf16-verify16-cutlass-m128-onetile-splitk-rewrite (pooler m=128), but this is pure cuBLAS dispatch at about 30 lines; if it wins, the CUTLASS gates measure against it. Not a retry of pooler-nn-layout-widen-m256 (flat), which kept one long-K GEMM.

## Result (after the run)
- job `ce808e5a055f41b4` (submit), status done, passed True, tests 9/9
- speedup 1.2470x, 95% CI [1.2283, 1.2657], score 3873.970827595536 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2490645809967658, "ci": [1.2303286122818142, 1.267800549711717], "job": "ee4f89f2f62c42d3"}; base result: muse_bf16_proj/hypotheses/done/neutral/pooler-nn-layout-widen-m256.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-batched-gemm-m65-m256.result.json`, `muse_bf16_proj/hypotheses/done/neutral/pooler-k-halves-batched-gemm-m65-m256.patch`; branch `hyp/muse_bf16_proj/pooler-k-halves-batched-gemm-m65-m256`
