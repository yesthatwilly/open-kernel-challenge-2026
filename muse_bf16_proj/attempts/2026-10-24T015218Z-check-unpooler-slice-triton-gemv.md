# muse_bf16_proj · 2026-10-24T015218Z · check · unpooler-slice-triton-gemv

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-23T225201Z-unpooler-slice-triton-gemv.md` (branch `hyp/muse_bf16_proj/unpooler-slice-triton-gemv`).

- **Job:** `4c2b113f81434570` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `7c05884`

## Goal
For layer == 'unpooler' and m <= 8 only (the sliced case: n=1024, k=4096, an 8.4 MB weight read), replace the _bf16_gemm_dispatch_impl(x, W[:1024], None) call in kernel_fn with one Triton launch and no split-K reduce: grid = (1024/BLOCK_N,) with BLOCK_N=8, i.e. 128 programs; each program streams its 8 weight rows over the full K=4096 in BLOCK_K=512 steps (num_stages=4), loads the x tile [16, BLOCK_K] with rows >= m masked, accumulates in fp32 with acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2) (FMA broadcast; tl.dot needs BLOCK_N >= 16 and this shape is launch-bound, not math-bound), and stores the [m, 8] bf16 slab into a torch.empty((m, 1024), bf16). prepare() unchanged; W[:1024] of the row-major weight is already contiguous. Kernels moved: nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT (4.0-4.3 us) + cublasLt::splitKreduce_kernel (1.5-1.6 us) and the gap between them become one kernel; the reported 6.9-7.1 us is 4x the 1.7 us floor, and the goal is 2 launches to 1 on a sub-10 us shape. Unpooler at m >= 16 (n=8192) and every other layer keep the cuBLAS dispatch.

## What changed, and where
Changed only `muse_bf16_proj/submission.py` (+66 lines, nothing removed):

- **Fast path in `kernel_fn`.** When `layer == "unpooler"` and `x.shape[0] <= 8` and `cfg["n"] == 1024` and `cfg["k"] == 4096`, the call goes to `_unpool_slice_gemv(x, state["unpooler"])`. Every other case falls through to the unchanged `_bf16_gemm_dispatch_impl` (cuBLAS) or router path. That includes unpooler at m >= 16 (n = 8192) and the m = 130 / m = 1000 tests.
- **New kernel `_unpool_slice_gemv_kernel` (Triton).**
  - Grid is `(1024 / BLOCK_N,)` = **128 programs**, with BLOCK_N = 8. There is one launch and no split-K reduce.
  - Each program streams its 8 contiguous weight rows (64 KB) over the full K = 4096. It reads the full `[8192, 4096]` row-major weight through its first 1024 rows, so no slice copy or repack is made.
  - Per step: `acc[BLOCK_M, 8] += tl.sum(x[:, None, :] * w[None, :, :], axis=2)`. Loads are bf16 and are upcast to fp32. Accumulation is fp32 and the store is bf16.
  - x rows `>= m` are masked: loads return 0 and those rows are not stored.
  - The output is `torch.empty((m, 1024), bf16)`.
  - Launch settings: num_warps = 4, num_stages = 4. There is no autotune.
- **Tile tune (inside the stated envelope, following the reviewer):**
  - **BLOCK_M = `triton.next_power_of_2(m)`** (1, 2, 4 or 8) instead of the fixed 16. decode_1 therefore does no wasted rows, and the FMA form no longer materialises a `[16, 8, 512]` fp32 product.
  - **BLOCK_K = `max(128, 512 // BLOCK_M)`**, which gives 512 at m = 1, 256 at m = 2 and 128 at m = 4..8. The per-CTA product stays at 4K-8K fp32 values (32-64 registers per thread at 4 warps), which avoids the spill the review warned about.
  - The spec's BLOCK_K = 512 is therefore kept only at m = 1. At m = 8 the K loop runs as 32 x 128-wide chunks.
  - BLOCK_N, the grid, num_stages and the no-split-K design are as specified.
- **`prepare()`.** The weights are untouched. The only addition is a pre-warm: each m in 1..8 runs once on synthetic `torch.zeros((m, 4096), bf16)` against the real weight tensor, so that every BLOCK_M / BLOCK_K and `M == 1` specialization is compiled and loaded before any timed call or graph capture. The outputs are discarded and nothing input-derived is kept.
- **Deviations.**
  - BLOCK_M and BLOCK_K were tuned as described above.
  - `x.contiguous()` was added, which is a no-op on judge inputs. The kernel assumes a unit k-stride, and this keeps any non-contiguous caller correct.
  - Not done: the tl.dot alternative and BLOCK_N = 4 (256 programs). These are left as follow-up tunes if the result lands between cuBLAS and the floor.
- **Not runnable here** (no GPU or Triton on this node). I checked the syntax with `ast.parse` and checked the logic by reading it. The first `okc check` is the real compile test.

## Rules check (each must be true before the run)
Verified against `git diff` in the worktree (hyp/muse_bf16_proj/unpooler-slice-triton-gemv):

- [x] CUDA-graph capturable: the new path has no `.item()`, `.cpu()`, sync or prints. Branching is only on `x.shape[0]`, `cfg["n"]`, `cfg["k"]` and `layer`. Tiles derive from m (a shape). The output is a fixed-shape `torch.empty`. The kernels are pre-compiled in `prepare()`, so nothing compiles under capture.
- [x] Nothing derived from `data` is cached across calls: there are no module-level containers. The warm-up in `prepare()` uses synthetic zeros, and its outputs are discarded.
- [x] Weights: the unpooler is read as stored (bf16, row-major, first 1024 rows, in place). There is no new repack or precision change, and the router hi/lo is unchanged.
- [x] Activations stay BF16 where SGLang keeps them: x is read as bf16 and upcast in registers, with fp32 accumulation and bf16 output, the same contract as cuBLAS.
- [x] Attention sees every key: N/A, since this problem has no attention.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the only new imports are `triton` and `triton.language`.
- [x] All work happens inside the timed call, with no output caching or replay: every call launches the kernel on the fresh x and writes a fresh output.

Lint: `python3 -I .claude/skills/churnsubs/bin/okc_lint.py muse_bf16_proj/submission.py --problem muse_bf16_proj` -> `CLEAN: submission.py`

## Risks / what could fail
medium
Risk: 128 programs each reading 64 KB may not pull 8.4 MB faster than cuBLAS's 4-way splitK; the shape is so small that launch latency (~2-3 us per graph node) dominates, so the win mostly comes from dropping the second node, which is why it should still be faster even at equal bandwidth. The FMA-broadcast form keeps register use at 16x8x512 fp32 = 64K values per program if written naively; tile it as BLOCK_K=128 inner chunks or use tl.dot with BLOCK_N=16 (64 programs) if Triton spills; either is a tune, not a new idea. Tests touched: m=5 unpooler and m=130/m=1000 unpooler (the latter two are n=8192, untouched path). Correctness: same bf16-in/fp32-acc/bf16-out as cuBLAS. CUDA graph: fixed-shape torch.empty only.

## Result (after the run)
- job `4c2b113f81434570` (check), status done, passed True, tests 9/9
- speedup 0.8801x, 95% CI [0.8669, 0.8933], score 5488.975268284764 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 0.9020752041800139, "ci": [0.8885440761173137, 0.9156063322427139], "job": "366189f0f1114da1"}; base result: muse_bf16_proj/hypotheses/done/gain/sglang-gemm-dispatch-parity.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/unpooler-slice-triton-gemv.result.json`, `muse_bf16_proj/hypotheses/done/regression/unpooler-slice-triton-gemv.patch`; branch `hyp/muse_bf16_proj/unpooler-slice-triton-gemv`
