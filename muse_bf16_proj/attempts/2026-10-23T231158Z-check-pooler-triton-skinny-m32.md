# muse_bf16_proj · 2026-10-23T231158Z · check · pooler-triton-skinny-m32

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-23T225201Z-pooler-triton-skinny-m32.md` (branch `hyp/muse_bf16_proj/pooler-triton-skinny-m32`).

- **Job:** `9f0ad00bbc874d31` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `d2840d0`

## Goal
For layer == 'pooler' and m <= 32 only, replace the _bf16_gemm_dispatch_impl call in kernel_fn with one Triton kernel, a split-K-free skinny GEMM that streams the 67 MB weight once: grid = (N/BLOCK_N,) with BLOCK_N=32 so 4096/32 = 128 programs (one wave on 132 SMs); each program owns 32 weight rows and loops the full K=8192 in BLOCK_K=256 steps with num_stages=4 (tl.load of a [BLOCK_N, BLOCK_K] bf16 weight tile from the row-major [N, K] weight as stored, and of the x tile [BLOCK_M, BLOCK_K] with BLOCK_M=16 or 32 from m, rows >= m masked to zero), acc += tl.dot(x, w.T) in fp32, then a masked bf16 store of [m, 32] into a torch.empty((m, 4096), bf16) output. prepare() stays as in the parity base (no repack). All other layers and pooler at m >= 128 keep the cuBLAS dispatch. Kernels moved: nvjet_sm90_tst_128x8/16/32_..._splitK_TNT (18.8-20.2 us) plus cublasLt::splitKreduce_kernel (1.6-1.9 us) become one launch; the reduce launch and its graph-node gap go away and the weight tile reads are fully sequential per program. The question the run answers: can a plain Triton full-K stream hit >= 4.0 TB/s (67 MB in <= 17 us) where nvjet splitK sits at 3.6 TB/s.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/pooler-triton-skinny-m32`, one file changed: `muse_bf16_proj/submission.py` (+94/-7; first line unchanged).

- `kernel_fn`: a new first branch, `layer == "pooler" and _pooler_skinny_applies(x, cfg)`, calls `_pooler_skinny_gemm(x, state["pooler"])`. The guard is config/shape only: `cfg m <= 32`, `cfg n == 4096`, `cfg k == 8192`, `x` 2-D of shape `(m, k)`, bf16, unit inner stride. Anything else (pooler at m > 32, any other n/k, qkv_gate, o, unpooler, router) falls through to the untouched parity code (`_bf16_gemm_dispatch_impl` / router hi+lo).
- `_pooler_skinny_kernel` (`@triton.jit(do_not_specialize=["M"])`): grid `(4096 // 32,) = (128,)`, one program per 32 output columns, so one wave on 132 SMs. Tiles `BLOCK_N=32`, `BLOCK_K=256`, `BLOCK_M=16` for m <= 16 and `BLOCK_M=32` for 17 <= m <= 32; `K=8192` is a constexpr so the loop is a static 32 steps. Each step `tl.load`s the `[32, 256]` bf16 weight tile straight from the row-major `[N, K]` weight as stored (unmasked: n and k divide exactly, enforced by the guard) and the `[BLOCK_M, 256]` x tile with rows >= M masked to zero (`other=0.0`), then `acc = tl.dot(x_tile, tl.trans(w_tile), acc)` in fp32. After the loop a single `acc.to(tl.bfloat16)` store masked to rows < M into a `torch.empty((m, 4096), bf16)`. **`num_warps=4`, `num_stages=4`** (per stage 16 KB weight + 8/16 KB x; 96/128 KB smem at 4 stages, 64 KB of weight in flight per SM as the hypothesis states).
- `M` is passed as a runtime int with `do_not_specialize`, so Triton does not mint an `M == 1` or divisible-by-16 variant: exactly two binaries exist (BLOCK_M 16 and 32). Strides are runtime ints (8192 / 8192 / 4096, all 16-divisible, identical between warm-up and the judge's contiguous inputs).
- `prepare()`: unchanged parity body (contiguous bf16 weights, router hi/lo split, no repack), plus `_pooler_skinny_warmup(state["pooler"])`, which launches the kernel on synthetic `torch.zeros((m, 8192), bf16)` for every m in 1..32, so both specializations (and any int specialization Triton might still apply) are compiled before the first timed call; outputs are dropped and nothing derived from inputs is retained.
- Deviations from the spec / review: (1) the reviewer's "consider BLOCK_M=16 for every m <= 32" was not taken: `tl.dot` needs M >= 16, so m = 32 at BLOCK_M=16 would need either a second M tile (a second pass over the 67 MB weight) or a second dot per step; two specializations is the smaller change and both are pre-warmed. (2) `pooler verify_32` (m = 256) is a flat control like verify_16; it stays on cuBLAS by the m <= 32 guard. No autotune; no TMA; no tf32/fp8 anywhere.
- Untested on hardware (no GPU/Triton on this node): syntax checked with `ast.parse`, lint CLEAN. The one Triton-specific thing to watch in the run log is the `tl.trans(w_tile)` operand feeding `tl.dot` inside the pipelined loop; it is the standard pattern (`tl.dot(q, tl.trans(k))` in the Triton attention tutorial) but has not been compiled here.

## Rules check (each must be true before the run)
Verified against `git diff` in the worktree (`muse_bf16_proj/submission.py`, +94/-7):

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints anywhere in the diff; the only allocation is `torch.empty((m, 4096))` from `x.shape`; all branching (`_pooler_skinny_applies`, `block_m`) is on `cfg` fields and `x.shape`/`stride`/`dtype`, never on values.
- [x] Nothing derived from `data` is cached across calls: `kernel_fn` writes no module-level state; `out` is fresh per call and returned; the warm-up tensors in `prepare()` are synthetic zeros that are not kept.
- [x] Weights: `prepare()` still only does `.contiguous()` on the bf16 weights (no repack, no dtype change); the Triton kernel reads `state["pooler"]` as stored; the router hi+lo split is the pre-existing parity code.
- [x] Activations BF16: `x` is loaded as bf16, `tl.dot` is bf16 x bf16 with an fp32 accumulator, the output is rounded to bf16 once at the store, the same chain cuBLAS runs; no tf32/fp8 casts in the diff.
- [x] Attention sees every key: not applicable, this problem is a dense GEMM; the diff computes the full `x @ W.T` over all 8192 K and all 4096 N (no skipping).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: imports are `torch`, `torch.nn.functional`, `triton`, `triton.language`, and the same `sglang` dispatch as before; lint CLEAN.
- [x] All work happens inside the timed call; no output caching or replay: `prepare()` only compiles the kernel on synthetic inputs (allowed untimed setup) and discards the result; every timed call streams the weight and writes a new output.

Lint: `python3 -I .claude/skills/churnsubs/bin/okc_lint.py muse_bf16_proj/submission.py --problem muse_bf16_proj` -> `CLEAN: muse_bf16_proj/submission.py`

## Risks / what could fail
medium
Risk: a full-K loop has only num_stages x BLOCK_N x BLOCK_K x 2 = 64 KB in flight per SM; if that is not enough memory-level parallelism to beat nvjet's splitK (which uses 4x more CTAs), the pooler shapes get slower, not faster, and the per-shape rows will show it directly (nothing else moves). Correctness: bf16 inputs, fp32 accumulate, one bf16 rounding at the store, same as cuBLAS; the m=7 pooler test exercises the masked-row path. CUDA graph: fixed-shape torch.empty only, no syncs. Compile: 3 BLOCK_M specializations, seconds. If faster, qkv-gate-triton-skinny-m32, unpooler-full-triton-skinny-m32, o-triton-skinny-m32 and pooler-skinny-tiled-weight-layout stack on it; if slower, they should be withdrawn or retuned (BLOCK_N=16 with 256 programs, or num_stages=6) before running.

## Result (after the run)
- job `9f0ad00bbc874d31` (check), status done, passed True, tests 9/9
- speedup 0.9983x, 95% CI [0.9833, 1.0132], score 4839.2372799770155 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 0.9020752041800139, "ci": [0.8885440761173137, 0.9156063322427139], "job": "366189f0f1114da1"}; base result: muse_bf16_proj/hypotheses/done/gain/sglang-gemm-dispatch-parity.result.json
- **outcome: gain** (gain)
- files: `muse_bf16_proj/hypotheses/done/gain/pooler-triton-skinny-m32.result.json`, `muse_bf16_proj/hypotheses/done/gain/pooler-triton-skinny-m32.patch`; branch `hyp/muse_bf16_proj/pooler-triton-skinny-m32`
