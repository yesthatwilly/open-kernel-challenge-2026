# muse_bf16_proj · 2026-10-24T071431Z · check · bf16-cublaslt-backend-toggle

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-23T225303Z-bf16-cublaslt-backend-toggle.md` (branch `hyp/muse_bf16_proj/bf16-cublaslt-backend-toggle`).

- **Job:** `5ed9ace10d9a4f68` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `800f606`

## Goal
A dispatch probe, pure PyTorch, no kernel of ours: in kernel_fn, for the four bf16 layers only, call torch.backends.cuda.preferred_blas_library('cublaslt') immediately before the _bf16_gemm_dispatch_impl(x, W[:n], None) call and restore torch.backends.cuda.preferred_blas_library('cublas') immediately after it (both are host-side settings, so inside the captured graph only the kernel chosen under the Lt backend is recorded; the restore guarantees SGLang's own measurement in the same process is taken with the default backend). ATen's torch.mm for bf16 goes through cublasGemmEx under the default backend and through cublasLtMatmul with PyTorch's own heuristic query and a 32 MB Lt workspace under 'cublaslt'; the two can select different nvjet tiles and splitK factors for the same shape. Router path and prepare() unchanged. Kernels possibly moved: every nvjet_sm90_tst_* + splitKreduce pair on the 20 bf16 shapes; the run's profile shows which shapes changed kernel name at all.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/bf16-cublaslt-backend-toggle`, file `muse_bf16_proj/submission.py` only (1 file, +23/-1).

- `kernel_fn`, non-router branch only: the `_bf16_gemm_dispatch_impl(x, state[layer][: cfg["n"]], None)` call is wrapped as `prev = torch.backends.cuda.preferred_blas_library()`; `try: preferred_blas_library("cublaslt"); return <call>` / `finally: preferred_blas_library(prev)`. Per the review, the previous backend is saved and restored in `finally`; 'cublas' is never hard-coded. The router branch (Triton and fp32 fallback) is unchanged.
- `prepare()`: after the existing router warm-up, one untimed warm-up of the Lt path, using the same save/set/restore in try/finally: `_bf16_gemm_dispatch_impl` on synthetic `randn` bf16 x [1, 8192] against `state["pooler"]`, so the Lt workspace is allocated before capture. It prints `torch.__version__` and the backend value before the set, after the set and after the restore (the review's open question 1). Nothing is set at import, and nothing stays set after prepare() returns.
- Deviations: the spec text said to restore with a hard-coded 'cublas' and to leave prepare() unchanged. Both were superseded by the review and by the dispatch instructions (save/restore prev, plus the prepare warm-up and prints). The warm-up is a single call on one layer (pooler, m=1), as instructed ("one untimed warm-up"), not one call per layer/m. The module docstring was not updated.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the added kernel_fn code is host-side backend get/set only; no `.item()`/`.cpu()`/`synchronize()`/tensor prints, and no allocation or control flow on input values. The prints are in prepare() and print only strings (version and backend enum), no tensors.
- [x] Nothing derived from `data` is cached across calls (`prev` is a local; state is untouched by kernel_fn).
- [x] Weights: unchanged; no new weight storage. The warm-up only reads `state["pooler"]`.
- [x] Activations BF16 where SGLang keeps BF16: the same `_bf16_gemm_dispatch_impl` bf16 GEMM, only the BLAS backend differs. KV cache/queries: n/a, since there is no attention in this problem.
- [x] Attention sees every key: n/a (no attention in this problem).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: `preferred_blas_library` is a documented torch API call, scoped to our own call and restored in `finally`. No loaded module is patched.
- [x] All work happens inside the timed call; no output caching or replay (the prepare warm-up output is discarded).

okc_lint output:
```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/bf16-cublaslt-backend-toggle/muse_bf16_proj/submission.py
```
`python3 -I -m py_compile` passed; the `__pycache__` it left was deleted.

## Risks / what could fail
low
Honest expectation: most shapes flat, because cublasGemmEx already consults the Lt heuristics internally on Hopper; this run exists to settle whether any PyTorch-level BLAS dispatch lever is left before all remaining bf16 effort goes into custom kernels. Reading: compare kernel names per shape against the parity profile; any shape whose nvjet name did not change and whose time moved is noise. Rule edges: a documented torch API call, not a monkeypatch of a loaded module and not harness state; the restore after each call keeps SGLang's baseline measurement unaffected (if the harness measured SGLang under our setting the paired ratios would be confounded, which is why the toggle is scoped to the single call). If the image's torch lacks preferred_blas_library or ignores it for bf16 mm, the run is a no-op at parity, costing one check. No correctness exposure: same cuBLAS math.

## Result (after the run)
- job `5ed9ace10d9a4f68` (check), status done, passed True, tests 9/9
- speedup 1.1505x, 95% CI [1.1333, 1.1678], score 4198.698300468958 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1510295903072447, "ci": [1.133764146452636, 1.1682950341618532], "job": "bd9fa14151644cee"}; base result: muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused-control-splitk-all-m.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-cublaslt-backend-toggle.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-cublaslt-backend-toggle.patch`; branch `hyp/muse_bf16_proj/bf16-cublaslt-backend-toggle`
