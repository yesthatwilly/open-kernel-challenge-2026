# muse_bf16_proj · 2026-10-24T164831Z · submit · bf16-cublaslt-algo-pick-in-prepare-m128-m256

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T140216Z-bf16-cublaslt-algo-pick-in-prepare-m128-m256.md` (branch `hyp/muse_bf16_proj/bf16-cublaslt-algo-pick-in-prepare-m128-m256`).

- **Job:** `26a8be34a5eb4426` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `a77cdec`

## Goal
Built on master (658d63b). For the four bf16 layers at exactly cfg m == 128 and cfg m == 256 (loads verify_16 and verify_32; 8 shapes), replace _bf16_gemm_dispatch_impl's cuBLAS call, which takes cuBLASLt heuristic result 0, with a cuBLASLt matmul whose algorithm is chosen ONCE in prepare() by timing the heuristic's other candidates on synthetic inputs, and replayed unchanged in kernel_fn. Mechanism: heuristic 0 is far from the floor at m=128 (pooler nvjet_sm90_tst_64x64_64x13_2x1, no split-K, 26.0 us = 2.6 TB/s vs 14.6 floor; qkv_gate a split-K nvjet 13.2 us plus a separate splitKreduce node 3.0 us vs 8.4 floor; unpooler 20.3 vs 14.6) and nothing has ever asked cuBLAS for its second-best answers; bf16-cublaslt-backend-toggle only switched torch's backend preference, which still takes result 0. Implementation: a small torch.utils.cpp_extension.load_inline C++ module (cpp only, include cublasLt.h, extra_ldflags ['-lcublasLt'], ATen's at::cuda::getCurrentCUDABlasLtHandle() and at::cuda::getCurrentCUDAStream()) with two functions. (a) candidates(n, k, m, max_algos=32, ws_bytes) builds exactly the problem torch issues for out = x @ W.T in column-major terms: D[n, m] = op(A)[n, k] * B[k, m] with A = W (bf16 col-major k x n, ld k, CUBLAS_OP_T), B = x (bf16 col-major k x m, ld k, CUBLAS_OP_N), D bf16 col-major n x m (ld n), CUBLAS_COMPUTE_32F, scale type CUDA_R_32F, alpha 1 beta 0, preference CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 32 MiB; calls cublasLtMatmulAlgoGetHeuristic and returns the returned cublasLtMatmulAlgo_t structs as a CPU int64 tensor [count, 8] (the struct is uint64_t data[8]). (b) run(x, W, out, algo_row, workspace) rebuilds the same descriptors (cached in a static map keyed by (n, k, m), shapes only), copies algo_row into a cublasLtMatmulAlgo_t and calls cublasLtMatmul on the current stream with the prepare()-allocated workspace tensor; no host sync, graph-capturable. prepare(): for each (layer, m) in {qkv_gate, o, pooler, unpooler} x {128, 256} with n the full width, allocate one 32 MiB uint8 workspace per key (state['lt_ws'][key]), get the candidates, and for the incumbent (_bf16_gemm_dispatch_impl) and each candidate: run once on synthetic randn bf16 x and drop any candidate whose output differs from torch's x @ W.T by rel L2 > 1e-3 or that raises; capture a CUDA graph of the call (AGENTS.md: time graph replays, eager misleads) and time 30 replays, each preceded by a write of a 256 MiB scratch buffer (cold L2, as the judge), median via torch.cuda.Event. Keep a candidate only if its median beats the incumbent's by >= 3%; store state['lt_algo'][(layer, m)] = algo row; print the table (layer, m, candidates, incumbent us, best us, chosen index or 'incumbent'). Free the flush buffer and graphs before returning. kernel_fn: for a bf16 layer whose (layer, cfg m) key is in state['lt_algo'] and x is bf16 [m, k] contiguous, out = torch.empty((m, n), bf16) and run(...); everything else keeps its master path unchanged (m <= 32 Triton/cuBLAS, router). Keys are exact m only, so nearby shapes the final review may run fall back to today's cuBLAS path, never to an algo chosen for another m. Rules: the choice depends on the weights' shapes and synthetic data only, is made inside the untimed prepare() (300 s budget), and kernel_fn branches only on cfg; no autotune or timing in kernel_fn, nothing derived from data cached, same bf16-in/fp32-accumulate/bf16-out precision as SGLang's call. Precedent: muse_decode_layer pooler-gemm-splitk-autotune did a prepare()-time graph-replay pick per T and was reviewed and run.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/bf16-cublaslt-algo-pick-in-prepare-m128-m256`, file `muse_bf16_proj/submission.py` only (+334 lines, 0 removed; uncommitted).

- New section (6) after `_swapab_prepare`: constants `LT_*`, the C++ source `_LT_CPP`, and `_lt_build`, `_lt_route_ok`, `_lt_forward`, `_lt_graph_median_us`, `_lt_rel`, `_lt_prepare`, `_lt_pick`. Module docstring gains one paragraph.
- `_LT_CPP` (cpp only, built with `load_inline(name="muse_bf16_lt_algo_v1", functions=["candidates","run"], with_cuda=True, extra_cflags=["-O2"], extra_ldflags=["-lcublasLt"])`):
  - descriptors cached in a static map keyed by (n, k, m): CUBLAS_COMPUTE_32F / CUDA_R_32F scale, TRANSA=T, TRANSB=N; A = W as col-major k x n ld k, B = x as col-major k x m ld k, D = out as col-major n x m ld n, all CUDA_R_16BF. alpha 1, beta 0, C = D.
  - `candidates(n, k, m, max_algos, ws_bytes)` sets PREF_MAX_WORKSPACE_BYTES = 32 MiB, calls cublasLtMatmulAlgoGetHeuristic, returns `[algos int64 [count, 8], info int64 [count, 8]]`. info = (CONFIG_ID, TILE_ID, STAGES_ID, SPLITK_NUM, REDUCTION_SCHEME, workspaceSize, wavesCount*1000, state), -1 for any attribute that cannot be read. A static_assert pins `sizeof(cublasLtMatmulAlgo_t) == 64`.
  - `run(x, w, out, algo_row, ws)` checks dtype, contiguity and shape metadata, memcpys the CPU int64 [8] row into a cublasLtMatmulAlgo_t, and calls cublasLtMatmul with `at::cuda::getCurrentCUDABlasLtHandle()` on `at::cuda::getCurrentCUDAStream()` and the caller's workspace. No sync and no allocation, so it can be captured.
- `_lt_prepare(state)` is called last in `prepare()`. It always sets `state["lt_algo"] = {}`, `state["lt_ws"] = {}` and `state["lt_ext"] = None` first. If the build fails, it prints `[lt] load_inline failed ... master path kept` and returns. For each (layer, m) in {qkv_gate, o, pooler, unpooler} x {128, 256} with n the full weight width, `_lt_pick` does the following:
  - synthetic x: randn, seeded generator 1234, cast to bf16. fp64 reference `x.double() @ w.double().T`.
  - incumbent = `_bf16_gemm_dispatch_impl(x, w, None)`: its rel err vs fp64 is recorded, and it is timed by graph replay.
  - candidate exclusion, each printed with its reason:
    - heuristic state != success
    - **split-K with reduction scheme != COMPUTE_TYPE (2)**. This rejects INPLACE (1) and OUTPUT_TYPE (4), i.e. bf16 reductions, and also NONE or unreadable when splitK > 1.
    - SPLITK_NUM unreadable
    - workspace > 32 MiB
    - raises
    - non-finite output
    - rel L2 vs torch's (incumbent) output > 1e-3
    - **extra guard**: rel L2 vs fp64 > 1.10 x the incumbent's rel L2. This catches any reduced-precision reduction the attributes might not reveal, e.g. opaque nvjet configs.
  - timing (`_lt_graph_median_us`): one CUDA graph per call, 30 replays, each preceded by `flush.zero_()` on a 256 MiB buffer. torch.cuda.Event records bracket the whole `g.replay()`, so split-K reduce nodes are included, as the review asked. The median is taken.
  - a candidate is kept only if `best_median <= 0.97 * incumbent_median`. It is stored as `state["lt_algo"][(layer, m)] = row` (CPU int64 [8]) and `state["lt_ws"][(layer, m)]` (32 MiB uint8). The workspaces of keys that keep the incumbent are freed.
  - a per-key exception prints `[lt] <layer> m=<m>: algo search failed, master path kept`. An exception outside the per-key loop clears all keys (`algo search aborted, master path kept everywhere`). The flush buffer and graphs are freed, then `empty_cache()` runs.
  - printed: per key, the candidate count, the incumbent time and its fp64 error. Per candidate: algo id, tile, stages, splitK, reduction, workspace, waves, then the time or the exclusion reason. A final table has layer, m, cands, valid, incumbent_us, best_us and chosen (`cand i` / `incumbent`). This answers the review's open questions 1 and 2 from the log.
- `kernel_fn`: in the `layer != "router"` branch, before master's `_bf16_gemm_dispatch_impl` line, `lt_key = (layer, cfg["m"])`. If `lt_key in state["lt_algo"]` and `_lt_route_ok`, it returns `_lt_forward(...)`, which does `torch.empty((m, n), bf16)` and then `ext.run`. `_lt_route_ok` checks that cfg n/k equal the weight's full shape and that x is a contiguous bf16 [cfg m, k] whose data_ptr is 256-byte aligned (pointer metadata, not values). Lookups use exact keys only: m=130, 512 and 1000 and every m <= 32 keep master's path unchanged.

Deviations from the spec, and why:
1. An extra precision guard, the fp64 error ratio of 1.10x the incumbent's. Correctness comes first on a scored submit, and the reduction-scheme attribute may not describe nvjet configs.
2. `candidates` returns an info tensor alongside the [count, 8] algo tensor, for the printed attribute table (review open questions).
3. An alignment guard in `_lt_route_ok` (the Lt heuristic's default 256-B alignment assumption). On a miss it falls back to master's path rather than issuing a possibly-invalid algo.
4. Only chosen keys keep their workspace.

The TMA/NN-layout probe and the enumeration of AlgoGetIds/ConfigSetAttribute (open question 1's follow-up) are not done: they are new ideas.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/bf16-cublaslt-algo-pick-in-prepare-m128-m256 diff` (334 insertions, 0 deletions, submission.py only):

- [x] CUDA-graph capturable. kernel_fn's new path is `_lt_route_ok`, which reads cfg, shape, dtype, contiguity and data_ptr alignment only, and `_lt_forward`: `torch.empty` and `ext.run`, which is one cublasLtMatmul on the current stream with a prepare()-allocated workspace and a host-side 64-byte memcpy of a CPU algo row. There is no `.item()`/`.cpu()`/sync/print in the timed path. Every sync, `.item()` and `.tolist()` in the diff is in `_lt_prepare`, `_lt_pick` or `_lt_graph_median_us`, all prepare()-only. Branching uses cfg and shapes only.
- [x] Nothing derived from `data` is cached across calls. `state["lt_algo"]`, `lt_ws` and `lt_ext` are written only in prepare(), from the weights' shapes and synthetic randn x. kernel_fn only reads them. The C++ static descriptor map is keyed by (n, k, m) shapes only.
- [x] Weights: unchanged. The bf16 weights are passed to cublasLtMatmul as stored (no repack, no precision change). The router path is untouched.
- [x] Activations bf16 where SGLang keeps bf16: bf16 x in, CUBLAS_COMPUTE_32F accumulate, bf16 out, the same as SGLang's cuBLAS call. Split-K algos that reduce in bf16 (INPLACE / OUTPUT_TYPE) are excluded, and an fp64-error guard backs that up. The KV-cache and query items are n/a: this problem has no attention.
- [x] n/a: no attention in muse_bf16_proj; every weight row is read (full K, full N).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. torch.cuda.Event and CUDAGraph are used, not patched, only inside prepare() for the selection. `load_inline` runs the compiler toolchain, as the accepted muse_attention_decode submission does. Noted as a residual uncertainty in the report.
- [x] All work happens inside the timed call; no output caching or replay. kernel_fn computes every output fresh with cublasLtMatmul. Only the *algorithm choice* (made from synthetic data in untimed prepare()) is reused.
- [x] First line is still `# okc: problem=muse_bf16_proj`. No triton.autotune, and no timing or autotune in kernel_fn.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
warn  host-sync    L845: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L850: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L858: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L866: return ((a64 - b64).norm() / b64.norm().clamp_min(1e-300)).item()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L883: flush = torch.empty(LT_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L899: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L915: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L928: ws = torch.empty(LT_WS_BYTES, dtype=torch.uint8, device=w.device)  (allowed only where SGLang does the same)
warn  host-sync    L936: algo_id, tile, stages, splitk, red, ws_need, waves, st = (int(v) for v in info[i].tolist())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L951: finite = bool(torch.isfinite(out).all().item())  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L956: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L966: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_bf16_proj/submission.py
```

None of the warnings needs a fix. Every host-sync warning is in prepare()-only helpers. The two uint8 dtype warnings are the L2 flush buffer and the cuBLASLt workspace byte buffers, not activations. `python3 -I -m py_compile` passed, and the `__pycache__` it left was removed.

## Risks / what could fail
medium
Rules call for the reviewer: TASK.md and the technical rules forbid caching anything derived from data and work outside the timed call, not prepare()-time selection from synthetic inputs; AGENTS.md itself tells you to tune by timing graph replays, and decode's pooler-gemm-splitk-autotune did exactly this. In my reading it is allowed (choose in prepare from synthetic data, replay the fixed choice; no autotune in kernel_fn). If the user rules otherwise, hold it. Risks: (1) the heuristic may return only near-identical nvjet variants, giving a flat run; that still settles whether cuBLAS can do better than its default here and tells whether m=128 needs a kernel of our own. (2) load_inline + cublasLt link adds ~60-90 s of compile inside the 10-minute run budget. (3) the timing table in prepare() is printed, so every changed row pairs with a non-incumbent pick in log_tail. verify_16 is 1 of 6 loads (~76 us of 5 shapes); ~5 us saved is ~+1% on the headline, so read rows. Overlaps the rewrite pooler-unpooler-m128-tma-wspec-rewrite on two rows; a later combo takes whichever row is faster.

## Result (after the run)
- job `26a8be34a5eb4426` (submit), status done, passed True, tests 9/9
- speedup 1.1879x, 95% CI [1.1701, 1.2057], score 4066.766244682083 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1958517819004566, "ci": [1.1779140051719497, 1.2137895586289633], "job": "eeb9985396764ac2"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-m32-swapab-wgmma-triton.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/bf16-cublaslt-algo-pick-in-prepare-m128-m256.result.json`, `muse_bf16_proj/hypotheses/done/neutral/bf16-cublaslt-algo-pick-in-prepare-m128-m256.patch`; branch `hyp/muse_bf16_proj/bf16-cublaslt-algo-pick-in-prepare-m128-m256`
