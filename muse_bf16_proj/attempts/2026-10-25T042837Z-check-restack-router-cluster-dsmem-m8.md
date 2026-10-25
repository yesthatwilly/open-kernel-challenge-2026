# muse_bf16_proj · 2026-10-25T042837Z · check · restack-router-cluster-dsmem-m8

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T042040Z-restack-router-cluster-dsmem-m8.md` (branch `hyp/muse_bf16_proj/restack-router-cluster-dsmem-m8`).

- **Job:** `d323ce8ef6a54fde` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `041d99b`

## Goal
Built on master (b99b1dd, the landed restack bf16-restack-m16-swapab-router-swapab-m32, 1.2215x; check 1e90292c7b954d96). Port ONLY the router cluster kernel from branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (job dc5d03c4: 16 clusters x 8 CTAs, swap-AB mma.sync, split-K reduced through DSMEM, torch.empty output, no memset, one node): its CUDA source string, the load_inline build helper (keep TORCH_CUDA_ARCH_LIST=9.0a around the build and -lineinfo; if extra_cuda_cflags are overridden they must contain -std=c++20, replacing any other -std), the prepare() self-check and the state['router_cluster_ok'] constant. Compile only the NT=8 instance. In the restack's kernel_fn router branch, insert before `path = _router_path(...)`: if state['router_cluster_ok'] and 1 <= x.shape[0] <= 8 and cfg n == 256 -> return the cluster forward. m 9..32 keeps the restack's _router_forward (swapab; it beat the cluster at m=16, 5.47 vs 5.87), m > 32 keeps _router_hilo_forward. Self-check at m in {1, 5, 8} only (same bounds as the source branch). Everything else byte-identical to the restack. At harvest confirm the cluster kernel name on router decode_1/verify_1 in the profile and the restack's _bf16_swapab_kernel/_router_swapab_kernel on the other rows (the 03:41Z harvest found nine runs that were really master).

## What changed, and where
Edited only `muse_bf16_proj/submission.py` in the worktree (450 insertions, 0 deletions, on the restack b99b1dd content).

What changed:
- Added `import os, time`, and a new section "(7) router, 1 <= m <= 8" before `prepare()`: the CUDA source string (`_RC_CPP`, `_RC_CUDA`), `_build_router_cluster()` (load_inline, `TORCH_CUDA_ARCH_LIST=9.0a` set around the build and restored in `finally`, arch flags printed, `-O3 -lineinfo`; no `-std` override, so the c++20 note does not apply), `_RC_EXT` built at import, `_router_cluster_route`, `_router_cluster_forward` (`torch.empty` output, one launch, no memset), `_rc_check`, `_router_cluster_prepare`.
- `prepare()`: `_router_cluster_prepare(state)` called right after `_router_prepare(state)` (so `router_stages` exists); sets the constant `state["router_cluster_ok"]`.
- `kernel_fn` router branch: before `path = _router_path(...)`, `if _router_cluster_route(x, cfg, state): return _router_cluster_forward(x, state)`. The route requires `router_cluster_ok`, 1 <= m <= 8, cfg n == 256, k == 8192, bf16 CUDA x, stride(1) == 1, (m == 1 or stride(0) % 8 == 0), 16 B alignment. m 9..32 stays `_router_forward` (swapab), m > 32 stays hilo. Everything else is untouched.

Reviewer instructions applied (they override the spec):
- Gated to m <= 8 (C++ side also TORCH_CHECKs m <= 8); only `router_kernel<8,8>` is instantiated. The CS=4 variant and NT=16/32 instances are dropped.
- Occupancy check kept: `router_info()` returns `cudaOccupancyMaxActiveClusters`; if CS=8 NT=8 reports < 16 active clusters, `router_cluster_ok` stays False and the swapab router runs (no CS=4 fallback).
- Stride/alignment preconditions kept in both the Python gate and the C++ TORCH_CHECKs.
- Self-check bounds kept (rel L2 <= 2e-5, worst row <= 1e-4, one-hot row within 1e-6 of hi+lo reconstruction, run 2 bit-identical), against `x.double() @ router.double().T`, at m in {1, 5, 8} (the source used 1, 8, 9, 16, 17, 32), each launch announced with a flushed print before and `torch.cuda.synchronize()` plus flushed print after.
- mode=0 diagnostic NOT ported: the `mode` kernel/launch argument is removed entirely, the kernel always does the DSMEM reduce with both barrier.cluster. The `_rc_time` timing diagnostics are not ported either.
- No memset on the m <= 8 path: output is `torch.empty`, the kernel writes every element (all m rows x 256 cols, one CTA per column pair), no `torch.zeros`/`zero_()` in the forward.
- PATH CHOSEN lines are printed per m = 1..8 and for the other ranges. Any build error, occupancy miss, self-check miss or exception lands on the swapab path and never raises. No subprocess/fork. No mbarrier or spin waits exist in the kernel (cp.async groups plus hardware cluster barriers), so nothing to bound. The kernel uses only baseline sm_90 PTX (mma.sync, cp.async.cg, mapa, barrier.cluster), so no `__CUDA_ARCH_FEAT_SM90_ALL` guard is needed; 9.0a is still set per the brief.
- Nothing was run (no GPU on this node); only py_compile and okc_lint.

Harvest criteria:
- Kill / promote on router `verify_1` (m=8): it must be below the restack's 5.47 us row (job 1e90292c7b954d96), expected about -7% (cluster read 5.08 us).
- Count router `decode_1` (m=1) as a win only if its per-shape speedup is above 1.954 (restack 1.925 plus 1.5% noise); a 5.30-5.38 us row is a tie, not a loss.
- In the profile confirm `router_kernel<8,8>` with NO memset node (no `vectorized_elementwise`) on router decode_1 and verify_1; `_router_swapab_kernel` on decode_16/decode_32; `_router_hilo_kernel` on m=128/256; `_bf16_swapab_kernel` on the restack's bf16 rows.
- Confirm qkv_gate decode_1 is near 12.2 us (13.5 us would mean a pre-restack build, as in the nine 03:41Z mis-builds).
- Expected headline about 1.2237 within CI; judge on the router rows only.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn's cluster path is `torch.empty` plus one `cudaLaunchKernelEx` on the current stream; no `.item()`/`.cpu()`/`synchronize()`/prints in kernel_fn (the only synchronize is in `_rc_check`, called from prepare()); no data-dependent control flow (gate uses cfg, shapes, strides, pointer alignment, and a prepare() constant).
- [x] Nothing derived from `data` is cached across calls: only `state["router_cluster_ok"]`, derived from synthetic data and weights.
- [x] Weights: hi/lo split unchanged from the restack, built in `prepare()` from the weights only; BF16 stays BF16.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: n/a, no attention or KV cache; router uses bf16 x with a bf16 hi+lo weight pair and fp32 accumulation, which the rules allow.
- [x] Attention sees every key: n/a, no attention.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocess/fork/spawn: checked with grep of the diff; load_inline runs nvcc in-process via torch's build (same as the source run dc5d03c4, which passed the judge).
- [x] All work happens inside the timed call; no output caching or replay: the output is a fresh `torch.empty` filled by the kernel on every call.

okc_lint output:
```
warn  host-sync    L1377: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_bf16_proj/submission.py
```

## Risks / what could fail
medium
Kill: router decode_1 and verify_1 rows not both below the restack's 1e90292c rows (5.38/5.47) with the cluster kernel in the profile. Expected headline ~1.224 (03:41Z substitution). Risk: the load_inline build adds ~40 s and an illegal-instruction fault would poison the context; the kernel already ran clean on the judge, so run as a check.

## Result (after the run)
- job `d323ce8ef6a54fde` (check), status done, passed True, tests 9/9
- speedup 1.2199x, 95% CI [1.2016, 1.2382], score 3960.107567010976 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "1e90292c7b954d96"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/restack-router-cluster-dsmem-m8.result.json`, `muse_bf16_proj/hypotheses/done/neutral/restack-router-cluster-dsmem-m8.patch`; branch `hyp/muse_bf16_proj/restack-router-cluster-dsmem-m8`
