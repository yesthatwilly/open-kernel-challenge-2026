# muse_bf16_proj · 2026-10-25T110032Z · submit · restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128

Hypothesis: `muse_bf16_proj/hypotheses/done/neutral/2026-10-25T054102Z-restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128.md` (branch `hyp/muse_bf16_proj/restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `9adc6715`

## Goal
Promote check-proven gain restack-three-cell-wins-router-unpooler-cluster-m8-nn-m128 (1.2366x [1.2181,1.2552], check job 84f17adc3f074a33) to the board.

Built on current master (b99b1dd, bf16-restack-m16-swapab-router-swapab-m32, check 1e90292c7b954d96, 1.2215). Stack the three row wins that were each checked alone on b99b1dd, each gated to exactly the (layer, m) cells it won; nothing else changes and no constant is retuned. Patches live in muse_bf16_proj/hypotheses/done/neutral/. (1) Unpooler-slice cluster at m<=8: churn 3-way-applies restack-unpooler-slice-cluster-dsmem-m8.patch (the depends_on parent, job e557d92969864449) as is: ouns_kernel NT=8 via load_inline (ext name muse_bf16_unpool_slice_cluster_v3, TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, -std=c++20), self-check at m {1,5,8}, the cold-flush graph-replay gate (41 reps, >=1.02x) that sets state['ouns_cl'], and the kernel_fn route "layer == 'unpooler' and n == 1024 and 1 <= m <= 8 and state['ouns_cl']" placed before the swapab16 check. o is NOT routed. (2) Router cluster at m<=8: then apply restack-router-cluster-dsmem-m8.patch (job d323ce8ef6a54fde) with git apply -3 and resolve by hand: the router_kernel<8,8> source and _RC_EXT build (its own ext name, distinct from the unpooler one), _router_cluster_prepare(state) called after _router_prepare in prepare() with its self-check at m {1,5,8} setting state['router_cluster_ok'], and the kernel_fn router branch "state['router_cluster_ok'] and 1 <= m <= 8 and n == 256 -> cluster forward" inserted before path = _router_path(...); m 9..32 keeps swapab, m > 32 keeps hilo. (3) NN layout at m=65..128: then apply restack-pooler-unpooler-nn-layout-m128.patch (job 92cd4e758a924ca0): prepare stores state['pooler_t'] and state['unpooler_t'] = W.t().contiguous() (lossless [K, N] bf16 copies; originals stay), and kernel_fn, just before the "if layer != 'router'" cuBLAS fallback, returns torch.mm(x, state[layer + '_t']) when layer in {pooler, unpooler}, 65 <= m <= 128 and n is the full width; m >= 129 keeps TN, m <= 32 keeps the swap-AB paths. The three edits touch disjoint (layer, m) cells; the only expected conflicts are adjacent insertions in prepare() and kernel_fn. Keep all three, in the order unpooler-cluster check, router-cluster check, NN branch, then master's existing fallbacks. STATE RULE (the judge prepares two states, setup copies 2, and rotates them): every device buffer, flag and handle lives in the state dict built by prepare(). Neither component patch holds device pointers in module or C++ static globals: the only C++ statics are host-side "static bool done" one-shot cudaFuncSetAttribute flags, and _RC_EXT / _OUNS_EXT are compiled-module handles. Keep it that way: if the merge introduces any module-level or static global holding a device pointer, tensor, workspace or TMA descriptor (the "static CUtensorMap g_tb" pattern from the DeepGEMM rewrite), move it into state and pass it as a kernel argument. Gate and self-check scratch (the 256 MB flush buffer, synthetic x) is local to prepare and released before it returns. Hard rules: kernel_fn stays CUDA-graph capturable, with no host syncs, no allocation keyed on input values beyond torch.empty of the output, no data-dependent control flow, no prints, no try/except, no per-call mutable state and nothing cached from inputs; routing reads only shapes and constants fixed in prepare; bf16 output with fp32 accumulation as master; every key attended (n/a, dense GEMM); no child processes (load_inline in-process only). Self-check before dispatch: prepare two states, capture a graph of kernel_fn per state at m in {1, 8, 128} for router, unpooler and pooler, and replay them alternately (A, B, A, B) with x refilled in place each time, comparing every output to a torch.mm reference; on any miss the affected component's flag falls back to master's path. At harvest, confirm in the profile: router_kernel<8,8> on router decode_1/verify_1, ouns_kernel on unpooler decode_1/verify_1, nvjet *_NNT on pooler/unpooler verify_16, and master's _bf16_swapab_kernel / _router_swapab_kernel / hilo on every other row (the 03:41Z harvest found runs that were really master).

## What changed, and where
All edits in the worktree's muse_bf16_proj/submission.py (diff vs master blob: +1226/-6).
- Unpooler-slice cluster: as 3-way-applied by the claim (ext muse_bf16_unpool_slice_cluster_v3, 9.0a, -lineinfo, gate sets state['ouns_cl'], route before swapab16).
- Router cluster: hand-merged from restack-router-cluster-dsmem-m8.patch as section (8): router_kernel<8,8> source, _RC_EXT (ext muse_bf16_router_cluster_m8_v1, distinct), _router_cluster_prepare after the ouns prepare, route before `_router_path`.
- NN layout: state['pooler_t'/'unpooler_t'] in prepare, branch just before the cuBLAS fallback; constants NN_GATED_*.
- Review fixes applied (review wins over spec where they differ):
  - distinct load_inline ext names (see above).
  - per-phase prepare timings printed (`[prepare-time] ...` lines per state, plus totals).
  - dispatch flags (`ouns_cl`, `router_cluster_ok`, new `nn_ok`) are read from `state` only; the `_OUNS_EXT is None` / `_RC_EXT is None` module checks were removed from the route functions (flags are already off when a module is missing).
  - `[ouns] PATH CHOSEN` and a single `[router_cluster] PATH CHOSEN` line print once per prepare() (the second A/B state inherits flags and prints none). The router patch's per-m PATH CHOSEN lines were collapsed to one.
  - ouns gate: finally block drops the 256 MB flush buffer, baseline graph and scratch; prepare() ends with synchronize + empty_cache.
  - ouns cudaOccupancyMaxActiveClusters (cl 8 and 4) and the router occupancy (needs 16) are reprinted at the end of prepare, after both modules are loaded.
  - kernel_fn has no try/except, print or sync.
- Added (spec + brief): `_ab_selfcheck` in prepare: builds a second state (`_prepare_state(weights, flags=...)`, inherits A's flags, skips gates), captures kernel_fn per state at router/unpooler/pooler x m {1,8,128} (unpooler m<=8 at n=1024, m=128 at n=8192), replays A,B,A,B with x refilled in place, compares to fp64 (router, 5e-5) / fp32 (bf16 layers, 6e-3) references, flushed prints around each capture. A miss or exception switches off that cell's component (router_cluster_ok / ouns_cl / nn_ok) in both states. Cells served by master's path are checked but have nothing to switch off. prepare() was split into `_prepare_state` + `prepare`.
- Deviations: no -std=c++20 added to either build (the ouns build compiled without it in e557d929 and the flags do not override gencode); no sm_90a-only PTX is used (mma.sync, cp.async.cg, cluster, mapa), so no __CUDA_ARCH_FEAT_SM90_ALL guard; arch list still pinned to 9.0a around both builds and restored. Router self-check keeps the patch's own m {1,5,8} check in addition to the A/B check.
- Not run (no GPU on this node). Main risks: prepare time (two cold load_inline builds + second state build + 18 graph captures against 300 s), second state memory.

## Rules check (each must be true before the run)
- [x] first line `# okc: problem=muse_bf16_proj` kept
- [x] no child processes (grep for subprocess/fork/multiprocessing/Popen/os.system: none); load_inline in-process only
- [x] no host sync, allocation on input values, print or try/except in kernel_fn (diff checked; only torch.empty of the output and cfg/shape branches)
- [x] no device pointers, tensors, TMA maps or workspaces in module/C++ globals: only `static bool done` flags and _RC_EXT/_OUNS_EXT module handles; everything else in state
- [x] nothing derived from inputs cached; routing reads cfg, shapes, strides, alignment and prepare() constants
- [x] precision: router fp32 output from hi+lo bf16 with fp32 accumulation as master; bf16 layers bf16 out, fp32 accumulation; NN copies are lossless
- [x] weights not stored at lower precision (pooler_t/unpooler_t are exact bf16 transposes)
- [x] CUDA-graph capturable kernel_fn; self-check captures graphs of it per state
- [x] every key attended: n/a, dense GEMM
- [x] sm_90a PTX guard: n/a, no wgmma/setmaxnreg in the CUDA code; arch list pinned anyway
- [x] mbarrier/spin waits: n/a, none in either CUDA kernel (cluster barriers only)
- [x] build error / self-check miss / unsupported shape falls to master's path without raising

okc_lint output: `OK with warnings: muse_bf16_proj/submission.py` (13 warnings: host-sync in prepare-only code, uint8 flush buffer in the gate; no errors). py_compile: ok, __pycache__ removed.

## Risks / what could fail
medium

## Result (after the run)
(pending)
