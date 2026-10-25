---
problem: muse_bf16_proj
slug: restack-router-cluster-dsmem-m8
created: 2026-10-25T042040Z
base: master
area: router GEMM (m<=8)
change: >
  Built on master (b99b1dd, the landed restack bf16-restack-m16-swapab-router-swapab-m32, 1.2215x;
  check 1e90292c7b954d96). Port ONLY the router cluster kernel from branch
  hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (job dc5d03c4: 16 clusters x 8 CTAs,
  swap-AB mma.sync, split-K reduced through DSMEM, torch.empty output, no memset, one node): its
  CUDA source string, the load_inline build helper (keep TORCH_CUDA_ARCH_LIST=9.0a around the
  build and -lineinfo; if extra_cuda_cflags are overridden they must contain -std=c++20, replacing
  any other -std), the prepare() self-check and the state['router_cluster_ok'] constant. Compile
  only the NT=8 instance. In the restack's kernel_fn router branch, insert before `path =
  _router_path(...)`: if state['router_cluster_ok'] and 1 <= x.shape[0] <= 8 and cfg n == 256 ->
  return the cluster forward. m 9..32 keeps the restack's _router_forward (swapab; it beat the
  cluster at m=16, 5.47 vs 5.87), m > 32 keeps _router_hilo_forward. Self-check at m in {1, 5, 8}
  only (same bounds as the source branch). Everything else byte-identical to the restack. At
  harvest confirm the cluster kernel name on router decode_1/verify_1 in the profile and the
  restack's _bf16_swapab_kernel/_router_swapab_kernel on the other rows (the 03:41Z harvest found
  nine runs that were really master).
expected:
  - shapes: layer=router load=decode_1 m=1
    direction: faster
    why: restack row 5.38 = swapab 3.1 + memset 0.8 + node; cluster read 5.24 on master (-8.7%, z -16)
  - shapes: layer=router load=verify_1 m=8
    direction: faster
    why: restack row 5.47; cluster read 5.08 on master (-14%, z -31)
  - shapes: layer=router load=decode_16 m=16
    direction: flat
    why: keeps restack swapab
  - shapes: layer=router load=decode_32 m=32
    direction: flat
    why: keeps restack swapab
  - shapes: layer=router load=verify_* m=12*
    direction: flat
    why: m=128 keeps hilo
  - shapes: layer=router load=verify_32 m=256
    direction: flat
    why: keeps hilo
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=o *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=unpooler *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T042320Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T042320Z
base_sha: 459e84861e1a2c35ffa8148cd1fe09f5ead28416
base_blob: 970b415b746fa03cb92e419d270be92588273178
branch: hyp/muse_bf16_proj/restack-router-cluster-dsmem-m8
worktree: .worktrees/muse_bf16_proj/restack-router-cluster-dsmem-m8
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T042837Z
impl_commit: 041d99b4bbca8464d43501a630a399e2afc877a6
submission_sha256: 94473893a8f96855d29583bcb00d85036237fb6d9b0558d36352d5cdb2b395e6
norm_sha256: 5516943688ba4ac5ae613c7840bd3c1e94ad07934c5fc33f5a5c31160314613b
submission_blob: c857e5300e8435a8deb056ba49439d4d23ea126b
lint_warnings: 1
attempt_file: 2026-10-25T042837Z-check-restack-router-cluster-dsmem-m8.md
prepared_at: 2026-10-25T042837Z
job: d323ce8ef6a54fde
dispatched_at: 2026-10-25T042850Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.2198658542111795
speedup_ci: [1.2015678663980118, 1.238163842024347]
harvested_at: 2026-10-25T043356Z
best_at_harvest: 1.2215006063849365
promotion: null
---

# restack-router-cluster-dsmem-m8

## Change

Built on master (b99b1dd, the landed restack bf16-restack-m16-swapab-router-swapab-m32, 1.2215x; check 1e90292c7b954d96). Port ONLY the router cluster kernel from branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (job dc5d03c4: 16 clusters x 8 CTAs, swap-AB mma.sync, split-K reduced through DSMEM, torch.empty output, no memset, one node): its CUDA source string, the load_inline build helper (keep TORCH_CUDA_ARCH_LIST=9.0a around the build and -lineinfo; if extra_cuda_cflags are overridden they must contain -std=c++20, replacing any other -std), the prepare() self-check and the state['router_cluster_ok'] constant. Compile only the NT=8 instance. In the restack's kernel_fn router branch, insert before `path = _router_path(...)`: if state['router_cluster_ok'] and 1 <= x.shape[0] <= 8 and cfg n == 256 -> return the cluster forward. m 9..32 keeps the restack's _router_forward (swapab; it beat the cluster at m=16, 5.47 vs 5.87), m > 32 keeps _router_hilo_forward. Self-check at m in {1, 5, 8} only (same bounds as the source branch). Everything else byte-identical to the restack. At harvest confirm the cluster kernel name on router decode_1/verify_1 in the profile and the restack's _bf16_swapab_kernel/_router_swapab_kernel on the other rows (the 03:41Z harvest found nine runs that were really master).

## Expected

- `layer=router load=decode_1 m=1` faster: restack row 5.38 = swapab 3.1 + memset 0.8 + node; cluster read 5.24 on master (-8.7%, z -16)
- `layer=router load=verify_1 m=8` faster: restack row 5.47; cluster read 5.08 on master (-14%, z -31)
- `layer=router load=decode_16 m=16` flat: keeps restack swapab
- `layer=router load=decode_32 m=32` flat: keeps restack swapab
- `layer=router load=verify_* m=12*` flat: m=128 keeps hilo
- `layer=router load=verify_32 m=256` flat: keeps hilo
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched

## Risk

Kill: router decode_1 and verify_1 rows not both below the restack's 1e90292c rows (5.38/5.47) with the cluster kernel in the profile. Expected headline ~1.224 (03:41Z substitution). Risk: the load_inline build adds ~40 s and an illegal-instruction fault would poison the context; the kernel already ran clean on the judge, so run as a check.

## Adversarial review

Reviewed 2026-10-25T042320Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a clean m<=8 router-only shape gate onto the landed restack (b99b1dd), and the per-shape router rows can be read without a companion. The m=1 claim is overstated: -8.7% is measured against old master (5.73 us), and against the restack's 5.38 us the cluster's 5.24 us is only -2.6%, close to the 1.5% noise. The headline cannot move outside its CI, so judge the run on the router decode_1 and verify_1 rows only.

### Justification
The mechanism holds up, but it is not the one the hypothesis states. In dc5d03c4 the cluster kernel itself is SLOWER than the restack's Triton kernel. router_kernel<8,8> profiles at 3.6 us (m=1) and 4.3 us (m=8), against _router_swapab_kernel at 3.1 / 3.0 us plus a 0.7-0.8 us memset node. The row still drops (5.24 vs 5.38, 5.08 vs 5.47) because a single graph node with torch.empty replaces the memset-plus-kernel pair, and each node costs about 1 us in a cold-L2 replay. So the gain comes from removing a node, not from a faster stream. That is plausible on H200, and the m=8 row is real (-0.39 us, per-shape speedup 5.18 vs 4.80, err ~20 ns). The two runs being compared were built on different bases (dc5d03c4 on 8b9ad43, the restack on b99b1dd), but SGLang's baselines agree within 1% (10432 vs 10368 ns, 26320 vs 26240 ns), so the cross-run row comparison is acceptable. Weakest link: the decode_1 m=1 expectation. The quoted -8.7% / z -16 is against old master's 5.73 us; against the restack it is 5.24 vs 5.38 = -2.6%, barely above the 1.5% per-shape noise threshold, so a re-measure could easily land flat. Second weakness: headline impact. Router decode_1 and verify_1 are ~5.4 us of ~48-49 us loads (weight 62 each), so 0.14 + 0.39 us is about +0.3% and +0.8% on two of six loads, about +0.18% geomean. Expect roughly 1.2237 against a CI of +/-1.5%. The headline will read within-ci whatever happens, and promotion has to rest on the router rows. Both rows remain 2.8-2.9x off the 1.75-1.78 us floor. The expected directions (router m=1/m=8 faster, everything else flat) are consistent with the code path: only router m in 1..8 changes.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32 (landed b99b1dd): it is the incumbent being displaced at m<=8 (_router_swapab_kernel plus the torch.zeros memset), so the change is defined relative to it. That makes it the comparison, not a confound. If its swap-AB router were ever made memset-free, most of this hypothesis's margin would disappear, because the cluster kernel's own time is higher (3.6/4.3 us vs 3.1/3.0 us). Judge against the restack's rows 5.38/5.47 (job 1e90292c7b954d96), not against master-era numbers.
- router-triton-hilo-fused-control-splitk-all-m, router-hilo-at-m8, sglang-gemm-dispatch-parity: superseded at m<=8 by the restack. They only set the hi/lo [256, 8192] bf16 row-major layout that the cluster kernel reads, and that layout is unchanged in b99b1dd, so there is no interaction.
- bf16-m32-swapab-wgmma-triton, bf16-shape-gated-triton-winners-combo: these change other layers only. Per-shape graphs are timed separately, so the only interaction is about 40 s more prepare/compile time inside the 10-minute budget. The non-router losses in dc5d03c4 (qkv/o 5-9% slower) came from its 8b9ad43 base lacking the restack, NOT from the cluster kernel. Here those rows must come back flat, and if they do not, the run did not build on the restack.

### Interference and A/B needs
No companions are needed. The change is a pure shape gate (router, 1 <= m <= 8, n == 256) inside the restack's kernel_fn, and the profile names a different kernel on exactly the two target rows. A gain or loss on router decode_1/verify_1 is therefore attributable to the swap of swapab+memset for cluster, and every other row is a built-in null control: if any non-router row moves beyond noise, the build is wrong, not the hypothesis. Reverting the restack (ab-revert) would only re-measure dc5d03c4. The informative follow-up is a different hypothesis, not a companion: a memset-free swap-AB router at m<=8 (e.g. SPLIT_K reduction in-kernel or a single-CTA-per-column variant with torch.empty) to test whether node removal alone captures the gain without a load_inline CUDA kernel. That would not be needed to read this run.

### Open questions
- Port faithfully: the source used state['router_cluster'] (NT bucket -> cluster size, 0 = fallback) plus a cudaOccupancyMaxActiveClusters gate (>=16 for CS=8). The hypothesis says 'router_cluster_ok' and 'compile only NT=8'. Keep the occupancy check and fall back to the restack's swapab when CS=8 NT=8 reports fewer than 16 active clusters. If the CS=4 variant is dropped, say so in the implementation notes.
- Insert the hook after _router_prepare in prepare() so router_stages exists, and have the self-check compare against x.double() @ router.double().T, the same bounds as the source (rel L2 <= 2e-5, worst row <= 1e-4, one-hot exact vs hi+lo, run 2 bit-identical). m in {1, 5, 8} matches the gate. The m=9 outlier test stays on swapab.
- Kill or promote on verify_1 m=8 above all (expected -7%). Treat decode_1 m=1 as a success only if its per-shape speedup beats the restack's 1.925 by more than 1.5% (above ~1.954). A 5.30-5.38 us row is a tie, not a loss.
- At harvest, confirm in the profile: router_kernel<8,8> on router decode_1/verify_1 with no vectorized_elementwise (memset) node; _router_swapab_kernel on router decode_16/decode_32; _router_hilo_kernel on m=128/256; _bf16_swapab_kernel on the restack's bf16 rows. Also confirm qkv_gate decode_1 is near 12.2 us and not 13.5 us; 13.5 would mean the build was on pre-restack master, as in the nine 03:41Z mis-builds.
- Why is the cluster kernel 3.6-4.3 us when swapab runs at 3.0-3.1 us? Cluster launch and scheduling cost and the DSMEM double barrier.cluster are likely causes. If this lands, the next stage should attack the kernel's own time, because the node saving has already been banked.


## Implementation notes

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

## Rules check

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

## Pre-flight audit

Audited 2026-10-24 by an opus subagent, by reading and numpy emulation; there is no GPU on this node.

**Verdict: dispatch.** No fixes were needed.

- Routing: the hook sits after every bf16-layer branch and immediately before `_router_path`. It fires only when `router_cluster_ok` is set, 1 <= m <= 8, `cfg["m"] == m`, `cfg["n"] == 256` and k == 8192, with stride and alignment preconditions met. Nothing earlier intercepts the router (`_swapab16_key` returns "" for the router, and `SWAPAB_CFG` has no router entry). m 9..32 still goes to `_router_forward` (swapab), m > 32 to `_router_hilo_forward`. The diff is 450 insertions and 0 deletions; the restack lines are byte-identical.
- No memset: the m <= 8 path is `torch.empty` plus a single `cudaLaunchKernelEx`. Every output element is written: 16 clusters x 16 columns = 256, and in each CTA, rank r writes columns {2r, 2r+1} for every tok < m. The tokens >= m are zero-filled by cp.async with src_bytes 0, so they cannot leak garbage into the result. The kernel never reads `out`.
- CUDA source: diffed against branch router-cuda-cluster-dsmem-splitk-rewrite (5abb1a1). The only differences are the mode=0 diagnostic and the dropped <8,16>/<8,32>/<4,*> instances. I re-derived the ldmatrix A/B fragment mapping, the XOR swizzle (store and load agree), the hi/lo row stacking (acc[0]+acc[2] gives hi+lo for the same column) and the warp-partial aliasing (2 KB <= 64 KB). All are correct.
- Gates: the occupancy check (< 16 active clusters keeps swapab), the self-check bounds, the seed and the one-hot construction are identical to the source's. A numpy emulation of hi+lo with fp32 per-rank partials at m 1/5/8 gives rel L2 of about 2.3-2.8e-6 and a worst row of about 2.8e-6, against bounds of 2e-5 and 1e-4. That is about 7x margin, so the gate cannot always fail. The one-hot row is exact up to one fp32 add, well inside 1e-6. Determinism holds because the warp and rank sums run in a fixed order.
- Fallbacks: a build error leaves `_RC_EXT` as None; any exception, occupancy miss or check miss leaves `router_cluster_ok` False. Each case prints `PATH CHOSEN` and lands on swapab. The C++ TORCH_CHECKs mirror the Python gate.
- Rules: rules line 96 explicitly allows `load_inline`/nvcc, and the file has no subprocess, multiprocessing or os.fork. There are no host syncs, data-dependent branches or cached data in kernel_fn, and the only `synchronize` is in prepare(). hi/lo is unchanged from the restack. okc_lint gives OK with one warning (the prepare() synchronize), and py_compile passes.

Residual risk: a device fault during the prepare() self-check would poison the CUDA context and fail the run instead of falling back (the same kernel already ran clean on the judge in dc5d03c4), and the m=1 win (about -2.6% against the restack) may land inside noise.

## Result

- job `d323ce8ef6a54fde` (check), status done, passed True, tests 9/9
- speedup 1.2199x, 95% CI [1.2016, 1.2382], score 3960.107567010976 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "1e90292c7b954d96"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/restack-router-cluster-dsmem-m8.result.json`, `muse_bf16_proj/hypotheses/done/neutral/restack-router-cluster-dsmem-m8.patch`; branch `hyp/muse_bf16_proj/restack-router-cluster-dsmem-m8`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 12.18 | 12.36 | 1.012 | flat |
| layer=o load=decode_1 m=1 | 62 | 5.89 | 5.95 | 0.9904 | flat |
| layer=pooler load=decode_1 m=1 | 62 | 18.84 | 18.85 | 1.009 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 6.15 | 5.92 | 1.0113 | flat |
| layer=router load=decode_1 m=1 | 62 | 5.38 | 4.98 | 1.0782 | gained |
| layer=qkv_gate load=decode_16 m=16 | 62 | 13.15 | 12.76 | 1.0325 | gained |
| layer=o load=decode_16 m=16 | 62 | 6.23 | 6.16 | 0.9873 | flat |
| layer=pooler load=decode_16 m=16 | 62 | 19.47 | 19.31 | 1.0083 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.51 | 18.73 | 0.9818 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.47 | 5.63 | 0.9717 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 14.19 | 14.0 | 0.9931 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.34 | 6.44 | 0.9666 | lost |
| layer=pooler load=decode_32 m=32 | 62 | 19.73 | 19.83 | 0.9966 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.08 | 18.14 | 0.9832 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.2 | 6.12 | 0.9985 | flat |
| layer=qkv_gate load=verify_1 m=8 | 62 | 12.5 | 12.53 | 1.0162 | flat |
| layer=o load=verify_1 m=8 | 62 | 6.13 | 6.49 | 0.9947 | flat |
| layer=pooler load=verify_1 m=8 | 62 | 18.89 | 18.95 | 1.0027 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.0 | 5.72 | 0.9922 | flat |
| layer=router load=verify_1 m=8 | 62 | 5.47 | 4.93 | 1.0994 | gained |
| layer=qkv_gate load=verify_16 m=128 | 62 | 17.05 | 16.92 | 0.9879 | flat |
| layer=o load=verify_16 m=128 | 62 | 6.98 | 7.51 | 0.9566 | lost |
| layer=pooler load=verify_16 m=128 | 62 | 25.0 | 25.21 | 1.0074 | flat |
| layer=unpooler load=verify_16 m=128 | 62 | 19.96 | 20.16 | 0.9885 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.28 | 7.48 | 0.9631 | lost |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.69 | 19.95 | 0.9931 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.26 | 9.24 | 0.9823 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 31.19 | 31.43 | 0.9797 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.42 | 24.26 | 1.0019 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.6 | 9.32 | 1.016 | flat |

gained on router decode_1, verify_1; qkv_gate decode_16; lost on o decode_32, verify_16; router verify_16. Expected hit: 8/10. Noise ±1.5%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: restack-router-cluster-dsmem-m8`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

