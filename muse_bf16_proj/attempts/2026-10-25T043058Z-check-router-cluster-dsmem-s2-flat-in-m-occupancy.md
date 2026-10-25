# muse_bf16_proj · 2026-10-25T043058Z · check · router-cluster-dsmem-s2-flat-in-m-occupancy

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T042040Z-router-cluster-dsmem-s2-flat-in-m-occupancy.md` (branch `hyp/muse_bf16_proj/router-cluster-dsmem-s2-flat-in-m-occupancy`).

- **Job:** `ef8a8bca31ef420e` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `647c57d`

## Goal
Build on branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (stage 1, dc5d03c4). Stage 1 won rows at m=1/8 only by deleting the memset node: its kernel was SLOWER than master's (3.6/4.3/4.6 us at m=1/8/16 vs hilo 3.2/3.2/3.5), i.e. 8.4 MB at ~2.3 TB/s with 4 warps and 1 CTA/SM, and it grew with m. Stage 2 changes only the kernel's decomposition, not the routing: (1) cluster dim 16 along K (set cudaFuncAttributeNonPortableClusterSizeAllowed=1; fall back to 8 if cudaOccupancyMaxActiveClusters returns 0 for 16, and print which): 16 clusters x 16 CTAs = 256 CTAs (2 per SM), each CTA owning K [512r, 512r+512): 32 KB of hi+lo weight; (2) 8 warps per CTA each taking a 64-K slice, so 16 warps/SM are in flight instead of 4; (3) load x for the CTA's K slice once into smem with tokens padded to NT and reuse the A (hi/lo) fragments across all NT/8 B fragments (decode-once, as stage 1 of the lossless line learned); (4) the DSMEM reduce stores each rank's slice of columns once over 16 ranks in fixed order. Keep NT 8/16/32, the same self-check bounds and run-twice bit-identical check, torch.empty output, one node. Routing stays as stage 1 (cluster at 1 <= m <= 32, master above). Print per-m kernel time (cold L2) and regs/spills/smem.

## What changed, and where
Nothing could be run here (no GPU, torch or Triton on the build node): the CUDA, the gate and every print are untested. Only py_compile and okc_lint ran.

**Base (review fix, review wins over the spec's stage-1 base).** The worktree was at stage 1's impl commit 0dea576, which predates master b99b1dd. `muse_bf16_proj/submission.py` is now a 3-way merge (`git merge-file master base s1`) of b99b1dd's file plus stage 1's change. Three conflicts, all resolved as "keep both": the module docstring paragraph, the end of master's section (6) swap-AB m<=16, and prepare()'s router lines (master's `_router_prepare(state)` kept, stage 1's warm-up loop dropped, `_router_cluster_prepare(state)` called after it). Result: `git diff b99b1dd` is 611 pure insertions, no deletions, so every non-router row and the m>32 router rows are byte-identical to master and the headline is comparable. Stage 1's `_rc_time` called the old `_router_forward(x, hi, lo)`; master's takes a stage count, so the new timing code passes `state["router_stages"][_router_config(m)]`.

**Kernel (section 7), what changed vs stage 1.** One templated kernel `router_kernel<CS, NT, NWARP, MINB>`; every cluster owns 16 output columns (16 clusters), CTA rank r owns K [r*8192/CS, +8192/CS). Three variants:
1. CS 16, 8 warps, `__launch_bounds__(256, 2)`: 256 CTAs, 32 KB weight per CTA, 16 warps/SM, one 64-K commit group per warp.
2. CS 16, 4 warps, `__launch_bounds__(128, 3)`: the review's fallback if variant 1 spills.
3. CS 8, 4 warps, 1 CTA/SM: the stage-1 geometry, kept as fallback and as the in-process reference line.
x tile loaded once into smem per CTA, A (hi/lo) fragments reused across all NT/8 token fragments (stage 1 already did this; unchanged). The DSMEM reduce now has each rank reduce 16/CS columns (1 at CS 16, 2 at CS 8) over ranks 0..CS-1 in fixed order. `cudaFuncAttributeNonPortableClusterSizeAllowed=1` is set for CS 16. Output, torch.empty, one node, fixed order, run-twice bit-identical self-check and the same bounds (2e-5 / 1e-4 / 1e-6): all kept.

**Review fixes applied.**
- Gate: a variant is a candidate only if `cudaOccupancyMaxActiveClusters >= 16` (all clusters co-resident in one wave), not merely > 0. Printed per (variant, NT) with regs, local/spill bytes and smem. A failing variant falls to the next (1, 2, 3), then master. Never a split wave.
- `__launch_bounds__(256, 2)` for variant 1, skipped if it spills at the 128-reg cap (spill printed); variant 2 is the review's 4-warp / 3 CTAs/SM fallback.
- DSMEM reduce tail: printed as full minus mode-0 (no-reduce) cold-L2 time per variant per m.
- Cold-L2 in-process timings (256 MB flush before each launch, cudaEvent median of 30, eager) printed per m in {1, 8, 16, 32} for every passing variant (full, no-reduce, tail) next to master's swap-AB + memset on the same x. This includes the stage-1 geometry line (variant 3) when it passes occupancy.
- Gate by m range: bucket NT 8 (m<=8, timed m=1 and 8), NT 16 (m=16), NT 32 (m=32) route to the CUDA kernel only if the best variant's cold-L2 time is <= 0.97 x master's swap-AB at every timed m of the bucket; otherwise master's swap-AB (m<=32) / hilo (m>32). `PATH CHOSEN` is printed per m 1..32 plus m>32. m=128/256 and every other layer are untouched.

**Deviations from the spec.** (a) Cluster dim 16 with fallbacks to a 4-warp 3-CTA/SM CS-16 variant and then the stage-1 CS-8 geometry instead of "cs 16 else cs 8", per the review. (b) The route is decided from prepare()-time timing on synthetic x, which the spec did not describe; it is a constant in `state`, never data-dependent in kernel_fn. (c) Stage 1's CS-4 instances are dropped (replaced by the two CS-16 variants). (d) I did not implement the review's optional "reduce in 8-rank halves" idea; the mode-0 diagnostic will show whether the tail matters. (e) load_inline module renamed `muse_bf16_router_cluster_v2`. Arch flags: builds with TORCH_CUDA_ARCH_LIST=9.0a (printed), `-O3 -lineinfo`; no sm_90a-only PTX is used (mma.sync, cp.async.cg, cluster launch, mapa, barrier.cluster are baseline sm_90), so no `__CUDA_ARCH_FEAT_SM90_ALL` guard is needed. No mbarrier or spin wait exists. No subprocess/fork.

**How to judge (review fix).** Judge on router rows (decode_1, verify_1, decode_16, decode_32) against restack b99b1dd's swap-AB rows (5.47 / 6.2 class) and on the in-process cold-L2 prints, not the headline and not profile top_kernels (about 0.7 us noise between identical instances in stage 1; M2 "flat within 0.3 us" can only be read from the prints). At harvest confirm `PATH CHOSEN` per m and the kernel name `router_kernel<16,NT,...>` on the routed rows, and no memset node. If it wins, this branch is not for landing: a separate restack swaps this kernel body into restack-router-cluster-dsmem-m8's port and widens its gate to the m ranges that won. Biggest risk: occupancy at CS 16 may report < 16 clusters for every variant (stage 1 saw 15 for 8-SM clusters), in which case every bucket falls to master or variant 3 and the run is a no-op or a stage-1 repeat; second, the load_inline build is now 9 instances (about 60-90 s est.).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn path adds one cluster launch via cudaLaunchKernelEx on the current stream; no `.item()`/`.cpu()`/`synchronize()` in kernel_fn (the two synchronize calls are in `_rc_check` / `_rc_time_and_gate`, prepare() only); verified in the diff.
- [x] Nothing derived from `data` is cached across calls: the per-bucket variant is a constant from prepare()-time synthetic x and the weights; kernel_fn only branches on cfg/shape/stride/alignment.
- [x] Weights: only the existing lossless hi + lo bf16 split of the fp32 router (already in master's prepare()); no new repack.
- [x] Activations BF16 where SGLang keeps BF16: x stays bf16, fp32 accumulation, fp32 router output as master.
- [x] Attention: n/a, GEMM problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocess/fork/multiprocessing (grep clean).
- [x] All work happens inside the timed call; no output caching or replay; output is a fresh torch.empty per call.

okc_lint output:
```
warn  host-sync    L1456: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1487: flush = torch.empty(ROUTER_CLUSTER_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L1502: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings
```
The uint8 flush buffer is a prepare()-only timing scratch, never an operand. py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Judge against stage 1's rows (dc5d03c4) and the restack's router rows. Milestone M2: kernel time within 0.3 us from m=1 to m=32 and <= 3.0 us. Kill: m=16 and m=32 rows not below the restack's 5.47/6.2. Built on master (stage 1's base); a win restacks onto the restack afterwards.

## Result (after the run)
- job `ef8a8bca31ef420e` (check), status done, passed True, tests 9/9
- speedup 1.2058x, 95% CI [1.1878, 1.2239], score 4006.1772791068784 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "1e90292c7b954d96"}; base result: muse_bf16_proj/hypotheses/done/regression/router-cuda-cluster-dsmem-splitk-rewrite.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-cluster-dsmem-s2-flat-in-m-occupancy.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-cluster-dsmem-s2-flat-in-m-occupancy.patch`; branch `hyp/muse_bf16_proj/router-cluster-dsmem-s2-flat-in-m-occupancy`
