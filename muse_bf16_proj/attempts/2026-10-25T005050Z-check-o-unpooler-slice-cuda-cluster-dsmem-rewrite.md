# muse_bf16_proj · 2026-10-25T005050Z · check · o-unpooler-slice-cuda-cluster-dsmem-rewrite

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T003826Z-o-unpooler-slice-cuda-cluster-dsmem-rewrite.md` (branch `hyp/muse_bf16_proj/o-unpooler-slice-cuda-cluster-dsmem-rewrite`).

- **Job:** `3f3330f70b6e4cdc` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `6ca6fa9`

## Goal
Built on master (8b9ad43, blob a8bb42a); restack not landed and not needed. WHY: o (16.8 MB, floor 3.5 us) and the unpooler slice (8.4 MB, floor 1.75 us) are the shapes furthest from their floors in ratio (1.7x and 3.4x on rows). REWRITE_READ_2026-10-25 section 2 found why the gemv rewrite lost on exactly these: they are latency-bound (each CTA streams only 32-64 KB, a ring never fills) and the cost grows with m in the per-row partial store and the serialized global ticket-reduce tail. This stage removes both: every CTA issues ALL its weight loads at once (no ring), and split-K partials are reduced through thread-block-cluster distributed shared memory instead of a global workspace + ticket. STAGE 1: one CUDA kernel template compiled with torch.utils.cpp_extension.load_inline (reuse the gemv/lossless scaffold: compile-seconds print, per-route in-process self-check, print-and-fall-back to master; TORCH_CUDA_ARCH_LIST=9.0a around the build, -lineinfo; mma.sync m16n8k16 swap-AB with weight rows on M and tokens on N, NT 8 or 16 template instances, cp.async.cg 16 B only, cluster launch via cudaLaunchKernelEx with cluster dim 8 along K, mapa + ld.shared::cluster, barrier.cluster; all baseline sm_90). Decompositions: o (n=8192, k=1024): 32 clusters x 8 CTAs = 256 CTAs (2 per SM), cluster owns 256 output rows, rank r owns K [128r, 128r+128): 256 x 128 x 2 B = 64 KB weight per CTA, issued in 2 commit groups. Unpooler slice (n=1024 of the full [8192, 4096] weight, k=4096): 16 clusters x 8 CTAs = 128 CTAs, cluster owns 64 rows, rank r owns K [512r, 512r+512): 64 KB per CTA, 4 commit groups. Each CTA: 4 or 8 warps split its K range, reduce warp partials in smem, write its [rows, NT] fp32 partial to smem, cluster barrier (release/acquire), then rank r reduces its 1/8 of the cluster rows over ranks 0..7 in fixed order via DSMEM and stores bf16 out[m, n] (transposed store, tokens < m only); a second cluster barrier before exit. torch.empty output, no memset, no atomics, one node. ROUTING in kernel_fn: o at 1 <= m <= 16 and unpooler with cfg n == 1024 at 1 <= m <= 8, when state cluster_ok[layer] (a prepare-time constant); everything else keeps master byte for byte. Tests through it: m=5 unpooler. SELF-CHECK in prepare per route at m in {1, 5, 8, 16 (o only)}, twice: rel L2 <= 3e-3 vs fp64, one-hot probe rows exact within 1e-4, finite, run 2 bit-identical; synchronize and print after each launch. DIAGNOSTICS: cudaFuncGetAttributes regs/spills/smem, cudaOccupancyMaxActiveClusters, warm-L2 and cold-L2 (flush > L2 first) event times vs master on the same synthetic x, and the m=1 vs m=8/16 kernel time (the gemv rewrite cost grew 1.6-1.7 us with m; this design should keep it flat). Stage 2 (later): extend to o m 17..32 (NT 32) and qkv_gate m<=8 if M2 is met. Run as a check.

## What changed, and where
Only `muse_bf16_proj/submission.py` in the worktree changed (+~520 lines, nothing removed): a new section (6) before `prepare()`, a `_oun_prepare(state)` call at the end of `prepare()`, two imports (`os`, `time`) and one routing branch in `kernel_fn` placed before the existing unpooler/o branches. Master's paths are untouched and remain the fallback.

What it is: one templated CUDA kernel `oun_kernel<NT, ROWS, KPER, CL, NG>`, built at import with `load_inline` (extension `muse_bf16_o_unpool_cluster_v1`, `-O3 -lineinfo`, plain sm_90 PTX: mma.sync m16n8k16 swap-AB, cp.async.cg 16 B, ldmatrix, cluster launch via `cudaLaunchKernelEx`, `mapa` + `ld.shared::cluster.v4`, `barrier.cluster` arrive.release / wait.acquire; no wgmma, so no 9.0a override and no `__CUDA_ARCH_FEAT_SM90_ALL` guard, the body is only guarded by `__CUDA_ARCH__ >= 900`). 256 threads/CTA, 8 warps split rows and K; every CTA issues all its weight + x loads up front in NG commit groups (o NG 2, slice NG 4) and consumes them group by group. The CTA's fp32 partial `part[t][row]` (pitch ROWS+4) is aliased onto the weight buffer; K-group warps add into it in fixed order; one cluster barrier; rank r reduces its ROWS/CL rows over ranks 0..CL-1 in fixed order through DSMEM and stores bf16 `out[m, n]` (tokens < m only, 8 B coalesced stores); a second cluster barrier before exit. `torch.empty` output, no memset, no atomics, no global workspace. Instances: o NT 8/16 (m<=8 / 9..16) x CL 8 (ROWS 256, K/CTA 128) or CL 4 (ROWS 128, K/CTA 256); slice NT 8 x CL 8 (ROWS 64, K/CTA 512) or CL 4 (ROWS 32, K/CTA 1024). Routing: o at 1<=m<=16 and the unpooler with n==1024 at 1<=m<=8, only when `state["oun_cl"][layer]` (its own prepare-time constant, 0 = master) is nonzero; the weight is read in place (stride 4096 for the slice).

Review fixes applied (the review wins where it corrects the spec):
- XOR swizzle (chunk c of row r at c ^ (r & 7)) in both cp.async destinations and ldmatrix addresses, for the weight and the x tile (row pitches 256 B - 2 KB, all multiples of 128 B).
- `cudaOccupancyMaxActiveClusters` queried per (layer, cluster dim) in prepare(): CL 8 needs 32 (o) / 16 (slice) clusters, else CL 4 needs 64 / 32, else the layer keeps master. Never launches more clusters than reported; one `PATH CHOSEN` line per layer prints the path and cluster dim. A CL whose self-check fails also falls to the next CL, then to master.
- fp32 partial aliased onto the consumed weight buffer (static_assert it fits); smem ~68 KB (o), 72 KB (slice CL 8), 80 KB (slice CL 4), 2 CTAs/SM fits for o.
- Plain `cp.async.cg` (no createpolicy/cache hints), `-lineinfo`, flushed print before and `torch.cuda.synchronize()` + flushed print after each self-check launch.
- Unique extension name and own route constant (`oun_cl`), distinct from the router sibling.
- Baselines corrected: the Expected/replaces text quoted the restack run 60b20fd0bf924267. Master's run is 1602a4e947e64090: o kernels 4.6/5.2/4.8 us (m 1/8/16), rows 6.32/6.32; slice kernels 4.5/4.6 us, rows 5.94/6.13. Judge M2 against those (rows), and also report vs the restack. M3 on the slice (<= 4.2) remains a stretch since slice rows sit ~1.5 us above kernel time.

Self-check (prepare, in-process): o at m in {1, 5, 8, 16}, slice at m in {1, 5, 8}, plus an m=1 one-hot case, each launched twice: rel L2 <= 3e-3 vs fp64, one-hot probe row within 1e-4 (exact weight column), finite, run 2 bit-identical. Diagnostics printed: cudaFuncGetAttributes (regs, local/spill, smem), occupancy, warm-L2 and cold-L2 (256 MB flush) eager event times vs master's path at m=1 and the top m.

Deviations: (1) NT 16 instances only for o (the slice is m<=8, NT 8). (2) Slice at 256 CTAs x 32 KB (review open question) not built; CL 4 slice (128 CTAs x 64 KB, K 1024) is the fallback instead. (3) Kernel was not compiled or run (no GPU/nvcc here): first compile and launch happen on the judge, run it as `okc check`.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn only calls `_oun_forward` (torch.empty + one cudaLaunchKernelEx on the current stream); syncs/prints only in prepare(); branches on cfg/shapes/strides/pointer alignment and the prepare constant `oun_cl`, never on input values (checked in `git diff`).
- [x] Nothing derived from `data` is cached: `oun_cl` comes from weights and synthetic x in prepare(); self-check inputs are generated, not from data.
- [x] Weights: no repack, no copy; bf16 weights read in place at their own precision (slice: first 1024 rows of the [8192, 4096] weight).
- [x] Activations BF16 (x bf16, fp32 accumulate, bf16 output, as SGLang); KV cache n/a: n/a (no attention here).
- [x] Attention sees every key: n/a: dense projection, no attention.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses (no subprocess/multiprocessing/fork; load_inline is in-process as in earlier accepted builds) or network.
- [x] All work happens inside the timed call (the full GEMM + reduce in one launch); no output caching or replay.

okc_lint (python3 -I okc_lint.py submission.py --problem muse_bf16_proj):
```
warn  host-sync    L1064: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1086: flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=w.device)  (allowed only where SGLang does the same)
warn  host-sync    L1096: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1103: e1.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1111: e1.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1168: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_bf16_proj/submission.py
```
py_compile OK, __pycache__ removed. The uint8 flush buffer is a prepare-only cold-L2 diagnostic, never used in kernel_fn or its output.

## Risks / what could fail
high
Counter-evidence to weigh: bf16-small-m-cuda-gemv-rewrite lost on exactly these rows (o m=8 6.32 -> 9.52, slice m=8 6.13 -> 7.55); this stage differs in the two mechanisms that read named as the cause (no ring on 32-64 KB streams; no global partial store + ticket tail). Risks: a first-launch fault poisons the context (LESSONS 2026-10-25), so prints around each self-check launch and check mode only; cluster co-residency (print cudaOccupancyMaxActiveClusters; fall back if below the grid). Milestones: M1 9/9 tests with m=5 unpooler through the kernel; M2 every routed row at or below master; M3 o rows <= 5.0-5.2, slice rows <= 4.2. Headline at M3: ~1 us o + ~2 us slice on decode_1 and verify_1, ~1 us o on decode_16: about +4-6% on decode_1/verify_1, +1.5% decode_16.

## Result (after the run)
- job `3f3330f70b6e4cdc` (check), status done, passed True, tests 9/9
- speedup 1.1710x, 95% CI [1.1535, 1.1886], score 4125.265010279047 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/o-unpooler-slice-cuda-cluster-dsmem-rewrite.result.json`, `muse_bf16_proj/hypotheses/done/regression/o-unpooler-slice-cuda-cluster-dsmem-rewrite.patch`; branch `hyp/muse_bf16_proj/o-unpooler-slice-cuda-cluster-dsmem-rewrite`
