---
problem: muse_bf16_proj
slug: o-unpooler-slice-cuda-cluster-dsmem-rewrite
created: 2026-10-25T003826Z
base: master
area: o GEMM + unpooler GEMM (latency-bound small streams)
change: >
  Built on master (8b9ad43, blob a8bb42a); restack not landed and not needed. WHY: o (16.8 MB,
  floor 3.5 us) and the unpooler slice (8.4 MB, floor 1.75 us) are the shapes furthest from their
  floors in ratio (1.7x and 3.4x on rows). REWRITE_READ_2026-10-25 section 2 found why the gemv
  rewrite lost on exactly these: they are latency-bound (each CTA streams only 32-64 KB, a ring
  never fills) and the cost grows with m in the per-row partial store and the serialized global
  ticket-reduce tail. This stage removes both: every CTA issues ALL its weight loads at once (no
  ring), and split-K partials are reduced through thread-block-cluster distributed shared memory
  instead of a global workspace + ticket. STAGE 1: one CUDA kernel template compiled with
  torch.utils.cpp_extension.load_inline (reuse the gemv/lossless scaffold: compile-seconds print,
  per-route in-process self-check, print-and-fall-back to master; TORCH_CUDA_ARCH_LIST=9.0a around
  the build, -lineinfo; mma.sync m16n8k16 swap-AB with weight rows on M and tokens on N, NT 8 or
  16 template instances, cp.async.cg 16 B only, cluster launch via cudaLaunchKernelEx with cluster
  dim 8 along K, mapa + ld.shared::cluster, barrier.cluster; all baseline sm_90). Decompositions:
  o (n=8192, k=1024): 32 clusters x 8 CTAs = 256 CTAs (2 per SM), cluster owns 256 output rows,
  rank r owns K [128r, 128r+128): 256 x 128 x 2 B = 64 KB weight per CTA, issued in 2 commit
  groups. Unpooler slice (n=1024 of the full [8192, 4096] weight, k=4096): 16 clusters x 8 CTAs =
  128 CTAs, cluster owns 64 rows, rank r owns K [512r, 512r+512): 64 KB per CTA, 4 commit groups.
  Each CTA: 4 or 8 warps split its K range, reduce warp partials in smem, write its [rows, NT]
  fp32 partial to smem, cluster barrier (release/acquire), then rank r reduces its 1/8 of the
  cluster rows over ranks 0..7 in fixed order via DSMEM and stores bf16 out[m, n] (transposed
  store, tokens < m only); a second cluster barrier before exit. torch.empty output, no memset, no
  atomics, one node. ROUTING in kernel_fn: o at 1 <= m <= 16 and unpooler with cfg n == 1024 at 1
  <= m <= 8, when state cluster_ok[layer] (a prepare-time constant); everything else keeps master
  byte for byte. Tests through it: m=5 unpooler. SELF-CHECK in prepare per route at m in {1, 5, 8,
  16 (o only)}, twice: rel L2 <= 3e-3 vs fp64, one-hot probe rows exact within 1e-4, finite, run 2
  bit-identical; synchronize and print after each launch. DIAGNOSTICS: cudaFuncGetAttributes
  regs/spills/smem, cudaOccupancyMaxActiveClusters, warm-L2 and cold-L2 (flush > L2 first) event
  times vs master on the same synthetic x, and the m=1 vs m=8/16 kernel time (the gemv rewrite
  cost grew 1.6-1.7 us with m; this design should keep it flat). Stage 2 (later): extend to o m
  17..32 (NT 32) and qkv_gate m<=8 if M2 is met. Run as a check.
expected:
  - shapes: layer=o load=decode_1 m=1
    direction: faster
    why: "incumbent skinny kernel 4.1 us, row 5.9; floor 3.5. All 64 KB per CTA in flight, DSMEM reduce: M2 row <= 5.9, M3 row <= 5.0"
  - shapes: layer=o load=verify_1 m=8
    direction: faster
    why: incumbent 4.2 us, row 6.1; floor 3.5; M3 <= 5.0
  - shapes: layer=o load=decode_16 m=16
    direction: faster
    why: incumbent 4.4 us, row 6.2; floor 3.6; M3 <= 5.2
  - shapes: layer=unpooler load=decode_1 m=1
    direction: faster
    why: "incumbent split-K-8 ticket kernel 4.3 us, row 6.2; floor 1.7: M2 row <= 5.9, M3 row <= 4.0"
  - shapes: layer=unpooler load=verify_1 m=8
    direction: faster
    why: incumbent 4.4 us, row 6.0; floor 1.8; M3 <= 4.2 (no m-proportional tail)
  - shapes: layer=o load=decode_32 m=32
    direction: flat
    why: m > 16 keeps master swap-AB
  - shapes: layer=unpooler load=decode_16 m=16
    direction: flat
    why: full weight, master
  - shapes: layer=router *
    direction: flat
    why: untouched
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
risk: high
depends_on: null
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 1
replaces: o at m<=16 (_pooler_skinny_kernel, 4.1-4.4 us kernel, rows 5.9-6.3) and the unpooler slice at m<=8 (_bf16_splitk_kernel/ticket split-K 8, 4.3-4.4 us kernel, rows 5.9-6.1)
review: proceed
status: running
review_at: 2026-10-25T004153Z
review_companions: null
stage: implemented
claimed_at: 2026-10-25T004153Z
base_sha: dcd2ff817f67ee97bdc52e92ec44df3a3752ccf1
base_blob: a8bb42ac483318e9fbc84bc7826a27b7a704d1fb
branch: hyp/muse_bf16_proj/o-unpooler-slice-cuda-cluster-dsmem-rewrite
worktree: .worktrees/muse_bf16_proj/o-unpooler-slice-cuda-cluster-dsmem-rewrite
mode: null
restack_of: null
stale_base: null
implemented_at: 2026-10-25T005050Z
impl_commit: 6ca6fa934e5ff0ef054baef30cc261eb9ad818ea
submission_sha256: 86f5e904f75d000bb58ee88f983282cc49c71e46bd9eee996ddc2f0bb0ac9393
norm_sha256: bf100462c0369acafb87e40b68b6f33c1bb7d039aef602e76a0863d1e420c812
submission_blob: 3ba071c2ec0b65200650da8027388a0e73be5fb1
lint_warnings: 6
---

# o-unpooler-slice-cuda-cluster-dsmem-rewrite

## Change

Built on master (8b9ad43, blob a8bb42a); restack not landed and not needed. WHY: o (16.8 MB, floor 3.5 us) and the unpooler slice (8.4 MB, floor 1.75 us) are the shapes furthest from their floors in ratio (1.7x and 3.4x on rows). REWRITE_READ_2026-10-25 section 2 found why the gemv rewrite lost on exactly these: they are latency-bound (each CTA streams only 32-64 KB, a ring never fills) and the cost grows with m in the per-row partial store and the serialized global ticket-reduce tail. This stage removes both: every CTA issues ALL its weight loads at once (no ring), and split-K partials are reduced through thread-block-cluster distributed shared memory instead of a global workspace + ticket. STAGE 1: one CUDA kernel template compiled with torch.utils.cpp_extension.load_inline (reuse the gemv/lossless scaffold: compile-seconds print, per-route in-process self-check, print-and-fall-back to master; TORCH_CUDA_ARCH_LIST=9.0a around the build, -lineinfo; mma.sync m16n8k16 swap-AB with weight rows on M and tokens on N, NT 8 or 16 template instances, cp.async.cg 16 B only, cluster launch via cudaLaunchKernelEx with cluster dim 8 along K, mapa + ld.shared::cluster, barrier.cluster; all baseline sm_90). Decompositions: o (n=8192, k=1024): 32 clusters x 8 CTAs = 256 CTAs (2 per SM), cluster owns 256 output rows, rank r owns K [128r, 128r+128): 256 x 128 x 2 B = 64 KB weight per CTA, issued in 2 commit groups. Unpooler slice (n=1024 of the full [8192, 4096] weight, k=4096): 16 clusters x 8 CTAs = 128 CTAs, cluster owns 64 rows, rank r owns K [512r, 512r+512): 64 KB per CTA, 4 commit groups. Each CTA: 4 or 8 warps split its K range, reduce warp partials in smem, write its [rows, NT] fp32 partial to smem, cluster barrier (release/acquire), then rank r reduces its 1/8 of the cluster rows over ranks 0..7 in fixed order via DSMEM and stores bf16 out[m, n] (transposed store, tokens < m only); a second cluster barrier before exit. torch.empty output, no memset, no atomics, one node. ROUTING in kernel_fn: o at 1 <= m <= 16 and unpooler with cfg n == 1024 at 1 <= m <= 8, when state cluster_ok[layer] (a prepare-time constant); everything else keeps master byte for byte. Tests through it: m=5 unpooler. SELF-CHECK in prepare per route at m in {1, 5, 8, 16 (o only)}, twice: rel L2 <= 3e-3 vs fp64, one-hot probe rows exact within 1e-4, finite, run 2 bit-identical; synchronize and print after each launch. DIAGNOSTICS: cudaFuncGetAttributes regs/spills/smem, cudaOccupancyMaxActiveClusters, warm-L2 and cold-L2 (flush > L2 first) event times vs master on the same synthetic x, and the m=1 vs m=8/16 kernel time (the gemv rewrite cost grew 1.6-1.7 us with m; this design should keep it flat). Stage 2 (later): extend to o m 17..32 (NT 32) and qkv_gate m<=8 if M2 is met. Run as a check.

## Expected

- `layer=o load=decode_1 m=1` faster: incumbent skinny kernel 4.1 us, row 5.9; floor 3.5. All 64 KB per CTA in flight, DSMEM reduce: M2 row <= 5.9, M3 row <= 5.0
- `layer=o load=verify_1 m=8` faster: incumbent 4.2 us, row 6.1; floor 3.5; M3 <= 5.0
- `layer=o load=decode_16 m=16` faster: incumbent 4.4 us, row 6.2; floor 3.6; M3 <= 5.2
- `layer=unpooler load=decode_1 m=1` faster: incumbent split-K-8 ticket kernel 4.3 us, row 6.2; floor 1.7: M2 row <= 5.9, M3 row <= 4.0
- `layer=unpooler load=verify_1 m=8` faster: incumbent 4.4 us, row 6.0; floor 1.8; M3 <= 4.2 (no m-proportional tail)
- `layer=o load=decode_32 m=32` flat: m > 16 keeps master swap-AB
- `layer=unpooler load=decode_16 m=16` flat: full weight, master
- `layer=router *` flat: untouched
- `layer=qkv_gate *` flat: untouched
- `layer=pooler *` flat: untouched

## Risk

Counter-evidence to weigh: bf16-small-m-cuda-gemv-rewrite lost on exactly these rows (o m=8 6.32 -> 9.52, slice m=8 6.13 -> 7.55); this stage differs in the two mechanisms that read named as the cause (no ring on 32-64 KB streams; no global partial store + ticket tail). Risks: a first-launch fault poisons the context (LESSONS 2026-10-25), so prints around each self-check launch and check mode only; cluster co-residency (print cudaOccupancyMaxActiveClusters; fall back if below the grid). Milestones: M1 9/9 tests with m=5 unpooler through the kernel; M2 every routed row at or below master; M3 o rows <= 5.0-5.2, slice rows <= 4.2. Headline at M3: ~1 us o + ~2 us slice on decode_1 and verify_1, ~1 us o on decode_16: about +4-6% on decode_1/verify_1, +1.5% decode_16.

## Milestones

Rewrite stage 1 of `o at m<=16 (_pooler_skinny_kernel, 4.1-4.4 us kernel, rows 5.9-6.3) and the unpooler slice at m<=8 (_bf16_splitk_kernel/ticket split-K 8, 4.3-4.4 us kernel, rows 5.9-6.1)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T004153Z by an adversarial subagent (model opus).

**Verdict: proceed.** Sound stage-1 rewrite aimed at the two shapes furthest from their floors; it fixes both causes REWRITE_READ section 2 found for the gemv loss (a ring that never fills; a global partial store plus a ticket tail that grows with m). Proceed as a check. Before building, correct the incumbent baselines (it quotes the restack's kernels, not master's), and add an smem swizzle and a cluster-residency gate.

### Justification
o (16.8 MB) and the unpooler slice (8.4 MB) are latency-bound 32-64 KB/CTA streams; issuing each CTA's whole slice at once puts the full weight in flight in one wave, and the DSMEM fixed-order reduce removes the gpu-scope ticket and fp32 global workspace (the m-proportional costs the read measured). Uses only baseline sm_90 (mma.sync, cp.async.cg, cluster launch, mapa, ld.shared::cluster, barrier.cluster); no wgmma. Weak points: (1) M3 needs the slice row at 4.0-4.2 us, but slice rows sit ~1.5 us above kernel time (master 4.5->5.94, restack 4.3->6.2), needing a ~2.5 us cold kernel for 8.4 MB plus cluster fixed cost; M2 credible, M3 on the slice a stretch, o <=5.0 plausible. (2) Baselines: master 8b9ad43 routes o m<=16 to _pooler_skinny_kernel and the slice to _bf16_splitk_kernel split-K 8; run 1602a4e947e64090 has o kernels 4.6/5.2/4.8 rows 6.32/6.32, slice kernels 4.5/4.6 rows 5.94/6.13. The quoted figures come from the restack run 60b20fd0bf924267. Judge M2 against master and also report vs the restack. (3) Bank conflicts: 256 B / 1 KB power-of-two row pitches make unswizzled ldmatrix ~8-way conflicted (~2 us); XOR-swizzle or +16 B pad the cp.async destination and ldmatrix addresses. Cluster residency: o needs 32 co-resident 8-CTA clusters at 2 CTAs/SM, an exact fit on 132 SMs; fall back to cluster dim 4 (64 x 4, K 256/rank) when cudaOccupancyMaxActiveClusters < 32, not straight to master; the slice (16 clusters at 1 CTA/SM) has the same risk. Smem ~80 KB, 2 CTAs/SM fits; alias the fp32 partial onto consumed weight buffer. Judge milestones on rows (warm-L2 caveat).

### Confounders on the current kernel
- bf16-shape-gated-triton-winners-combo (landed): supplies the replaced kernels; the true incumbent; compare to run 1602a4e947e64090.
- bf16-restack-m16-swapab-router-swapab-m32 (not landed, pending user): re-routes the same shapes; report deltas vs both.
- bf16-m32-swapab-wgmma-triton (landed): owns o at 17..32; rewrite stops at m=16, so o decode_32 flat is valid.
- router-hilo / sglang-gemm-dispatch-parity / router-hilo-at-m8: other layers; no interference.

### Interference and A/B needs
No companion. Judge on replaced kernels' time per shape vs master (1602a4e947e64090) and restack (60b20fd0bf924267). The queued router-cuda-cluster-dsmem-splitk-rewrite uses the same design: give each its own load_inline extension name and its own route-ok constant (LESSONS 2026-10-25).

### Open questions
- Fix the expected/replaces text to master's numbers before the build.
- cudaOccupancyMaxActiveClusters on the judge for cluster 8 at 2 CTAs/SM (o needs 32) and 1 CTA/SM (slice needs 16)? Fall back to cluster dim 4.
- Is the smem tile swizzled or padded for ldmatrix?
- Slice at 256 CTAs (32 KB, 2/SM) instead of 128 x 64 KB? Worth a prepare diagnostic.
- Use the prepare cold-L2 event time to decide whether slice M3 is reachable before stage 2.


## Implementation notes

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

## Rules check

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

## Pre-flight audit

Audited 2026-10-24 (opus) by reading the diff plus a numpy emulation of the kernel's exact smem/fragment indexing (cp.async swizzled destinations, ldmatrix x4 lane addressing, mma.sync m16n8k16 A/B/C fragment layouts, K-group partial adds, DSMEM rank reduce) for all six template instances (o NT 8/16 x CL 8/4, slice CL 8/4, m 1/5/8/13/16): all match x @ W.T to fp32 roundoff (max abs err <= 8e-5), no unwritten output.

**Verdict: dispatch-with-fixes** (run as `okc check`).

Checked and fine: mapa rank/offset (16 B aligned: PR*4 and RPR*4 multiples of 16) and arrive.release/wait.acquire pairing, executed by all threads; second cluster barrier before exit; fp32 partial aliased onto the weight buffer only after the last cp.async.wait_group 0 + __syncthreads (all ldmatrix done), and peers read only partials after barrier 1; XOR swizzle identical in cp.async and ldmatrix (row & 7 == lane & 7 in both operands, CHG multiple of 8); cudaLaunchKernelEx on the current torch stream with cluster attribute, MaxDynamicSharedMemorySize set in prep() before occupancy/launch, C10_CUDA_KERNEL_LAUNCH_CHECK, graph-capturable; occupancy gate CL 8 -> CL 4 -> master never launches more clusters than reported; static_asserts pass for all six instances; self-check gates (rel L2 3e-3, exact one-hot probe, bit-identical rerun) are passable; routing on cfg/shape/stride/alignment and a prepare constant only; torch.empty output, fp32 accumulate, fixed-order reduce, bf16 store; no host sync, no cache from data, no process beyond load_inline. No ptxVersion gate exists.

Fixes:
- `_build_oun`: set `TORCH_CUDA_ARCH_LIST=9.0a` around `load_inline` and restore the previous value (or unset) in `finally`, as the spec and the router sibling do. The implementation compiled for whatever the image's arch list holds, which can multiply nvcc time inside the 10-minute run limit; the kernel uses no wgmma so 9.0a is safe. Build print now also shows the previous value.

Residual risk: a first-launch fault still poisons the context unrecoverably (check mode only); o CL 8 may report < 32 co-resident clusters on chips with 14-SM GPCs and drop to CL 4, and the 128-CTA slice may pack 2 CTAs/SM (launch_bounds 256,2), costing bandwidth rather than correctness.
