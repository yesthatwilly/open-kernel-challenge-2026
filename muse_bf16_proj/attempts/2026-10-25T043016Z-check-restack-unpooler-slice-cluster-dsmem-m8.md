# muse_bf16_proj · 2026-10-25T043016Z · check · restack-unpooler-slice-cluster-dsmem-m8

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T042040Z-restack-unpooler-slice-cluster-dsmem-m8.md` (branch `hyp/muse_bf16_proj/restack-unpooler-slice-cluster-dsmem-m8`).

- **Job:** `e557d92969864449` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `5f0b298`

## Goal
Built on master (b99b1dd, the landed restack). Port ONLY the unpooler-slice half of the oun kernel from branch hyp/muse_bf16_proj/o-unpooler-slice-cuda-cluster-dsmem-rewrite (job 3f3330f7: 16 clusters x 8 CTAs = 128 CTAs, cluster owns 64 rows of W[:1024], rank r owns K [512r, 512r+512), all 64 KB in flight, DSMEM reduce, transposed bf16 store, torch.empty, one node). Keep its load_inline scaffold (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, -std=c++20 in extra_cuda_cflags replacing any other -std), self-check at m in {1, 5, 8} and print-and-fall-back. Compile only the unpooler instantiation (NT=8); do NOT route o (o lost +14..+48% to the incumbent). Gate the route on a COLD-L2 prepare timing (flush a buffer > 50 MB before each event, median of 20, synthetic x): enable only if the cluster beats the restack's own unpooler-slice path (_swapab16_forward for unpooler at m<=8) at m=1 and m=8; print both. The previous run gated on warm eager times. In kernel_fn insert, before the swapab16 check: if layer == 'unpooler' and cfg n == 1024 and 1 <= m <= 8 and state['oun_ok'] -> cluster forward. Everything else byte-identical to the restack.

## What changed, and where
Edited only `<worktree>/muse_bf16_proj/submission.py` (+586 lines, nothing removed). The oun branch is based on an old master, so section (6) was ported by hand, not merged.

What changed:
- New section (7) before `prepare()`: `ouns_kernel` ported unchanged from job 3f3330f7's `oun_kernel` (swizzle, DSMEM reduce, transposed bf16 store, `torch.empty`, one launch), but only the unpooler instantiation (NT=8; CL 8 = ROWS 64 / K 512, CL 4 fallback = ROWS 32 / K 1024). No o instantiation is compiled and o is not routed.
- Fresh extension name `muse_bf16_unpool_slice_cluster_v3` (distinct from `muse_bf16_o_unpool_cluster_v1` and the router/bulk siblings). `TORCH_CUDA_ARCH_LIST=9.0a` set around `load_inline` and restored; flags `-O3 -lineinfo`, printed in the build line. No `-std` flag is passed. Any build failure prints the last 3000 chars of the nvcc error and keeps master's path (never raises out of the import).
- Occupancy gate kept: CL 8 needs 16 co-resident clusters, else CL 4 needs 32, else off. Self-check kept at m = 1, 5, 8 plus a final m = 1, one-hot probe, each launch run twice and bit-identical, flushed print before and synchronize + print after each launch.
- `kernel_fn`: one new branch before the swapab16 check: `layer == "unpooler"` and `_ouns_route_ok` (n == 1024, 1 <= m <= 8, k 4096, alignment, `state["ouns_cl"]` > 0) -> `_ouns_forward`. `prepare()` calls `_ouns_prepare(state)` after `_swapab16_prepare`. All other paths are byte-identical to the restack.
- Prepare gate (`_ouns_gate`), the main change from the spec, per the review's fixes (the review wins over the spec's "flush > 50 MB, eager" wording):
  - flush 256 MB (`OUNS_FLUSH_BYTES`) before every replay, not 50 MB;
  - both paths timed by CUDA-graph replay (each captured once at m = 1 and m = 8 on synthetic x), so Triton's Python launch cost is out of the comparison;
  - the two graphs are interleaved per rep with the order swapped each rep, 41 reps, median;
  - an empty-graph baseline (a trivial `add_` graph, same flush, same timing) is subtracted from both medians, mirroring the judge's fixed-cost subtraction; raw medians, the baseline and the net values are all printed;
  - the cluster route is enabled only if incumbent / cluster >= 1.02 at BOTH m = 1 and m = 8, the incumbent being `_swapab16_forward("unpooler_slice")` with its prepared stage count (falls back to `_unpool_forward` only if swapab16 did not prepare);
  - eager-cold medians (20 reps, 256 MB flush) are also printed for continuity with 3f3330f7 but never decide;
  - a capture or timing exception leaves the restack path (the gate does not fall back to eager timing, which is biased toward the cluster kernel);
  - loud `[ouns] gate m=... cluster median X us, swapab16 median Y us ... -> WIN/LOSS` lines, then one `[ouns] PATH CHOSEN` line.
- Deviation: the spec's `-std=c++20` is dropped per the review (3f3330f7 compiled without it); the build-failure tail print covers the case where torch demands it. No mbarrier or spin-wait exists in this kernel (cluster barriers only), so there is nothing to bound; no sm_90a-only PTX is used (baseline sm_90 cluster/mapa/mma.sync/cp.async), so the `__CUDA_ARCH_FEAT_SM90_ALL` guard is not applicable (the arch list is still 9.0a and printed).

How to read the result: judge on the unpooler decode_1 and verify_1 rows (3f3330f7 read 5.02 / 5.56 us vs master 6.15 / 6.00 us), not the headline (inside the +/-0.018 CI). The profile's kernel time will NOT show the gain (`ouns_kernel` ~4.4/4.8 us vs the restack's `_bf16_swapab_kernel` 4.3/4.4 us); the gain lives in row time minus kernel time. If the m=8 row lands at or above 6.0 while m=1 wins, stage 2 should be the 256-CTA x 32 KB slice variant. If the gate prints LOSS, the run measures the restack and the printed graph-replay medians decide whether the design is dead.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: `kernel_fn` adds only a shape/cfg branch and one `torch.empty` + cluster launch on the current stream; the syncs and prints are all in prepare() (diff checked: lines in `_ouns_*` and `_build_ouns`).
- [x] Nothing derived from `data` is cached across calls: state holds only the prepare-time int `ouns_cl`; gate and self-check use synthetic x.
- [x] Weights: unchanged, read in place (bf16, first 1024 rows of the full unpooler weight); nothing repacked.
- [x] Activations BF16 where SGLang keeps BF16: bf16 in, fp32 accumulate in mma.sync, one bf16 store.
- [x] Attention sees every key: n/a, no attention in this problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added; `load_inline` is an in-process build (no `subprocess`/`os.fork` written by this diff).
- [x] All work happens inside the timed call; no output caching or replay: one fresh `torch.empty` output per call, kernel computes it.

okc_lint output:
```
warn  host-sync    L1404: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  dtype        L1450: flush = torch.empty(OUNS_FLUSH_BYTES, dtype=torch.uint8, device=dev)  (allowed only where SGLang does the same)
warn  host-sync    L1459: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1465: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1469: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1497: ev[1].synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L1557: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
```
The uint8 warning is the prepare-only 256 MB L2-flush buffer (never an activation or weight). py_compile passed; __pycache__ deleted.

## Risks / what could fail
medium
Kill: unpooler decode_1/verify_1 rows not below restack 6.15/6.00, or the cold-L2 gate disables the route (then the run measures the restack; the printed cold numbers decide whether the design is dead). Estimated headline +0.003.

## Result (after the run)
- job `e557d92969864449` (check), status done, passed True, tests 9/9
- speedup 1.2260x, 95% CI [1.2077, 1.2444], score 3940.1601354408376 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "1e90292c7b954d96"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/restack-unpooler-slice-cluster-dsmem-m8.result.json`, `muse_bf16_proj/hypotheses/done/neutral/restack-unpooler-slice-cluster-dsmem-m8.patch`; branch `hyp/muse_bf16_proj/restack-unpooler-slice-cluster-dsmem-m8`
