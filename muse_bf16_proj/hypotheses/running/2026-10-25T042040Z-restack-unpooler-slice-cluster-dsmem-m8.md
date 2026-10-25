---
problem: muse_bf16_proj
slug: restack-unpooler-slice-cluster-dsmem-m8
created: 2026-10-25T042040Z
base: master
area: unpooler GEMM (sliced n=1024, m<=8)
change: >
  Built on master (b99b1dd, the landed restack). Port ONLY the unpooler-slice half of the oun
  kernel from branch hyp/muse_bf16_proj/o-unpooler-slice-cuda-cluster-dsmem-rewrite (job 3f3330f7:
  16 clusters x 8 CTAs = 128 CTAs, cluster owns 64 rows of W[:1024], rank r owns K [512r,
  512r+512), all 64 KB in flight, DSMEM reduce, transposed bf16 store, torch.empty, one node).
  Keep its load_inline scaffold (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, -std=c++20 in
  extra_cuda_cflags replacing any other -std), self-check at m in {1, 5, 8} and
  print-and-fall-back. Compile only the unpooler instantiation (NT=8); do NOT route o (o lost
  +14..+48% to the incumbent). Gate the route on a COLD-L2 prepare timing (flush a buffer > 50 MB
  before each event, median of 20, synthetic x): enable only if the cluster beats the restack's
  own unpooler-slice path (_swapab16_forward for unpooler at m<=8) at m=1 and m=8; print both. The
  previous run gated on warm eager times. In kernel_fn insert, before the swapab16 check: if layer
  == 'unpooler' and cfg n == 1024 and 1 <= m <= 8 and state['oun_ok'] -> cluster forward.
  Everything else byte-identical to the restack.
expected:
  - shapes: layer=unpooler load=decode_1 m=1
    direction: faster
    why: restack row 6.15; oun read 5.02 on master (-15.5%, z -33); floor 1.7
  - shapes: layer=unpooler load=verify_1 m=8
    direction: faster
    why: restack row 6.00; oun read 5.56 on master (-9.4%, z -21)
  - shapes: layer=unpooler load=decode_16 m=16
    direction: flat
    why: full weight, untouched
  - shapes: layer=unpooler load=decode_32 m=32
    direction: flat
    why: untouched
  - shapes: layer=unpooler load=verify_* m=*
    direction: flat
    why: m>=128 untouched (verify_1 excepted above)
  - shapes: layer=o *
    direction: flat
    why: o not routed
  - shapes: layer=qkv_gate *
    direction: flat
    why: untouched
  - shapes: layer=pooler *
    direction: flat
    why: untouched
  - shapes: layer=router *
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: running
review_at: 2026-10-25T042349Z
review_companions: null
stage: dispatched
claimed_at: 2026-10-25T042349Z
base_sha: fd92cb3a1268dd03948af28bfe64ae31f385757e
base_blob: 970b415b746fa03cb92e419d270be92588273178
branch: hyp/muse_bf16_proj/restack-unpooler-slice-cluster-dsmem-m8
worktree: .worktrees/muse_bf16_proj/restack-unpooler-slice-cluster-dsmem-m8
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T043016Z
impl_commit: 5f0b29891e442c7242e78519042f722b90594de4
submission_sha256: db26d0d70cb123c200e7ea3fe6fa87430aa97412aaa34cf0f446a532a255556e
norm_sha256: e9b3280b0bb3092c0fa286a242450f515529241c794cf8dc26e43ac38e8ecb9d
submission_blob: 5b37668b80c0d70c2839a12eccae7da0b9cd5795
lint_warnings: 7
attempt_file: 2026-10-25T043016Z-check-restack-unpooler-slice-cluster-dsmem-m8.md
prepared_at: 2026-10-25T043016Z
job: e557d92969864449
dispatched_at: 2026-10-25T043024Z
---

# restack-unpooler-slice-cluster-dsmem-m8

## Change

Built on master (b99b1dd, the landed restack). Port ONLY the unpooler-slice half of the oun kernel from branch hyp/muse_bf16_proj/o-unpooler-slice-cuda-cluster-dsmem-rewrite (job 3f3330f7: 16 clusters x 8 CTAs = 128 CTAs, cluster owns 64 rows of W[:1024], rank r owns K [512r, 512r+512), all 64 KB in flight, DSMEM reduce, transposed bf16 store, torch.empty, one node). Keep its load_inline scaffold (TORCH_CUDA_ARCH_LIST=9.0a, -lineinfo, -std=c++20 in extra_cuda_cflags replacing any other -std), self-check at m in {1, 5, 8} and print-and-fall-back. Compile only the unpooler instantiation (NT=8); do NOT route o (o lost +14..+48% to the incumbent). Gate the route on a COLD-L2 prepare timing (flush a buffer > 50 MB before each event, median of 20, synthetic x): enable only if the cluster beats the restack's own unpooler-slice path (_swapab16_forward for unpooler at m<=8) at m=1 and m=8; print both. The previous run gated on warm eager times. In kernel_fn insert, before the swapab16 check: if layer == 'unpooler' and cfg n == 1024 and 1 <= m <= 8 and state['oun_ok'] -> cluster forward. Everything else byte-identical to the restack.

## Expected

- `layer=unpooler load=decode_1 m=1` faster: restack row 6.15; oun read 5.02 on master (-15.5%, z -33); floor 1.7
- `layer=unpooler load=verify_1 m=8` faster: restack row 6.00; oun read 5.56 on master (-9.4%, z -21)
- `layer=unpooler load=decode_16 m=16` flat: full weight, untouched
- `layer=unpooler load=decode_32 m=32` flat: untouched
- `layer=unpooler load=verify_* m=*` flat: m>=128 untouched (verify_1 excepted above)
- `layer=o *` flat: o not routed
- `layer=qkv_gate *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=router *` flat: untouched

## Risk

Kill: unpooler decode_1/verify_1 rows not below restack 6.15/6.00, or the cold-L2 gate disables the route (then the run measures the restack; the printed cold numbers decide whether the design is dead). Estimated headline +0.003.

## Adversarial review

Reviewed 2026-10-25T042349Z by an adversarial subagent (model opus).

**Verdict: proceed.** A narrow, attributable port. Only the unpooler n=1024 route at m<=8 changes, and job 3f3330f7 already measured that kernel's rows at 5.02/5.56 us (1.38x/1.28x paired vs SGLang) against the restack's 6.15/6.00. Proceed with no companion. Fix the gate: use a flush of at least 256 MB, not '>50 MB', and preferably graph-replay timing, so the gate does not wrongly disable the route.

### Justification
Mechanism: the 8.4 MB slice is latency-bound. Each 128-program Triton swap-AB split-K-8 CTA streams a small chunk through a 3-stage ring and then a semaphore tail. The cluster kernel puts all 64 KB per CTA in flight and reduces over DSMEM, which suits a cold-L2 graph replay. Evidence: on 3f3330f7 (base 8b9ad43) the slice rows were 5.02 us (m=1) and 5.56 us (m=8), against SGLang's 6.93/7.14, a paired 1.381x/1.284x. Restack master (b99b1dd) has 6.15/6.00 against 7.3/7.1, about 1.19x/1.18x. Each shape is a separate graph timed in pairs against SGLang, so the slice row does not depend on what other layers run, and the cross-base comparison holds. Floors are 1.75/1.76 us, so there is still headroom (the rows are 2.9x/3.2x the floor). The expected directions match TASK: m=1 decode_1 and m=8 verify_1 are the only shapes with n=1024 and every other row stays untouched. My headline estimate is about +0.005 to +0.007, not +0.003: -1.1 us on a ~48.5 us decode_1 load (~2.3%) and -0.45 us on verify_1 (~0.9%), geomean over 6 loads. That is still inside the ±0.018 CI, so read the result on the two rows' per-shape z, not the headline. WEAKEST LINK: the profile's kernel times do not show the mechanism. oun_kernel<8,64,512,8,4> reads 4.4/4.8 us, no faster than the restack's _bf16_swapab_kernel at 4.3/4.4 us. The whole gain is in the row-minus-kernel gap (0.6 us vs ~1.8 us), which the profiler's kernel time does not capture. The gain is real in the scored metric, but the prepare gate has to measure what the judge measures or it can disable a winning route. Prior evidence that cold eager is the right family: on 3f3330f7, cold eager ranked both routes correctly (o: cluster 13.11 vs master 12.19, and o lost; slice: 10.28/10.69 vs 11.05/11.19, and the slice won). Warm eager wrongly favoured o. But that run flushed 256 MB. 'A buffer > 50 MB' is about the size of H100's 50 MB L2 and will not reliably cold-flush. Eager timing also includes Triton's Python launch cost (~10 us, visible in master warm 17 us vs cluster 7 us). A short flush does not hide that cost, which biases the gate toward the cluster kernel. The judge's CUDA-graph replay has no such cost.

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32 (landed, b99b1dd): supplies the incumbent for exactly these two rows (_swapab16_forward 'unpooler_slice', grid (16,8), split-K 8, 3 stages, ws/cnt semaphore). The new branch must sit before the swapab16 check in kernel_fn, or it never fires. The incumbent is the comparison target, not a confounder of the measured delta. If the restack were reverted, the slice would fall to _unpool_forward (splitk8 ticket, 5.94/6.13 rows), which oun also beat on 3f3330f7.
- bf16-shape-gated-triton-winners-combo (landed, 658d63b): its _unpool_forward slice path is still in the file but shadowed by swapab16. The new route shadows both. No interaction beyond the fallback order.
- bf16-m32-swapab-wgmma-triton, router-triton-hilo-fused-control-splitk-all-m, router-hilo-at-m8, sglang-gemm-dispatch-parity (landed): different layers or m ranges. Each shape is its own captured graph with a cold L2, so they cannot change the slice row. The one shared cost is prepare/compile time: load_inline took ~31 s on 3f3330f7 plus Triton warmups, well inside the 10-minute limit.

### Interference and A/B needs
No companion is warranted. The change routes one new kernel on exactly two scored shapes (unpooler decode_1 m=1 and verify_1 m=8) plus the m=5 test, and leaves every other path byte-identical to master. A gain or a loss on those two rows is therefore attributable to the kernel alone, and any movement elsewhere is noise. Reverting a landed hypothesis (ab-revert) would only change the incumbent, and 3f3330f7 already gives the oun-vs-old-master comparison. A control that isolates a sub-change is not meaningful, because the kernel is one unit already measured. The only ambiguous outcome is the gate turning the route off. The run then measures the restack, and the printed cold timings decide. Make the gate faithful (below) so that case is informative rather than a false negative.

### Open questions
- BUILDER: flush at least 256 MB (as 3f3330f7 did), not '>50 MB'. H100 L2 is 50 MB.
- BUILDER: preferably time both candidates by CUDA-graph replay. Capture each forward once at m=1 and m=8, and replay with a 256 MB flush before each event pair. This removes Triton's launch cost from the gate and mirrors the judge. Print the eager-cold numbers too for continuity with 3f3330f7.
- BUILDER: -std=c++20 was not needed for oun; 3f3330f7 compiled in 31.4 s with torch's default flag. Keep it only if it compiles. Wrap load_inline so that a build failure prints the nvcc stderr tail and falls back, as the bulk s2 run does. Use a fresh extension name, distinct from muse_bf16_o_unpool_cluster_v1 and the router/bulk siblings, so a stale cache is not reused.
- BUILDER: keep the occupancy gate (CL 8 needs 16 clusters; it reported 45 on the judge) and the CL 4 fallback, the XOR swizzle, and the 9.0a arch-list set and restore. Self-check at m in {1, 5, 8} plus the one-hot probe, run twice and bit-identical.
- Readout: judge on the two rows' paired speedup vs SGLang. Expected unpooler decode_1 ~5.0 us and verify_1 ~5.5 us. If the m=8 row lands at or above 6.0 while m=1 wins, the m-dependence the rewrite targeted is still present at NT=8, and stage-2 work should be a 256-CTA x 32 KB slice variant (the previous review's open question).


## Implementation notes

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

## Rules check

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

## Pre-flight audit

Audited 2026-10-25 by an opus subagent, by reading only (no GPU here). **Verdict: dispatch.** No fixes were made.

- Kernel: the CUDA source is byte-identical to 3f3330f7's `oun_kernel`, apart from the rename and the dispatch losing the o instantiations (checked by diff). Fragment layout, XOR swizzle, partial aliasing (2176 B within a 64 KB weight buffer), the DSMEM reduce and the 8 B-aligned bf16 store all check out for CL 8 (72 KB smem) and CL 4 (80 KB). Rows with token >= m are zero-filled (src_bytes 0) and are never stored.
- Routing: `layer == "unpooler"` plus `_ouns_route_ok` (n 1024, k 4096, 1 <= m <= 8, shapes, strides and alignment) branches only on cfg, shapes and the prepare-time int `ouns_cl`. It cannot shadow router, pooler, qkv_gate or o. The reference gives the unpooler n = 1024 only at m <= 8, so m 16/32/128/256/130/1000 keep their paths. The m = 5 test takes the new route and is covered by the self-check.
- kernel_fn path: no prints, no try/except, no per-call Python state. Each call does `torch.empty` and one `cudaLaunchKernelEx` (the static `prep()` flag was already set in prepare). The only host-side checks are TORCH_CHECK and `cudaGetLastError`.
- Gate: the flush is a 256 MB `zero_()` outside the graph, before `ev[0]`, the same convention 3f3330f7 used. It evicts the 8.4 MB slice from the 50 MB L2. The judge reads its flush buffer instead, so up to ~50 MB of dirty lines are written back during the timed replay. That affects both paths, so it is noted but not changed. Both paths are warmed 3x before capture (Triton M is do_not_specialize and m 1..8 were pre-warmed, so nothing compiles during capture). `torch.cuda.graph` always ends the capture on exit, and any exception returns False, which keeps the restack path, so a failed capture cannot leave a capture open. The gate doubles as a capturability probe. Cost: about 285 flushes of 256 MB plus 6 captures, well under 1 s. Ticket counters self-reset and are re-zeroed afterwards. A 1.02x threshold against the expected ~1.2x/1.08x is a fair gate, not one that always fails.
- Rules: OK. Self-check and gate use synthetic x, weights are read in place in bf16, and load_inline is the sanctioned build. Lint is OK with warnings, all of which are in prepare. py_compile passes.

Residual risk: m = 8 may land near the 1.02x line in the graph-replay gate (the expected margin is only ~1.08x), and the gate would then switch the route off and the run would measure the restack.
