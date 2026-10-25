---
problem: muse_bf16_proj
slug: router-cluster-dsmem-s2-flat-in-m-occupancy
created: 2026-10-25T042040Z
base: master
area: router GEMM (m<=32, cluster DSMEM kernel)
change: >
  Build on branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (stage 1, dc5d03c4).
  Stage 1 won rows at m=1/8 only by deleting the memset node: its kernel was SLOWER than master's
  (3.6/4.3/4.6 us at m=1/8/16 vs hilo 3.2/3.2/3.5), i.e. 8.4 MB at ~2.3 TB/s with 4 warps and 1
  CTA/SM, and it grew with m. Stage 2 changes only the kernel's decomposition, not the routing:
  (1) cluster dim 16 along K (set cudaFuncAttributeNonPortableClusterSizeAllowed=1; fall back to 8
  if cudaOccupancyMaxActiveClusters returns 0 for 16, and print which): 16 clusters x 16 CTAs =
  256 CTAs (2 per SM), each CTA owning K [512r, 512r+512): 32 KB of hi+lo weight; (2) 8 warps per
  CTA each taking a 64-K slice, so 16 warps/SM are in flight instead of 4; (3) load x for the
  CTA's K slice once into smem with tokens padded to NT and reuse the A (hi/lo) fragments across
  all NT/8 B fragments (decode-once, as stage 1 of the lossless line learned); (4) the DSMEM
  reduce stores each rank's slice of columns once over 16 ranks in fixed order. Keep NT 8/16/32,
  the same self-check bounds and run-twice bit-identical check, torch.empty output, one node.
  Routing stays as stage 1 (cluster at 1 <= m <= 32, master above). Print per-m kernel time (cold
  L2) and regs/spills/smem.
expected:
  - shapes: layer=router load=decode_1 m=1
    direction: faster
    why: stage-1 row 5.24 (kernel 3.6); floor 1.8; target kernel <= 2.8, row <= 4.6
  - shapes: layer=router load=verify_1 m=8
    direction: faster
    why: stage-1 row 5.08 (kernel 4.3); target flat-in-m kernel <= 2.9
  - shapes: layer=router load=decode_16 m=16
    direction: faster
    why: stage-1 row 5.87 (4.6) lost to restack swapab 5.47; target <= 5.0
  - shapes: layer=router load=decode_32 m=32
    direction: faster
    why: stage-1 fell back to hilo 6.85; now routed, target <= 5.5 (restack swapab 6.2)
  - shapes: layer=router load=verify_* m=12*
    direction: flat
    why: m=128 keeps master
  - shapes: layer=router load=verify_32 m=256
    direction: flat
    why: keeps master
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
depends_on: router-cuda-cluster-dsmem-splitk-rewrite
priority: 2
kind: rewrite
controls_for: null
rewrite_stage: 2
replaces: _router_swapab_kernel + memset at router m<=32 (restack) / _router_hilo_kernel + memset (master)
review: proceed
status: done
review_at: 2026-10-25T042412Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T042413Z
base_sha: 0dea576ae73cfb9d114326146a90d40766b24f0e
base_blob: 441dcedb5cd6286e6851afb629598d27c30f6c82
branch: hyp/muse_bf16_proj/router-cluster-dsmem-s2-flat-in-m-occupancy
worktree: .worktrees/muse_bf16_proj/router-cluster-dsmem-s2-flat-in-m-occupancy
mode: check
restack_of: null
stale_base: true
implemented_at: 2026-10-25T043058Z
impl_commit: 647c57de3b765569021a4563ae5aa6c20452e5a6
submission_sha256: 0e9da739b6792b33b87518ad246b0fa848811b4886bd239cbfa9ecb4c596ba7f
norm_sha256: aebf5f794e8af4af09adcb2558e059c6c780cb86fb9739c54e0a737b3035f1de
submission_blob: 9f00e0ed1a2bbb4ff08465385f6c583d07c83efa
lint_warnings: 3
attempt_file: 2026-10-25T043058Z-check-router-cluster-dsmem-s2-flat-in-m-occupancy.md
prepared_at: 2026-10-25T043058Z
job: ef8a8bca31ef420e
dispatched_at: 2026-10-25T043105Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.205837800836652
speedup_ci: [1.1877502338241022, 1.2239253678492017]
harvested_at: 2026-10-25T053759Z
best_at_harvest: 1.2215006063849365
promotion: null
---

# router-cluster-dsmem-s2-flat-in-m-occupancy

## Change

Build on branch hyp/muse_bf16_proj/router-cuda-cluster-dsmem-splitk-rewrite (stage 1, dc5d03c4). Stage 1 won rows at m=1/8 only by deleting the memset node: its kernel was SLOWER than master's (3.6/4.3/4.6 us at m=1/8/16 vs hilo 3.2/3.2/3.5), i.e. 8.4 MB at ~2.3 TB/s with 4 warps and 1 CTA/SM, and it grew with m. Stage 2 changes only the kernel's decomposition, not the routing: (1) cluster dim 16 along K (set cudaFuncAttributeNonPortableClusterSizeAllowed=1; fall back to 8 if cudaOccupancyMaxActiveClusters returns 0 for 16, and print which): 16 clusters x 16 CTAs = 256 CTAs (2 per SM), each CTA owning K [512r, 512r+512): 32 KB of hi+lo weight; (2) 8 warps per CTA each taking a 64-K slice, so 16 warps/SM are in flight instead of 4; (3) load x for the CTA's K slice once into smem with tokens padded to NT and reuse the A (hi/lo) fragments across all NT/8 B fragments (decode-once, as stage 1 of the lossless line learned); (4) the DSMEM reduce stores each rank's slice of columns once over 16 ranks in fixed order. Keep NT 8/16/32, the same self-check bounds and run-twice bit-identical check, torch.empty output, one node. Routing stays as stage 1 (cluster at 1 <= m <= 32, master above). Print per-m kernel time (cold L2) and regs/spills/smem.

## Expected

- `layer=router load=decode_1 m=1` faster: stage-1 row 5.24 (kernel 3.6); floor 1.8; target kernel <= 2.8, row <= 4.6
- `layer=router load=verify_1 m=8` faster: stage-1 row 5.08 (kernel 4.3); target flat-in-m kernel <= 2.9
- `layer=router load=decode_16 m=16` faster: stage-1 row 5.87 (4.6) lost to restack swapab 5.47; target <= 5.0
- `layer=router load=decode_32 m=32` faster: stage-1 fell back to hilo 6.85; now routed, target <= 5.5 (restack swapab 6.2)
- `layer=router load=verify_* m=12*` flat: m=128 keeps master
- `layer=router load=verify_32 m=256` flat: keeps master
- `layer=qkv_gate *` flat: untouched
- `layer=o *` flat: untouched
- `layer=pooler *` flat: untouched
- `layer=unpooler *` flat: untouched

## Risk

Judge against stage 1's rows (dc5d03c4) and the restack's router rows. Milestone M2: kernel time within 0.3 us from m=1 to m=32 and <= 3.0 us. Kill: m=16 and m=32 rows not below the restack's 5.47/6.2. Built on master (stage 1's base); a win restacks onto the restack afterwards.

## Milestones

Rewrite stage 2 of `_router_swapab_kernel + memset at router m<=32 (restack) / _router_hilo_kernel + memset (master)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T042412Z by an adversarial subagent (model opus).

**Verdict: proceed.** Worth running as stage 2, but stage 1's own occupancy prints show the planned 16x16-at-2-CTAs/SM layout probably will not fit in one wave. Stage 1 reported max active clusters = 15 for 8-SM clusters, and the fallback in the spec only fires when the query returns 0. Fix the gate (>= 16, else a 4-warp cs=8 instance, never a split wave), cap registers for 2 CTAs/SM, preferably build on current master b99b1dd, and judge on router rows and in-process cold-L2 prints, not the headline.

### Justification
Mechanism, partly supported. Stage 1 (dc5d03c4) ran cs=8 NT 8/16 at 162-168 regs, 128 threads and 1 CTA/SM. Profile kernels were 3.6/4.6/4.3 us at m=1/16/8 against the restack's _router_swapab_kernel at 3.1/3.4/3.0 plus a 0.7-0.8 us memset. Stage 1 already issued all 8.4 MB of weight up front (64 KB per CTA x 128), so 'more bytes in flight' is not what stage 2 adds. Stage 2 adds more warps to overlap cp.async issue, ldmatrix/MMA and the tail, and halves per-CTA bytes. That is plausible, but the in-process split in stage 1's log contradicts the framing. Cold-L2 eager: m=1 cluster 10.08 vs no-DSMEM-reduce 9.02; m=8 10.21 vs 8.96; m=16 10.82 vs 9.60. So the cluster barriers plus DSMEM reduce cost about 1.1-1.25 us, about a third of the kernel. Going to cluster 16 makes that tail worse: 16-CTA barriers twice, and each output element summed over 16 remote ranks instead of 8. Only the main loop (about +0.6 us from NT 8 to 16) is what the occupancy change addresses. WEAKEST LINK: co-residency. Stage 1 printed max active clusters = 15 for cs=8 NT=32 (133 KB smem, 1 CTA/SM, so 8 SMs per cluster) and 30 for cs=8 NT=8 (2 CTAs/SM, 4 SMs per cluster). This H200 therefore exposes about 15 disjoint 8-SM cluster slots. A 16-CTA cluster at 2 CTAs/SM also needs 8 SMs, so cudaOccupancyMaxActiveClusters(16) will most likely return about 15, not 16. The spec falls back to cs=8 only on 0, so it would launch 16 clusters with one in a second wave and roughly double the kernel time. The same geometry rules out the obvious alternative of 32 clusters x 8 at 2 CTAs/SM: 256 CTAs need 32 slots of 4 SMs against 30 available. Registers are a second limit: 2 CTAs x 256 threads needs <= 128 regs/thread, but stage 1 ran 162-198 at 128 threads. Without __launch_bounds__(256,2) the kernel drops to 1 CTA/SM, which leaves about 7 clusters of 16 co-resident. Milestone readability: stage 1's identical router_kernel<8,8> instance read 3.6 us at m=1 and 4.3 us at m=8, and row minus kernel was 1.64 vs 0.78 us. Profile kernel times therefore carry about 0.7 us of noise, and M2 ('within 0.3 us from m=1 to m=32') cannot be read from top_kernels; it must come from the in-process cold-L2 prints. Targets: a kernel <= 2.8 us is 3.0 TB/s on 8.4 MB (floor 1.75 us at about 4.8 TB/s), within the ~3.5 TB/s ceiling seen here, so it is reachable. Rows <= 4.6 (m=1) and <= 5.0/5.5 (m=16/32) follow if the kernel lands at 2.8-3.0, because the memset node is gone. The expected directions match TASK shapes (all router rows weight 62). The m=128/256 and other-layer flats hold only if the base is current master (see interference).

### Confounders on the current kernel
- bf16-restack-m16-swapab-router-swapab-m32 (b99b1dd, current master): the change is built on the stage-1 branch, cut from 8b9ad43 BEFORE the restack. qkv_gate/o/unpooler m<=16 will run the old Triton paths and lose 5-9% against the restack base, as stage 1 did (qkv decode_1 0.919, o decode_16 0.912), so the headline lands near 1.19 and the harvester will label it a regression whatever the router does. Router m=9..32 rows are compared against _router_swapab_kernel rows (5.47/6.2) that are not in this build. Frontmatter says base: master while the change says the stage-1 branch; resolve to master.
- router-triton-hilo-fused-control-splitk-all-m: supplies router_hi/router_lo [256, 8192] bf16 row-major built in prepare(), the operand layout the cluster kernel reads, and the _router_hilo_kernel + memset fallback for m>32 and for any gate failure. If the gate silently falls back (as stage 1 did at m=32), the row is master's hilo, not the new kernel. Confirm the kernel name per row.
- bf16-m32-swapab-wgmma-triton / bf16-shape-gated-triton-winners-combo: other layers only, but together with the extra load_inline build (~40 s) they share the prepare() budget and the first-launch fault risk for the CUDA context. No numeric interaction.
- router-hilo-at-m8, sglang-gemm-dispatch-parity: superseded on the router path. No confound.

### Interference and A/B needs
No companion is warranted; this is a rewrite stage and the next stage is the follow-up. The same-run A/B already exists: stage 1's prepare() prints cold-L2 eager medians for the cluster kernel, the cluster kernel with the reduce removed (mode=0) and master's _router_forward at m in {1,8,16,32}. These attribute main loop vs reduce vs incumbent without a second run, provided stage 2 keeps them and adds m=32 and a stage-1-instance (cs=8, 4 warps) line. The one ambiguity a companion would otherwise address is the base: built on the stage-1 branch, a headline gain or loss mixes the router change with the missing restack. Fix that by building on master b99b1dd, porting the scaffold exactly as restack-router-cluster-dsmem-m8 does, rather than with an ab run. With that base, the non-router rows and the m>32 router rows are byte-identical to the base and the headline is interpretable. Relation to restack-router-cluster-dsmem-m8 (minted alongside): that run is the promotable primary. It puts the UNCHANGED stage-1 kernel (cs=8, NT=8) on b99b1dd at m<=8 only and keeps the restack's swapab at m 9..32. This stage 2 is the kernel-improvement line, which tries to beat both stage 1 at m<=8 and the restack swapab at m 9..32. They do not duplicate each other and can run in parallel. restack-m8's router decode_1/verify_1 rows on b99b1dd are the correct m<=8 incumbent for judging stage 2, and stage 2 should be judged on those rather than on stage 1's 5.24/5.08 from the old base. If stage 2 wins, the landing is a swap of the kernel body inside restack-m8's port plus widening its gate to m<=32 where the new rows beat 5.47/6.2. Mint that as its own restack. Do not land stage 2's branch directly.

### Open questions
- Fix the gate: require cudaOccupancyMaxActiveClusters(cs=16) >= 16 (every cluster co-resident in one wave), not > 0. Stage 1's 15 for 8-SM clusters predicts about 15 here. If it fails, fall back to stage 1's cs=8 4-warp instance (or master), never to a split wave. Print the value per NT.
- Declare __launch_bounds__(256, 2) (<= 128 regs) and print regs/spills. Stage 1 sat at 162-198 regs with 128 threads. If 2 CTAs/SM cannot be had without spills, the better occupancy path given 15 slots of 8 SMs is 4 warps/CTA with 3 CTAs/SM (__launch_bounds__(128,3), <= 170 regs, 12 warps/SM). That gives about 5.3 SMs per 16-CTA cluster, so 16 clusters should fit.
- The DSMEM reduce tail measured about 1.1-1.25 us in stage 1 (cluster vs mode=0, cold-L2 eager). Cluster 16 doubles the ranks per sum and widens the barrier. Keep the mode=0 diagnostic so the run shows whether the main-loop gain is eaten by a larger tail. Consider reducing only within 8-rank halves and finishing with one ld.shared::cluster pair.
- Read M2 (flat within 0.3 us, <= 3.0 us) from the in-process cold-L2 prints, not from profile top_kernels. Stage 1's identical NT=8 instance read 3.6 vs 4.3 us at m=1 vs m=8.
- Base: build on master b99b1dd (port the scaffold as restack-router-cluster-dsmem-m8 does) instead of the stage-1 branch from 8b9ad43, so non-router rows and the headline are comparable to BEST.
- At harvest, confirm PATH CHOSEN per m, the router_kernel<16,NT> name (not <8,NT> and not _router_hilo_kernel) on decode_1/verify_1/decode_16/decode_32, and no vectorized_elementwise memset node before reading rows.


## Implementation notes

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

## Rules check

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

## Result

- job `ef8a8bca31ef420e` (check), status done, passed True, tests 9/9
- speedup 1.2058x, 95% CI [1.1878, 1.2239], score 4006.1772791068784 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "1e90292c7b954d96"}; base result: muse_bf16_proj/hypotheses/done/regression/router-cuda-cluster-dsmem-splitk-rewrite.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-cluster-dsmem-s2-flat-in-m-occupancy.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-cluster-dsmem-s2-flat-in-m-occupancy.patch`; branch `hyp/muse_bf16_proj/router-cluster-dsmem-s2-flat-in-m-occupancy`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| layer=qkv_gate load=decode_1 m=1 | 62 | 13.56 | 12.75 | 1.0647 | gained |
| layer=o load=decode_1 m=1 | 62 | 6.26 | 5.94 | 1.0626 | gained |
| layer=pooler load=decode_1 m=1 | 62 | 19.58 | 19.15 | 1.007 | flat |
| layer=unpooler load=decode_1 m=1 | 62 | 6.55 | 5.67 | 1.1044 | gained |
| layer=router load=decode_1 m=1 | 62 | 5.24 | 5.3 | 0.9702 | flat |
| layer=qkv_gate load=decode_16 m=16 | 62 | 14.07 | 13.13 | 1.0532 | gained |
| layer=o load=decode_16 m=16 | 62 | 6.67 | 6.15 | 1.0801 | gained |
| layer=pooler load=decode_16 m=16 | 62 | 19.69 | 19.55 | 1.0148 | flat |
| layer=unpooler load=decode_16 m=16 | 62 | 18.63 | 18.53 | 0.9953 | flat |
| layer=router load=decode_16 m=16 | 62 | 5.87 | 5.64 | 1.0162 | flat |
| layer=qkv_gate load=decode_32 m=32 | 62 | 13.93 | 13.82 | 0.9979 | flat |
| layer=o load=decode_32 m=32 | 62 | 6.45 | 6.42 | 1.0146 | flat |
| layer=pooler load=decode_32 m=32 | 62 | 20.03 | 19.75 | 1.0001 | flat |
| layer=unpooler load=decode_32 m=32 | 62 | 18.54 | 18.1 | 1.0023 | flat |
| layer=router load=decode_32 m=32 | 62 | 6.85 | 5.55 | 1.1836 | gained |
| layer=qkv_gate load=verify_1 m=8 | 62 | 13.68 | 12.72 | 1.0794 | gained |
| layer=o load=verify_1 m=8 | 62 | 6.53 | 6.14 | 1.0519 | gained |
| layer=pooler load=verify_1 m=8 | 62 | 19.33 | 19.42 | 1.0096 | flat |
| layer=unpooler load=verify_1 m=8 | 62 | 6.29 | 6.31 | 1.0301 | gained |
| layer=router load=verify_1 m=8 | 62 | 5.08 | 5.62 | 0.9077 | lost |
| layer=qkv_gate load=verify_16 m=128 | 62 | 16.59 | 17.14 | 0.9875 | flat |
| layer=o load=verify_16 m=128 | 62 | 7.69 | 7.69 | 1.0084 | flat |
| layer=pooler load=verify_16 m=128 | 62 | 24.81 | 25.61 | 0.9593 | lost |
| layer=unpooler load=verify_16 m=128 | 62 | 20.0 | 20.4 | 0.9788 | flat |
| layer=router load=verify_16 m=128 | 62 | 7.54 | 7.75 | 0.9926 | flat |
| layer=qkv_gate load=verify_32 m=256 | 62 | 19.83 | 19.66 | 1.0068 | flat |
| layer=o load=verify_32 m=256 | 62 | 9.83 | 9.44 | 0.9996 | flat |
| layer=pooler load=verify_32 m=256 | 62 | 31.82 | 31.09 | 1.0224 | flat |
| layer=unpooler load=verify_32 m=256 | 62 | 24.75 | 24.78 | 0.9955 | flat |
| layer=router load=verify_32 m=256 | 62 | 9.63 | 9.53 | 1.0015 | flat |

gained on qkv_gate decode_1, decode_16, verify_1; o decode_1, decode_16, verify_1; unpooler decode_1, verify_1; router decode_32; lost on router verify_1; pooler verify_16. Expected hit: 4/10. Noise ±1.5%.

## Follow-ups

- rewrite stage 2 of `_router_swapab_kernel + memset at router m<=32 (restack) / _router_hilo_kernel + memset (master)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_bf16_proj "<what to fix or tune>" --kind rewrite --depends-on router-cluster-dsmem-s2-flat-in-m-occupancy --stage 3`; a slow or failed first cut is the expected start, not a do-not-retry.
- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: router-cluster-dsmem-s2-flat-in-m-occupancy`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

