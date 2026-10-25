# muse_attention_decode · 2026-10-25T044042Z · check · full-decode1-sk4-cluster-dsmem-merge

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T042334Z-full-decode1-sk4-cluster-dsmem-merge.md` (branch `hyp/muse_attention_decode/full-decode1-sk4-cluster-dsmem-merge`).

- **Job:** `f519261b7c2b435f` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `2c94c5c`

## Goal
Stage 4 of the full-decode rewrite, built on master (stage 3 two-level ticket landed, c3c1b01). Full decode_1 is 22.6 us with the kernel at 18.6 us (0.89 TB/s) vs a 4.3 us floor; in stage 3 prepare grid 2xSM was slower than grid SM (24.7 vs 23.1), so more CTAs do not help and the tail is the merge chain: each of 132 CTAs writes a 4.2 KB fp32 partial, __threadfence, group atomic, the group last arrival reads ~11 partials, fence, second atomic, last group merges ~12 group slots + norm-gate. Change, at batch 1 only (config): launch the kernel with a thread-block cluster of C CTAs (cudaLaunchKernelEx, cluster dims C, grid a multiple of C: C=4 -> grid 132, C=2 -> 132, C=8 -> 128) and replace the FIRST ticket level with an on-chip merge: each CTA writes (m[16], l[16], acc[16x64]) to its own smem, barrier.cluster.arrive.release + wait.acquire, rank 0 reads the C-1 peer partials through mapa + ld.shared::cluster and LSE-merges, then writes ONE cluster partial to global, __threadfence, atomicAdd on ticket[0]; the last cluster merges the grid/C cluster partials, applies v_scale, the landed norm-gate epilogue, stores bf16 and resets the ticket (graph replay). No group tickets are used at B=1. Peers stay resident until rank 0 has read them (second barrier.cluster before exit). The stream-K tile split, ring and math are the landed code. Prepare: build C in {2,4,8}, time each against the landed two-level kernel (cold L2, median of 10), print all, choose the fastest and assert it beats the landed kernel by >= 3%, else keep the landed two-level path (PATH CHOSEN: full decode_1 cluster C=<c> or two-level (fallback: <timings>)). Self-check: the stage-3 B=1 cases (80000/4097/63/1 keys, kv 1.0 and 0.25) for each C, two replays to prove the ticket resets. B > 1 unchanged. BUILD/SAFETY as landed (sm_90a only, -std=c++20, -lineinfo, in-process canary, never a child process). First measurement is an okc check.

## What changed, and where
Edited only `muse_attention_decode/submission.py` in the worktree (uncommitted, +474/-12). Review wins over the spec where they differ (stated below).

- Kernel: `splitkv_attn_sk_kernel` gained two trailing template params `<..., int CL = 0, bool CGRP = false>`. Every landed instantiation (variants 0-3) has CL=0 and its code text is unchanged (the new branches are `if constexpr`); `sk_attn`, `sk_pick`, `sk_attrs` and variant 3 are untouched, so the landed two-level kernel is the byte-for-byte fallback.
- New device code `cluster_finish<CL, CGRP>`: each CTA writes its 4-warp-merged (M, L, o[16x64]) to its own smem (the dead ring, 4.25 KB), `barrier.cluster.arrive.release` + `wait.acquire`, rank 0 reads the CL-1 peers with `mapa` + `ld.shared::cluster` and LSE-merges, writes ONE cluster partial to global slot `cid`, `__threadfence`, acq_rel ticket over `nclu` clusters; the last cluster merges, applies v_scale + the landed `finalize_store` norm-gate and resets the ticket. Peers wait on a second barrier phase that rank 0 arrives on right after its DSMEM reads, so they stay resident until read. CGRP: group ticket over cluster partials (Gs = ceil(sqrt(nclu))) then a second level, as the landed two-level path.
- Empty CTAs: CTAs with no tile (bx >= min(grid, T)) inside a live cluster still run the cluster barriers with an empty partial (M=-inf, L=0, o=0); a cluster with no tile exits whole. This avoids exited-CTA barrier/DSMEM hazards.
- Host: `sk_attn_cluster` (cudaLaunchKernelEx, cluster dim C, B==1 and grid % C == 0 checked) and `sk_cluster_attrs` (regs, local, CTAs/SM cluster-less, `cudaOccupancyMaxActiveClusters` for (grid, C)). C <= 8 only, so NonPortableClusterSizeAllowed is not needed (noted in `cv_prepare`).
- Review fixes applied: (1) grid 2xSM cluster variants added: C=2/4/8 at grid ~SM (132/132/128) and 2xSM (264); (2) "cluster merge + group ticket over cluster partials" for C in {2,4} at both grids (10 candidates); (3) `cudaOccupancyMaxActiveClusters` printed per (C, grid) with resident CTAs per SM, plus a NOTE when 2 CTAs/SM does not hold (said in the [sk4] lines; the winner's line is in PATH CHOSEN); (4) the kill note data: a line compares grid 2xSM + C=8 with the landed kernel.
- Prepare (`_sk_cluster_select`, only once the landed B=1 path validated, all in try/except): each candidate runs 8 self-check cases (80000/4097/201/63/1 keys, kv 1.0 and 0.25, grids C*3, 2C and its own) and 3 graph-replay checks of kernel_fn (two replays each, NaN-poisoned output) with ticket sums asserted 0. Survivors are timed with the landed two-level kernel (grid SM) in CUDA-graph replay, cold L2 (256 MB flush before EVERY replay, order rotated each rep), median of 10. Deviation from the stage-3 diag timer, which flushes once per rep and lets later kernels ride a warm L2. A cluster variant is used only if <= 0.97x of the landed time; else PATH CHOSEN names the fallback with timings.
- `kernel_fn`: B=1 path reads the module constant `_B1_CL` (set at prepare) and calls `_sk_launch(variant 10+cvar, grid)`, else the landed variant 3. No per-call state, prints or try/except. B>1 untouched.
- Module name bumped to `muse_full_decode_splitkv_v4` (arch flags unchanged: TORCH_CUDA_ARCH_LIST=9.0a around load_inline, restored; -O3 -lineinfo as landed; no wgmma/setmaxnreg here).
- Uncompiled and unrun (no nvcc/GPU on this node). Five new kernel instantiations lengthen the build.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: kernel_fn only reads the `_B1_CL` constant; cudaLaunchKernelEx is capturable and the prepare replays prove it; no sync/.item() in kernel_fn (lint: those are prepare-only warnings).
- [x] Nothing derived from `data` is cached across calls: `_B1_CL` is chosen on synthetic data from shapes/hardware; tickets are device-only counters reset by the last arrival.
- [x] Weights: n/a: attention problem, no weights touched.
- [x] Activations/KV/queries: unchanged (FP8 KV, FP8 E4M3 query, fp32 accumulation, partials fp32); only the merge location changed.
- [x] Attention sees every key: the stream-K tile split is the landed one; the cluster merge combines all CTA partials (empty ones with weight 0).
- [x] No timing patches, harness access, subprocess or network: checked `git diff` (lint reports none).
- [x] All work inside the timed call; no output caching or replay (prepare-time timing uses synthetic inputs only).

okc_lint: `OK with warnings: submission.py` (warnings are only prepare-time host-sync notes outside kernel_fn's call graph); `python3 -I -m py_compile` ok, `__pycache__` removed.

## Risks / what could fail
medium
Kill: if no C beats the landed two-level kernel in prepare by 3%, the merge chain is not the B=1 limiter; next stage would be per-phase clock64 stamps. Risk: cluster co-scheduling at 95 KB smem and 1 CTA/SM; peer smem lifetime.

## Result (after the run)
- job `f519261b7c2b435f` (check), status done, passed True, tests 16/16
- speedup 1.4388x, 95% CI [1.4172, 1.4603], score 2617.7410874172037 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.4321773019894537, "ci": [1.4106946424596118, 1.4554282820505813], "job": "2a3241780c7b4c46"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode1-streamk-s3-two-level-ticket.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk4-cluster-dsmem-merge.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk4-cluster-dsmem-merge.patch`; branch `hyp/muse_attention_decode/full-decode1-sk4-cluster-dsmem-merge`
