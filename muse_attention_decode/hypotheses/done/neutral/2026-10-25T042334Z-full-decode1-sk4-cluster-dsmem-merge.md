---
problem: muse_attention_decode
slug: full-decode1-sk4-cluster-dsmem-merge
created: 2026-10-25T042334Z
base: master
area: full-layer attention (batch-1 decode, stream-K kernel merge)
change: >
  Stage 4 of the full-decode rewrite, built on master (stage 3 two-level ticket landed, c3c1b01).
  Full decode_1 is 22.6 us with the kernel at 18.6 us (0.89 TB/s) vs a 4.3 us floor; in stage 3
  prepare grid 2xSM was slower than grid SM (24.7 vs 23.1), so more CTAs do not help and the tail
  is the merge chain: each of 132 CTAs writes a 4.2 KB fp32 partial, __threadfence, group atomic,
  the group last arrival reads ~11 partials, fence, second atomic, last group merges ~12 group
  slots + norm-gate. Change, at batch 1 only (config): launch the kernel with a thread-block
  cluster of C CTAs (cudaLaunchKernelEx, cluster dims C, grid a multiple of C: C=4 -> grid 132,
  C=2 -> 132, C=8 -> 128) and replace the FIRST ticket level with an on-chip merge: each CTA
  writes (m[16], l[16], acc[16x64]) to its own smem, barrier.cluster.arrive.release +
  wait.acquire, rank 0 reads the C-1 peer partials through mapa + ld.shared::cluster and
  LSE-merges, then writes ONE cluster partial to global, __threadfence, atomicAdd on ticket[0];
  the last cluster merges the grid/C cluster partials, applies v_scale, the landed norm-gate
  epilogue, stores bf16 and resets the ticket (graph replay). No group tickets are used at B=1.
  Peers stay resident until rank 0 has read them (second barrier.cluster before exit). The
  stream-K tile split, ring and math are the landed code. Prepare: build C in {2,4,8}, time each
  against the landed two-level kernel (cold L2, median of 10), print all, choose the fastest and
  assert it beats the landed kernel by >= 3%, else keep the landed two-level path (PATH CHOSEN:
  full decode_1 cluster C=<c> or two-level (fallback: <timings>)). Self-check: the stage-3 B=1
  cases (80000/4097/63/1 keys, kv 1.0 and 0.25) for each C, two replays to prove the ticket
  resets. B > 1 unchanged. BUILD/SAFETY as landed (sm_90a only, -std=c++20, -lineinfo, in-process
  canary, never a child process). First measurement is an okc check.
expected:
  - shapes: "*layer=full load=decode_1 *"
    direction: faster
    why: removes one global fence/atomic/read level and cuts global partial traffic by C; kernel 18.6 -> ~15.5 us, shape 22.6 -> ~19.5. M1 tests pass; M2 kernel <= 16.5 us; M3 <= 14 us
  - shapes: "*layer=full load=decode_16* *layer=full load=decode_32*"
    direction: flat
    why: batch > 1 keeps the landed single-level kernel
  - shapes: "*layer=sliding* *load=verify_*"
    direction: flat
    why: outside the gate
risk: medium
depends_on: null
priority: 1
kind: rewrite
controls_for: null
rewrite_stage: 4
replaces: "full decode batch 1: master splitkv_attn_sk_kernel two-level global ticket (grid SM)"
review: proceed
status: done
review_at: 2026-10-25T043142Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T043142Z
base_sha: aafeb8f4d969d4a791dafeca1ad8dda986c3e7c1
base_blob: d8769ef25caf845a0e207538ccc5c62394b6ee3b
branch: hyp/muse_attention_decode/full-decode1-sk4-cluster-dsmem-merge
worktree: .worktrees/muse_attention_decode/full-decode1-sk4-cluster-dsmem-merge
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-25T044042Z
impl_commit: 2c94c5c0f5309c4a67503f30d7fd3fa710407c06
submission_sha256: cecebe4df2e9f4cc2e7fc66cb75c1d99810384377d749a5ba300f9015a3a1572
norm_sha256: 405c2d4ba202c3a1ae83d31b590b4c9b52ec099ab2630ac9f3eb495dc5ff52c0
submission_blob: 3dea00b0efb40e1019685585557f439ee4ccd0d6
lint_warnings: 20
attempt_file: 2026-10-25T044042Z-check-full-decode1-sk4-cluster-dsmem-merge.md
prepared_at: 2026-10-25T044042Z
job: f519261b7c2b435f
dispatched_at: 2026-10-25T044049Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.438759554221622
speedup_ci: [1.4171781609082976, 1.4603409475349463]
harvested_at: 2026-10-25T053756Z
best_at_harvest: 1.4321773019894537
promotion: null
---

# full-decode1-sk4-cluster-dsmem-merge

## Change

Stage 4 of the full-decode rewrite, built on master (stage 3 two-level ticket landed, c3c1b01). Full decode_1 is 22.6 us with the kernel at 18.6 us (0.89 TB/s) vs a 4.3 us floor; in stage 3 prepare grid 2xSM was slower than grid SM (24.7 vs 23.1), so more CTAs do not help and the tail is the merge chain: each of 132 CTAs writes a 4.2 KB fp32 partial, __threadfence, group atomic, the group last arrival reads ~11 partials, fence, second atomic, last group merges ~12 group slots + norm-gate. Change, at batch 1 only (config): launch the kernel with a thread-block cluster of C CTAs (cudaLaunchKernelEx, cluster dims C, grid a multiple of C: C=4 -> grid 132, C=2 -> 132, C=8 -> 128) and replace the FIRST ticket level with an on-chip merge: each CTA writes (m[16], l[16], acc[16x64]) to its own smem, barrier.cluster.arrive.release + wait.acquire, rank 0 reads the C-1 peer partials through mapa + ld.shared::cluster and LSE-merges, then writes ONE cluster partial to global, __threadfence, atomicAdd on ticket[0]; the last cluster merges the grid/C cluster partials, applies v_scale, the landed norm-gate epilogue, stores bf16 and resets the ticket (graph replay). No group tickets are used at B=1. Peers stay resident until rank 0 has read them (second barrier.cluster before exit). The stream-K tile split, ring and math are the landed code. Prepare: build C in {2,4,8}, time each against the landed two-level kernel (cold L2, median of 10), print all, choose the fastest and assert it beats the landed kernel by >= 3%, else keep the landed two-level path (PATH CHOSEN: full decode_1 cluster C=<c> or two-level (fallback: <timings>)). Self-check: the stage-3 B=1 cases (80000/4097/63/1 keys, kv 1.0 and 0.25) for each C, two replays to prove the ticket resets. B > 1 unchanged. BUILD/SAFETY as landed (sm_90a only, -std=c++20, -lineinfo, in-process canary, never a child process). First measurement is an okc check.

## Expected

- `*layer=full load=decode_1 *` faster: removes one global fence/atomic/read level and cuts global partial traffic by C; kernel 18.6 -> ~15.5 us, shape 22.6 -> ~19.5. M1 tests pass; M2 kernel <= 16.5 us; M3 <= 14 us
- `*layer=full load=decode_16* *layer=full load=decode_32*` flat: batch > 1 keeps the landed single-level kernel
- `*layer=sliding* *load=verify_*` flat: outside the gate

## Risk

Kill: if no C beats the landed two-level kernel in prepare by 3%, the merge chain is not the B=1 limiter; next stage would be per-phase clock64 stamps. Risk: cluster co-scheduling at 95 KB smem and 1 CTA/SM; peer smem lifetime.

## Milestones

Rewrite stage 4 of `full decode batch 1: master splitkv_attn_sk_kernel two-level global ticket (grid SM)`. Judged on the replaced kernels' own time per
shape (profile top_kernels) against the incumbent and the speed-of-light floor, not on the headline:
1. correct: passes every test (a failure is a fix for the next stage, not a dead end);
2. at or below the incumbent kernels' time on the target shapes;
3. within ~1.5x of the floor on the target shapes, and the headline CI clears 1.0 (promotable).
The next stage is minted with `--kind rewrite --depends-on <this slug> --stage N+1`.

## Adversarial review

Reviewed 2026-10-25T043142Z by an adversarial subagent (model opus).

**Verdict: proceed.** A sound stage 4 for the full decode_1 rewrite: replacing the first global ticket level with a DSMEM cluster merge is physically cheaper and has no rules risk. As written, though, it only times grid ~SM, and at C=2 and C=4 its single global level is longer than the landed second level. Add grid-2xSM cluster variants and a two-level-over-clusters option so the prepare can find the real win.

### Justification
Mechanism: the stage-3 diagnostic (REWRITE_READ 03:41Z) prices the merge structure directly. Single-level at grid SM ran 34.8 us and two-level 23.1 us, so serial slot reads cost ~0.1 us per 4.2 KB slot (11.7 us over ~109 fewer serial reads). The landed chain is ~12 group-slot reads plus ~11 final reads plus two fence/atomic rounds. The proposal is C-1 DSMEM peer reads plus grid/C global reads plus one fence/atomic. That gives C=8 -> 16 global reads (a clear cut), C=4 -> 33 (about equal to today's 23 plus one fence) and C=2 -> 66 (worse). So only C=8 shortens the chain, and the expected 18.6 -> 15.5 us (-3 us) is more than the whole merge chain is worth at 0.1 us/slot (~2.3 us plus two fences). WEAKEST LINK: the claim that the merge chain is the B=1 limiter. 20.5 MB in 18.6 us at grid SM means 1 CTA/SM with a 4-stage 16 KB ring, half the bytes in flight of the 2-CTA/SM B=16 config that reaches 3 TB/s. Bytes in flight at 1 CTA/SM explains most of the gap to the 4.3 us floor. grid 2xSM lost in stage 3 (24.7 vs 23.1) because it doubled the partials the global merge chain reads, and a cluster merge removes exactly that cost. The design's real upside is therefore cluster merge plus grid 2xSM (264 CTAs, C=8 -> 33 cluster partials), which the hypothesis does not time. Co-scheduling: tickets are last-arrival and never spin, so non-resident clusters cannot deadlock. barrier.cluster is intra-cluster and co-scheduled by hardware, so the design is safe at any grid. Milestones are readable from the profile row (sk kernel time on full decode_1). M3 (<=1.5x of 4.3 us) is not reachable this stage and needs the bytes-in-flight fix.

### Confounders on the current kernel
- full-decode1-streamk-s3-two-level-ticket (c3c1b01): this is the incumbent being replaced. B=1 grid and variant selection live in _full_decode_sk (variant 3, grid SM), and the change must leave B>1 (variant 0) byte-identical. Read the result only on the full decode_1 sk-kernel row.
- full-decode-splitkv-s2-streamk-ticket-inproc (landed inside c3c1b01): it shares splitkv_attn_sk_kernel and its smem layout (95264 B, 2 CTAs/SM). A cluster launch attribute on the B=1 instantiation must not change B>1 regs/smem, so print attrs for every variant and compare with the landed log (ticket 119-121 regs).
- In flight, not landed: full-decode-splitkv-s3-fused-preattn edits the same TU for B>1, and full-decode1-sk-pdl-from-preattn edits the same B=1 launch. Neither confounds a run against master, but sk4 + PDL must later share one cudaLaunchKernelEx (cluster dim + programmatic serialization attributes are compatible).

### Interference and A/B needs
No companion. This is a rewrite stage, its prepare already times every C against the landed two-level kernel on the same input, and PATH CHOSEN names what ran. A gain or flat on the full decode_1 row is attributable to the chosen variant. The grid/merge-shape ambiguity is better handled inside the prepare (extra variants) than with a second check.

### Open questions
- BUILDER: extend the prepare matrix to grid 2xSM (264 CTAs, C=4 and C=8, co-residency permitting; print cudaOccupancyMaxActiveClusters per C at 95 KB smem and 2 CTAs/SM). Also time, for C in {2,4}, a 'cluster merge + group ticket over cluster partials' variant (Gs = ceil(sqrt(grid/C))). A single global level over 33-66 cluster slots is no shorter than the landed 12+11 chain. Choose by the printed timings, with the 3% bar as written.
- BUILDER: confirm that a cluster launch keeps 2 CTAs/SM at 95264 B smem. If clusters force 1 CTA/SM at grid 2xSM, print it and say so in PATH CHOSEN.
- If no variant clears 3%, the kill note should record whether grid 2xSM + C=8 closed the gap. That tells whether stage 5 is bytes-in-flight (more stages or 2 CTAs/SM at B=1) rather than per-phase stamps.


## Implementation notes

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

## Rules check

- [x] CUDA-graph capturable: kernel_fn only reads the `_B1_CL` constant; cudaLaunchKernelEx is capturable and the prepare replays prove it; no sync/.item() in kernel_fn (lint: those are prepare-only warnings).
- [x] Nothing derived from `data` is cached across calls: `_B1_CL` is chosen on synthetic data from shapes/hardware; tickets are device-only counters reset by the last arrival.
- [x] Weights: n/a: attention problem, no weights touched.
- [x] Activations/KV/queries: unchanged (FP8 KV, FP8 E4M3 query, fp32 accumulation, partials fp32); only the merge location changed.
- [x] Attention sees every key: the stream-K tile split is the landed one; the cluster merge combines all CTA partials (empty ones with weight 0).
- [x] No timing patches, harness access, subprocess or network: checked `git diff` (lint reports none).
- [x] All work inside the timed call; no output caching or replay (prepare-time timing uses synthetic inputs only).

okc_lint: `OK with warnings: submission.py` (warnings are only prepare-time host-sync notes outside kernel_fn's call graph); `python3 -I -m py_compile` ok, `__pycache__` removed.

## Pre-flight audit

Audited 2026-10-25 by an opus subagent (read-only emulation; no GPU/nvcc on this node).

**Verdict: dispatch-with-fixes.**

Checked and found sound:
- Landed variants 0-3 (CL=0): every new branch is `if constexpr` or folds on the constant CL (`nclu`, the `bx >= gx` exit, `t0`/`N`, the epilogue `else`). Same code text, so same codegen. `sk_pick`/`sk_smem`/`sk_prepare`/`sk_attrs` are unchanged and B>1 is untouched.
- Cluster barriers: a cluster with no tile exits whole (`bx / CL >= nclu`). In a live cluster, rank 0 always owns >= 1 tile (`cid*CL < gx`, and `gx <= T` gives N >= 1). Empty peers call `cluster_finish` with M=-inf, L=0, o=0, which get weight 0. The arrive/wait sequence is matched (2 phases) on every rank and is CTA-uniform, so `.aligned` is valid. Rank 0 reads DSMEM only after the phase-1 `arrive.release`/`wait.acquire`, so every peer has started and written. Peers block on phase 2 until rank 0 arrives after its last `ld.shared::cluster`. `mapa.shared::cluster.u32` is applied to the CTA-local shared address.
- Staging area: `part` (4.25 KB at RING offset 0) is dead when it is written. All non-empty cp.async groups have completed at n=N-1 (`wait_group<STAGES-2>`, and the last two groups are empty). All warps passed the epilogue `__syncthreads` after their last ring reads. ACC/ML/FLAG do not overlap it.
- Tickets: the single level is reset by the last of `nclu` arrivals (`ticket[0]=0`). CGRP resets `gticket[gi]` per group and `ticket[0]` at the last group. ngrp <= 11 < 32 slots. Each candidate gets 3x2 graph replays plus a ticket-sum==0 assert, so it self-resets per call under replay with refilled inputs. Tickets are re-zeroed after any failed candidate and after the selection.
- `kernel_fn` only reads the module constant `_B1_CL`. There is no per-call state, print or try/except in its call graph. cudaLaunchKernelEx with a cluster attribute is graph-capturable. C <= 8 is portable, so NonPortableClusterSizeAllowed is not needed. The cudaLaunchKernelEx template arguments match SkFn (20 params).
- Compile: the sk module built in ~37 s with 4 instantiations. 5 more should add ~25-40 s, which fits in the 15 min run.

Fixes:
1. `_sk_cluster_select`: per-candidate try/except around `_sk_cl_validate`. Before, a TORCH_CHECK or launch-config error in one candidate (e.g. a refused C=8 grid-264 cluster launch) aborted the whole matrix to the landed path. Now only that candidate is dropped, with a loud `[sk4] ... FAILED (raised ...)` line, and the tickets are re-zeroed. okc_lint OK (prepare-only host-sync warnings); py_compile OK; `__pycache__` removed.

Residual risk: the code has never been compiled. A ptxas error in the cluster PTX would fail the whole sk module build and lose stages 2-3 for this run (falling back to splitkv v1/FA3, loud), and a device fault in a cluster candidate is sticky and cannot be caught in process.

## Result

- job `f519261b7c2b435f` (check), status done, passed True, tests 16/16
- speedup 1.4388x, 95% CI [1.4172, 1.4603], score 2617.7410874172037 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.4321773019894537, "ci": [1.4106946424596118, 1.4554282820505813], "job": "2a3241780c7b4c46"}; base result: muse_attention_decode/hypotheses/done/neutral/full-decode1-streamk-s3-two-level-ticket.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk4-cluster-dsmem-merge.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode1-sk4-cluster-dsmem-merge.patch`; branch `hyp/muse_attention_decode/full-decode1-sk4-cluster-dsmem-merge`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 13.15 | 13.45 | 0.9679 | lost |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 22.6 | 22.83 | 1.0063 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.33 | 15.03 | 1.0154 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 111.14 | 110.23 | 1.0071 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 24.33 | 24.47 | 1.0003 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 226.42 | 226.47 | 1.0016 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 17.78 | 17.1 | 1.0523 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.51 | 29.83 | 1.0198 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.68 | 22.84 | 0.9996 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 162.8 | 162.61 | 0.9993 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 28.34 | 28.77 | 0.9915 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.2 | 302.12 | 0.9985 | flat |

gained on sliding verify_1; lost on sliding decode_1. Expected hit: 0/3. Noise ±1.5%.

## Follow-ups

- rewrite stage 4 of `full decode batch 1: master splitkv_attn_sk_kernel two-level global ticket (grid SM)`: read the replaced kernels' own us per shape (profile top_kernels) against the incumbent and the floor, and check which Milestone it reached. The next stage is `/hypothesize muse_attention_decode "<what to fix or tune>" --kind rewrite --depends-on full-decode1-sk4-cluster-dsmem-merge --stage 5`; a slow or failed first cut is the expected start, not a do-not-retry.
- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: full-decode1-sk4-cluster-dsmem-merge`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

