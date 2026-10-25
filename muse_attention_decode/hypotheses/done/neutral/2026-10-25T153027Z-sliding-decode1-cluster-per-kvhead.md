---
problem: muse_attention_decode
slug: sliding-decode1-cluster-per-kvhead
created: 2026-10-25T153027Z
base: master
area: sliding-layer attention (batch-1 decode, cluster kernel work split)
change: >
  Built on current master (no depends_on). Sliding decode_1 (weight 46, 60% of the decode_1 load)
  runs sliding_cluster_kernel at C=16: ONE cluster of 16 CTAs for the whole step, 10.8 us kernel /
  11.9 us shape, floor 0.1 us. Each CTA owns 128 old keys of BOTH kv heads and, before it can
  compute, builds all 16 query heads (RMSNorm + RoPE with accurate sincosf, 32 B local stack)
  while its gather flies; the last rank also builds the new k/v rows. With only 16 CTAs on 132
  SMs, per-CTA latency is the whole shape. CHANGE (B=1 band only): a template flag PERHEAD (false
  = master, byte-identical) where the grid is 2 clusters of C=16 (32 CTAs), cluster k serving kv
  head k only: each CTA gathers only its head's 64-byte half of every K and V row (16-byte
  cp.async on the 4 chunks of that half; same SW pattern), builds only its 8 query heads, runs the
  same mma.sync QK/PV for one head with all 8 warps splitting the 128 keys (instead of warps
  alternating heads), and the DSMEM reduce-scatter covers 8 rows instead of 16. The new k/v build
  stays on the last rank of EACH cluster for its own head (each writes its half of the 128-byte
  cache row; the two halves are disjoint bytes). Output: rows of head k only. Gate in prepare: B=1
  x 80000, interleaved cold-L2 graph replay (empty graph subtracted, median of 31) PERHEAD vs
  master C=16; route at >= 1.03x and rel L2 < 1e-2 vs master; self-check on master's slide cases
  (lens 1/10/2047/2048/2049/80000, kv 0.25) with cache rows compared to master's, plus the
  two-state alternating replay; else master. B=17..32 and every other path untouched. RULES (every
  path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values
  (shape/config only), no data-dependent Python control flow, no prints, no try/except, no
  per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32
  softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in
  kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a
  device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32
  ticket arrays, zeroed once and returned to 0 by every launch, are fine).
expected:
  - shapes: "*layer=sliding load=decode_1 *"
    direction: faster
    why: "halve each CTA's q-build and compute on the latency chain (32 CTAs, one head each); 10.8 us kernel, target <= 9.5 us; floor 0.1 us"
  - shapes: "*layer=sliding load=decode_16 *"
    direction: flat
    why: Triton route, untouched
  - shapes: "*layer=sliding load=decode_32 *"
    direction: flat
    why: sk window route, untouched
  - shapes: "*layer=full *"
    direction: flat
    why: untouched
  - shapes: "*load=verify_* *"
    direction: flat
    why: untouched
risk: medium
depends_on: null
priority: 2
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-25T161935Z
review_companions: null
stage: harvested
claimed_at: 2026-10-25T161936Z
base_sha: a5f0c41fcf6d081ab9a5f2262b3505c796565fea
base_blob: 5cf8fae31e5b84c36efeeb4569763eb9b06e1c8a
branch: hyp/muse_attention_decode/sliding-decode1-cluster-per-kvhead
worktree: .worktrees/muse_attention_decode/sliding-decode1-cluster-per-kvhead
mode: submit
restack_of: null
stale_base: null
implemented_at: 2026-10-25T162957Z
impl_commit: 037e0696dc223ca43c8c3460f3dc2fd7ed7b573a
submission_sha256: 5875b86506261122bba2f7c47dd333ab98d9b2163b230ecc341e5d52b0aac932
norm_sha256: 7b5894f405355819a421a39d1af1293dd5826a93e7c0949ab330f60db4497c5a
submission_blob: 6d6afa4722731ceb9cb8216d0334a03f1a96ad2b
lint_warnings: 60
attempt_file: 2026-10-25T162957Z-submit-sliding-decode1-cluster-per-kvhead.md
prepared_at: 2026-10-25T162957Z
job: 4b608df595b046fb
dispatched_at: 2026-10-25T163005Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.5119748206149146
speedup_ci: [1.4892951983056908, 1.5346544429241382]
harvested_at: 2026-10-25T185514Z
best_at_harvest: 1.5016522435853301
promotion: merged
hand_landed: true
promotion_commit: 5d528bf313ed6deb0094c453214f1584fea0033b
---

# sliding-decode1-cluster-per-kvhead

## Change

Built on current master (no depends_on). Sliding decode_1 (weight 46, 60% of the decode_1 load) runs sliding_cluster_kernel at C=16: ONE cluster of 16 CTAs for the whole step, 10.8 us kernel / 11.9 us shape, floor 0.1 us. Each CTA owns 128 old keys of BOTH kv heads and, before it can compute, builds all 16 query heads (RMSNorm + RoPE with accurate sincosf, 32 B local stack) while its gather flies; the last rank also builds the new k/v rows. With only 16 CTAs on 132 SMs, per-CTA latency is the whole shape. CHANGE (B=1 band only): a template flag PERHEAD (false = master, byte-identical) where the grid is 2 clusters of C=16 (32 CTAs), cluster k serving kv head k only: each CTA gathers only its head's 64-byte half of every K and V row (16-byte cp.async on the 4 chunks of that half; same SW pattern), builds only its 8 query heads, runs the same mma.sync QK/PV for one head with all 8 warps splitting the 128 keys (instead of warps alternating heads), and the DSMEM reduce-scatter covers 8 rows instead of 16. The new k/v build stays on the last rank of EACH cluster for its own head (each writes its half of the 128-byte cache row; the two halves are disjoint bytes). Output: rows of head k only. Gate in prepare: B=1 x 80000, interleaved cold-L2 graph replay (empty graph subtracted, median of 31) PERHEAD vs master C=16; route at >= 1.03x and rel L2 < 1e-2 vs master; self-check on master's slide cases (lens 1/10/2047/2048/2049/80000, kv 0.25) with cache rows compared to master's, plus the two-state alternating replay; else master. B=17..32 and every other path untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## Expected

- `*layer=sliding load=decode_1 *` faster: halve each CTA's q-build and compute on the latency chain (32 CTAs, one head each); 10.8 us kernel, target <= 9.5 us; floor 0.1 us
- `*layer=sliding load=decode_16 *` flat: Triton route, untouched
- `*layer=sliding load=decode_32 *` flat: sk window route, untouched
- `*layer=full *` flat: untouched
- `*load=verify_* *` flat: untouched

## Risk

Risk: 64-byte half-row gathers halve sector use per request but the shape is latency-bound (0.5 MB window); if the gate keeps master the timing table still tells whether per-CTA work is part of the 10.8 us. Interacts with the running epilogue-gate-l2-prefetch-cuda (gate prefetch in the CUDA epilogues): disjoint lines, both may stack.

## Adversarial review

Reviewed 2026-10-25T161935Z by an adversarial subagent (model opus).

**Verdict: proceed.** This is a plausible, self-gated primary. It halves each CTA's q-build, MMA tiles and gather bytes on the only sliding decode_1 kernel (10.8 us) and touches no other shape. Run it with the fixes listed: the reduce-scatter row mapping (HQ/C goes to 0), a C=16 occupancy >= 2 check, and making PERHEAD conditional on C=16 having routed. The headline will sit inside the CI, so read the result from the [slide-gate] PERHEAD-vs-C=16 line and the sliding decode_1 row, not from the geomean.

### Justification
Mechanism: at B=1 the sliding layer is one 16-CTA cluster on 132 SMs, so the shape's time is one CTA's latency chain. Each CTA gathers 128 old keys (one SK=128 sub-chunk, about 32 KB of K+V). Each warp builds 2 query heads (loop h = warp, warp+8, each with a cold qkv load, a shfl reduce and sincosf; at pos*fr <= 80000 rad sincosf stays on the fast Cody-Waite path, so the cost is the load more than the math). Each warp then attends 2 of the 8 tiles for its head. PERHEAD gives each warp 1 head and 1 tile and halves the gather to 4 x 16 B chunks per row. If the compiler does not hoist the second qkv load over the first iteration's smem store, the change also removes one serialised DRAM round trip per warp, which is the most likely source of a real gain (~0.5-1 us). The profile supports a latency-bound kernel: 10.8 us for 0.5 MB is about 48 GB/s, roughly 1.5% of HBM, and the floor is 0.1 us. The expectations are consistent with TASK.md: only the sliding decode_1 row (weight 46) moves. That is about 46 x 1.3 us = 60 us of the 1578 us decode_1 load, around 0.6% on the six-load geomean, well inside the [1.479, 1.530] CI. WEAKEST LINK: nothing shows that per-CTA work is a meaningful part of the 10.8 us. The chain is dominated by things PERHEAD does not shorten: dependent DRAM round trips (seq_lens -> page_table indices -> K/V gather), cluster launch and scheduling of a 16-wide non-portable cluster, two cluster.sync, and the DSMEM reduce. Halving the bytes per row does not halve the latency of one gather round trip. The 9.5 us target is optimistic, a 3% gate is plausible, and 'flat' is the most likely alternative. The gate makes a loss impossible by construction (master is kept), so a non-routing result is still informative if the gate line prints both times. Co-schedulability (extra note): the only measured occupancy in profiles shows cudaOccupancyMaxActiveClusters C=16 -> 14, so 2 clusters fit. The two clusters also have no cross-cluster dependency: disjoint output rows, disjoint 64-byte halves of the new cache row, no shared ticket, and the new k/v is read from smem, never from the cache. So serialisation could only slow it down, never deadlock or corrupt it, and the timing gate would catch the slowdown.

### Confounders on the current kernel
- restack-sk-pseudo-cluster-plus-v3-fused-preattn (landed C=16 at B=1): this is the incumbent PERHEAD is gated against. The comparison is clean only if C=16 itself routed in the same process. If C=16 fell back to Triton (canary, self-check or its own timing gate), PERHEAD must stay off or must be gated against what is actually routed. As written ('vs master C=16') it can route against a C=16 that was never enabled.
- restack-sliding-decode-sk-window-on-cluster-master (landed sk-window decode route): kernel_fn checks _sd_gate before _slide_gate. Today _SD_ROUTE only fills band 3 (B=17..32), so B=1 is not intercepted. The builder must keep that true; if band 1 can ever be filled, PERHEAD is dead code.
- fused-preattention-triton / full-decode stream-K landings: no shared launch or memory traffic with the sliding B=1 single-node graph; no confound.
- In-flight epilogue-gate-l2-prefetch-cuda (named by the hypothesis): it touches the CUDA epilogues, possibly including this kernel's gate read. If it lands first, the absolute times shift for both PERHEAD and C=16 alike; the interleaved gate ratio stays valid.

### Interference and A/B needs
No companion. The change is internally A/B'd: prepare times PERHEAD against master C=16 interleaved, with cold L2, the empty graph subtracted and the median of 31, and routes only at >= 1.03x. The printed gate line is therefore the attribution, independent of the headline. A scored companion could not resolve anything the gate line does not already resolve, and the headline delta (~0.6% expected) is below the CI either way. Requirement instead: the [slide-gate] PERHEAD line must print both times and the ratio whether or not it routes, and the profile's sliding decode_1 top kernel must show which template instance ran (e.g. by name or by the 32-CTA grid).

### Open questions
- Does master's q-build loop issue its two qkv loads per warp back to back, or serialised behind the first iteration's smem store and sincosf? If they already overlap, the q-build half of the argument is mostly gone. SASS or a quick ncu source-level view of the existing C=16 kernel would settle it before the build.
- Is the 10.8 us dominated by the seq_lens -> page_table -> gather dependency chain plus cluster launch? If so, a follow-up should attack that chain (for example, prefetching page indices for both clusters, or PDL from nothing). Neither more CTAs nor less work per CTA shortens it.
- Do the two 16-CTA clusters land on separate GPCs, or do they share SMs (2 CTAs/SM at 101 KB smem)? If 32 CTAs pack onto 16 SMs, per-SM gather bandwidth and issue are shared and the halving partly cancels. Printing %smid or %clusterid once in the canary would show it.
- Would C=8 per head (2 clusters x 8 CTAs = 16 CTAs, 256 keys each) be a cheaper variant to time in the same prepare gate? It needs no new occupancy assumption and keeps DSMEM fan-in at 8.


## Implementation notes

Base a5f0c41f; master 43e5ca46 differs from it only in this md (submission.py identical), so no merge was needed. Edits only in `.worktrees/muse_attention_decode/sliding-decode1-cluster-per-kvhead/muse_attention_decode/submission.py` (diff: +235 / -56).

CUDA (`_SLIDE_CUDA`):
- `sliding_cluster_kernel` is now `template <bool PERHEAD>`. With `false` (master), every branch reduces to master's expressions (`b = blockIdx/C`, `kh = 0`, same loops and constants). `load_idx<PH>` and `issue_sub<PH>(..., kh)` are templated too. In PH, a thread handles token `tid/4 (+64 i)` and chunk `4 kh + tid%4`: 16-byte cp.async on the 4 chunks of the head's 64-byte half, 2 passes per 128-key sub-chunk, same smem row layout and XOR swizzle. Padding rows still zero both halves, and the unloaded half is never read: attend_tile reads only chunks `4g..4g+3`.
- PH grid: `B * HKV * C` CTAs. Cluster `cl = blockIdx/C` gives `b = cl / HKV`, `kh = cl % HKV`. q-build: warp w builds head `kh*8 + w` (one head per warp). On the last rank, warps 0/1 build k/v of head kh only and write only that head's 64-byte half of the cache row (disjoint bytes across the two clusters). Compute: `g = kh`, `sub = warp`, tile stride `NSL = 8` (master 4). The new-key tile goes to slice `nt & (NSL-1)`.
- Review fix (2), reduce-scatter: in PH the CTA partial has G = 8 local rows. The in-CTA merge (threads 0..127, 16 per row) loops over all 8 warps (`mw[WARPS]`). In the DSMEM reduce, `NROWS = G` and `NR = max(1, G/C)` = 1 at C = 16. `owns = rank*NR < NROWS`, so ranks 0..7 own one row each. Ranks 8..15 skip the whole reduce/write block but still execute both cluster.sync. Partials are read at the local row `hl`; gate and output use the global head `h = kh*8 + hl`.
- Review fix (6), cluster.sync audit (done by hand plus a grep of the kernel body): the kernel has no `return`. Exactly two `cluster.sync()` calls exist, both at the top level of the kernel body and outside every branch, so every rank of every cluster runs exactly 2 on both instances. Every `__syncthreads` is at a CTA-uniform point: the loop bound `nsub` and the condition `s+NS<nsub` are CTA-uniform, and the staging barrier comes before the `if (PERHEAD)`. The `owns` branch is CTA-uniform and contains only warp shuffles (warp 0, full warp inside). The two PH clusters share no barrier, ticket or memory dependency.
- Host: `occupancy_of<PH>`, `set_attrs<PH>` (max dynamic smem, carveout and non-portable cluster size, for both instances). `slide_setup()` appends `[PERHEAD occ C=16, regs, local]` at indices 8..10 and leaves master's indices 0..7 unchanged. `slide_attn(..., int64 perhead)` picks the instance and sets the grid to `B*C*(ph?HKV:1)`. The profile names the instance (`sliding_cluster_kernel<true>`, 32 CTAs).

Python:
- `_SLIDE_PH = False` (module flag). `_slide_forward` passes `_SLIDE_PH and c == 16`, and C = 16 serves only B = 1. kernel_fn is unchanged: no prints, no try/except, no new mutable state. It only reads the frozen flag.
- Review fix (4): `_SLIDE_PH` is set once at import, in `_slide_init`, after the C loop and before the PATH CHOSEN prints. It is never toggled per call. The two-state replay sets it provisionally to True at import and restores it in a `finally`. The init-exception fallback resets it to False.
- Review fixes (1) and (3): `_slide_ph_init` runs only if `_SLIDE_MOD[16] is not None`, i.e. only if C = 16 itself routed in this run (canary, self-check, its own two-state replay and its timing gate against Triton). It checks, in order:
  (a) the PERHEAD instance's `cudaOccupancyMaxActiveClusters` at C = 16 is >= 2;
  (b) a PERHEAD canary;
  (c) an eager self-check at B = 1 on lens 1/10/2047/2048/2049/80000 at kv 0.25 and 2049 at 1.0: rel L2 vs fp32 reference < 2e-2 (master's bound), vs master C = 16 on identical inputs < 1e-2, and new cache rows vs master C = 16's rel < 1e-2 (byte match printed);
  (d) the two-state alternating graph replay through kernel_fn with `_SLIDE_PH` on (`_alt_check`, gate `_slide_gate`);
  (e) an output check at 80000 vs C = 16, then the timing gate: interleaved cold-L2 graph replays (`_interleaved_us`, reps = 31, empty graph subtracted), routing only at C16/PERHEAD >= 1.03x.
  Any failure or exception leaves `_SLIDE_PH = False`, which keeps master C = 16, never Triton.
- Review fix (3), second half: `_sd_setup_body` now runs `route.pop(1, None)` before freezing `_SD_ROUTE`, so the sk-window decode route can never take B = 1. Today it only fills band 3 anyway.
- Review fix (5): `[slide-gate] PERHEAD-vs-C=16 B=1 x 80000: [cluster C=16] X us, [PERHEAD 2 x C=16] Y us -> C=16 / PERHEAD r x (ROUTE PERHEAD | ... master C=16)` prints whenever the gate is reached, routed or not. There is also an output rel-L2 line, plus `[slide] PATH CHOSEN: sliding B=1 PERHEAD ON|OFF (reason)` and the per-band PATH CHOSEN line (`PERHEAD 2 x C=16` when on).
- Every PERHEAD launch at import is announced, synced and confirmed (`_slide_run`), and the build already has `-lineinfo`. PH adds no new PTX: same cp.async.cg, mma.sync and cvt forms as master. No mbarrier or spin waits. No module or static global holds device pointers: the PH instance takes everything from the launch arguments.

Deviations / not done:
- Open question "print %smid/%clusterid in the canary": not done, to avoid new PTX in a scored run.
- Open question "C=8 per head variant": not built (out of scope; candidate follow-up).
- The spec's "rel L2 < 1e-2 vs master" is applied both in the self-check cases and at the 80000 gate input.
- Because this is a scored submit, a hard fault in the first PERHEAD launch (illegal address) would poison the context with no fallback. The canary is the first PH launch, so a fault would be attributed to it, but it is still a risk.

## Rules check
- [x] CUDA-graph capturable: kernel_fn -> `_slide_forward` -> one `cudaLaunchKernelEx` (shape-only grid). No sync, `.item()` or print in that path. The flag is a host bool frozen at import. Allocation is `torch.empty` of the config shape only.
- [x] Nothing derived from `data` is cached across calls: no new caches. `_SLIDE_INFO` holds host ints from `slide_setup`, not data.
- [x] n/a: weights: this problem has no weights; no repack.
- [x] Activations: qkv/out BF16; KV cache FP8 E4M3 (new rows written as E4M3 halves); q rounded once to E4M3 (same as master); fp32 softmax/merges.
- [x] Every key: each cluster attends every key of the 2048 window for its kv head (same ck split over 16 ranks, all tiles masked only by window/causal bounds), plus the new key on the last rank; no skipping.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network (lint clean on these).
- [x] All work inside the timed call; no output caching or replay. Import-time work is checks and timing on synthetic data only.
- [x] Two-state rotation: no device pointer, map or workspace in module/static globals. The two-state alternating replay with refilled inputs is run for PERHEAD through kernel_fn.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <wt>/muse_attention_decode/submission.py --problem muse_attention_decode`):
```
warn  host-sync  (59 lines, all "outside kernel_fn's call graph; fine in prepare()")
OK with warnings: submission.py
```
py_compile: OK; __pycache__ removed. An AST scan of the changed functions found no undefined names.

## Pre-flight audit

Audited 2026-10-25 by an Opus subagent, by reading only (no GPU here). **Verdict: dispatch-with-fixes.**

Fixes (all in the worktree submission.py):
- **The PERHEAD build is now in its own module.** The builder had templated master's own `sliding_cluster_kernel` inside master's module (`muse_sliding_cluster_v2`). An nvcc failure on `<true>` would have lost master's C=16 (B=1) and C=4 (B=17..32) routes and sent those shapes to Triton, a regression. `_SLIDE_CPP` / `_SLIDE_CUDA` are now restored byte-identical to master (checked by AST string equality against base a5f0c41f). The templated source is `_SLIDE_PH_CPP` / `_SLIDE_PH_CUDA`, with host entry points renamed `slide_ph_setup` / `slide_ph_attn` and built as `muse_sliding_cluster_v2_ph[_f16qk]` by `_build_slide(fp8qk, ph=True)`. That build happens inside `_slide_ph_init`, only after C=16 routed. A build failure returns "PERHEAD module build failed; master module untouched" and leaves master C=16. `_slide_launch` calls `slide_ph_attn(..., 1)` only when ph, and otherwise calls master's `slide_attn` with master's signature. `_slide_forward` picks `_SLIDE_PH_MOD` (set once at import, reset to None when PERHEAD does not route). The master side of the self-check, the gate output check and the interleaved timing now run master's real module (`mext`). `_SLIDE_INFO` was removed. The load_inline lines print the module name, and a new `[slide] modules: master C=16 ..., C=4 ..., PERHEAD ...` line prints after PATH CHOSEN.
- **Host bounds check before every PERHEAD launch.** In `_slide_run` with ph (prepare-time only), `_slide_bounds` checks that the page indices in each request's window [lo, seq) and cache_loc lie in [0, cache rows), and that seq <= page_table width. If not, it raises, which lands on master C=16 (the canary catches it; the self-check, replay and gate errors are caught in `_slide_init`). This prevents a device fault from poisoning the context.

Verified, no change needed:
- With PERHEAD=false, the templated code folds to master's expressions. It is moot now, since master's module source is byte-identical.
- Cache-row half-writes: on the last rank of cluster kh, warp 0 writes K and warp 1 writes V at bytes kh*64 + 2*lane (32 lanes x 2 B = 64 B). Cluster 0 covers bytes 0..63 and cluster 1 covers 64..127, so each byte of the K row and of the V row is written exactly once. The new key is never re-read from the cache (each cluster uses its smem XK/XV, of which only the kh half of row 0 is read; rows 1..15 are zeroed). There is no race.
- cluster.sync: there are exactly 2 calls, both at the top level, and the kernel has no return. The `owns` branch (ranks 0..7) contains only warp-0 shuffles, `__syncthreads` sits at CTA-uniform points, and the PH in-CTA merge (tid < 128) contains no barrier.
- Reduce-scatter at C=16: NR=1, ranks 0..7 own local row hl = rank, the output and gate use head kh*8+hl, and the 16 heads are covered exactly once across the 2 clusters.
- Gather: tokens tid/4 + 64 i (i < 2) cover 128 rows, chunks 4kh..4kh+3 match what attend_tile reads (QK g*4..g*4+3, V vlc = g*4 + quad/2), and pad rows are zeroed in both halves.
- `_SLIDE_PH` is set only in `_slide_init` / `_slide_ph_init` at import (provisional True for the replay, restored in finally) and never per call.
- kernel_fn purity: kernel_fn reads two frozen module globals, does no sync or print, allocates only by shape, and has no data-dependent control flow. Lint is clean apart from 59 prepare-time host-sync warnings.
- The `route.pop(1, None)` in `_sd_setup_body` is harmless (`route` is always defined, and band 1 was never filled).

Residual risk: about 45 s more nvcc at import (wall 470-490 s plus PERHEAD checks, still well under the 15-min limit); a device fault inside PERHEAD on in-bounds data would still poison the context, with no fallback.

## Result

- job `4b608df595b046fb` (submit), status done, passed True, tests 16/16
- speedup 1.5120x, 95% CI [1.4893, 1.5347], score 2490.9806358205487 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.5016522435853301, "ci": [1.47912745993155, 1.530423257736894], "job": "252abb7f11ec48c3"}; base result: muse_attention_decode/hypotheses/done/neutral/restack-sliding-decode-sk-window-on-cluster-master.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-decode1-cluster-per-kvhead.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-decode1-cluster-per-kvhead.patch`; branch `hyp/muse_attention_decode/sliding-decode1-cluster-per-kvhead`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 11.88 | 10.87 | 1.0985 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 22.4 | 22.7 | 0.9921 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 15.51 | 15.24 | 1.0146 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 111.12 | 110.73 | 0.9999 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 17.68 | 17.8 | 1.0058 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 227.87 | 229.07 | 0.9964 | flat |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 13.17 | 13.85 | 0.9737 | flat |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 30.48 | 30.31 | 0.9913 | flat |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 22.83 | 23.23 | 0.9947 | flat |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 163.0 | 163.76 | 1.0001 | flat |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 29.21 | 28.83 | 1.0154 | flat |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 302.31 | 302.56 | 1.0005 | flat |

gained on sliding decode_1. Expected hit: 5/5. Noise ±1.5%.

## Follow-ups

- unproven-vs-best: the change is not proven either way; a stacked hypothesis (`depends_on: sliding-decode1-cluster-per-kvhead`) that adds a further change in the same area is the only way to learn more (identical code is never re-run).

