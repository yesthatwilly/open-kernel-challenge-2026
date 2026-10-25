# muse_attention_decode · 2026-10-25T162957Z · submit · sliding-decode1-cluster-per-kvhead

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-25T153027Z-sliding-decode1-cluster-per-kvhead.md` (branch `hyp/muse_attention_decode/sliding-decode1-cluster-per-kvhead`).

- **Job:** `4b608df595b046fb` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `037e0696`

## Goal
Built on current master (no depends_on). Sliding decode_1 (weight 46, 60% of the decode_1 load) runs sliding_cluster_kernel at C=16: ONE cluster of 16 CTAs for the whole step, 10.8 us kernel / 11.9 us shape, floor 0.1 us. Each CTA owns 128 old keys of BOTH kv heads and, before it can compute, builds all 16 query heads (RMSNorm + RoPE with accurate sincosf, 32 B local stack) while its gather flies; the last rank also builds the new k/v rows. With only 16 CTAs on 132 SMs, per-CTA latency is the whole shape. CHANGE (B=1 band only): a template flag PERHEAD (false = master, byte-identical) where the grid is 2 clusters of C=16 (32 CTAs), cluster k serving kv head k only: each CTA gathers only its head's 64-byte half of every K and V row (16-byte cp.async on the 4 chunks of that half; same SW pattern), builds only its 8 query heads, runs the same mma.sync QK/PV for one head with all 8 warps splitting the 128 keys (instead of warps alternating heads), and the DSMEM reduce-scatter covers 8 rows instead of 16. The new k/v build stays on the last rank of EACH cluster for its own head (each writes its half of the 128-byte cache row; the two halves are disjoint bytes). Output: rows of head k only. Gate in prepare: B=1 x 80000, interleaved cold-L2 graph replay (empty graph subtracted, median of 31) PERHEAD vs master C=16; route at >= 1.03x and rel L2 < 1e-2 vs master; self-check on master's slide cases (lens 1/10/2047/2048/2049/80000, kv 0.25) with cache rows compared to master's, plus the two-state alternating replay; else master. B=17..32 and every other path untouched. RULES (every path): CUDA-graph capturable; kernel_fn does no host sync, no allocation sized by input values (shape/config only), no data-dependent Python control flow, no prints, no try/except, no per-call mutable Python state; nothing cached from inputs; KV FP8, q at most FP8 E4M3, fp32 softmax/merges, bf16 out; every key in each window attended; no child processes, no timing in kernel_fn. The judge prepares TWO states and rotates them: no module/static global may hold a device pointer into a state's tensors, a TMA map or a state-sized buffer (self-resetting int32 ticket arrays, zeroed once and returned to 0 by every launch, are fine).

## What changed, and where
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

## Rules check (each must be true before the run)
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

## Risks / what could fail
medium
Risk: 64-byte half-row gathers halve sector use per request but the shape is latency-bound (0.5 MB window); if the gate keeps master the timing table still tells whether per-CTA work is part of the 10.8 us. Interacts with the running epilogue-gate-l2-prefetch-cuda (gate prefetch in the CUDA epilogues): disjoint lines, both may stack.

## Result (after the run)
- job `4b608df595b046fb` (submit), status done, passed True, tests 16/16
- speedup 1.5120x, 95% CI [1.4893, 1.5347], score 2490.9806358205487 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.5016522435853301, "ci": [1.47912745993155, 1.530423257736894], "job": "252abb7f11ec48c3"}; base result: muse_attention_decode/hypotheses/done/neutral/restack-sliding-decode-sk-window-on-cluster-master.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-decode1-cluster-per-kvhead.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-decode1-cluster-per-kvhead.patch`; branch `hyp/muse_attention_decode/sliding-decode1-cluster-per-kvhead`
