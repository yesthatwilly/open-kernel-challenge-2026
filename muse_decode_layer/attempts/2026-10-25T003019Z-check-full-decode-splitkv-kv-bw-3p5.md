# muse_decode_layer · 2026-10-25T003019Z · check · full-decode-splitkv-kv-bw-3p5

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-25T001251Z-full-decode-splitkv-kv-bw-3p5.md` (branch `hyp/muse_decode_layer/full-decode-splitkv-kv-bw-3p5`).

- **Job:** `6af3b0e3311943ee` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `2e50a1c`

## Goal
Stage 2 of full-decode-cuda-splitkv-port (REWRITE_READ_2026-10-25 section 2, item 2), built on the restacked code from full-decode-splitkv-restack-on-pdl. Kernel A (splitkv_attn_kernel) is 93-96% of the replaced time and reaches only 2.9 TB/s at B=16 (112.0 us) and 2.6 TB/s at B=32 (249.5 us) vs a KV floor of ~63/126 us; it has STAGES=4 x 16 KB ring at 2 CTAs/SM (~96 KB in flight per SM) over a persistent 2xSM grid of CHUNK=1024-token items, and at B=32 the agentic-spread tail leaves SMs idle. Change only kernel A and its host grid/chunk constants: (a) deeper ring: STAGES 6 if per-CTA smem (ring + idx [2][CHUNK] + q + acc/ml) stays <= ~113 KB so 2 CTAs/SM hold, else STAGES 5; keep the cp.async.cg 16-byte form and the wait_group<STAGES-2> logic; (b) smaller chunk where the tail dominates: make CHUNK a template parameter with instantiations 1024 and 512 and pick 512 when total KV tokens / (2 x SMs) is below ~3 chunks of 1024 or batch >= 24 (host-side from config and cache_seqlens-free sizes only: decide from batch, no host sync), with the combine's split count sized accordingly (S per sequence doubles; combine reads ws by item exactly as now); (c) issue the next item's page-index prefetch two items ahead instead of one. Combine kernel, gate (_splitkv_gate) and everything else unchanged. Self-check: rerun the existing _splitkv_self_check cases (B=5 and B=33, kv_scale 1.0/0.25, device grid and grid 3) for every (STAGES, CHUNK) instantiation used; any failure falls back to the stage-1 configuration (STAGES 4, CHUNK 1024), not to FA3. Print [splitkv-s2] stages/chunk/occupancy and kernel-A CUDA-event time + effective TB/s at B=16 and 32 for old and new configs.

## What changed, and where
Built on the worktree base b2f95e63a6 (the hand-merged full-decode-splitkv-restack-on-pdl). Only the split-KV section of `muse_decode_layer/submission.py` changed: kernel A, the host launcher, the combine's chunk argument, the self-check, and prepare-time selection and timing. The gate (`_splitkv_gate`), the kernel_fn call site and the combine's math are unchanged.

- **Kernel A is templated as `splitkv_attn_kernel<STAGES, CHUNK, AHEAD>`** with a `KCfg` smem layout struct. Six instantiations are dispatched by a variant id: 0 = S4/C1024/A1 (stage 1, byte-identical logic), 1 = S5/C1024/A1, 2 = S5/C1024/A2, 3 = S4/C512/A1, 4 = S5/C512/A1, 5 = S5/C512/A2. The extension was renamed `muse_full_decode_splitkv_v2` and gained `splitkv_variant_info(v)`, which returns `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and the smem bytes.
- **(a) Ring depth: STAGES 5, not 6 (review fix).** S6 would need ~126 KB/CTA and drop to 1 CTA/SM. S5 is 109,584 B at C1024/A1 and 114,704 B with the third meta buffer. The cp.async.cg 16-B form and `wait_group<STAGES-2>` are unchanged.
- **(b) CHUNK 512 by batch only, applied to the small-batch class (review fix; the review overrides the spec's rule).** The spec's rule (512 when batch >= 24) put the small chunk on B=32, where the round-robin tail loss is ~3%, and skipped B=16, where it is ~15%. The rule is now CHUNK 512 when batch <= 23 and 1024 above it. The combine takes the chunk as a runtime argument (n = ceil(len/chunk)), and `max_chunks` and the workspace are sized per chunk.
- **(c) Two-ahead index/query prefetch (AHEAD=2), kept per the spec with the third buffer the review flagged.** It uses `[NBUF=3][CHUNK]` idx and q buffers indexed `k % NBUF`, so there is no k&1 race. The prologue waits for items 0..AHEAD-1, and meta(k+AHEAD) rides item k's first tile and reuses item k-1's buffer. `static_assert`s enforce `AHEAD*TILES >= STAGES-1` and `TILES > STAGES-1`.
- **Selection (prepare, batch only):**
  - Small batch prefers 5 (S5/C512/A2), then 4, then stage 1.
  - Large batch prefers 2 (S5/C1024/A2), then 1, then stage 1.
  - A variant qualifies only if it passes the self-check and has occupancy >= 2.
  - If the stage-1 variant fails its self-check, the path is OFF (FA3). Any other variant's failure falls back to stage 1, not to FA3.
  - The log prints `[splitkv-s2] PATH CHOSEN = ...`.
- **Self-check:** the existing 4 cases (B=5 at kv_scale 1/0.25 on the device grid and grid 3, plus B=33) run for every variant with that variant forced. Each launch is announced before, synced, and printed after.
- **In-run timings (review fix):** `_splitkv_s2_timing` prints every self-checked variant's kernel-A CUDA-event time and TB/s at B=16 and B=32 on synthetic lengths with a ~74K mean, plus the combine time at each chunk. This covers S4/S5 × C1024/C512 and two-ahead on/off, so the kept sub-change can be decided from the log. The timing only informs; it does not choose the variant (no autotuning). Caveat: warm L2 and a synthetic page table.
- **Not done:** the review's acc_s-aliasing idea to make S6 fit (it is a new idea, outside scope). No wgmma or setmaxnreg is used, so `TORCH_CUDA_ARCH_LIST` is untouched. The existing `-arch=sm_90a -lineinfo` flags are kept. There are no mbarriers; the only waits are cp.async wait_group, which are bounded.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/full-decode-splitkv-kv-bw-3p5 diff`:
- [x] CUDA-graph capturable. kernel_fn's path adds only a dict lookup on `batch` (config) and a `torch.empty` sized from shapes. The syncs and `.elapsed_time` calls are in prepare-only self-check and timing code.
- [x] Nothing derived from `data` is cached. `_SPLITKV_SEL` is set once in prepare from synthetic checks and the device occupancy.
- [x] Weights are untouched by this change.
- [x] Activations and precision are unchanged: the query is FP8 E4M3, the KV stays FP8, the combine output is bf16, and accumulation is fp32.
- [x] Attention sees every key: all chunks cover [0, len) and only positions >= len are masked.
- [x] No timing patches, harness access, module edits, reference import, network, or processes. There is no subprocess, multiprocessing, fork or spawn.
- [x] All work happens inside the timed call; nothing is cached or replayed.
- [x] Fallbacks never raise. Build failure or a stage-1 self-check failure gives FA3; a new-variant failure gives stage 1. The timing is wrapped in try/except.

okc_lint output:
```
(host-sync warnings, all "outside kernel_fn's call graph; fine in prepare()")
warn  dtype        L2348/L2350: torch.randint(..., dtype=torch.uint8) -- synthetic FP8 cache bytes for prepare-time timing only
OK with warnings: submission.py
```
`python3 -I -m py_compile` passed. `__pycache__` was removed.

## Risks / what could fail
medium
Medium: 6 stages may drop residency to 1 CTA/SM (print occupancy; the fallback rule picks 5). A 512 chunk doubles combine work (8-9 us now); if combine grows past ~12 us, keep 1024 at B=16. Must not start until full-decode-splitkv-restack-on-pdl lands (it needs a hand merge); build on whatever that run measured.

## Result (after the run)
- job `6af3b0e3311943ee` (check), status done, passed True, tests 13/13
- speedup 1.2663x, 95% CI [1.2473, 1.2853], score 17862.977708744875 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2521147631671776, "ci": [1.23333304171967, 1.270896484614685], "job": "39bcf05afc4e4dd3"}; base result: muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-restack-on-pdl.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-kv-bw-3p5.result.json`, `muse_decode_layer/hypotheses/done/neutral/full-decode-splitkv-kv-bw-3p5.patch`; branch `hyp/muse_decode_layer/full-decode-splitkv-kv-bw-3p5`
