# muse_attention_decode · 2026-10-24T184625Z · check · splitkv-full-line-gather-mapping

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183847Z-splitkv-full-line-gather-mapping.md` (branch `hyp/muse_attention_decode/splitkv-full-line-gather-mapping`).

- **Job:** `cfa8c89e4f2b488d` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `7dafeb4`

## Goal
One change inside splitkv_attn_kernel's issue_tile lambda (master full decode, 1 < batch <= 64): remap threads so that each warp-wide cp.async instruction covers whole 128-byte cache rows instead of half rows. Today thread tid copies token tt = tid >> 2, 16-byte chunks c = 2 * (tid & 3) + {0, 1}, so one warp instruction touches 8 tokens x 64 bytes (two sectors of each line) and the other half of every line arrives from the next instruction. New mapping: for j in {0, 1}: tt = (tid >> 3) + 32 * j, c = tid & 7; the destination stays dst = tt * ROW + ((c ^ (tt & 7)) << 4) for K and TILE * ROW + dst for V, the validity test and the page index read idx_s[(k & 1) * CHUNK + t * TILE + tt] are computed per tt as now, and the source is ks + c * 16 / vs + c * 16 with src_bytes ok ? 16 : 0. Each thread still issues 4 cp.async16 per tile (2 tokens x K and V), the ring layout, swizzle, commit/wait counts and all compute are unchanged, so the smem contents per stage are byte-identical to master's. Rename the load_inline module to muse_full_decode_splitkv_v2 (no stale build). The import self-check runs unchanged.

## What changed, and where
In splitkv_attn_kernel's issue_tile (submission.py), thread mapping is now tt = (tid >> 3) + 32 * j for j in {0,1}, c = tid & 7. The validity test, page-index read and row pointers are computed per tt (inside the j loop). dst swizzle, K/V offsets, src_bytes, ring layout and commit/wait counts are unchanged, so smem contents are byte-identical. The module is renamed to muse_full_decode_splitkv_v2. The comment above issue_tile is updated. No deviation from the spec. The review's mechanism correction (win is fewer duplicate L2 sector requests, not DRAM merging) is a judging note only.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff only changes cp.async addressing inside the kernel; no syncs, allocation or value-dependent control flow added.
- [x] Nothing derived from `data` is cached across calls: diff adds no state.
- [x] Weights: n/a: no weight handling changed.
- [x] Activations/KV dtype: n/a: unchanged; the same FP8 bytes are copied.
- [x] Attention sees every key: the same tokens are covered (64 per tile via 2 x 32), same validity test.
- [x] No timing patches, harness access, subprocesses or network: diff touches only the gather mapping and the module name.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `CLEAN: submission.py`. py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs). The rewrite's BULK=0 fallback uses this same full-line mapping, so a gain here also tells stage 2 of that rewrite which copy form to keep.

## Result (after the run)
- job `cfa8c89e4f2b488d` (check), status done, passed True, tests 16/16
- speedup 1.4160x, 95% CI [1.3948, 1.4423], score 2659.7552901313115 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/splitkv-full-line-gather-mapping.result.json`, `muse_attention_decode/hypotheses/done/neutral/splitkv-full-line-gather-mapping.patch`; branch `hyp/muse_attention_decode/splitkv-full-line-gather-mapping`
