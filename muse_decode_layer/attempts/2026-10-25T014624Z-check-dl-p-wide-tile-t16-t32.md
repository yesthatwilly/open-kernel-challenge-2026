# muse_decode_layer · 2026-10-25T014624Z · check · dl-p-wide-tile-t16-t32

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T224506Z-dl-p-wide-tile-t16-t32.md` (branch `hyp/muse_decode_layer/dl-p-wide-tile-t16-t32`).

- **Job:** `9ad8e2cf869c4b4f` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `c50bee2`

## Goal
Kernel P (dl_pooler_router_kernel<NF>) streams only 2.85 TB/s at T=32 and 3.41 at T=8 (prepare timing in 81ee6714b7614a99.log: 26.4 us vs 20.7 at T=1) because each 64-row CTA re-reads its x K-chunk from L2: x is NF x 2 KB per 16 KB weight stage (50% extra at NF=4). For NF = 2 and 4 only (T 9..32) instantiate a wide variant: BLOCK_N 128 weight rows per CTA (8 warps x 16 rows, THREADS 256), BLOCK_K 64 per stage (W stage 16 KB, x stage NF x 1 KB), 4 stages = 72/80 KB so 2 CTAs/SM; grid 36 n-tiles x SPLIT 6 (K-chunks of 21/21/21/21/21/23 x 64) = 216 CTAs in one wave; x bytes per weight byte halve. Router tiles stay 2 x [16 hi; 16 lo] pairs inside a 128-row tile (tile layout of W_pr unchanged: rows 4096..4607 = 4 wide tiles of 2 router sub-tiles each), split-K ticket per wide tile, same fixed-order reduce and epilogue mapping, ws sized SPLIT 6. PDL pre-wait prefetch of stages 0-2 unchanged. NF=1 (T <= 8) keeps the current kernel. Self-check: the existing P checks at T 9, 16, 32 (ei rel L2 <= 2e-3, logits vs fp64); print occupancy and P timing for the wide variant at T=16/32; fall back per NF to the current kernel if occupancy < 2 or the check fails.

## What changed, and where
Edited only `.worktrees/muse_decode_layer/dl-p-wide-tile-t16-t32/muse_decode_layer/submission.py` (+367/-12). No GPU/nvcc on the build node: nothing was compiled or run.

- New kernel `dl_pooler_router_wide_kernel<NF>` (NF = 2, 4 only) in the `_DL_CUDA` TU, a distinct symbol (review open question 1). 128 weight rows per CTA (8 warps x 16), BLOCK_K 64, 4 stages = 72 / 80 KB, `__launch_bounds__(256, 2)`, grid 36 n-tiles x 6 K-chunks = 216 CTAs. Same instruction forms as master's P (cp.async.cg, ldmatrix.x4, mma.sync m16n8k16), same XOR swizzle (8 chunks per 128 B row), same PDL pre-wait weight prefetch of stages 0-2, same split-K ticket and fixed-order reduce (p = 0..5). Router tiles: rows 4096.. are 4 wide tiles, expert = 64 x (tile - 32) + sub x 16 + j; W_pr layout unchanged.
- Deviation from spec: the K-chunk split uses the master formula `kb = s * 128 / 6` = 21/21/22/21/21/22 blocks, not 21 x 5 + 23. Same sum, better balanced.
- Master's P is untouched and still serves T <= 8. `dl_pooler_router` takes a new `wide` bit mask (2 = NF 2, 4 = NF 4); `_dl_forward` zeroes it for T > `DL_WIDE_MAX_TOKENS` (new constant, = 32, separate from `DL_MAX_TOKENS`, which is not edited). `ws` is allocated for SPLIT 6 (`DL_WIDE_SPLIT_K`); `cnt_p` already has 72 counters.
- prepare(): `_dl_wide_info` prints regs, local/spill bytes (cudaFuncGetAttributes), smem, CTAs/SM and warps/SM (at the occupancy limit and as resident for the real grid on this GPU's SM count) for old and wide P at NF=2/4, so the 8 -> 16 warps/SM lever is visible next to the 2 CTAs/SM number. `_dl_wide_check` runs old vs wide P at T 9/12/16 and 17/24/32, two launches each, with a flushed print before and after a `torch.cuda.synchronize()` for every launch; passes at ei rel L2 <= 2e-3 and logits max-rel <= 1e-4 vs master's P (bit-exactness is printed, not required, since the split-K order differs) and counters all reset. Then the existing full `_dl_self_check` (fp64 logits, R, Marlin y, graph replay at T=32) is rerun with the wide path on. Per NF fallback to master's P if occupancy < 2, the check fails or raises, or the full self-check fails. `_dl_time_wide` prints CUDA-event graph timings (20 launches x 5 replays, pdl=0) of old vs wide P at T=16 and 32. Prepare timings undercount the profile by 20-25% (LESSONS 2026-10-25): rank on them, judge on the profile row.
- `PATH CHOSEN for P:` line per shape class (T<=8, 9..16, 17..32). `-lineinfo` added to the nvcc flags. The wide self-check never raises into prepare().
- Review fixes applied: distinct symbol; old and wide timed in the same run at T=16/32 (T=9 is covered by the check, not the timing); fallback per NF on occupancy < 2. Not applied (not in this change): `griddepcontrol.launch_dependents` after the K loop, which the review says the wide kernel must get only if dl-route-warp-topk-pdl lands first.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the wide kernel is launched with `cudaLaunchKernelEx` on the current stream; no `.item()`/`.cpu()`/sync/print in kernel_fn; the wide mask is a prepare() constant and the branch depends on T (shape) only. The self-check captures a graph at T=8/32 with wide on.
- [x] Nothing derived from `data` is cached across calls: only `dl["wide"]` (a prepare-time constant of kernel availability) was added; ws/cnt are scratch, not input-derived.
- [x] Weights: W_pr unchanged (bf16 pooler rows + router hi/lo, built in prepare()); no new weight copies.
- [x] Activations BF16, fp32 accumulation, same epilogue as master's P; KV cache and queries untouched (n/a: attention is not edited).
- [x] Attention sees every key: n/a, attention not touched.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network: the diff adds none (the build keeps `load_inline` in process, as master does).
- [x] All work happens inside the timed call; no output caching or replay: the wide P reads x and writes ei/logits on every call.
- [x] Only proven instruction forms: cp.async.cg, ldmatrix.x4, mma.sync m16n8k16, griddepcontrol.wait, atomicAdd, threadfence, all already in master's P. No wgmma, so no 9.0a arch change.

okc_lint output (warnings are prepare()-only host syncs, all outside kernel_fn):
```
33 x "warn host-sync ... (outside kernel_fn's call graph; fine in prepare())"
OK with warnings: muse_decode_layer/submission.py
```
`python3 -I -m py_compile` ok, __pycache__ removed.

## Risks / what could fail
medium
Medium: register pressure at 8 warps x NF=4 fragments; if occupancy drops to 1 CTA/SM the 216-CTA grid takes two waves and loses. Small headline effect (~+0.5%), judge on the P timing line and the decode_16/32 rows.

## Result (after the run)
- job `9ad8e2cf869c4b4f` (check), status done, passed True, tests 13/13
- speedup 1.2457x, 95% CI [1.2270, 1.2644], score 18158.55872270687 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.2521147631671776, "ci": [1.23333304171967, 1.270896484614685], "job": "39bcf05afc4e4dd3"}; base result: muse_decode_layer/hypotheses/done/neutral/stack-align-par-sidestream-sliding1.result.json
- **outcome: neutral** (within-ci)
- files: `muse_decode_layer/hypotheses/done/neutral/dl-p-wide-tile-t16-t32.result.json`, `muse_decode_layer/hypotheses/done/neutral/dl-p-wide-tile-t16-t32.patch`; branch `hyp/muse_decode_layer/dl-p-wide-tile-t16-t32`
