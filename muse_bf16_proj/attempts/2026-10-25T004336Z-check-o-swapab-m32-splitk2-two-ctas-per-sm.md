# muse_bf16_proj · 2026-10-25T004336Z · check · o-swapab-m32-splitk2-two-ctas-per-sm

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T003858Z-o-swapab-m32-splitk2-two-ctas-per-sm.md` (branch `hyp/muse_bf16_proj/o-swapab-m32-splitk2-two-ctas-per-sm`).

- **Job:** `75f09fa68f9c4415` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `39a42db`

## Goal
Built on master (8b9ad43, blob a8bb42a). o at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG o = (8192, 1024, SPLIT_K 1, BLOCK_K 256, 4 stages): grid (128, 1) = 128 CTAs, each streaming 64 x 1024 x 2 B = 128 KB in 4 K-steps, one CTA per SM (4 x 48 KB = 192 KB smem), 4.8 us kernel against a 3.6 us floor (row 6.3 at decode_32). With one CTA per SM and only 4 steps, the pipeline is mostly fill. Change the tuple to (8192, 1024, 2, 256, 2): grid (128, 2) = 256 CTAs, K_SLICE 512 (2 K-steps, both in flight with 2 stages), 96 KB smem per CTA so 2 CTAs fit per SM (264 slots, one wave); each CTA streams 64 KB and the two CTAs on an SM overlap their fill. The existing ticket reduce handles SPLIT_K 2 (sum of 2 fp32 [32, 64] partials, fixed order). _swapab_buffers returns state swapab_ws_o / swapab_cnt_o for layer o; _swapab_prepare allocates swapab_ws_o = (2, SWAPAB_MAX_M, 8192) fp32 (2 MB) and swapab_cnt_o = zeros(128) int32 and re-zeroes it after warm-up like the others. Nothing else changes (o at m<=16 keeps _pooler_skinny_kernel; m=33 o test and m>=128 keep cuBLAS).

## What changed, and where
In `muse_bf16_proj/submission.py` (worktree only, +11/-1):
- `SWAPAB_CFG["o"]` is now `(8192, 1024, 2, 256, 2)`: grid (128, 2), K_SLICE 512, 2 stages.
- `_swapab_buffers` returns `swapab_ws_o` and `swapab_cnt_o` for layer o.
- `_swapab_prepare` allocates both from constant shapes. `swapab_ws_o` is fp32 (2, 32, 8192). `swapab_cnt_o` is int32 zeros(128). It zeroes `swapab_cnt_o` again after the warm-up loop.
- The `report=True` print in prepare() now also shows `compiled.metadata.shared` (must be <= ~113 KB for 2 CTAs/SM), `n_regs`, and the number of `cp.async.commit_group` before the first `cp.async.wait_group` in the PTX. It prints for every swap-AB layer, in prepare() only.
- No deviation from the spec. `kernel_fn` is untouched. If 2 stages hit OutOfResources, the existing fallback uses 3 stages, which would break the 2 CTAs/SM goal. The printed shared size shows this.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: the diff adds no `.item()`, `.cpu()` or sync to `kernel_fn`. The new prints are in prepare() only, and `ws` and `cnt` are preallocated.
- [x] Nothing derived from `data` is cached: the new state is constant-shape buffers only.
- [x] Weights: unchanged, BF16 stays BF16. The split-K workspace is fp32 accumulation scratch, as for pooler and qkv.
- [x] Activations BF16 where SGLang keeps BF16: n/a, no activation dtype change.
- [x] Attention: n/a, projection GEMMs only.
- [x] No timing patches, harness access, subprocesses or network: checked in the diff.
- [x] All work is inside the timed call: no output caching or replay.

okc_lint: `CLEAN: muse_bf16_proj/submission.py`. py_compile OK, `__pycache__` removed.

## Risks / what could fail
medium
Counter-evidence: the CUDA gemv rewrite ran o with SPLIT_K 2 and lost at m=8 (the m-proportional ticket-reduce tail); here the reducer reads only 2 partials of 8 KB and the tail is one CTA per tile, but if the decode_32 row is not below 6.3 the tail cost is confirmed and o-unpooler-slice-cuda-cluster-dsmem-rewrite (DSMEM reduce) is the route instead. One scored shape, attributable from its row.

## Result (after the run)
- job `75f09fa68f9c4415` (check), status done, passed True, tests 9/9
- speedup 1.1860x, 95% CI [1.1682, 1.2038], score 4073.096050881423 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/o-swapab-m32-splitk2-two-ctas-per-sm.result.json`, `muse_bf16_proj/hypotheses/done/regression/o-swapab-m32-splitk2-two-ctas-per-sm.patch`; branch `hyp/muse_bf16_proj/o-swapab-m32-splitk2-two-ctas-per-sm`
