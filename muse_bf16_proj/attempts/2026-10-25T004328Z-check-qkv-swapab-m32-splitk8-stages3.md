# muse_bf16_proj · 2026-10-25T004328Z · check · qkv-swapab-m32-splitk8-stages3

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-25T003858Z-qkv-swapab-m32-splitk8-stages3.md` (branch `hyp/muse_bf16_proj/qkv-swapab-m32-splitk8-stages3`).

- **Job:** `a61778a08ec04ea6` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `e9ec7d9`

## Goal
Built on master (8b9ad43, blob a8bb42a). qkv_gate at 17 <= m <= 32 runs _bf16_swapab_kernel with SWAPAB_CFG qkv_gate = (2304, 8192, SPLIT_K 4, BLOCK_K 128, 4 stages): grid (36, 4) = 144 CTAs on 132 SMs (12 SMs get a second CTA, the rest idle half their smem), 24 KB per stage, 11.8 us kernel = 3.2 TB/s against an 8.0 us floor (row 14.2 at decode_32). Change only that tuple to (2304, 8192, 8, 128, 3): grid (36, 8) = 288 CTAs, K_SLICE 1024 (8 K-steps of 128), 3 stages = 72 KB smem per CTA so 3 CTAs fit per SM (396 slots >= 288, one wave), roughly doubling the weight bytes in flight per SM. The ticket reduce is unchanged code; it now sums 8 fp32 partials of [32, 64] per tile in fixed order 0..7. In _swapab_prepare, allocate state swapab_ws_qkv as (8, SWAPAB_MAX_M, 2304) fp32 (2.4 MB) instead of (4, ...); swapab_cnt_qkv is unchanged (36 counters). Nothing else changes: other layers, m<=16 qkv path (_qkv_splitk_sem_kernel), m>=128 cuBLAS.

## What changed, and where
- SWAPAB_CFG qkv_gate (2304, 8192, 4, 128, 4) -> (2304, 8192, 8, 128, 3): grid (36, 8) = 288 CTAs, K_SLICE 1024.
- _swapab_prepare: swapab_ws_qkv is sized from SWAPAB_CFG["qkv_gate"][2] (constant, = 8) -> (8, 32, 2304) fp32; swapab_cnt_qkv unchanged (36).
- Counters: prepare() already zeroes swapab_cnt_pooler / swapab_cnt_qkv after all warm-ups; kept as is.
- Per the review, the prepare()-time report in _swapab_forward now also prints compiled shared bytes, n_regs and ctas_per_sm_fit (min of 227 KB smem incl. 1 KB/CTA reserve, and 64K regs / (regs*128 threads)); ctas_per_sm_fit >= 3 means 3 CTAs/SM fit. Not runnable here (no GPU): the expected value is 3 (72 KB + pipeline, ~219 KB), unverified.
- No other deviation. Reduce kernel code untouched (loops over SPLIT_K).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff has no .item/.cpu/sync in kernel_fn; the diagnostic print is in prepare() only (report=True), no tensor prints.
- [x] Nothing derived from data cached: ws/cnt sized from constants; synthetic warm-up inputs only.
- [x] Weights: unchanged, BF16 stays BF16; fp32 partials are workspace only.
- [x] Activations BF16: n/a, unchanged.
- [x] Attention: n/a, no attention touched.
- [x] No timing patches, harness access, subprocesses or network in the diff.
- [x] All work inside the timed call; no output caching or replay.

okc_lint: `CLEAN: submission.py`; py_compile OK; __pycache__ removed.

## Risks / what could fail
medium
One scored shape (qkv decode_32), plus the m=33 o test is unaffected. Risk: the 8-way reduce tail (8 x 8 KB reads by the last CTA) and more partial-store traffic may eat the gain; qkv weight (37.7 MB) is L2-resident so judge on the cold-L2 row, not the profile. LESSONS: Triton streams top out near 3.5-3.7 TB/s here, so the realistic ceiling is ~10.5 us kernel.

## Result (after the run)
- job `a61778a08ec04ea6` (check), status done, passed True, tests 9/9
- speedup 1.1963x, 95% CI [1.1784, 1.2143], score 4038.0550826865397 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.2215006063849365, "ci": [1.2031780972891624, 1.2398231154807104], "job": "60b20fd0bf924267"}; base result: muse_bf16_proj/hypotheses/done/neutral/bf16-restack-m16-swapab-router-swapab-m32.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/qkv-swapab-m32-splitk8-stages3.result.json`, `muse_bf16_proj/hypotheses/done/regression/qkv-swapab-m32-splitk8-stages3.patch`; branch `hyp/muse_bf16_proj/qkv-swapab-m32-splitk8-stages3`
