# muse_attention_decode · 2026-10-24T184625Z · check · splitkv-combine-wide-mlp

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183848Z-splitkv-combine-wide-mlp.md` (branch `hyp/muse_attention_decode/splitkv-combine-wide-mlp`).

- **Job:** `47967b6c4d98454b` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `34daa03`

## Goal
Rework splitkv_combine_kernel (master full decode, 1 < batch <= 64) for more loads in flight; grid (B, 16) and the math stay. Today each (request, head) block has 4 warps; warp w walks partials w, w+4, ... (~20 per warp at 79 partials), each iteration loading m, then l and a float2 of acc, so the loop is a chain of L2 round trips. New: 8 warps (256 threads, __launch_bounds__(256), red_m/red_l/red_acc sized [8], loop 1 strides by 256). Loop 2: warp w handles partials c = w + 8 * i; process them in groups of 4: first load the 4 m values and 4 l values and the 4 float2 acc values into registers (masked by c < n, m = -inf when out of range), then compute the 4 weights and accumulate, so up to 12 independent loads are in flight per lane. The final warp-0 merge sums 8 entries; the epilogue (v_scale, scaleless RMSNorm with eps inside the sqrt, sigmoid(gate), bf16 store) is unchanged. Host launch uses 256 threads. Rename the module muse_full_decode_splitkv_v2.

## What changed, and where
Edited only `.worktrees/muse_attention_decode/splitkv-combine-wide-mlp/muse_attention_decode/submission.py`. In `splitkv_combine_kernel`: `__launch_bounds__(256)`, red_m/red_l/red_acc sized [8], M pass strides by 256, 8-way max of red_m, loop 2 rewritten as groups of 4 (c = warp + 8*j from c0, step 32): all m, l, float2 acc loaded into registers (masked c < n, m = -inf, l = 0, acc = 0 out of range) before weights/accumulate; final warp-0 merge sums 8 entries; host launch 256 threads; module renamed muse_full_decode_splitkv_v2; header comment updated. Epilogue unchanged. Deviation: none. Not included (review's optional suggestions, new ideas rather than implementation details): gate prefetch at kernel start and single-pass online-rescale merge.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff only touches kernel code and launch block size; no syncs, allocation or value-dependent host control flow.
- [x] Nothing derived from `data` is cached across calls: diff adds no state.
- [x] Weights: n/a: attention problem, no weights touched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: unchanged; combine still fp32 merge and bf16 store.
- [x] Attention sees every key: merge still covers all n partials (masked loads only for c >= n); no skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none in diff.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] No triton.autotune, no host syncs in kernel_fn; first-line marker kept.

okc_lint output: `CLEAN: muse_attention_decode/submission.py`; py_compile OK, __pycache__ removed.

## Risks / what could fail
low
Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs).

## Result (after the run)
- job `47967b6c4d98454b` (check), status done, passed True, tests 16/16
- speedup 1.3836x, 95% CI [1.3629, 1.4044], score 2722.0682137655576 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/splitkv-combine-wide-mlp.result.json`, `muse_attention_decode/hypotheses/done/neutral/splitkv-combine-wide-mlp.patch`; branch `hyp/muse_attention_decode/splitkv-combine-wide-mlp`
