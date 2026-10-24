# muse_attention_decode · 2026-10-24T184726Z · check · splitkv-chunk2048-batch32

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183855Z-splitkv-chunk2048-batch32.md` (branch `hyp/muse_attention_decode/splitkv-chunk2048-batch32`).

- **Job:** `7cfc8c5be3a04369` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `9c8119b`

## Goal
Shape-gated chunk size for master's CUDA split-KV path: at batch >= 32 use 2048-token work items instead of 1024, halving the partials written by kernel A and merged by the combine. (1) template <int CHUNK> on splitkv_attn_kernel and splitkv_combine_kernel (TILES, smem offsets and SMEM_A derived from CHUNK; issue_meta loops `for (int e = tid; e < CHUNK; e += THREADS)`); kernel bodies otherwise unchanged. At 2048 the idx double buffer grows from 8 KB to 16 KB, so SMEM_A is ~99 KB, still 2 CTAs per SM. (2) Host functions take an int64 chunk argument (update _SPLITKV_CPP) and dispatch 1024 or 2048, else TORCH_CHECK. Module renamed muse_full_decode_splitkv_v2. (3) Python: _full_decode_splitkv uses chunk = 2048 if batch >= 32 else 1024 (config only), max_chunks = ceil(page_table.shape[1] / chunk). (4) _splitkv_self_check adds one pass at chunk 2048 by running the existing ragged lens list repeated to 32 requests, grid device and 3.

## What changed, and where
All in the worktree's muse_attention_decode/submission.py (+76/-41).
- CUDA: removed global CHUNK/TILES/IDX_OFF..SMEM_A; added `template <int C> struct SmemA` (offsets, BYTES) and `template <int CHUNK>` on splitkv_attn_kernel and splitkv_combine_kernel (TILES and offsets are kernel-local constexprs; issue_meta loops `for (int e = tid; e < CHUNK; e += THREADS)`, unrolled). Bodies otherwise unchanged.
- Host: splitkv_attn / splitkv_combine take an int64 `chunk` (1024 or 2048, else TORCH_CHECK); launch_attn<CHUNK> / launch_combine<CHUNK> helpers, with the max-dynamic-smem attribute set once per instantiation (SMEM_A about 99 KB at 2048). _SPLITKV_CPP updated; module renamed muse_full_decode_splitkv_v2.
- Python: `chunk = 2048 if batch >= 32 else 1024` (config only); `max_chunks = ceil(page_table.shape[1] / chunk)`, so ws_acc [batch * max_chunks, 16, 64] and ws_ml [batch * max_chunks, 2, 16] are sized for the chunk in use (half the 1024 size at batch 32); the same chunk is passed to both kernels.
- Self-check: added two passes at batch 32 (grid device kv 1.0, grid 3 kv 0.25) on lens32 = ([9000, 1025, 70, 4097, 1023, 2048, 4096, 2047, 2049] * 4)[:32], covering exact 2048/4096 boundaries, 4097, 1025 and a chunk mismatch between kernel A and the combine. Existing batch-5 passes still exercise 1024.
- No deviation from the spec. Review's analytic tail-quantization suggestion not implemented (not an implementation detail).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds no .item()/.cpu()/synchronize(); chunk and max_chunks come from config and page_table.shape only; the attribute call is per-instantiation, set at the first (self-check) call at import, before any capture.
- [x] Nothing derived from `data` cached across calls (the only statics are the per-instantiation smem attribute flags).
- [x] Weights: n/a: no weights in this problem.
- [x] Activations/KV unchanged: FP8 KV cache, FP8 query, fp32 accumulate; diff touches no dtype.
- [x] Attention sees every key: item schedule covers every token at either chunk; the combine merges all ceil(len/chunk) partials; no skipping.
- [x] No timing patches, harness edits, reference import, subprocesses or network in the diff.
- [x] All work inside the timed call; no output caching or replay; no triton.autotune.

Lint output: `CLEAN: muse_attention_decode/submission.py`. py_compile OK, __pycache__ removed.
Uncertainty: not compiled or run (no GPU/nvcc here).

## Risks / what could fail
low
Touches master's splitkv_attn_kernel/splitkv_combine_kernel at full decode 1 < B <= 64, which full-cuda-onelaunch-warpspec-bulk-gather (in the judge now) replaces outright; if that rewrite lands and self-checks, this idea is moot for those shapes and should be held. If the rewrite fails or falls back, this is a pure tuning of the incumbent. Judge on the splitkv kernel rows in top_kernels, not the headline (full decode_32 medians drifted 6% between identical runs).

## Result (after the run)
- job `7cfc8c5be3a04369` (check), status done, passed True, tests 16/16
- speedup 1.3838x, 95% CI [1.3631, 1.4046], score 2721.637802006452 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/splitkv-chunk2048-batch32.result.json`, `muse_attention_decode/hypotheses/done/neutral/splitkv-chunk2048-batch32.patch`; branch `hyp/muse_attention_decode/splitkv-chunk2048-batch32`
