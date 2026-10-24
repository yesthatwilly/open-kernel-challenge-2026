# muse_attention_decode · 2026-10-24T184624Z · submit · full-decode1-cuda-splitkv-chunk320

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183847Z-full-decode1-cuda-splitkv-chunk320.md` (branch `hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk320`).

- **Job:** `8d4771feca9b4e18` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `183eb55`

## Goal
Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through master's existing CUDA split-KV path with a smaller work chunk, instead of unpaired FA3 + FlashAttnFwdCombine + sglang norm_gate. (1) In _SPLITKV_CUDA make the chunk a template parameter: template <int CHUNK> on splitkv_attn_kernel and on splitkv_combine_kernel (TILES = CHUNK / TILE, IDX_OFF/Q_OFF/... offsets and SMEM_A become constexpr functions of CHUNK; the idx double buffer is 2 * CHUNK ints). Replace issue_meta's loop `for j < CHUNK / THREADS` with `for (int e = tid; e < CHUNK; e += THREADS)` so CHUNK need not be a multiple of 256. Everything else in the kernels (ring, swizzle, mma, softmax, item-end merge, partial slot b * max_chunks + chunk) stays byte for byte. (2) Give splitkv_combine_kernel a second template parameter NW (warps per block): red_m/red_l/red_acc sized [NW], loop 1 strides by NW * 32, loop 2 strides by NW, the final warp-0 merge sums NW entries, __launch_bounds__(NW * 32). (3) Host functions splitkv_attn and splitkv_combine take a new int64 chunk argument (update _SPLITKV_CPP too) and dispatch: chunk 1024 -> <1024> (combine <1024, 4>, i.e. today's code), chunk 320 -> <320> (combine <320, 16>, 512 threads), anything else TORCH_CHECK fails. cudaFuncSetAttribute is set once per instantiation. Rename the module to muse_full_decode_splitkv_v2 so no stale build is reused. (4) Python: _splitkv_gate widens 1 < batch to 1 <= batch; _full_decode_splitkv picks chunk = 320 if batch == 1 else 1024 (config only), max_chunks = ceil(page_table.shape[1] / chunk), and passes chunk to both calls; grid stays 2 x SM count. At 80K keys that is 250 items of 5 tiles, one per CTA, versus 79 items at 1024. kernel_fn is unchanged otherwise: batch 1 is already unpaired, so _pre_attention_kernel writes the [1, 16, 64] FP8 q the split-KV path takes. (5) Self-check: add a batch-1 pass to _splitkv_self_check with lens [80000] and [5000] at kv_scale 1 and 0.25, grid device and grid 3, same rel L2 < 2e-2 bound; batch > 1 cases unchanged. Print '[splitkv] batch-1 chunk 320 path ON/OFF' once at import.

## What changed, and where
All in the worktree's muse_attention_decode/submission.py (+108/-57). _SPLITKV_CUDA: CHUNK is now a template parameter of splitkv_attn_kernel<CHUNK> and splitkv_combine_kernel<CHUNK, NW>; smem offsets are constexpr functions of CHUNK; issue_meta loops `for (e = tid; e < CHUNK; e += THREADS)`; static_asserts on CHUNK % TILE and TILES >= STAGES-1. Combine: red_m/red_l/red_acc sized [NW], loop 1 strides NW*32, loop 2 strides NW, warp-0 merge sums NW entries, max over red_m is a loop, __launch_bounds__(NW*32), launched with NW*32 threads (the trap from the brief). Host: launch_attn<CHUNK>/launch_combine<CHUNK,NW> templates, dispatch 1024 -> <1024>/<1024,4>, 320 -> <320>/<320,16>, else TORCH_CHECK; cudaFuncSetAttribute once per instantiation (function-local static). New int64 chunk arg in both functions and _SPLITKV_CPP; module renamed muse_full_decode_splitkv_v2. Python: gate 1 <= batch (batch 1 additionally requires _B1_ON), chunk = 320 if batch == 1 else 1024, max_chunks from chunk, chunk passed to both calls.

Deviation (self-check, per the brief's fall-back rule): the batch-1 pass (lens [80000] and [5000], kv_scale 1 and 0.25, device grid and grid 3, same rel L2 < 2e-2 bound) is separate from the batch>1 pass, and a batch-1 failure or exception only clears _B1_ON (batch 1 returns to the master FA3 path) without disabling the batch > 1 CUDA path; it never raises. Import prints '[splitkv] batch-1 chunk 320 path ON/OFF'. Nothing could be compiled or run on this node (no GPU/nvcc).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no .item()/.cpu()/synchronize()/prints in kernel_fn; chunk and grid come from shapes/config/device only (checked in git diff).
- [x] Nothing derived from `data` is cached across calls: the only new statics are the per-instantiation cudaFuncSetAttribute flag and _B1_ON, set at import from synthetic data.
- [x] Weights: n/a: no weights touched.
- [x] Activations BF16, KV cache FP8, queries FP8 E4M3: unchanged (same pre-attention kernel, same FP8 q).
- [x] Attention sees every key: items cover all tokens (ceil(seq / chunk)), no skipping; self-check compares against a full FP32 reference.
- [x] No timing patches, harness access, reference import, subprocesses or network: none added (load_inline was already there).
- [x] All work happens inside the timed call; no output caching or replay: workspaces and outputs are allocated and computed per call.

okc_lint.py output: `CLEAN: muse_attention_decode/submission.py`; py_compile OK, __pycache__ removed.

## Risks / what could fail
medium
Full decode_1 is the one scored shape none of the three CUDA rewrites in the judge touches (full-cuda-onelaunch-warpspec-bulk-gather gates 1 < batch, its batch 1 is stage 3). Precedent: the Triton gather at batch 1 lost 5% to FA3 at 0.89 TB/s (full-fused-triton-decode1-gate); the CUDA kernel streams 2.94 TB/s at decode_16, so the batch-1 risk is parallelism, which the 320-token chunk addresses. Weakest link: the combine merging 250 partials per head from L2 (the 4-warp combine already costs 8 us at B=16); read its own row. Sibling: full-decode1-cuda-splitkv-chunk640 is the A/B on chunk size; run whichever the reviewer prefers first.

## Result (after the run)
- job `8d4771feca9b4e18` (submit), status done, passed True, tests 16/16
- speedup 1.3954x, 95% CI [1.3744, 1.4163], score 2699.129510489057 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode1-cuda-splitkv-chunk320.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode1-cuda-splitkv-chunk320.patch`; branch `hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk320`
