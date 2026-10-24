# muse_attention_decode · 2026-10-24T184625Z · check · full-decode1-cuda-splitkv-chunk640

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T183847Z-full-decode1-cuda-splitkv-chunk640.md` (branch `hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk640`).

- **Job:** `91d541778b8f45f3` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `299b00d`

## Goal
Same mechanism as full-decode1-cuda-splitkv-chunk320, with fewer, longer items. Route FULL-layer decode at batch 1 (window == 0, q_len 1, tp8, d 64, page 1) through master's CUDA split-KV path instead of FA3. (1) template <int CHUNK> on splitkv_attn_kernel and splitkv_combine_kernel (TILES, smem offsets and SMEM_A derived from CHUNK; issue_meta loops `for (int e = tid; e < CHUNK; e += THREADS)`); kernel bodies otherwise byte for byte. (2) splitkv_combine_kernel gets template <int CHUNK, int NW> (red arrays [NW], loop strides NW * 32 and NW, __launch_bounds__(NW * 32)). (3) Host functions take an int64 chunk argument (update _SPLITKV_CPP): 1024 -> <1024> attn and <1024, 4> combine (today's code); 640 -> <640> attn and <640, 8> combine; anything else TORCH_CHECK fails. Module renamed muse_full_decode_splitkv_v2. (4) Python: _splitkv_gate allows 1 <= batch; _full_decode_splitkv uses chunk = 640 if batch == 1 else 1024, max_chunks = ceil(page_table.shape[1] / chunk), grid unchanged (2 x SM count; at 80K keys only 125 items exist, so 125 CTAs get one 10-tile item each and the rest exit at the `bx >= total` check). (5) _splitkv_self_check adds batch-1 cases lens [80000] and [5000], kv_scale 1 and 0.25, grid device and 3, rel L2 < 2e-2. Print '[splitkv] batch-1 chunk 640 path ON/OFF' once at import.

## What changed, and where
Only `muse_attention_decode/submission.py` in the worktree changed (+108/-61).
- CUDA: `CHUNK`/`TILES` global constants removed; `splitkv_attn_kernel<int CHUNK>` derives TILES and the smem offsets from a `Lay<CHUNK>` struct (SMEM_A per instantiation); `issue_meta` loops `for (e = tid; e < CHUNK; e += THREADS)`. Bodies otherwise unchanged, so `<1024>` is today's kernel.
- `splitkv_combine_kernel<CHUNK, NW>`: red arrays [NW], first-pass stride NW*32, warp stride NW, warp-0 final loop NW, and the red_m max reduction is now a loop over NW (the review's trap), `__launch_bounds__(NW*32)`, launched with NW*32 threads.
- Host: `launch_attn<CHUNK>` / `launch_combine<CHUNK,NW>` templates (per-instantiation cudaFuncSetAttribute flag); `splitkv_attn`/`splitkv_combine` take int64 chunk (1024 -> <1024>/<1024,4>, 640 -> <640>/<640,8>, else TORCH_CHECK). `_SPLITKV_CPP` updated; module `muse_full_decode_splitkv_v2`.
- Python: gate allows batch 1 only when `_B1_OK`; chunk = 640 at batch 1 else 1024; max_chunks from that chunk; grid unchanged.
- Deviation (per review open question 2): batch-1 self-check (lens [80000] and [5000], kv_scale 1 and 0.25, device and 3-CTA grids, rel L2 < 2e-2, finite) runs after the batch>1 check, and a failure there only sets `_B1_OK = False` (batch 1 stays on FA3, batch>1 split-KV stays on). Prints `[splitkv] batch-1 chunk 640 path ON/OFF`. Gate never raises. Not run (no GPU).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no .item()/.cpu()/synchronize() in kernel_fn; the only new host work is shape/config arithmetic; the self-check is import-time only.
- [x] Nothing derived from `data` is cached across calls (only a per-instantiation static attribute flag; `_B1_OK` is set from synthetic data at import, not from inputs).
- [x] Weights: n/a: no weights touched.
- [x] Activations/KV/query dtypes unchanged (FP8 KV, FP8 e4m3 q, bf16 out).
- [x] Attention sees every key: positions < cache_seqlens all processed (chunk 640 items cover ceil(len/640) chunks); no skipping.
- [x] No timing patches, harness access, reference import, subprocess or network (diff checked).
- [x] All work inside the timed call; no output caching or replay.

okc_lint output: `CLEAN: submission.py`

## Risks / what could fail
medium
A/B sibling of full-decode1-cuda-splitkv-chunk320: 320 maximizes CTAs (250, ~2 per SM) at the cost of 250 partials; 640 halves the partials and keeps ~1 CTA per SM. The two differ only in the chunk/NW constants, so comparing their kernel A and combine rows attributes the trade. Untouched by every rewrite in the judge.

## Result (after the run)
- job `91d541778b8f45f3` (check), status done, passed True, tests 16/16
- speedup 1.3805x, 95% CI [1.3598, 1.4052], score 2728.254245433283 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/sliding-deep-splits-on-cuda-splitkv.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode1-cuda-splitkv-chunk640.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode1-cuda-splitkv-chunk640.patch`; branch `hyp/muse_attention_decode/full-decode1-cuda-splitkv-chunk640`
