# muse_attention_decode · 2026-10-24T165946Z · submit · full-decode-cuda-fp8-qk-mma

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T165302Z-full-decode-cuda-fp8-qk-mma.md` (branch `hyp/muse_attention_decode/full-decode-cuda-fp8-qk-mma`).

- **Job:** `b939681d3ab549d7` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `3a7b503`

## Goal
Built on master (full-decode-cuda-gather-splitkv). One change inside splitkv_attn_kernel (full-layer decode at 1 < batch <= 64, the only kernel touched): compute S = Q K^T with the native FP8 tensor-core op mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instead of widening Q and K to f16 for m16n8k16. A operand: the FP8 query bytes read straight from q_s (row quad = query head g*8 + quad; a0 = 4 bytes at dims ks*32 + 4*qi, a2 = dims ks*32 + 16 + 4*qi; a1 = a3 = 0 for the zero rows 8-15), loaded once per item at t == 0 into 4 uint32 registers (replacing qa[4][2] and its 8 cvt per lane). B operand: the raw K bytes of token slice + nb*8 + quad, kv head g, from the swizzled ring: for k-step ks (2 per 64 dims) b0 = the 32-bit word at dims ks*32 + 4*qi and b1 = at ks*32 + 16 + 4*qi, i.e. 16-byte chunk index g*4 + ks*2 + {0,1} XOR (tt & 7), byte offset 4*qi, so each lane still reads one contiguous 32-bit word per register and the swizzle stays conflict-free. Per (warp, tile) QK goes from 8 mma16816 + 8 K cvt-pairs to 4 mma16832 + 0 cvt; the natural dim order is used for both operands, so the k-permutation trick and its comment go away. The C fragment layout of m16n8k32 f32 is the same as m16n8k16 (c0,c1 = row quad, cols 2*qi, 2*qi+1), so the masking, online softmax, P packing, P@V (f16, V widened exactly as today), item-end merge, kernel B, gate, grid, smem and pipeline are unchanged byte for byte. Rename the load_inline module (muse_full_decode_splitkv_v1 -> _v1fp8qk) so a cached build is not reused; keep the import-time self-check and its thresholds as they are (it now validates the fp8 QK path).

## What changed, and where
Worktree: `.worktrees/muse_attention_decode/full-decode-cuda-fp8-qk-mma`, file `muse_attention_decode/submission.py` only (diff: 85 insertions, 15 deletions; uncommitted).

- **Kernel A QK (`splitkv_attn_kernel`) under `#if SPLITKV_FP8QK`:** a new `mma16832_e4m3` helper wraps `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. `qa` is `uint32_t qa[2][2]` (4 regs). It is loaded at `t == 0` straight from `q_s` with no cvt: `qa[s][0]` = word at dims `s*32 + 4*qi`, `qa[s][1]` = word at `s*32 + 16 + 4*qi`, row = query `g*G + quad`, and a1 = a3 = 0. K B operand per k-step `ks` (2 of them): `b0` = word at chunk `(g*4 + ks*2) ^ (tt&7)`, `b1` = chunk `(g*4 + ks*2 + 1) ^ (tt&7)`, byte offset `4*qi`. That is 4 mma16832 per (warp, tile) and 0 K cvt. Natural dim order. Everything else in kernel A, kernel B, the grid, smem, the pipeline, the masking, the softmax and P@V is byte-identical.
- **Fallback (REQUIRED by the review):** the same CUDA string is compiled with `#define SPLITKV_FP8QK 1` or `0` prepended. `0` reproduces master's f16 QK code exactly: the old `qa[4][2]`, the cvt and m16n8k16, with the k-permutation comment kept in the `#else` branch. At import the loop tries fp8 first (module `muse_full_decode_splitkv_v1fp8qk`). If the build or the unchanged self-check (`err < 2e-2`, finite, same 3 configs vs FP32 ref) fails, it builds and self-checks the f16 kernel (module `muse_full_decode_splitkv_v1`). Only if that also fails does it drop to FA3. It prints `[splitkv] active path: fp8 QK (mma.sync m16n8k32 e4m3)` or `... f16 QK fallback ...`, and `SELF-CHECK FAILED (fp8qk)` when the fp8 kernel builds but is wrong.
- **Build evidence:** a new host function `splitkv_info()` (C++ `cudaFuncGetAttributes` on kernel A, in-process, no subprocess) is printed after each build. It shows SPLITKV_FP8QK, regs/thread, local (spill) bytes/thread, SASS binaryVersion and PTX version. **Deviation / limit:** native QMMA vs F2FP+HMMA emulation cannot be read in-process, because that needs `cuobjdump -sass`, which is a subprocess and a rules violation from submission.py. This node has no nvcc/cuobjdump either, so the offline SASS check was not done. Per the review, a flat kernel-A time must be recorded as "emulated-or-memory-bound (undetermined)". The regs/thread line gives a partial hint: emulation would likely raise register use relative to the f16 build (the f16 build prints its own count only when the fallback is taken).
- Comments: the kernel-A header block and the mma comments are updated. The k-permutation comment survives only in the f16 `#else` branch.
- The self-check thresholds and synthetic data are unchanged. In the fp8 build they now validate the fp8 QK path.
- Compile-time cost: the fallback build runs only on failure. The worst case is two nvcc builds at import, well inside the 15-minute budget if master's single build was ~1 min.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/full-decode-cuda-fp8-qk-mma diff`:

- [x] CUDA-graph capturable: the diff touches no host code in `kernel_fn` / `_full_decode_splitkv`. The new `splitkv_info()` and prints run only at import. No `.item()`/`.cpu()`/sync, no allocation or branching on input values. The fp8/f16 choice is an import-time build flag, not data.
- [x] Nothing derived from `data` is cached across calls: no new state. `_EXT` is a compiled module, and the self-check uses synthetic data only.
- [x] Weights: n/a: this problem has no weights; no prepare() repack.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3. Q stays the same E4M3 bytes (now consumed natively, not widened). K and V stay FP8, accumulation is fp32 and the output is BF16. Nothing is coarser than E4M3.
- [x] Attention sees every key: the tile/item loop, masking (`pos < cseq`) and grid are unchanged, and every token's full 64-dim dot product is computed (2 k-steps x 32 dims). No skipping.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. `load_inline` (already on master) is the only build mechanism. The SASS check via cuobjdump was deliberately NOT added.
- [x] All work happens inside the timed call; no output caching or replay. Only the build and self-check run at import, as on master.
- [x] First line `# okc: problem=muse_attention_decode` kept; no triton.autotune added.

okc_lint output:

```
CLEAN: /home/user/git/musespark/.worktrees/muse_attention_decode/full-decode-cuda-fp8-qk-mma/muse_attention_decode/submission.py
```

`python3 -I -m py_compile`: OK (its `__pycache__` was removed from the worktree).

## Risks / what could fail
medium
Single attributable change on the measured kernel: read kernel A time in top_kernels at decode_16/32 against 111.4/245.5 us. If flat, kernel A is DRAM/TLB-bound on the random 128-byte rows, not issue-bound, and further ALU tuning of the decode kernel should stop (next levers would be bytes in flight: stages / CTAs per SM). Numerics: e4m3 x e4m3 products are exact in fp32 and the f16 path also widened them exactly, so only the tensor-core accumulation order/precision differs (Hopper fp8 MMA accumulates with reduced internal precision; for 64-term logits that is ~1e-4 relative, far inside the 3% bound; the self-check rel L2 should stay ~1.7e-3). PTX m16n8k32 e4m3 needs sm_89+, the build already uses -arch=sm_90a. Overlaps in source with full-verify-cuda-streamk-splitkv (which adds new kernels to the same CUDA string and uses the same fp8 QK op in its own kernel); whichever lands second rebases with a textual merge, shapes are disjoint.

## Result (after the run)
- job `b939681d3ab549d7` (submit), status done, passed True, tests 16/16
- speedup 1.3439x, 95% CI [1.3237, 1.3641], score 2802.516509201515 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3948738389144513, "ci": [1.3739507313307344, 1.415796946498168], "job": "698d4cd3838c4a63"}; base result: muse_attention_decode/hypotheses/done/gain/full-decode-cuda-gather-splitkv.result.json
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/full-decode-cuda-fp8-qk-mma.result.json`, `muse_attention_decode/hypotheses/done/neutral/full-decode-cuda-fp8-qk-mma.patch`; branch `hyp/muse_attention_decode/full-decode-cuda-fp8-qk-mma`
