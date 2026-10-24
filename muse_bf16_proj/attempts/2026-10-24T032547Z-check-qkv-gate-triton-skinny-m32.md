# muse_bf16_proj · 2026-10-24T032547Z · check · qkv-gate-triton-skinny-m32

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-23T225317Z-qkv-gate-triton-skinny-m32.md` (branch `hyp/muse_bf16_proj/qkv-gate-triton-skinny-m32`).

- **Job:** `abcd45776f244c8c` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `2a69a2d`

## Goal
Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'qkv_gate' at m <= 32 (n=2304, k=8192, a 37.7 MB weight): in kernel_fn route qkv_gate at m <= 32 to the same kernel with BLOCK_N=16 (2304/16 = 144 programs, the nearest power-of-two tile that keeps about one wave) and the same BLOCK_K=256, num_stages=4, BLOCK_M from m, fp32 accumulate, bf16 store into torch.empty((m, 2304), bf16). The pooler path and everything else stay as the parent left them; qkv_gate at m >= 128 keeps cuBLAS. Kernels moved: nvjet_sm90_tst_128x8/64x16/64x32_..._splitK_TNT (7.8-9.6 us) + cublasLt::splitKreduce_kernel (1.6-1.8 us) become one launch on the four m <= 32 qkv_gate shapes; reported 14.1-14.5 us against a 7.9 us floor, so the splitK main kernel is at 3.9 TB/s and the reduce + gap is ~3 us of the total.

## What changed, and where
Worktree `.worktrees/muse_bf16_proj/qkv-gate-triton-skinny-m32`, file `muse_bf16_proj/submission.py` only (diff: 56 insertions, 12 deletions), on top of the parent impl d2840d0.

- The parent's `_pooler_skinny_kernel` (unchanged, still `do_not_specialize=["M"]`) is now launched through a generic `_skinny_gemm(x, w, n, k, block_n, block_k, num_warps, num_stages)`; `_pooler_skinny_gemm` calls it with exactly the parent's constants (BLOCK_N 32, BLOCK_K 256, 4 warps, 4 stages, grid 128), so the pooler launch is byte-for-byte the same configuration.
- New constants `QKV_SKINNY_MAX_M=32, QKV_N=2304, QKV_K=8192, QKV_BLOCK_N=16, QKV_BLOCK_K=256, QKV_NUM_WARPS=4, QKV_NUM_STAGES=4` and `_qkv_skinny_gemm` (grid 2304/16 = 144, BLOCK_M 16 for m<=16 else 32, fp32 accumulate, bf16 store into `torch.empty((m, 2304), bf16)`).
- Guard `_qkv_skinny_applies` shares `_skinny_applies` with the pooler: config and shape only (cfg m <= 32, n == 2304, k == 8192, x 2-D [m, k], x.stride(1) == 1, bf16). m=128/256 verify loads and the m=512 outlier test stay on `_bf16_gemm_dispatch_impl`.
- `kernel_fn`: after the pooler branch, `layer == "qkv_gate" and _qkv_skinny_applies(...)` -> `_qkv_skinny_gemm(x, state["qkv_gate"])`.
- `prepare()`: `_qkv_skinny_warmup` runs the qkv path for every m in 1..32 on synthetic zeros (both BLOCK_M specializations compiled; outputs dropped), guarded on the weight shape, mirroring the parent's pooler warm-up which is kept as is.
- Deviations: none from the change paragraph. m=32 is routed as specified (not dropped), per the review note; the reviewer expects decode_32 qkv_gate to be flat-or-slower (parent's BLOCK_M=32 ran at 2.7 TB/s). BLOCK_N=16 as specified (BLOCK_N=8 is invalid for tl.dot); the follow-on retune axis if this is flat/slow is num_stages=8 at BLOCK_N=16 (restores 64 KB/SM in flight) or BLOCK_N=32 (72 programs). Record achieved TB/s = 37.7 MB / Triton kernel us from the profile, and reported-minus-kernel gap.
- Not run: no GPU/Triton on this node; only lint and py_compile.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/qkv-gate-triton-skinny-m32 diff`.

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints added; the only new branch is on `cfg["layer"]`, cfg m/n/k and x shape/stride/dtype; the only allocation is the fixed-shape output `torch.empty((m, 2304))`, same as the parent pattern.
- [x] Nothing derived from `data` is cached across calls: state gains no new entries; the warm-up uses synthetic zeros in prepare().
- [x] Weights: unchanged, used as stored (bf16 [2304, 8192] contiguous, no repack); router hi+lo split untouched.
- [x] Activations BF16 where SGLang keeps BF16: x read as bf16, bf16 x bf16 tl.dot with fp32 accumulate, single bf16 rounding at store.
- [x] n/a: Attention sees every key: this problem has no attention; GEMM computes the full K for every output column, nothing skipped.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds only constants, Python helpers and a kernel launch.
- [x] All work happens inside the timed call; no output caching or replay: warm-up outputs are discarded, each kernel_fn call launches the kernel on its own inputs.
- [x] No triton.autotune; first-line marker `# okc: problem=muse_bf16_proj` intact; `do_not_specialize=["M"]` kept.

Lint (`okc_lint.py <worktree>/muse_bf16_proj/submission.py --problem muse_bf16_proj`):

```
CLEAN: submission.py
```

`python3 -I -m py_compile`: OK; the `__pycache__` it left was removed.

## Risks / what could fail
medium
Only worth running if the parent is faster on pooler; it reuses the parent's kernel with one tile change so a loss here attributes to the 144-program/BLOCK_N=16 tiling of this layer, not to the kernel. 144 programs is 1.09 waves, so a tail of 12 programs runs alone; BLOCK_N=8 (288 programs, 2.2 waves) is the ab-tune if this is flat. Tests touched: m=1 qkv_gate and m=512 qkv_gate outliers (the latter stays on cuBLAS). Same correctness and graph properties as the parent.

## Result (after the run)
- job `abcd45776f244c8c` (check), status done, passed True, tests 9/9
- speedup 0.9238x, 95% CI [0.9100, 0.9377], score 5229.064394393541 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1510295903072447, "ci": [1.133764146452636, 1.1682950341618532], "job": "bd9fa14151644cee"}; base result: muse_bf16_proj/hypotheses/done/neutral/pooler-triton-skinny-m32.result.json
- **outcome: regression** (regression)
- files: `muse_bf16_proj/hypotheses/done/regression/qkv-gate-triton-skinny-m32.result.json`, `muse_bf16_proj/hypotheses/done/regression/qkv-gate-triton-skinny-m32.patch`; branch `hyp/muse_bf16_proj/qkv-gate-triton-skinny-m32`
