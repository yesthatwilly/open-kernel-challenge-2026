# muse_bf16_proj · 2026-10-24T032616Z · check · unpooler-full-triton-skinny-m32

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-23T225303Z-unpooler-full-triton-skinny-m32.md` (branch `hyp/muse_bf16_proj/unpooler-full-triton-skinny-m32`).

- **Job:** `635c2693bd4147a4` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `fbfb795`

## Goal
Extend the skinny Triton GEMM from pooler-triton-skinny-m32 to layer == 'unpooler' at m in {16, 32} (the full weight: n=8192, k=4096, 67 MB): in kernel_fn route unpooler at 8 < m <= 32 to the same kernel with BLOCK_N=64 (8192/64 = 128 programs, one wave), BLOCK_K=256, num_stages=4, BLOCK_M from m, fp32 accumulate, bf16 store into torch.empty((m, 8192), bf16). Unpooler at m <= 8 (the 1024-row slice) and at m >= 128 keep whatever they have (cuBLAS, or unpooler-slice-triton-gemv if it landed); pooler keeps the parent's path. Kernel moved: nvjet_sm90_tst_64x16/64x32_64x16_2x1_v_bz_TNT, a single non-splitK kernel at 18.5-18.8 us for 67 MB (3.6 TB/s) against a 14.1 us floor; unlike pooler there is no reduce node to drop, so this run isolates the pure streaming-bandwidth question: does the full-K Triton stream beat nvjet's 2x1 cluster kernel on an N-heavy, K-short (4096) weight.

## What changed, and where
- File: `muse_bf16_proj/submission.py` in the worktree (only file touched; 53 insertions, 2 deletions). Built on d2840d0 (the parent's impl), so the pooler path is byte-identical.
- Reuses the parent's `@triton.jit _pooler_skinny_kernel` unchanged (K, BLOCK_M, BLOCK_N, BLOCK_K are constexpr, so the unpooler call is a separate specialization; M stays unspecialized).
- Added `UNPOOLER_*` constants: N=8192, K=4096, BLOCK_N=64 (128 programs, one wave), BLOCK_K=256 (16 K-steps), num_warps=4, num_stages=4; routed m range 9..32.
- Added `_unpooler_skinny_applies(x, cfg)`: true only for 9 <= cfg m <= 32, cfg n == 8192, cfg k == 4096, plus x 2-D bf16, x.shape == (m, k), x.stride(1) == 1. Keys on cfg, never on x.shape[0] alone. Therefore m=1/8 (n=1024 slice), the m=5 test, and m=128/130/256/1000 stay on `_bf16_gemm_dispatch_impl` (cuBLAS).
- Added `_unpooler_skinny_gemm(x, w)`: BLOCK_M = 16 if m <= 16 else 32 (BLOCK_M=32 at m=32, as specified; the BLOCK_M=16-everywhere control is the queued companion and is NOT folded in), fp32 accumulate in the kernel, bf16 store into `torch.empty((m, 8192), bf16)`.
- `prepare()`: sets `state["unpooler_skinny"]` from the weight's shape (weights-derived, not data-derived) and, when true, `_unpooler_skinny_warmup` runs every m in 9..32 on synthetic zeros so both specializations compile before graph capture and any compile/launch failure surfaces in prepare().
- `kernel_fn`: one new branch after the pooler branch: `layer == "unpooler" and state["unpooler_skinny"] and _unpooler_skinny_applies(x, cfg)`.
- Smem risk: at BLOCK_M=32, 4 stages x (64x256 + 32x256) x 2 B = 192 KB per program (160 KB at BLOCK_M=16), near the H200's 227 KB per-block limit. If Triton's pipeliner needs more than that (e.g. extra layout-conversion buffers), the BLOCK_M=32 specialization raises OutOfResources during warmup in prepare() (the run fails visibly rather than at capture) — check the run log. num_stages=6 is not possible at this tile (288 KB). No fallback was added, to keep the run attributable.
- Open question from review (what BLOCK_N=64 / 16 K-steps changes vs the parent's 20.2/25.2 us): the one real difference is 2x the weight bytes in flight per SM (128 KB of weight per program across 4 stages vs 64 KB), which may raise per-SM memory-level parallelism; fewer K-steps make the pipeline prologue/epilogue a larger fraction. The parent's data predict flat-to-loss at m=16 and a loss at m=32; judge per row against the parent's rows (18.71 / 18.63 us) and record kernel us.
- Not run (no GPU/torch/Triton on this node); only py_compile and the lint below.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in the diff; the new branch depends only on cfg, x shape/stride/dtype, and a prepare()-time bool; the only allocation is the fixed-shape `torch.empty((m, 8192))` output, same as the parent's pooler path.
- [x] Nothing derived from `data` is cached across calls: state gains only `unpooler_skinny` (from the weight shape in prepare()); warmup uses synthetic zeros and drops outputs.
- [x] Weights: unpooler used as stored (bf16, contiguous, no repack); router hi+lo split unchanged from parent.
- [x] Activations BF16 where SGLang keeps BF16: x is bf16 into tl.dot, fp32 accumulator, bf16 output. KV cache / FP8 query: n/a: no attention in this problem.
- [x] Attention sees every key: n/a: no attention in this problem; the GEMM reads the full K and every weight row.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds only constants, two helpers, a warmup and one branch.
- [x] All work happens inside the timed call; no output caching or replay: the output is computed fresh each call by the Triton kernel.
- [x] No triton.autotune (fixed config), first-line marker `# okc: problem=muse_bf16_proj` kept.

Lint (`okc_lint.py ... --problem muse_bf16_proj`):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/unpooler-full-triton-skinny-m32/muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` OK; worktree __pycache__ removed.

## Risks / what could fail
medium
Because the cuBLAS kernel here is already a single launch, a gain can only come from bandwidth, which makes this the cleanest test of the streaming kernel itself: if pooler wins but this is flat, the pooler gain was the dropped reduce node, which should go into FINDINGS. K=4096 means only 16 BLOCK_K steps per program; raise num_stages to 6 if the parent's profile shows the pipeline draining. Tests touched: m=130 unpooler outliers (n=8192, m=130 stays on cuBLAS), m=5 unpooler (slice path, untouched). Same correctness and graph properties as the parent.

## Result (after the run)
- job `635c2693bd4147a4` (check), status done, passed True, tests 9/9
- speedup 1.0053x, 95% CI [0.9902, 1.0204], score 4805.354625162871 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1510295903072447, "ci": [1.133764146452636, 1.1682950341618532], "job": "bd9fa14151644cee"}; base result: muse_bf16_proj/hypotheses/done/neutral/pooler-triton-skinny-m32.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/unpooler-full-triton-skinny-m32.result.json`, `muse_bf16_proj/hypotheses/done/neutral/unpooler-full-triton-skinny-m32.patch`; branch `hyp/muse_bf16_proj/unpooler-full-triton-skinny-m32`
