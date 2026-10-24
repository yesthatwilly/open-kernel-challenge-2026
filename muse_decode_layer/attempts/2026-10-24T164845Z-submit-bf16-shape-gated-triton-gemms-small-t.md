# muse_decode_layer · 2026-10-24T164845Z · submit · bf16-shape-gated-triton-gemms-small-t

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T140116Z-bf16-shape-gated-triton-gemms-small-t.md` (branch `hyp/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t`).

- **Job:** `6076ae1f250143aa` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `060c5bd`

## Goal
Port muse_bf16_proj's landed bf16-shape-gated-triton-winners-combo (1.176x vs 1.151x there; code is muse_bf16_proj/submission.py on master: _bf16_splitk_kernel, _pooler_forward, _qkv_splitk_sem_kernel/_qkv_forward, _unpool_forward, _o_skinny_gemm and their constants POOLER_* (BLOCK_K 256, SPLIT_K 2), QKV_* (SPLIT_K 4), UNPOOL_* (SPLIT_K 8), O_* (full K, 128 programs)) to muse_decode_layer's own dense GEMMs, which have exactly bf16_proj's shapes: qkv_gate [T,8192]x[2304,8192]^T, o [T,1024]x[8192,1024]^T, pooler [T,8192]x[4096,8192]^T, unpooler slice [T,4096]x[1024,4096]^T (rows 0-1023, row stride 4096). Copy the kernels and launch helpers verbatim; the area is all four dense-GEMM call sites in kernel_fn (labelled pooler GEMM, the largest of them). Routing, by tokens and weight shape only (no data-dependent branch), mirroring the combo's gates: qkv = _qkv_forward at 1 <= T <= 16, else _gemm; o = _o_skinny_gemm at T <= 16, else _gemm; pooler (main stream, still overlapping the side-stream router) = _pooler_forward at T <= 16, else _gemm; unpooler slice = _unpool_forward on dense['unpooler'][:UNPOOL_SLICE] at T <= 8 (followed by the existing F.pad all-gather), else unchanged. T >= 32 (decode_32, verify_16, verify_32) keeps _bf16_gemm_dispatch_impl everywhere as the control. Each path keeps its own fp32 split-K workspace and int32 ticket counter (pooler ws [2,16,4096] cnt [64]; qkv ws [4,16,2304] cnt [36]; unpooler ws [8,16,1024] cnt [16]), allocated in prepare(); prepare() warms every routed specialization on synthetic bf16 inputs (T 1..16 for qkv/o/pooler, 1..8 for the unpooler slice), synchronises and zero_()s every cnt, as the combo's prepare() does. The router (side stream, hi/lo torch.mm) and everything else are untouched.

## What changed, and where
Worktree `.worktrees/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t`, `muse_decode_layer/submission.py` only (uncommitted; +377/-4).

- **Kernels copied verbatim** from `muse_bf16_proj/submission.py` (master): `_bf16_splitk_kernel`, `_qkv_splitk_sem_kernel` and `_pooler_skinny_kernel`. Their constants are unchanged: POOLER_* (BLOCK_N 64, BLOCK_K 256, SPLIT_K 2, 4 warps, 4 stages), QKV_* (BLOCK_N 64, BLOCK_K 128, SPLIT_K 4, 4 stages), UNPOOL_* (BLOCK_N 64, BLOCK_K 128, SPLIT_K 8, 3 stages) and O_* (BLOCK_N 64, BLOCK_K 256, full K, 128 programs, 4 stages). The launch helpers `_pooler_forward`, `_qkv_forward`, `_unpool_forward` and `_o_skinny_gemm` are the combo's, with one change: BLOCK_M is the literal 16, because only m <= 16 is ever routed. bf16_proj's `16 if m <= 16 else 32` therefore always picked 16. The m = 32 swap-AB code (`_bf16_swapab_kernel`, `_swapab_*`) and the router kernel were **not** ported, as the review requires.
- **Guards rewritten (review REQUIRED).** decode_layer's Config has no m/n/k keys, so the guards (`_bf16_x_ok`, `_bf16_w_ok`, then `_pooler_shape_ok(x, w)`, `_qkv_shape_ok(x, w)`, `_unpool_shape_ok(x, w)` and `_o_shape_ok(x, w)`) read only tensor shapes, dtype and inner stride. x must be bf16 2-D with stride(1) == 1 and 1 <= T <= 16 at the right K. w must be the exact [N, K] row-major weight; for the unpooler it is the leading >= 1024 rows of a [*, 4096] row-major weight.
- **Routing in kernel_fn** uses token count only: `small_t = 1 <= tokens <= SMALL_T_TRITON_GEMM (16)`.
  - qkv_gate: `_qkv_forward`
  - o: `_o_skinny_gemm`
  - pooler: `_pooler_forward` on the main stream. It is still launched after `side_stream.wait_stream(main_stream)`, and the side-stream router block is unchanged.
  - unpooler slice: `_unpool_forward(expert_output, dense["unpooler"][:UNPOOL_SLICE])` inside the existing `tokens <= SLICED_UNPOOL_MAX_TOKENS` branch, followed by the unchanged `F.pad` all-gather.
  - T >= 17 (decode_32, verify_16, verify_32 and the T = 24/33/40 tests) never enters these paths and keeps `_gemm`.
- **Deviation: a logged fallback instead of a silent one or an assert.** The review's open question asked whether attn_input, the o_norm_gate output and ffn_input are contiguous at every tested T. I can't run anything here, so I took a middle path that favours correctness on a scored submit. If a routed T's tensor fails its precondition, `_fallback()` prints one line per (layer, shape, stride, dtype) and the call then takes `_gemm`. It keys only on shapes and strides, never on values, and the print is a Python string, not a tensor, so it is safe under capture. **Read the run log for `[small-t-gemm]` lines.** Any such line means that GEMM silently stayed on cuBLAS, so its expected gain is absent.
- **prepare()** now calls `_prepare_small_t_gemms(state)` before the final synchronize. It allocates `state["small_t"]`, which holds a separate ws/cnt pair per path: pooler ws [2,16,4096] / cnt [64], qkv ws [4,16,2304] / cnt [36], unpool ws [8,16,1024] / cnt [16]. o has no workspace. It warms qkv, pooler and o at T 1..16 and the unpooler slice at T 1..8 on synthetic randn bf16 inputs against the real weights, then calls `torch.cuda.synchronize()`, `zero_()`s every cnt, and synchronises again. M is `do_not_specialize` and BLOCK_M is fixed, so in practice this is one compile per kernel variant.
- The module docstring got one added bullet under "Differs from SGLang".

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t diff`:

- [x] CUDA-graph capturable. kernel_fn has no `.item()`, `.cpu()`, synchronize or tensor print. The only additions are `torch.empty` outputs, the same kind of allocation `_gemm` and the rest of kernel_fn already make. Branches read `tokens` (config), tensor shapes, strides and dtype; none reads a value. The ticket counters self-reset in-kernel, so graph replays start from 0. The synchronize calls are in prepare() only.
- [x] Nothing derived from `data` is cached across calls. The ws/cnt buffers are allocated in prepare(), every partial a reducer reads is written in the same launch (masked rows), and the outputs are fresh. `_FALLBACK_SEEN` stores only (layer, shape, stride, dtype) tuples for a one-time log line, never values or tensors.
- [x] Weights. The bf16 weights are used as stored (`dense[...]`, contiguous, unchanged); the unpooler slice is a view read in place. The NVFP4 experts and the router hi/lo split are untouched.
- [x] Activations stay BF16 where SGLang keeps BF16. Each kernel is bf16 x bf16 with fp32 accumulation and stores bf16, like cuBLAS. The split-K partials are fp32 and are summed in a fixed order. The KV cache and FP8 query path are untouched.
- [x] n/a: attention is untouched. FA3 still sees every key, with the same window and context.
- [x] No timing patches, harness frames or I/O, module edits, reference import, subprocesses or network. The only new import-level code is `@triton.jit` functions and constants.
- [x] All work happens inside the timed call. prepare() runs only warm-ups on synthetic inputs and discards their outputs. There is no output caching or replay.
- [x] The first-line marker `# okc: problem=muse_decode_layer` is kept, and no `triton.autotune` is used.
- [x] Only the named area changed: the four dense-GEMM call sites and their support code. The router side stream, routing, experts and norms are unchanged in the diff.

okc_lint output:

```
warn  host-sync    L580: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L584: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
warn  host-sync    L627: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: .worktrees/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t/muse_decode_layer/submission.py
```

`python3 -I -m py_compile` passed, and the `__pycache__` it left was removed.

## Risks / what could fail
medium
Expected ~+1% geomean: modest but the four kernels are measured winners at these exact shapes, and each replaces an nvjet+splitKreduce pair (two graph nodes) with one launch. Risks: (1) in this layer the pooler overlaps the side-stream router; a 128-CTA Triton pooler may contend with the router's GEMMs differently than nvjet does; read the pooler's own top_kernels time against the bf16 run. (2) The judge uses a cold L2 before each replay here too, as in bf16_proj, so per-kernel times should carry over. (3) The ticket-semaphore reduce is shared-pattern code that passed in bf16_proj; counters are per path, never shared. If the headline is flat, read the per-GEMM rows (nvjet 128x8/64x8 splitK and splitKreduce rows should disappear at T<=16).

## Result (after the run)
- job `6076ae1f250143aa` (submit), status done, passed True, tests 13/13
- speedup 1.1688x, 95% CI [1.1513, 1.1863], score 19353.402702518822 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.1664628806987323, "ci": [1.1489659374882513, 1.183959823909213], "job": "a017e10aec544c55"}; base result: muse_decode_layer/hypotheses/done/gain/post-expert-norm-triton-small-t.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_decode_layer/hypotheses/done/neutral/bf16-shape-gated-triton-gemms-small-t.result.json`, `muse_decode_layer/hypotheses/done/neutral/bf16-shape-gated-triton-gemms-small-t.patch`; branch `hyp/muse_decode_layer/bf16-shape-gated-triton-gemms-small-t`
