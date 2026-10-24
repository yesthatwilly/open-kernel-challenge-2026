# muse_bf16_proj · 2026-10-24T104413Z · check · pooler-splitk2-tma-weight-stream

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T092648Z-pooler-splitk2-tma-weight-stream.md` (branch `hyp/muse_bf16_proj/pooler-splitk2-tma-weight-stream`).

- **Job:** `4e89311eefdc4bc3` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `954d393`

## Goal
From branch hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore (pooler at m <= 32: _bf16_splitk_kernel, grid (64, 2) = 128 CTAs, BLOCK_N=64, SPLIT_K=2, ticket-semaphore reduce; harvested neutral at 20.2-20.7 us per pooler row, ~3.3 TB/s, the same rate as every plain-tl.load Triton weight stream so far), change only how the weight tile reaches shared memory: load it with a Hopper TMA tensor descriptor instead of per-thread tl.load (cp.async). Build the descriptor once in prepare() from the pooler weight alone (shape [4096, 8192], strides [8192, 1], block [BLOCK_N=64, BLOCK_K=256], bf16): with Triton >= 3.4 use triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(W, [64, 256]) passed as a kernel argument and w = w_desc.load([n0, k]) in the K loop; on older Triton use triton.tools.experimental_descriptor.create_2d_tma_descriptor and tl._experimental_descriptor_load(desc, [n0, k], [64, 256], tl.bfloat16). print(triton.__version__) and the chosen API in prepare(). The tile arrives [BLOCK_N, BLOCK_K] (K-contiguous, as stored); feed acc = tl.dot(x_tile, tl.trans(w), acc) (the transpose is a smem-descriptor/ldmatrix.trans view, not a copy). BLOCK_K 256 (16 K-steps per 4096 K-slice) and num_stages 4: 32 KB of weight per stage, 128 KB of weight in flight per CTA, moved by one elected thread per stage on an mbarrier instead of ~64 cp.async.16 per thread, which leaves the warps free to issue the next stage and the MMA. x stays a plain masked tl.load ([BLOCK_M, 256], L2-resident). Smem 4 x (32 KB + BLOCK_M x 512 B) = 160 KB at BLOCK_M=16, 192 KB at BLOCK_M=32 (drop to num_stages 3 for the BLOCK_M=32 specialization only if it does not fit). Everything else stays as the parent: routing (pooler m <= 32 only), grid, SPLIT_K, the fp32 workspace, barrier + acq_rel ticket + fixed-order last-CTA reduce + atomic_xchg reset, warm-up and counter re-zero in prepare(). No autotune; the descriptor depends only on the weight.

## What changed, and where
Worktree file: `.worktrees/muse_bf16_proj/pooler-splitk2-tma-weight-stream/muse_bf16_proj/submission.py` (base cee590d, the parent pooler-triton-splitk2-blockn64-semaphore). Diff: 1 file, +155/-8. Pooler path only; router, cuBLAS dispatch and routing (pooler m <= 32) untouched.

- New `_bf16_splitk_tma_kernel`: a copy of `_bf16_splitk_kernel` in which only the weight load changed. `w_desc` replaces `w_ptr`/`stride_wn`; per K-step `w_tile = w_desc.load([n0, k])` (constexpr `NEW_DESC_API`) or `tl._experimental_descriptor_load(w_desc, [n0, k], [64, 256], tl.bfloat16)`; `acc = tl.dot(x_tile, tl.trans(w_tile), acc)`. x is still a masked `tl.load` of [BLOCK_M, 256]. Workspace store, debug_barrier, acq_rel gpu ticket, fixed-order last-CTA reduce, bf16 store and atomic_xchg reset are byte-for-byte the parent's. Grid (64, 2), SPLIT_K 2, BLOCK_N 64, num_warps 4, BLOCK_K 256 (`POOLER_TMA_BLOCK_K`), num_stages 4.
- API selection by feature at import: `triton.tools.tensor_descriptor.TensorDescriptor` if importable (the >= 3.4 API, `from_tensor(W, [64, 256])`), else `triton.tools.experimental_descriptor.create_2d_tma_descriptor(W.data_ptr(), 4096, 8192, 64, 256, 2)`. The descriptor is built once in prepare() from the pooler weight alone and kept in `state["pooler_w_desc"]`. prepare() prints `triton.__version__` and the chosen API.
- Deviation 1 (fallback): if neither API imports, `POOLER_TMA_API = None` and the parent's cp.async kernel (BLOCK_K 128) runs unchanged, so the run does not crash. The md says "no run" in that case, but this node has no Triton and the judge's Triton version is recorded nowhere in the repo (grep of attempts/hypotheses/logs found nothing), so it cannot be known before a run. SGLang on H200 needs torch >= 2.5 (Triton >= 3.1), where at least the experimental API exists, so the fallback should never trigger. If the prepare() log says `none (cp.async fallback)`, the run is the parent and must be read as "no TMA API", not as a TMA result.
- Deviation 2 (stages at BLOCK_M=32, per the md's "only if it does not fit"): prepare() compiles BLOCK_M=16 and 32 first; if the BLOCK_M=32 launch raises `triton.runtime.errors.OutOfResources` at 4 stages, it retries at 3 and records that in `state["pooler_stages"]` (a constant from compile resources, not input-derived). The m=32 log line says which.
- Diagnostics (from the review's open questions): prepare() prints per BLOCK_M `n_regs`, `n_spills` and PTX counts of `cp.async.bulk.tensor`, `ldmatrix`, `.trans`, `mbarrier.`, non-bulk `cp.async`, `wgmma.mma_async` and `mma.sync` from the launch handle's `asm['ptx']`. It is wrapped in try/except so it can never fail prepare().
- `_pooler_forward` now returns `(out, handle)`; kernel_fn takes `[0]`. The warm-up over m = 1..32 and the counter re-zero are as in the parent.
- Box [64, 256]: inner extent 512 B, more than the 128 B swizzle span. Triton's host fill for the experimental path clamps the box to 128 B and the codegen issues several copies. The new API's lowering splits it too. Not verifiable offline. The PTX `cp.async.bulk.tensor` count shows how many copies were issued per stage.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/pooler-splitk2-tma-weight-stream diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn. Prints happen only in prepare(), and they print strings and ints, never tensors. The output `torch.empty` is the parent's per-call allocation. The branches are on shapes (`_pooler_shape_ok`, `_pooler_block_m`) and on prepare()-fixed state (descriptor present, stages), never on input values. The TMA loads are device-side and the descriptor is a fixed host param.
- [x] Nothing derived from `data` is cached across calls: the descriptor is built from the pooler weight only. `pooler_stages` comes from compile resources. The workspace is rewritten every call, and the counters self-reset as in the parent.
- [x] Weights: the pooler weight is unchanged bf16. The descriptor is a view of it, not a copy. The router hi+lo split is unchanged.
- [x] Activations BF16 where SGLang keeps BF16: x is bf16, the accumulation is fp32 and the output is bf16, as in the parent.
- [x] n/a: attention. This problem has no attention, KV cache or queries.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The new imports are triton.tools.* only.
- [x] All work happens inside the timed call; no output caching or replay. The warm-up runs on synthetic x in prepare() and its outputs are discarded.
- [x] No triton.autotune. The configs are static constants.
- [x] First line `# okc: problem=muse_bf16_proj` kept.

okc_lint output:

```
CLEAN: submission.py
```

`python3 -I -m py_compile`: OK, and the `__pycache__` it created in the worktree was removed.

## Risks / what could fail
high
High risk: the judge image Triton version is not recorded anywhere in the repo. Both API branches are named; if neither exists the implementer stops and records it (no run). It changes two things relative to the parent (TMA, and BLOCK_K 128 -> 256, which TMA needs for a 32 KB stage); the held pooler-triton-splitk2-blockn64-semaphore-control-blockk256 is the BLOCK_K-only companion if the reviewer wants the split. Device-side TMA with mbarriers is graph-capturable; the host descriptor is built once from the weight pointer (not input-derived). Readout: the kernel us per pooler row against the parent 20.2/19.9/20.7/22.2 us; a stream above 4 TB/s would carry over to qkv_gate and unpooler at m <= 32.

## Result (after the run)
- job `4e89311eefdc4bc3` (check), status done, passed True, tests 9/9
- speedup 1.1693x, 95% CI [1.1517, 1.1868], score 4131.470128729046 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/neutral/pooler-triton-splitk2-blockn64-semaphore.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/pooler-splitk2-tma-weight-stream.result.json`, `muse_bf16_proj/hypotheses/done/neutral/pooler-splitk2-tma-weight-stream.patch`; branch `hyp/muse_bf16_proj/pooler-splitk2-tma-weight-stream`
