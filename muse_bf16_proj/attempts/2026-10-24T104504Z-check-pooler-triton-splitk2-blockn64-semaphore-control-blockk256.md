# muse_bf16_proj · 2026-10-24T104504Z · check · pooler-triton-splitk2-blockn64-semaphore-control-blockk256

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T075347Z-pooler-triton-splitk2-blockn64-semaphore-control-blockk256.md` (branch `hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore-control-blockk256`).

- **Job:** `75d595a29a4c4b57` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `af3b6a1`

## Goal
From the parent's branch (pooler-triton-splitk2-blockn64-semaphore), change exactly one constant: the pooler split-K kernel's BLOCK_K from 128 to 256, so each CTA's K-slice of 4096 is 16 K-steps instead of 32. Keep num_stages=4, num_warps=4, BLOCK_N=64, SPLIT_K=2, grid (64, 2) = 128 CTAs, BLOCK_M=16 for m<=16 and 32 for 17..32. Keep the direct transposed weight addressing, the fp32 workspace, the barrier + acq_rel ticket + last-CTA fixed-order reduce and atomic_xchg reset, the routing (pooler m<=32 only), and the warm-up plus re-zero in prepare(). Smem becomes 4 x (64x256 + BLOCK_M x256) x 2 B = 160 KB at BLOCK_M=16 and 192 KB at BLOCK_M=32. This is exactly the unpooler-full-triton-skinny-m32 per-CTA program, and it fits H200's 227 KB. If Triton rejects 192 KB at BLOCK_M=32, use num_stages=3 for the BLOCK_M=32 specialization only and record that.

## What changed, and where
Worktree: `.worktrees/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore-control-blockk256` (base cee590d). Diff stat: `muse_bf16_proj/submission.py | 39 ++++++++++++++++++++++++++++++++-------` (32 insertions, 7 deletions).

- **The change:** `POOLER_BLOCK_K = 128` became `256`. Each CTA now covers its 4096-wide K-slice in 16 steps. `num_stages=4`, `num_warps=4`, `BLOCK_N=64`, `SPLIT_K=2`, grid (64, 2), the `BLOCK_M` 16/32 split, the direct `[BLOCK_K, BLOCK_N]` transposed weight addressing, the fp32 workspace, the barrier + acq_rel ticket + fixed-order last-CTA reduce + atomic_xchg reset, the routing (pooler m<=32 only) and the prepare() warm-up + re-zero are all unchanged. The kernel body is byte-identical.
- **num_stages:** kept at 4 for both `BLOCK_M` specializations, giving 192 KB of smem at `BLOCK_M=32`. Nothing can be compiled on this node. The unpooler kernel compiled and ran with the same 4 x (64x256 + 32x256) x 2 B footprint, so no fallback was added. If the check run fails with an out-of-resources / shared-memory error at m in 17..32, apply the spec's fallback (`num_stages=3` for `BLOCK_M=32` only) as a follow-up and read decode_32 separately (96 KB in flight, two constants changed).
- **Diagnostics (review open question 2):** `_pooler_forward` takes a `diag=False` keyword. With `diag=True` it also returns the Triton launch handle. Only prepare() passes `diag=True`. The timed path in kernel_fn is unchanged and returns only `out`. A new helper `_pooler_compile_diag(m, handle)` runs after the m=16 and m=32 warm-ups. It prints one `[pooler diag]` line per `BLOCK_M` specialization: n_regs, n_spills, metadata.shared, and PTX line counts for ldmatrix, ldmatrix with `.trans`, cp.async.cg and cp.async. It is wrapped in try/except so a Triton-version API difference cannot fail prepare(). These lines will appear in the run log (`profiles/submission.<job>.log`). Compare them with the unpooler kernel's lowering to settle whether the `[BLOCK_K, BLOCK_N]` addressing lowers the same way as `[BLOCK_N, BLOCK_K]` + `tl.trans`.
- The module docstring's pooler paragraph now records BLOCK_K=256 and the smem budget.
- **Deviation:** the only addition beyond the one constant is the prepare()-only diagnostic print, which the adversarial review asked for. It adds no work to the timed call.

## Rules check (each must be true before the run)
Checked against `git -C <worktree> diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints, and no allocation or control flow on input values. The diff touches kernel_fn's path only through a constant. The `diag` branch is a Python bool that only prepare() sets. The prints run only in prepare() and print compile metadata, not tensors.
- [x] Nothing derived from `data` is cached across calls. The launch handle and diagnostics come from prepare()'s synthetic warm-up inputs and are not stored in state.
- [x] Weights: repacked losslessly in `prepare()` only, and BF16 stays BF16. Weight handling is unchanged by the diff.
- [x] Activations BF16 where SGLang keeps BF16, with fp32 accumulation and a bf16 output, as before. n/a for the KV-cache and query parts: this problem has no attention.
- [x] n/a: attention sees every key. This problem has no attention, and the GEMM still reads all of K (16 steps x 256 x 2 slices = 8192).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff adds only `print` in prepare().
- [x] All work happens inside the timed call, with no output caching or replay. The workspace is still fully rewritten for rows < m on every call.
- [x] No `triton.autotune`. The first-line marker `# okc: problem=muse_bf16_proj` is intact.

okc_lint output:

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore-control-blockk256/muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` passed. The `__pycache__` it left was removed.

## Risks / what could fail
low
Separates stage depth (64 KB vs 128 KB of weight in flight per CTA, the only measured difference between the 3.3 and 3.73 TB/s Triton configs) from the parent's own design (split-K 2 + BLOCK_N=64 + semaphore reduce). Parent about 20.3 us and control about 18.5 us: the parent's spec, not its design, was at fault, and the control is the one to promote. Both about 20.3 us: split-K/semaphore costs as much as it saves. Both about 18.5 us: BLOCK_N=64 + split-K is what matters.

## Result (after the run)
- job `75d595a29a4c4b57` (check), status done, passed True, tests 9/9
- speedup 1.1620x, 95% CI [1.1446, 1.1795], score 4157.2306484072415 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/neutral/pooler-triton-splitk2-blockn64-semaphore.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/pooler-triton-splitk2-blockn64-semaphore-control-blockk256.result.json`, `muse_bf16_proj/hypotheses/done/neutral/pooler-triton-splitk2-blockn64-semaphore-control-blockk256.patch`; branch `hyp/muse_bf16_proj/pooler-triton-splitk2-blockn64-semaphore-control-blockk256`
