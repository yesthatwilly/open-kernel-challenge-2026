# muse_moe_w4a16 · 2026-10-24T115932Z · submit · post-expert-norm-triton-small-m

Hypothesis: `muse_moe_w4a16/hypotheses/done/gain/2026-10-24T075126Z-post-expert-norm-triton-small-m.md` (branch `hyp/muse_moe_w4a16/post-expert-norm-triton-small-m`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `75f21a8`

## Goal
Promote check-proven gain post-expert-norm-triton-small-m (1.1297x [1.1128,1.1467], check job bacd1f502f4e46fa) to the board.

For num_tokens < 32 only (bs=1, 8, 16; the bs=4 test too), replace the call to SGLang's fused_post_expert_norm_weighted_sum at the end of kernel_fn with one launch of a new Triton kernel _post_norm_combine_kernel defined in submission.py; bs>=32 keeps SGLang's call untouched as the in-run control. Evidence for the target: in master's profile (job fb2cc736613747df) SGLang's Triton _post_expert_norm_weighted_sum_kernel is in the top kernels at bs=32/128/256 (4.1/6.5/6.1 us) but absent at bs=1/8/16, where instead at::native reduce_kernel<512,1> (5.6/8.9/8.9 us), elementwise_kernel<128,2> (4.9/6.5/8.6 us) and unrolled_elementwise_kernel (5.4/5.8/6.1 us) appear, i.e. ~16 us at bs=1 (20% of decode_1) and ~21-24 us at bs=8/16; they are also unchanged in align-single-token-ep's bs=1 profile, so they do not come from the align. The reading is that the SGLang wrapper takes an eager-PyTorch path (float cast, broadcast mul by norm weight, pow/mean reduce, rsqrt, weighted slot sum, bf16 cast) at small M. Kernel: grid (num_tokens,), num_warps=8, BLOCK_H=4096 (= H, constexpr). Program t zeroes an fp32 acc[BLOCK_H]; loads nw = 1 + post_norm_w (bf16 original kept in state as state['post_norm_w'], cast to fp32 in-kernel, so no new-precision copy of a weight is stored); loops s over 8 slots (tl.static_range): eid = load(local_ids[t*8+s]); g = load(topk_weights[t*8+s]); a scalar "if eid >= 0:" branch (remote slots are skipped entirely, never loaded, because cache3 rows for remote slots are uninitialised torch.empty memory and 0*NaN would poison the sum) loads y[t, s, :] (the w2 Marlin output viewed [M, 8, H]) as fp32, v = y * nw, r = rsqrt(sum(v*v)/H + 1e-8), acc += g * v * r. Store acc as bf16 into a fresh torch.empty((M, H), bf16) output; tokens with no local slot get zeros as the reference requires. This matches reference.post_expert_norm exactly (scale by 1+w first, then normalise, then weight) with fp32 math. No atomics, no memset, one launch. Everything else in kernel_fn (align, both Marlin calls, silu_and_mul, block_size_m pick) is unchanged.

## What changed, and where
Worktree `.worktrees/muse_moe_w4a16/post-expert-norm-triton-small-m`, file `muse_moe_w4a16/submission.py` only (+37 lines, 0 removed). Not run: there is no GPU, torch or Triton on this node.

- Imports: `import triton`, `import triton.language as tl`.
- New module constant `SMALL_M_POST_NORM = 32` and a new `@triton.jit _post_norm_combine_kernel(y_ptr, ids_ptr, tw_ptr, w_ptr, out_ptr, H, eps, TOP_K_C: constexpr, BLOCK_H: constexpr)`. One program per token. It loads `nw = 1 + post_norm_w` (the bf16 weight cast to fp32 in-kernel) and zeroes an fp32 `acc[BLOCK_H]`. It then loops over the 8 slots with `tl.static_range`: it loads the local id; under a scalar `if eid >= 0:` it loads g and the y row `t*8+s` as fp32 (int64 row offset), computes `v = y*nw` and `r = rsqrt(sum(v*v)/H + eps)`, and adds `acc += g*v*r`. Remote slots are never loaded. The kernel always stores `acc` as bf16, so a token with no local slot is written as an explicit zero row (this answers the review's third open question).
- `prepare()`: adds `state["post_norm_w"] = weights["post_norm_w"].contiguous()`, the original BF16 tensor (same dtype, no new-precision copy). `state["norm_w"]` stays for the SGLang path.
- `kernel_fn`: after the w2 Marlin call, the new branch runs `if num_tokens < 32` (a shape branch). It allocates `out = torch.empty((M, H), bf16)` and makes one launch with grid `(M,)`, `num_warps=8`, `BLOCK_H=triton.next_power_of_2(size_k)` (= 4096), and returns `out`. bs>=32 still returns SGLang's `fused_post_expert_norm_weighted_sum` unchanged, as the control. Align, both Marlin calls, silu_and_mul and the block_size_m pick are untouched.
- Deviations from the spec, both minor: (1) `BLOCK_H` comes from `next_power_of_2(size_k)` with an `offs < H` mask instead of a hard-coded 4096. It equals 4096 at H=4096 (the mask is all-true and the mean still divides by the real H) and stays safe if H were ever not a power of two. (2) `y.contiguous()` is a no-op on the Marlin output (cache3 is contiguous). It guards the flat `row*H` addressing.
- No triton.autotune, no host syncs, no atomics, no memset, nothing cached from `data`.
- Still open, per the review: whether reduce_kernel<512,1>, reduce_kernel<128,4> and elementwise_kernel<128,2> disappear from the bs=1/8/16 profile. Record the full kernel list (including the "N more") in the attempt file.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/post-expert-norm-triton-small-m diff`:

- [x] CUDA-graph capturable: the diff adds no `.item()`/`.cpu()`/`synchronize()`/tensor prints. The only new host control flow is `num_tokens < 32`, a shape branch. The per-slot `if eid >= 0` runs on the device inside the Triton kernel. The `torch.empty` output has a fixed shape (M, H).
- [x] Nothing derived from `data` is cached across calls: `out` is allocated fresh on every call, and `state` gains only the weight tensor, set in `prepare()`.
- [x] Weights: `state["post_norm_w"]` is the original BF16 tensor, unchanged. The 1+w is added in fp32 registers inside the kernel. No NVFP4 weight is touched.
- [x] Activations BF16 where SGLang keeps BF16: y is read as the bf16 Marlin output, the math is fp32 (higher, not lower) and the output is bf16 as required. KV cache / FP8 queries: n/a, this is not an attention problem.
- [x] n/a: no attention in this problem, so no key skipping is possible. The kernel reads every local slot's full H row; remote slots are not this GPU's work (the reference skips them too).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds only triton imports and the kernel.
- [x] All work happens inside the timed call: the norm and combine run in kernel_fn on every call, with no output caching or replay.
- [x] First-line marker `# okc: problem=muse_moe_w4a16` kept; no `triton.autotune`.

okc_lint output:

```
warn  host-sync    L138: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
```

(The warning is the existing `torch.cuda.synchronize()` in `prepare()`, untimed, which this change does not touch.) `python3 -I -m py_compile` passed, and the `__pycache__` it left was removed.

## Risks / what could fail
low

## Result (after the run)
(pending)
