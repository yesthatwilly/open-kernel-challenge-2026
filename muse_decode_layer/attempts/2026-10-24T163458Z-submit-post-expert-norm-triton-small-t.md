# muse_decode_layer · 2026-10-24T163458Z · submit · post-expert-norm-triton-small-t

Hypothesis: `muse_decode_layer/hypotheses/done/gain/2026-10-24T140057Z-post-expert-norm-triton-small-t.md` (branch `hyp/muse_decode_layer/post-expert-norm-triton-small-t`).

- **Job:** `a017e10aec544c55` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `85caaf9`

## Goal
Promote check-proven gain post-expert-norm-triton-small-t (1.1635x [1.1461,1.1810], check job 9dcaec316b3045a7) to the board.

Port muse_moe_w4a16's landed post-expert-norm-triton-small-m (1.130x there, job bacd1f502f4e46fa; patch muse_moe_w4a16/hypotheses/done/gain/post-expert-norm-triton-small-m.patch) into muse_decode_layer/submission.py _experts. Today _experts ends with SGLang's fused_post_expert_norm_weighted_sum(y, topk_weights, norm_weight=state['post_expert_w'], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids). Master's profile (fb07a3bd3aa54baf) shows the same signature as moe's: SGLang's Triton _post_expert_norm_weighted_sum_kernel is in the top kernels only at T>=32 (3.6 us at decode_32, 5.7-5.8 us at verify_16/32), while at T=1/8/16 it is absent and at::native reduce_kernel<512,1> (5.3-5.5 us at T=1, 8.9-9.0 us at T=8/16), elementwise_kernel<128,2> x3 (4.9 / 6.6-6.7 / 8.6 us) and unrolled_elementwise_kernel x2 (5.7-6.6 us) appear instead: the wrapper's eager-PyTorch path at small M. Change: add the verbatim _post_norm_combine_kernel from that patch (grid (T,), num_warps=8, BLOCK_H=next_power_of_2(4096), fp32 acc, static_range over the 8 slots, scalar 'if eid >= 0' so remote slots' uninitialised cache3 rows are never loaded, nw = 1 + w added in fp32 in-kernel, explicit zero row for tokens with no local slot, bf16 store into a fresh torch.empty((T, 4096), bf16)), a constant SMALL_T_POST_NORM = 32, and in prepare() state['post_expert_norm_w'] = weights['post_expert_norm'].contiguous() (the original BF16 weight, no new-precision copy; post_expert_w stays for the SGLang path). In _experts, after the w2 Marlin call: if num_tokens < 32 launch the kernel on y (the w2 output viewed [T, 8, 4096], .contiguous()), local_ids (bank['expert_map'][topk_ids], int32, -1 for remote), topk_weights (fp32) and return its output; else keep SGLang's call unchanged as the in-run control. Shape branch only. Everything else (local_ids arithmetic, moe_align_block_size, block_size_m pick with cap 16, both Marlin calls, silu_and_mul, pre-expert norm, unpooler, boundaries, attention) is untouched.

## What changed, and where
Worktree `.worktrees/muse_decode_layer/post-expert-norm-triton-small-t`, `muse_decode_layer/submission.py` only (+37 lines, 0 removed):

- Imports: `import triton` and `import triton.language as tl`, after `torch.nn.functional`.
- Constant `SMALL_T_POST_NORM = 32`, after `MARLIN_BLOCK_SIZE_M_CAP`.
- `_post_norm_combine_kernel` is copied verbatim from muse_moe_w4a16 post-expert-norm-triton-small-m and sits between `_marlin_bank` and `prepare`. Its math matches `reference.post_expert_norm`:
  - `y.float() * (1 + w.float())`
  - `rsqrt(mean(v^2) + 1e-8)`
  - weighted by the fp32 topk_weights and summed over local slots
  
  A scalar branch skips remote slots (`eid < 0`), so they are never loaded. A token with no local slot stores the zero `acc`.
- `prepare()`: `state["post_expert_norm_w"] = weights["post_expert_norm"].contiguous()`. This is the original BF16 [4096] weight, not a copy at a new precision. `post_expert_w` stays for the SGLang path.
- `_experts()`, after the w2 Marlin call: `if num_tokens < SMALL_T_POST_NORM`, the code calls `y.contiguous()` and `torch.empty((T, 4096), bf16)`. It then launches grid `(T,)` with `TOP_K_C=8`, `BLOCK_H=next_power_of_2(4096)=4096` and `num_warps=8`, and returns the result. The inputs are `local_ids` (`bank["expert_map"][topk_ids]`, int32, -1 for remote slots) and the fp32 contiguous `topk_weights`, both already computed earlier in `_experts`. Otherwise the code falls through to the SGLang `fused_post_expert_norm_weighted_sum` call, unchanged, which serves as the T>=32 control.
- Deviations: none in the code. In decode_layer, cache3 aliases the start of cache13, which cache1 also uses. Remote rows therefore hold stale w13 output, not truly uninitialised memory. They must not be read either way, and the `eid >= 0` skip handles that. I kept the docstring wording verbatim.
- Untouched: the local_ids arithmetic in kernel_fn, moe_align_block_size, the block_size_m pick and its cap, both Marlin calls, silu_and_mul, the pre-expert norm, the unpooler, the boundaries and attention.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/post-expert-norm-triton-small-t diff`:

- [x] The new path can be captured in a CUDA graph. There are no `.item()`, `.cpu()`, synchronize calls or tensor prints. The only host branch is on `num_tokens`, which is a shape. `torch.empty` is sized by shape. The in-kernel `eid >= 0` is device-side control flow, not a host branch.
- [x] Nothing derived from `data` is cached across calls. The only addition to state is a weight copy made in `prepare()`.
- [x] Weights: `post_expert_norm_w` is the original BF16 tensor with only `.contiguous()` applied, so its precision is unchanged. `1 + w` is computed in fp32 inside the kernel.
- [x] Activations: y is read as BF16 from Marlin, math runs in fp32, and the output is stored as BF16, the same dtype the SGLang wrapper returns. The KV cache and queries are untouched.
- [x] n/a: attention is untouched, so every key is still attended.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network. The only new imports are triton and triton.language.
- [x] All the work happens inside the timed call, and nothing is cached or replayed. The kernel runs on each call with that call's y, ids and weights.
- [x] No `triton.autotune`. The config is fixed: num_warps=8 and a constexpr BLOCK_H.
- [x] The first line is still `# okc: problem=muse_decode_layer`.

Lint (`okc_lint.py ... --problem muse_decode_layer`):

```
warn  host-sync    L277: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_decode_layer/submission.py
```

The warning predates this change: it flags the existing `torch.cuda.synchronize()` at the end of `prepare()`. `python3 -I -m py_compile` passes, and the __pycache__ it left has been removed.

## Risks / what could fail
low

## Result (after the run)
Promotion passed: 1.1665x [1.1490,1.1840], board: {"rank": 405, "team": "yesthatwilly", "speedup": 1.1665, "first": false, "ahead": 404, "to_first": 1.8691}
