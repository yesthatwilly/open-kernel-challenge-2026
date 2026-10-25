# muse_moe_w4a16 · 2026-10-25T003435Z · check · post-norm-triton-hoisted-decode32

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T183635Z-post-norm-triton-hoisted-decode32.md` (branch `hyp/muse_moe_w4a16/post-norm-triton-hoisted-decode32`).

- **Job:** `47d186dd91974d84` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `4aab018`

## Goal
num_tokens == 32 only (decode_32); bs<32 keeps master's _post_norm_combine_kernel unchanged and bs>=33 (incl. the bs=33 test) keeps SGLang's fused_post_expert_norm_weighted_sum. Add a new Triton kernel _post_norm_combine_hoisted_kernel with master _post_norm_combine_kernel's signature and math (out[t] = sum over local slots of g * rmsnorm(y[t,s] * (1 + w)), fp32, remote slots never read, zero row for a token with no local slot) but with every slot load issued before any reduction: load the 8 ids and 8 weights of the token as vectors, then in tl.static_range(TOP_K_C) extract eid_s/g_s with tl.sum(tl.where(arange==s, vec, 0)), load v_s = tl.load(y + row_s + offs, mask=mask & (eid_s >= 0), other=0.0) with no 'if', and accumulate tl.where(eid_s >= 0, g_s, 0.0) * v_s*nw * rsqrt(sum((v_s*nw)^2)/H + eps). Launch grid (32,), BLOCK_H = 4096, num_warps=16. In kernel_fn, at the end: if num_tokens < SMALL_M_POST_NORM: master's path unchanged; elif num_tokens == 32: y = y.contiguous(), out = torch.empty((32, H), bf16) and launch the new kernel with state['post_norm_w'] (as master's small-M path does); else SGLang's call unchanged. Compile it in prepare() once on a dummy y/ids (untimed) so the first call does not JIT.

## What changed, and where
- submission.py only (+41 lines): new `_post_norm_combine_hoisted_kernel` (master kernel's signature and math), constant `HOISTED_POST_NORM_TOKENS = 32`, a `num_tokens == 32` branch at the end of kernel_fn (y.contiguous(), torch.empty out, grid (32,), BLOCK_H=4096, num_warps=16, state['post_norm_w']), and an untimed dummy-data compile in prepare(). bs<32 and bs>=33 paths are unchanged.
- Deviation from the spec: instead of a static_range of per-slot scalar extractions (tl.sum(tl.where(arange==s, ...))) with a Python list of loads, the kernel loads all 8 slot rows as one [8, BLOCK_H] tile (mask = id>=0 & offs<H) in a single tl.load, then does a row-wise sum of squares, coef = where(id>=0, g, 0) * rsqrt(ss/H+eps), and acc = sum(v * coef[:, None], axis=0). Same hoisting intent (all loads before any reduction, no 'if', remote rows never read), but avoids list-in-jit risk and scalar extraction. Cost: about 64 fp32 regs/thread for the tile at 512 threads.
- Cannot be run here (no GPU/Triton); compile and numerics untested.

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: diff adds only a Triton launch and torch.empty with shape-only size; no .item()/.cpu()/sync in kernel_fn; branch on num_tokens (shape) only.
- [x] Nothing derived from `data` is cached across calls (prepare dummy compile uses zeros, not inputs).
- [x] Weights: n/a: no weight repack changed; post_norm_w is the same BF16 tensor.
- [x] Activations BF16 where SGLang keeps BF16: y read as BF16, fp32 math, BF16 output as master's small-M path.
- [x] Attention: n/a: MoE problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network (checked diff).
- [x] All work happens inside the timed call; no output caching or replay.

okc_lint: `OK with warnings` (only host-sync warnings in pre-existing prepare()-only code, lines 355-491); py_compile OK, __pycache__ removed.

## Risks / what could fail
low
Shape-disjoint sibling of post-norm-combine-hoisted-slot-loads (bs<32): the two restack by 3-way merge into one hoisted kernel for bs<=32. decode_32 Marlin rows are seed-noisy (147-265 us), so judge only on the post-norm kernel row in top_kernels. post-norm-triton-all-m (old serial kernel at every M) was within CI; this differs by the hoisted loads and by bs=32 only. Overlap: the queued cuda-fused-moe-midm-single-launch-routed would replace it; the running swap-AB and two-stream keep this post-norm, so it stacks.

## Result (after the run)
- job `47d186dd91974d84` (check), status done, passed True, tests 5/5
- speedup 1.3248x, 95% CI [1.3050, 1.3447], score 10150.97842656069 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.3278240305297715, "ci": [1.307906670071825, 1.3477413909877178], "job": "1947e3ec2abb4bdf"}; base result: muse_moe_w4a16/hypotheses/done/neutral/align-parallel-per-expert-verify.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/post-norm-triton-hoisted-decode32.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/post-norm-triton-hoisted-decode32.patch`; branch `hyp/muse_moe_w4a16/post-norm-triton-hoisted-decode32`
