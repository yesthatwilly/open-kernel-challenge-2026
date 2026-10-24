# muse_moe_w4a16 · 2026-10-24T124312Z · check · post-norm-triton-all-m

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T112212Z-post-norm-triton-all-m.md` (branch `hyp/muse_moe_w4a16/post-norm-triton-all-m`).

- **Job:** `2ec5387ffe8748c9` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `051681e`

## Goal
On top of marlin-block-m-two-smaller-on-postnorm, remove the SMALL_M_POST_NORM = 32 split at the end of kernel_fn so every num_tokens (bs=32/33/128/256/257 too) combines with master's _post_norm_combine_kernel instead of SGLang's fused_post_expert_norm_weighted_sum. The kernel itself is unchanged: grid (num_tokens,), num_warps=8, BLOCK_H = next_power_of_2(H) = 4096, one program per token looping over its 8 slots with tl.static_range, skipping remote slots (local id -1) without loading them, fp32 math, (1 + w) applied in-kernel from the bf16 post_norm_w, bf16 store into a fresh [M, H] output. y.contiguous() stays (a no-op on Marlin's contiguous output). state['norm_w'] and the fused_post_expert_norm_weighted_sum import become unused; leave them or delete them, nothing else changes (block_size_m picks, align, both Marlin calls, silu_and_mul as in the dependency).

## What changed, and where
Implemented inline by churnsubs on master (marlin-block-m-two-smaller-on-postnorm): the `num_tokens <
SMALL_M_POST_NORM` test before the Triton combine became `if True:` (with a comment), so every num_tokens
takes master's unchanged `_post_norm_combine_kernel` (grid (num_tokens,), 8 warps, BLOCK_H 4096). The
SGLang `fused_post_expert_norm_weighted_sum` return after it is now unreachable and left in place, as the
spec allows; `state['norm_w']` stays. A minimal diff keeps the attribution to the one branch.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/post-norm-triton-all-m diff` (one condition line plus a comment).

- [x] CUDA-graph capturable: no new host code; the kernel launch is master's; no `.item()`/`.cpu()`/`synchronize()` added.
- [x] Nothing derived from `data` is cached across calls: no state writes added.
- [x] Weights: `post_norm_w` stays the original bf16 tensor; (1+w) formed in fp32 in-kernel, as on master.
- [x] Activations BF16 where SGLang keeps BF16: bf16 output, fp32 math, as on master; KV/query items n/a.
- [x] Attention sees every key: n/a: no attention; every local slot is combined, remote slots skipped as before.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line marker kept; no triton.autotune added.

okc_lint: OK with warnings (only the existing `torch.cuda.synchronize()` in prepare()).

## Risks / what could fail
low
Expected effect is small (~0.5-2 us per shape, well under the ~1.5% geomean resolution in LESSONS): it is a cleanup-and-measure run whose verdict should come from the replaced kernel's own time (SGLang _post_expert_norm_weighted_sum_kernel 3.7/6.1/5.2 us at bs=32/128/256 vs _post_norm_combine_kernel on the same shapes), not the headline. Run it after the restack so the bs>=128 shapes are smaller and the combine is a larger share. Risk: our kernel's per-token serial 8-row loop could lose at bs=256 if 256 programs x 8 dependent 8 KB loads underfill bandwidth; if _post_norm_combine_kernel exceeds SGLang's time at bs>=128 the right follow-up is the threshold at 128, not a rewrite. Correctness: same kernel already passes the 5 tests at bs<32; bs=257 just adds programs.

## Result (after the run)
- job `2ec5387ffe8748c9` (check), status done, passed True, tests 5/5
- speedup 1.2483x, 95% CI [1.2296, 1.2670], score 10773.321822675993 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 1.2535643764852882, "ci": [1.2347609108380089, 1.2723678421325675], "job": "c00ef7295c154867"}; base result: muse_moe_w4a16/hypotheses/done/gain/marlin-block-m-two-smaller-on-postnorm.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/post-norm-triton-all-m.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/post-norm-triton-all-m.patch`; branch `hyp/muse_moe_w4a16/post-norm-triton-all-m`
