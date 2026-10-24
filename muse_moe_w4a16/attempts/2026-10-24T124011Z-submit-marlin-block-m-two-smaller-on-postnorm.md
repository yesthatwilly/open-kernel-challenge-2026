# muse_moe_w4a16 · 2026-10-24T124011Z · submit · marlin-block-m-two-smaller-on-postnorm

Hypothesis: `muse_moe_w4a16/hypotheses/done/gain/2026-10-24T112137Z-marlin-block-m-two-smaller-on-postnorm.md` (branch `hyp/muse_moe_w4a16/marlin-block-m-two-smaller-on-postnorm`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `98aa5c6`

## Goal
Promote check-proven gain marlin-block-m-two-smaller-on-postnorm (1.2536x [1.2348,1.2724], check job c00ef7295c154867) to the board.

Restack the landed-but-unmerged marlin-block-size-m-verify-two-smaller (1.092x on the pre-post-norm base, job ed837a4a46d44a95; patch at muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.patch) onto current master (post-expert-norm-triton-small-m, 192f225d8f). In kernel_fn only, keep SGLang's selection loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) unchanged (first candidate with num_tokens*8/bank['count']/block_size_m < 0.9, loop exhausting at 64), remember the index idx it stopped at, and for num_tokens >= 128 only set block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - 2, 0)]. Picks with bank count 32: bs=128 48 -> 16 (thread_m_blocks 3 -> 1), bs=256/257 64 -> 32 (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps master's pick (8 or 16) and is the in-run control. The chosen value feeds moe_align_block_size (bank count 32, as on master; bs>=128 does not take the moe_align_single_token branch) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls (w13 and w2). Nothing else changes: the post-norm split at SMALL_M_POST_NORM = 32 (Triton _post_norm_combine_kernel below, SGLang's fused_post_expert_norm_weighted_sum at bs>=32) stays exactly as on master, as do silu_and_mul, the workspace and both Marlin calls' other kwargs. The branch is on num_tokens only (shape), no sync, nothing input-dependent.

## What changed, and where
Implemented inline by churnsubs on master (which has post-expert-norm-triton-small-m): the same 5-line hunk
as marlin-block-size-m-verify-two-smaller, placed right after SGLang's unchanged block_size_m loop in
kernel_fn: at num_tokens >= 128 the pick moves two candidates down (bs=128 48 -> 16, bs=256/257 64 -> 32).
The bs=256 pick stays 32: marlin-block-m-cap-16-port's bs=256 row (1.329 vs SGLang) was within 3% of
two-smaller's (1.320), the reviewer's threshold for switching to 16. The post-norm path (bs<32) is untouched.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/marlin-block-m-two-smaller-on-postnorm diff` (+5 lines in kernel_fn).

- [x] CUDA-graph capturable: a Python index pick on `num_tokens` (a shape); no `.item()`/`.cpu()`/`synchronize()` added; no control flow on input values.
- [x] Nothing derived from `data` is cached across calls: no state writes added.
- [x] Weights: n/a: prepare() and the Marlin repack untouched; NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged; KV cache / query items n/a: no attention in this problem.
- [x] Attention sees every key: n/a: no attention; alignment still covers every local route.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune added.

okc_lint: OK with warnings (only the existing `torch.cuda.synchronize()` in prepare()).

## Risks / what could fail
low

## Result (after the run)
(pending)
