# muse_moe_w4a16 · 2026-10-24T162227Z · submit · triton-map-align-on-block-m-restack

Hypothesis: `muse_moe_w4a16/hypotheses/done/gain/2026-10-24T132314Z-triton-map-align-on-block-m-restack.md` (branch `hyp/muse_moe_w4a16/triton-map-align-on-block-m-restack`).

- **Job:** `<job id>` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `efe040c`

## Goal
Promote check-proven gain triton-map-align-on-block-m-restack (1.3117x [1.2920,1.3313], check job bc56182a80444d16) to the board.

Stack the two measured, shape-disjoint moe wins: start from master (marlin-block-m-two-smaller-on-postnorm, 1.254x: block_size_m two candidates down at num_tokens >= 128) and add triton-map-align-small-m-port (1.185x on the older post-norm base, gained 4.5-12.6% at bs=1/8/16/32, flat at bs=128/256) unchanged: the _align_small_kernel single launch for expert_map gather + align at num_tokens <= 32, with its prepare() warm-up and self-check. The result is the 3-way merge of the map-align branch onto master; the only conflict is the block_size_m pick, resolved by keeping map-align's _block_size_m() call and applying master's num_tokens >= 128 two-step adjustment right after it. The two changes touch disjoint shapes (<=32 vs >=128), so each shape should read as its own parent did.

## What changed, and where
Built by churnsubs as a 3-way merge (`git merge-file`) of hyp/muse_moe_w4a16/triton-map-align-small-m-port's
submission.py (on its base, post-expert-norm master) onto current master (marlin-block-m-two-smaller-on-postnorm).
The one conflict was the block_size_m pick: resolved by keeping map-align's `_block_size_m()` call and the
`_align_small` dispatch, with master's `num_tokens >= 128` two-step adjustment placed right after the pick.
The result differs from the map-align branch's file only by that 5-line block (checked with diff). No other
deviation; the reviewer confirmed the merge.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/triton-map-align-on-block-m-restack diff` (the map-align diff, already
rules-checked in its own run, plus master's 5-line block-m pick).

- [x] CUDA-graph capturable: both branches are on `num_tokens`/`expert_map` (shape/config); no `.item()`/`.cpu()`/`synchronize()` in kernel_fn.
- [x] Nothing derived from `data` is cached across calls: bank start/count come from prepare(); no new state from inputs.
- [x] Weights: NVFP4 stays NVFP4; Marlin repack unchanged.
- [x] Activations BF16 where SGLang keeps BF16; KV/query items n/a.
- [x] Attention sees every key: n/a: no attention; every local route is aligned, remote routes dropped as on master.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] First line marker kept; no triton.autotune.

okc_lint: OK with warnings (host-syncs only in prepare()/the align self-check).

## Risks / what could fail
low

## Result (after the run)
(pending)
