# muse_decode_layer · 2026-10-24T111950Z · check · triton-flash-decode-full-oversub-kv-prefetch

Hypothesis: `muse_decode_layer/hypotheses/running/2026-10-24T100416Z-triton-flash-decode-full-oversub-kv-prefetch.md` (branch `hyp/muse_decode_layer/triton-flash-decode-full-oversub-kv-prefetch`).

- **Job:** `e55908a05eec4000` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `016401c`

## Goal
Build on stage 1 (branch hyp/muse_decode_layer/triton-flash-decode-full-paired-kv, muse_decode_layer/submission.py) and change only _full_decode_splits and the loop of _full_decode_split_kernel, to put more K/V bytes in flight per SM on full DECODE (window == 0, q_len == 1). Verify, sliding layers and everything else stay byte-identical to stage 1 (verify keeps FA3). Stage 1 measured kernel 1 at 200.8 us (decode_16 full, 1.63 TB/s over ~328 MB) and 394.8 us (decode_32 full, 1.66 TB/s over ~655 MB), against FA3 188.8 / 374.4 us. Its grid is the limiter: S = ceil(264/B) gives B=16: 17x16 = 272 CTAs and B=32: 9x32 = 288 CTAs, about 2 per SM (and 288 > 264 leaves a 24-CTA second wave). Each CTA has one 16 KB K+V tile in flight, because the loop issues tile i's K/V loads and consumes them in the same iteration (only the page_table indices are fetched one tile ahead). Under spread=agentic the split length comes from page_table.shape[1], so the CTAs of short requests finish early and leave their slots idle for the rest of the single wave. Two changes, both aimed at bytes in flight. (1) Oversubscribe the grid: FD_SPLIT_CTAS = 1024 and S = max(1, min(128, 1024 // B)), from B only, so B=1: 128 (unchanged), B=16: 64, B=32: 32, B=33: 31. That is about 4 resident CTAs per SM over about 2 waves, and empty or short splits drain and free their slots for the long requests. split_len stays ceil(page_table.shape[1]*P / S) rounded up to BLOCK_N (B=16: 1280 tokens = 20 tiles, B=32: 2560 tokens = 40 tiles). (2) Prefetch one tile ahead in K/V, not only in indices. Before the loop, load tile 0's indices, rows, K and V (uint8 [64,128] each), plus tile 1's indices and rows. In each iteration, first issue the K and V loads for tile i+1 (its rows are already in hand; masked by its valid, other=0), then load tile i+2's page_table indices and rows, then run tile i's QK^T, online softmax and PV on the K/V loaded in the previous iteration, then rotate (k_cur, v_cur, valid_cur) <- (k_next, v_next, valid_next) and rows_next <- rows_next2. While one tile is computed, the next tile's 16 KB and the index load after it are outstanding. At about 4 CTAs per SM that is about 64 KB in flight per SM (about 8.4 MB chip-wide, about 2x stage 1's ~4 MB), towards LESSONS' 3.73 TB/s at 128 KB per CTA in bf16. Keep BLOCK_N=64 and num_warps=4. Set num_stages=1, because the prefetch is explicit and the compiler must not add its own smem ring on top. Everything else is stage 1's code: the block-diagonal [16,128] Q2 with explicit zeros, FP8 q, fp8->bf16 K/V, the exp2-domain online softmax with its -inf guard, fp32 partials [B,S,16,64] plus (m,l), and combine kernel 2 with BLOCK_S = next_pow2(S) (64 and 32 at B=16 and 32; prepare() already warms every BLOCK_S up to 128). Every key t < cache_seqlens[b] is still visited exactly once. The loop bound and masks are unchanged, so a prefetch of a tile past end is fully masked and never dereferenced (-1 entries are never read). No autotune, no host syncs, and scratch is allocated per call from shapes only. Report kernel 1 + kernel 2 us per full-decode shape and KV bytes / kernel-1 time.

## What changed, and where
Worktree `.worktrees/muse_decode_layer/triton-flash-decode-full-oversub-kv-prefetch`, on stage 1 (`bbbf316`). Diff: `muse_decode_layer/submission.py | 58 +++++----` (37+, 21-), all inside the full-decode path.

- Constants: `FD_NUM_SMS = 132` became `FD_SPLIT_CTAS = 1024` (FD_NUM_SMS had no other user). Its comment line was updated. `FD_MAX_SPLITS = 128` and `FD_BLOCK_N = 64` are unchanged.
- `_full_decode_splits(batch)` now returns `max(1, min(FD_MAX_SPLITS, FD_SPLIT_CTAS // batch))`. That gives B=1: 128, B=16: 64, B=32: 32 and B=33: 31, from B only. split_len is unchanged: ceil(max_pages*page / S), rounded up to BLOCK_N.
- `_full_decode_split_kernel` loop:
  - Before the loop it loads tile 0's indices, rows, K and V (`k_cur`, `v_cur`, `valid_cur`), plus tile 1's indices and rows (`rows_next`, `valid_next`).
  - Each iteration does four things in order. (1) Issue tile i+1's K and V loads, masked by `valid_next`, other=0. (2) Load tile i+2's page-table indices and rows (`valid_next2`, `rows_next2`, masked, with the same `t // PAGE < max_pages` guard as before). (3) Run tile i's fp8->bf16, QK^T, -inf masking by `valid_cur`, exp2 online softmax and PV, on `k_cur`/`v_cur`. (4) Rotate `k_cur, v_cur, valid_cur <- k_next, v_next, valid_next` and `rows_next, valid_next <- rows_next2, valid_next2`.
  - Valid moves with its tile. The loop bound `range(start, end, BLOCK_N)` and the masks are unchanged, so every key t < seqlen is still visited exactly once. A prefetch past end is fully masked, so no -1 page entry is read and no out-of-range row is dereferenced.
  - The math body, Q2 construction, partial stores and combine kernel are byte-identical to stage 1.
- Launch: `num_stages=3` became `num_stages=1`. num_warps=4 and BLOCK_N=64 are unchanged. Combine still uses BLOCK_S = next_pow2(S): 64 at B=16, 32 at B=32 and B=33, 128 at B=1. `_warm_full_decode` already compiles every BLOCK_S up to 128 and both PAGE variants. Its own split launch (B=1, S=128) compiles the new num_stages=1 variant.
- Not touched: verify (q_len 8), sliding layers, FA3 calls, everything outside the full-decode path.
- Deviations: none from the spec. The only extra edit is the stale constant comment.
- Uncertainties (cannot run here):
  - `k_cur`/`v_cur`/`k_next`/`v_next` are loop-carried uint8 [64,128] tensors. That should be about 32 extra regs/thread at 128 threads. If the compiler spills or residency drops below 4 CTAs/SM, the gain shrinks. Check kernel-1 bandwidth against stage 1's 1.63/1.66 TB/s.
  - With num_stages=1, Triton's pipeliner should leave the explicit prefetch as written. Whether stage 1's num_stages=3 already pipelined the indirect loads stays open, per the review.
  - For an empty split (start >= end), the pre-loop K/V loads are issued fully masked, as stage 1's pre-loop index load already was. That is harmless.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/triton-flash-decode-full-oversub-kv-prefetch diff`:

- [x] CUDA-graph capturable: the diff adds no `.item()`/`.cpu()`/`synchronize()`/tensor prints. Grid (B, S) and split_len come from `q8.shape` and `page_table.shape` only. Kernel-side masking uses seqlens on device, with no host control flow on input values.
- [x] Nothing derived from `data` is cached across calls: scratch (acc, m, l, out) is still allocated per call from shapes, and no new state was added.
- [x] Weights: n/a. The diff touches no weight or prepare() repacking; prepare() is unchanged.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3. Unchanged: K/V are read as FP8 and widened to bf16 in-kernel, q stays FP8, partials are fp32 and the output is bf16.
- [x] Attention sees every key: the loop bound and masks are unchanged, and each tile in [start, end) is computed exactly once. No top-k, sparse, block-skip or weight-based skipping; prefetched tiles past end are masked and never computed as valid.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds none.
- [x] All work happens inside the timed call, with no output caching or replay. Both kernels launch inside kernel_fn per call.
- [x] First-line marker `# okc: problem=muse_decode_layer` kept; no `triton.autotune` added.

Lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_decode_layer/submission.py --problem muse_decode_layer`):

```
warn  host-sync    L260: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_decode_layer/submission.py
```

The warning comes from the existing prepare() sync, which stage 1 already had. `python3 -I -m py_compile` passed, and the __pycache__ it left was removed.

## Risks / what could fail
medium
Milestones: kernel 1 + combine <= 150 us at decode_16 full and <= 300 us at decode_32 full (0.8x FA3); decode_1 full <= 25 us. If it misses, read kernel 1's bandwidth. Under ~2 TB/s means either registers capped residency below 4 CTAs/SM (the held next K/V tile adds ~32 regs/thread at num_warps=4) or the gather is latency-bound per row; the next stage is then BLOCK_N=128 with num_warps=8, or a two-deep ring. Combine cost grows with S: partials are B*S*16*64*4 B = 4 MB at B=16/32 (written, then read), about 2-3 us, and kernel 2 reads 64 or 32 partials per (b, h). Count it in the bar. Correctness: the rotation must carry valid with its tile, because a tile past end has valid all false and its qk becomes -inf. Tests on this path: batch 1 ctx 3000 full (page 64, the PAGE constexpr path), batch 33 ctx 20000 full spread mixed (S=31, many empty splits), batch 16 ctx 20000 full, batch 1 ctx 80000 full. Grid and split_len come from B and page_table.shape only; nothing input-derived is cached.

## Result (after the run)
- job `e55908a05eec4000` (check), status done, passed True, tests 13/13
- speedup 1.0413x, 95% CI [1.0257, 1.0569], score 21722.604910781774 µs vs SGLang 22620.3 µs
- best at harvest: {"speedup": 1.0873503247934666, "ci": [1.0710400699215645, 1.1036605796653685], "job": "fb07a3bd3aa54baf"}; base result: muse_decode_layer/hypotheses/done/neutral/triton-flash-decode-full-paired-kv.result.json
- **outcome: regression** (regression)
- files: `muse_decode_layer/hypotheses/done/regression/triton-flash-decode-full-oversub-kv-prefetch.result.json`, `muse_decode_layer/hypotheses/done/regression/triton-flash-decode-full-oversub-kv-prefetch.patch`; branch `hyp/muse_decode_layer/triton-flash-decode-full-oversub-kv-prefetch`
