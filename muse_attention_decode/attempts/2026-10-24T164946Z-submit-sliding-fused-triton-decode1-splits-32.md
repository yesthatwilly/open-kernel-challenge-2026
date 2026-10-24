# muse_attention_decode · 2026-10-24T164946Z · submit · sliding-fused-triton-decode1-splits-32

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-24T140241Z-sliding-fused-triton-decode1-splits-32.md` (branch `hyp/muse_attention_decode/sliding-fused-triton-decode1-splits-32`).

- **Job:** `f9ac830932cb4eb6` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `5c273c6`

## Goal
Built on master. One config-only change in _sliding_forward: at batch 1 and q_len 1 (the only shape master routes to _sliding_fused_kernel) use S = 32 splits instead of _num_splits(1, 2) = 16, so each program owns ceil(2047/32) rounded up to 64 = 64 old keys, exactly one BLOCK_N=64 tile, and the grid is (1, 2, 32) = 64 CTAs instead of 32 CTAs of two serial tiles. To keep the combine compact, the two tl.static_range(S) loops in the last-ticket combine may stay static (32 iterations) or become a runtime range(S) when S > 16 behind a constexpr branch; the S <= 16 instantiation must compile unchanged. Workspace and counters already scale with S from config (ws_acc 64 x 16 x 64 fp32 = 256 KB). Leave _num_splits for other batches, BLOCK_N, num_warps=4, num_stages=2 and everything else as master.

## What changed, and where
- One line in `_sliding_forward` (worktree `muse_attention_decode/submission.py`, ~line 355):
  `splits = _num_splits(batch, hkv)` became
  `splits = 32 if batch == 1 and length == 1 else _num_splits(batch, hkv)`, with a one-line comment.
  It is written as an explicit batch-1/q_len-1 override, per the review coordination note, so it
  3-way merges with sliding-deep-splits-on-cuda-splitkv. The override branches on config only.
- Kernel unchanged. Both combine loops stay `tl.static_range(S)` (32 unrolled iterations at
  S=32). The review calls this "simpler and fine", and it leaves the S<=16 instantiation
  byte-identical. No constexpr runtime-range branch was added.
- Workspace (`ws_acc` [64, 16, 64] fp32 = 256 KB, `ws_ml` [64, 2, 16]) and the counters
  ([B*Hkv], unchanged) already scale with `splits`. `_num_splits`, BLOCK_N=64, num_warps=4 and
  num_stages=2 are all unchanged.
- Edge case checked by reading: for short contexts (test "batch 1, ctx 10, sliding"),
  chunk = 64 and every split s >= 1 gets start >= seq, so its loop is empty and its partial is
  m=-inf, l=0, acc=0. Split S-1 holds the new keys, so it is finite, and the combine weights
  -inf partials 0. This is the same path S=16 already took.
- Deviation from the spec: none.
- Read the result from the `_sliding_fused_kernel` profile row against master's 10.3 us, not
  from the headline.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/sliding-fused-triton-decode1-splits-32 diff`
(1 file, +2/-1, `_sliding_forward` only):

- [x] CUDA-graph capturable: the new branch reads `batch`/`length` from `cfg` (config), not input
      values. It adds no `.item()`/`.cpu()`/`synchronize()`/prints. Allocations are shape-only
      (`torch.empty` sized from config), as before.
- [x] Nothing derived from `data` is cached across calls. The only cache is `_COUNTERS`, keyed by
      (batch, hkv, device), and it holds no data (it is reset in-kernel each launch). The diff
      leaves it unchanged.
- [x] n/a: weights. This problem has no weights, and the diff does not touch prepare or any
      weight handling.
- [x] Activations stay BF16 where SGLang keeps BF16. The KV cache is still stored as E4M3.
      Queries are still bf16 operands (finer than E4M3). The kernel is unchanged.
- [x] Attention sees every key. 32 * 64 = 2048 >= 2047 old keys, and the chunks tile
      [lo, seq-1] contiguously. The new keys are still attended by split S-1. Nothing is skipped.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses
      or network. The diff adds none of these.
- [x] All work happens inside the timed call. There is no output caching or replay.
- [x] First-line marker `# okc: problem=muse_attention_decode` kept. There is no `triton.autotune`.

Lint (`okc_lint.py ... --problem muse_attention_decode`):

```
CLEAN: /home/user/git/musespark/.worktrees/muse_attention_decode/sliding-fused-triton-decode1-splits-32/muse_attention_decode/submission.py
```

`python3 -I -m py_compile`: OK. The `__pycache__` it left was removed from the worktree.

## Risks / what could fail
low
Cheap single-knob probe of split depth on the one shape the fused kernel owns today; it also calibrates how far deep splitting goes before the combine and the atomics cost more than a tile, which informs sliding-fused-triton-decode-batched-deep-splits (B>1, S=16; disjoint shapes, the two can be 3-way merged). Risk: 64 partials of fresh tiles may be dominated by the fixed seq/page-table/gather/ticket latency chain, so the gain may be under 1 us (within ~1.5% per-shape drift); read the kernel row in the profile.

## Result (after the run)
- job `f9ac830932cb4eb6` (submit), status done, passed True, tests 16/16
- speedup 1.3520x, 95% CI [1.3317, 1.3764], score 2785.749776544196 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.3492076314709118, "ci": [1.328969516998848, 1.3729544014677721], "job": "de486093311d4bb9"}; base result: muse_attention_decode/hypotheses/done/gain/full-decode-cuda-gather-splitkv.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_attention_decode/hypotheses/done/neutral/sliding-fused-triton-decode1-splits-32.result.json`, `muse_attention_decode/hypotheses/done/neutral/sliding-fused-triton-decode1-splits-32.patch`; branch `hyp/muse_attention_decode/sliding-fused-triton-decode1-splits-32`
