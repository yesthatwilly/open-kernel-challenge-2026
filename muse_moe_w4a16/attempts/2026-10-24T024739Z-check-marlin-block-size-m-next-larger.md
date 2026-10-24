# muse_moe_w4a16 · 2026-10-24T024739Z · check · marlin-block-size-m-next-larger

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-23T225502Z-marlin-block-size-m-next-larger.md` (branch `hyp/muse_moe_w4a16/marlin-block-size-m-next-larger`).

- **Job:** `2639e3ca7f5549de` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `82f0d6b`

## Goal
Change only the block_size_m pick in kernel_fn. SGLang's loop breaks on the first candidate in (8, 16, 32, 48, 64) where num_tokens*8/32/block_size_m < 0.9, which yields 8 at bs=1/8/16, 16 at bs=32, 48 at bs=128 and 64 at bs=256. Replace it with the next larger candidate (cap 64): 16 at bs=1/8/16, 32 at bs=32, 64 at bs=128, 64 at bs=256 (unchanged, a control). The same block size feeds moe_align_block_size and the moe_block_size kwarg of both Marlin calls; nothing else changes. Mechanism: Marlin MoE re-reads an expert's whole NVFP4 weight for every m-block of that expert, so fewer, larger m-blocks mean fewer weight passes (L2 is cold at every replay, so a second block re-reads from HBM), at the cost of more padded rows, whose compute is free while the kernel is memory-bound.

## What changed, and where
Worktree `.worktrees/muse_moe_w4a16/marlin-block-size-m-next-larger`, `muse_moe_w4a16/submission.py`, uncommitted, on base 6b5469bd92 (diff stat: 1 file, +8 -1).

- SGLang's selection loop is kept verbatim except that it now enumerates, so it yields SGLang's index `idx` (first candidate with `num_tokens*8/32/block_size_m < 0.9`, else the last). The block size actually used is then `BLOCK_SIZE_M_CANDIDATES[min(idx + BLOCK_SIZE_M_STEP(num_tokens), 4)]`.
- `BLOCK_SIZE_M_STEP = lambda num_tokens: 1` is a new module constant beside the candidates. Adopting part of the change per shape is a one-line edit there (e.g. `lambda num_tokens: 0 if num_tokens < 32 else 1`), as the review asked.
- Picks checked offline, SGLang -> new: bs=1/4/8/16: 8 -> 16; bs=32/33: 16 -> 32; bs=128: 48 -> 64; bs=256/257: 64 -> 64 (the control). These match the `change` paragraph.
- The same `block_size_m` still feeds `moe_align_block_size` (over the bank's 32 experts, as on master) and the `moe_block_size` kwarg of both Marlin calls. Align, workspace, caches, and everything else are unchanged. There is no other tuning, and the "64 everywhere" ab-tune is not folded in.
- For the review's open question, the profile should record the Marlin template (thread_m_blocks) for each shape, to confirm which config ran.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/marlin-block-size-m-next-larger diff`:

- [x] CUDA-graph capturable: the pick is pure Python on `num_tokens` (a shape) and a constant. There is no `.item()`/`.cpu()`/sync, and no control flow on tensor values.
- [x] Nothing derived from `data` is cached across calls: no new state.
- [x] Weights: n/a. prepare() is untouched, and NVFP4 stays NVFP4 via the same Marlin repack.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: n/a for KV/queries (MoE problem). Activation dtypes are unchanged.
- [x] Attention sees every key: n/a (no attention in this problem). The diff skips no routes or experts. Padding only grows.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the diff adds none.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line `# okc: problem=muse_moe_w4a16` kept. `python3 -m py_compile` passes.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_moe_w4a16/submission.py --problem muse_moe_w4a16`), rc=0:

```
warn  host-sync    L113: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
```

The one warning was already on master: it is in prepare(), not this diff, so there was nothing to fix.

## Risks / what could fail
low
Risks: Marlin's thread_m_blocks grows with moe_block_size, so a bigger block changes the kernel config (tile shape, smem) and could lose occupancy at small M; bs=1/8 could regress slightly from doubled A-tile padding. Pure dispatch knob: no correctness or rule exposure. If this is a gain, the obvious follow-up ab-tune is 64 everywhere.

## Result (after the run)
- job `2639e3ca7f5549de` (check), status done, passed True, tests 5/5
- speedup 0.9178x, 95% CI [0.9041, 0.9316], score 14652.30828522236 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: regression** (regression)
- files: `muse_moe_w4a16/hypotheses/done/regression/marlin-block-size-m-next-larger.result.json`, `muse_moe_w4a16/hypotheses/done/regression/marlin-block-size-m-next-larger.patch`; branch `hyp/muse_moe_w4a16/marlin-block-size-m-next-larger`
