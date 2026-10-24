# muse_moe_w4a16 · 2026-10-24T100132Z · check · marlin-block-size-m-verify-two-smaller

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T075257Z-marlin-block-size-m-verify-two-smaller.md` (branch `hyp/muse_moe_w4a16/marlin-block-size-m-verify-two-smaller`).

- **Job:** `ed837a4a46d44a95` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `4df94f3`

## Goal
Change only the block_size_m pick in kernel_fn, for num_tokens >= 128 only: take SGLang's pick (the first of BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) with num_tokens*8/32/block_size_m < 0.9) and step TWO candidates down: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - 2, 0)]. Resulting picks, SGLang -> new: bs=128: 48 -> 16 (thread_m_blocks 3 -> 1); bs=256/257: 64 -> 32 (4 -> 2); every num_tokens < 128 (bs=1/4/8/16/32/33) keeps SGLang's pick as the in-run control. The same value feeds moe_align_block_size (bank count 32, as on master) and the moe_block_size kwarg of both Marlin calls; nothing else changes. Independent of marlin-block-size-m-next-smaller (one step down at bs=32/128/256) and run from master in parallel with it, so the two give the per-shape gradient of Marlin cost vs block size below SGLang's pick, the opposite direction of the next-larger regression (48 -> 64 lost 14% at bs=128).

## What changed, and where
- One edit in `kernel_fn` (worktree `muse_moe_w4a16/submission.py`), right after SGLang's candidate loop: `if num_tokens >= 128: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(BLOCK_SIZE_M_CANDIDATES.index(block_size_m) - 2, 0)]`, with a 3-line comment. 5 insertions, nothing else touched.
- The same `block_size_m` still feeds `moe_align_block_size(local_ids, block_size_m, bank["count"])` (count 32, as on master) and `moe_block_size` in `common`, so both Marlin calls get it.
- Picks checked by evaluating the loop + new line in plain Python (SGLang -> new): bs 1/4/8/16 8->8, bs 32/33 16->16, bs 128 48->16, bs 256 64->32, bs 257 64->32 (256/257 never break the loop, so they end on 64, index 4 -> 32). Matches the spec.
- No deviation from the spec. Per the review's open question, the attempt file should try to record n_padded at bs=128/256 from the profile/log if visible; no print was added to kernel_fn (it would sync), so this is left to the profile.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/marlin-block-size-m-verify-two-smaller diff` (one 5-line hunk in kernel_fn).

- [x] CUDA-graph capturable: the new branch is on `num_tokens` (a shape) and a Python tuple index; no `.item()`/`.cpu()`/`synchronize()`/tensor prints added; no allocation or control flow on input values.
- [x] Nothing derived from `data` is cached across calls: no state writes added.
- [x] Weights: n/a: prepare() and the repack are untouched; NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged (only the tile size of the same Marlin W4A16 calls changes); KV cache / query items n/a: no attention in this problem.
- [x] Attention sees every key: n/a: no attention in this problem; no routes are skipped (alignment still covers every local route).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds none.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune added.

okc_lint output:

```
warn  host-sync    L110: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
```

`python3 -I -m py_compile` passed; the `__pycache__` it left was removed. Not committed, staged or run.

## Risks / what could fail
medium
Risk: higher than one step down. At bs=128 the average local expert has ~32 routes (more on hot experts under skewed popularity), so 16-row blocks give 2-4+ m-blocks per expert, each streaming the expert's full NVFP4 w13/w2 slice; if Marlin at this M is weight-bandwidth-bound per m-block rather than per-block-latency-bound, this loses, possibly a lot. Read together with marlin-block-size-m-next-smaller: (one-step gain, two-step loss) brackets the optimum; (both lose) closes the knob at verify. Correctness unaffected; the bs=257 and skewed bs=128 tests run the new picks.

## Result (after the run)
- job `ed837a4a46d44a95` (check), status done, passed True, tests 5/5
- speedup 1.0916x, 95% CI [1.0752, 1.1080], score 12319.802694029935 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: gain** (gain)
- files: `muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.result.json`, `muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-verify-two-smaller.patch`; branch `hyp/muse_moe_w4a16/marlin-block-size-m-verify-two-smaller`
