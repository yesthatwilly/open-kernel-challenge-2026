# muse_moe_w4a16 · 2026-10-24T100227Z · check · marlin-block-size-m-next-smaller

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T075232Z-marlin-block-size-m-next-smaller.md` (branch `hyp/muse_moe_w4a16/marlin-block-size-m-next-smaller`).

- **Job:** `198f10f446a847cb` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `fa264a1`

## Goal
Change only the block_size_m pick in kernel_fn: keep SGLang's loop over BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64) (first candidate with num_tokens*8/32/block_size_m < 0.9) but take the candidate one index BELOW it, floored at index 0: block_size_m = BLOCK_SIZE_M_CANDIDATES[max(idx - BLOCK_SIZE_M_STEP_DOWN(num_tokens), 0)] with a module-level BLOCK_SIZE_M_STEP_DOWN = lambda num_tokens: 1 so a per-shape partial adoption is a one-line follow-up. Resulting picks, SGLang -> new: bs=1/4/8/16: 8 -> 8 (unchanged, already the smallest, the in-run control); bs=32/33: 16 -> 8; bs=128: 48 -> 32; bs=256/257: 64 -> 48. The same block_size_m feeds moe_align_block_size (still over the bank's 32 experts, as on master) and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls; nothing else changes. Mechanism: this is the mirror image of marlin-block-size-m-next-larger (regression 0.918x), which moved Marlin to the next thread_m_blocks config and lost 21% at bs=32 (16 -> 32, thread_m_blocks 1 -> 2), 14% at bs=128 (48 -> 64, 3 -> 4) and ~2% at bs=1 (8 -> 16, the 8-row special path to the 16-row path; inside noise), even though the larger block collapsed m-blocks. So at these M the per-m-block cost (tile shape, registers, smem stages) dominates over the number of weight passes; a smaller block (8-row path at bs=32; thread_m_blocks 2 at bs=128; 3 at bs=256) may be cheaper per block by more than the extra m-blocks cost (experts with more routes than block_size_m now get two m-blocks, each streaming that expert's weights, partly from L2).

## What changed, and where
- `muse_moe_w4a16/submission.py`, module level: added `BLOCK_SIZE_M_STEP_DOWN = lambda num_tokens: 1` (with a `# noqa: E731`) right after `BLOCK_SIZE_M_CANDIDATES`, so a per-shape partial adoption (for example keeping 16 at bs=32) is a one-line edit. This is the idx-table form the review asked for, matching next-larger's `BLOCK_SIZE_M_STEP`.
- `kernel_fn`, the block_size_m pick: SGLang's loop now records the index `idx` of the first candidate with `num_tokens*TOP_K/bank["count"]/candidate < 0.9`. If no candidate passes, `idx` defaults to the last index, which matches the old loop running out on 64. The pick is then `BLOCK_SIZE_M_CANDIDATES[max(idx - BLOCK_SIZE_M_STEP_DOWN(num_tokens), 0)]`. This is a host int that depends only on the shape and the bank count, so it adds no sync and no branching on input values.
- The same `block_size_m` feeds `moe_align_block_size` (still over `bank["count"]` = 32) and the `moe_block_size` kwarg shared by both `moe_wna16_marlin_gemm` calls. The `num_tokens == 1 and expert_map is None` single-token path is unchanged; it gets 8 before and after. Nothing else is touched.
- Picks checked offline with the same rule (count = 32), old -> new: bs 1/4/8/16 8->8, bs 32/33 16->8, bs 128 48->32, bs 256/257 64->48. These match the spec.
- No deviation from the spec. I did not apply the review's open questions (port cap-16, edit the Risk text) because they are new ideas or documentation, not implementation details.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/marlin-block-size-m-next-smaller diff` (1 file, +11/-2):

- [x] CUDA-graph capturable: the diff adds only host-int arithmetic on `num_tokens` (a shape) and `bank["count"]` (config). It adds no `.item()`, `.cpu()`, `synchronize()` or tensor print, and no allocation or control flow that depends on input values.
- [x] Nothing derived from `data` is cached across calls: the diff writes no new state. `BLOCK_SIZE_M_STEP_DOWN` is a constant function of the shape.
- [x] Weights: untouched by the diff. `prepare()` is unchanged, NVFP4 stays NVFP4 and the repack is SGLang's own.
- [x] Activations: the diff leaves every dtype unchanged, so they stay BF16 where SGLang keeps BF16 (x, cache1/2/3 and the output).
- [x] Attention sees every key: n/a, this is an MoE kernel with no attention. Also no expert or route is skipped: every local route is still computed, and only the m-tile size changes.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none appear in the diff.
- [x] All work happens inside the timed call, with no output caching or replay: the diff only changes which tile size the in-call Marlin launches use.
- [x] The first-line marker `# okc: problem=muse_moe_w4a16` is kept, and the diff adds no triton.autotune.

Lint (`okc_lint.py ... --problem muse_moe_w4a16`):

```
warn  host-sync    L114: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_moe_w4a16/submission.py
```

The warning is the existing `torch.cuda.synchronize()` at the end of `prepare()` on master, not something the diff added. `python3 -I -m py_compile` passed, and I removed the `__pycache__` it left.

## Risks / what could fail
medium
Risk: an expert with more local routes than block_size_m now needs an extra m-block, and every m-block streams that expert's full w13 (16 MB) and w2 (8 MB) slice; under skewed popularity at bs=256 the hot experts may cross 48 rows, so a weight-bandwidth loss can outweigh the config gain; read per shape. Correctness is unaffected (pure dispatch knob; the bs=33/257 and skewed bs=128 tests run the new picks). marlin-block-size-m-verify-two-smaller runs the two-step-down picks at bs=128/256 independently from master so the two runs together give the gradient; if all three lose here, SGLang's pick is a local optimum at bs>=32 and the block-size knob is closed. muse_decode_layer is testing a cap of 16 on the same Marlin call; do not merge their result into this one, the shapes and routing differ.

## Result (after the run)
- job `198f10f446a847cb` (check), status done, passed True, tests 5/5
- speedup 1.0394x, 95% CI [1.0238, 1.0550], score 12939.1686230573 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: gain** (gain)
- files: `muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-next-smaller.result.json`, `muse_moe_w4a16/hypotheses/done/gain/marlin-block-size-m-next-smaller.patch`; branch `hyp/muse_moe_w4a16/marlin-block-size-m-next-smaller`
