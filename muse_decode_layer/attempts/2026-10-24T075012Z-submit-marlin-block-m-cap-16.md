# muse_decode_layer · 2026-10-24T075012Z · submit · marlin-block-m-cap-16

Hypothesis: `muse_decode_layer/hypotheses/done/gain/2026-10-23T225545Z-marlin-block-m-cap-16.md` (branch `hyp/muse_decode_layer/marlin-block-m-cap-16`).

- **Job:** `2b1f82408ab64116` (`okc status <job>`)
- **File:** `muse_decode_layer/submission.py` at git `e0303f1`

## Goal
Promote check-proven gain marlin-block-m-cap-16 (1.0621x [1.0462,1.0780], check job 0730135cdcac43b7) to the board.

Cap SGLang's block_size_m choice for the two Marlin MoE GEMMs at 16. Today _experts walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and stops at the first size where T*8/32/size < 0.9, which yields 8 at T<=28, 16 at T=32, 48 at T=128 and 64 at T=256; the cap keeps 8 and 16 as chosen and clamps the T=128/256 picks to 16. Mechanism: routing is skewed (a few experts take most tokens), so with block 48/64 the many cold experts holding 1-4 tokens each are padded to a 48/64-row MMA tile and the hot expert gets only 4-6 tiles to spread over 132 SMs; block 16 pads cold experts 3-4x less and gives the hot expert 3-4x more m-blocks for the same 25 MB weight stream. The same block_size_m goes to moe_align_block_size and both moe_wna16_marlin_gemm calls (moe_block_size), as it must. Nothing else changes. Attribution: only T=128 and T=256 shapes (load=verify_16 and verify_32) can move; every other shape keeps its current block size and is the control.

## What changed, and where
- `muse_decode_layer/submission.py`: new module constant `MARLIN_BLOCK_SIZE_M_CAP = 16` beside `BLOCK_SIZE_M_CANDIDATES`, and in `_experts` one line after SGLang's unchanged selection loop: `block_size_m = min(block_size_m, MARLIN_BLOCK_SIZE_M_CAP)`. That single variable already feeds `moe_align_block_size(local_ids, block_size_m, N_EXPERTS)` and `common["moe_block_size"]`, which both `moe_wna16_marlin_gemm` calls (w13 and w2) take, so all three see the same capped size. Diff: 1 file, +3 lines.
- Branches only on `num_tokens` (a shape) and constants; no input values, no syncs, nothing cached.
- Picks per scored T (T*8/32/bsm < 0.9 rule, T/4 routes per local expert on average):
  | T | before | after |
  |---|---|---|
  | T <= 28 (decode_1, decode_16, verify_1, ...) | 8 | 8 |
  | 32 (decode_32) | 16 | 16 |
  | 128 (verify_16) | 48 | 16 |
  | 256 (verify_32; loop exhausts at 64) | 64 | 16 |
- Mechanism caveat (from the adversarial review, recorded so the result is read correctly): the reason to expect a gain is the per-m-block cost of larger thread_m_blocks Marlin templates, not padding bytes or SM parallelism. Sister problem muse_moe_w4a16 (job 2639e3ca7f5549de) measured going *up* in thread_m_blocks costs +18-28% on Marlin (1->2 at T=32: 208 -> 261 us; 3->4 at T=128: 462 -> 545 us) with the m-block count essentially unchanged. Block 16 does add m-blocks for hot experts (more weight passes), so the net is that per-block saving minus the m-block growth; opposite signs on verify_16 vs verify_32 would point at the latter at T=256.
- No deviation from the spec. The review's open question (log the launched thread_m_blocks) was not added: it would need a print/host-side side effect outside the one-knob change.
- moe_align_block_size's host-sized capacity shrinks with block size (review: 18176 -> 5888 slots at T=256), a bundled ~1-2 us side effect.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_decode_layer/marlin-block-m-cap-16 diff` (+3 lines, constant + `min()` on a host int).
- [x] CUDA-graph capturable: the cap is a Python `min` on a host int derived from `x.shape`; no `.item()`/`.cpu()`/sync/prints added; no allocation or control flow on input values.
- [x] Nothing derived from `data` is cached across calls (no state writes added).
- [x] Weights: n/a: weights and `prepare()` untouched; NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries at most FP8 E4M3: unchanged, same dtypes into Marlin, same fp32 reduce.
- [x] Attention sees every key: n/a: attention path untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] First-line marker `# okc: problem=muse_decode_layer` kept.

okc_lint output:
```
warn  host-sync    L247: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_decode_layer/marlin-block-m-cap-16/muse_decode_layer/submission.py
```
`python3 -I -m py_compile` passed; `__pycache__` removed.

## Risks / what could fail
low

## Result (after the run)
(pending)
