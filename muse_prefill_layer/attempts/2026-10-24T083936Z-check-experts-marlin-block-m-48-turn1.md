# muse_prefill_layer · 2026-10-24T083936Z · check · experts-marlin-block-m-48-turn1

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T075422Z-experts-marlin-block-m-48-turn1.md` (branch `hyp/muse_prefill_layer/experts-marlin-block-m-48-turn1`).

- **Job:** `0606f849d2f441ad` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `80dcbc0`

## Goal
Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py): when 512 < num_tokens <= 1024 use block_size_m = 48; every other token count keeps the parity loop's pick unchanged (64 at T=3012 and T=16384, and the parity picks at the tests outside the range). In the scored set this moves only the two turn_1 shapes (T=753, parity pick 64) and in the tests T=600 and T=753. The same block_size_m goes to moe_align_block_size and to the moe_block_size kwarg of both moe_wna16_marlin_gemm calls. Nothing else changes. Mechanism: Marlin's per-m-block cost grows with thread_m_blocks (muse_moe_w4a16 marlin-block-size-m-next-larger measured +18% for 3->4 and +28% for 1->2 at about one block per expert), so 64 -> 48 (thread_m_blocks 4 -> 3) should cut each block's cost, while 48 rows still hold, in one block, nearly every local expert at T=753 (mean ~23.5 rows per local expert, but routing is skewed by the router's alignment with the stream mean and router_bias, so hot experts reach 30-50 rows). It is the sibling of experts-marlin-block-m-local-rows (block 32 at T=753): 32 has the larger per-block saving but sends hot experts to 2 blocks (2 weight passes); 48 trades a smaller per-block saving for fewer passes. Run both; the faster lands and the other is its A/B.

## What changed, and where
- Where: `_experts` in `muse_prefill_layer/submission.py`, directly after the parity loop over `BLOCK_SIZE_M_CANDIDATES`. Three added lines: `if 512 < num_tokens <= 1024: block_size_m = 48` (plus a one-line comment). `num_tokens` is `x.shape[0]`, i.e. total tokens B*L, a shape, so this is a static branch under graph capture.
- The single `block_size_m` variable already feeds `moe_align_block_size(local_ids, block_size_m, N_EXPERTS)` and `common["moe_block_size"]`, which both `moe_wna16_marlin_gemm` calls take, so all three use 48 together. Nothing else changed.
- Before/after picks. The parity loop breaks only when T*8/32/bs < 0.9. That never happens for T >= 231 at bs=64, so master picks 64 for every T here:

  | T (case) | master pick | this branch |
  |---|---|---|
  | 753 (scored turn_1 sliding / full) | 64 | **48** |
  | 3012 (scored turn_4, B=4 x 753) | 64 | 64 |
  | 16384 (scored first_turn, all 4) | 64 | 64 |
  | 300 (test B1 sliding q300) | 64 | 64 |
  | 753 (test B1 ctx3000 full) | 64 | **48** |
  | 1506 (test B2 x 753 sliding) | 64 | 64 |
  | 600 (test B3 x 200 full, outliers) | 64 | **48** |
  | 4096 (test B1 q4096 sliding) | 64 | 64 |

- No deviation from the spec. I did not compute the routing histogram the review's open question asks for, because this node has no GPU or torch.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-marlin-block-m-48-turn1 diff` (+3 lines in `_experts`):
- [x] CUDA-graph capturable: the new branch reads `num_tokens = x.shape[0]`, a shape, not an input value. It adds no `.item()`, `.cpu()`, synchronize or prints.
- [x] Nothing derived from `data` is cached across calls. The diff does not touch `state`.
- [x] Weights: n/a. `prepare()` and the weight bank are untouched, and NVFP4 stays NVFP4.
- [x] Activations: BF16 where SGLang keeps BF16 and the KV cache is FP8. The diff changes only Marlin tiling, not dtypes or the cache.
- [x] Attention sees every key: n/a. Attention is untouched.
- [x] No timing patches, harness access, module edits, reference import, subprocesses or network. The diff adds no imports or calls.
- [x] All work happens inside the timed call. The diff adds no output caching or replay.
- [x] The first-line marker `# okc: problem=muse_prefill_layer` is kept, and there is no triton.autotune.

okc_lint output:
```
warn  host-sync    L261: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: muse_prefill_layer/submission.py
```
The warning comes from the existing line in `prepare()` and is unchanged from master. `py_compile` passed, and I removed the __pycache__ it left.

## Risks / what could fail
low
Same knob as experts-marlin-block-m-local-rows at a different value; the two cannot both land as written (they edit the same pick), so whichever loses becomes the comparison point. 48 is a less common Marlin MoE size (thread_m_blocks 3); if the build's moe_wna16_marlin_gemm has no tmb-3 instantiation for this config it should raise at the first test, which is a clean failure, not a silent fallback.

## Result (after the run)
- job `0606f849d2f441ad` (check), status done, passed True, tests 5/5
- speedup 1.0030x, 95% CI [0.9880, 1.0180], score 386244.9430555745 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.result.json
- **outcome: neutral** (within-ci)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-marlin-block-m-48-turn1.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-marlin-block-m-48-turn1.patch`; branch `hyp/muse_prefill_layer/experts-marlin-block-m-48-turn1`
