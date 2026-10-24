# muse_prefill_layer · 2026-10-24T083755Z · check · experts-marlin-block-m-local-rows

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-24T075409Z-experts-marlin-block-m-local-rows.md` (branch `hyp/muse_prefill_layer/experts-marlin-block-m-local-rows`).

- **Job:** `8e9943ec1b004b0f` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `ed2a571`

## Goal
Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py). The parity loop walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and breaks on the first size where num_tokens * TOP_K / LOCAL_EXPERTS / size < 0.9. num_tokens * 8 / 32 is the routes per local expert only if every route were local; under EP8 only 32 of the 256 experts are local, so the expected local routes per local expert are num_tokens * TOP_K / N_EXPERTS (= T / 32), 8x fewer. At T=753 the parity rule computes 188 rows per expert, never breaks, and picks 64 although each local expert holds ~23.5 rows on average. Replace LOCAL_EXPERTS with N_EXPERTS in that one expression (keep the 0.9 rule and the candidate list). Picks become: T=753 -> 32 (23.5/32 = 0.73 breaks), T=3012 -> 64 (94/64 = 1.47, unchanged), T=16384 -> 64 (unchanged); tests T=300 -> 16, T=600 -> 32, T=753 -> 32, T=1506 -> 64, T=4096 -> 64. The same block_size_m feeds moe_align_block_size and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls, as it must. Nothing else changes. Mechanism (empirical, from muse_moe_w4a16 marlin-block-size-m-next-larger, job 2639e3ca7f5549de): Marlin's per-m-block cost grows with thread_m_blocks even with the m-block count unchanged (+28% for 1->2, +18% for 3->4 at about one block per expert); here 64 -> 32 is thread_m_blocks 4 -> 2 while most local experts (~23 rows) still need one block. In the parity profile (job 62025c368c3c4e8e) Marlin is 730/681 us of the turn_1 shapes (66%/45%) and streams ~0.9 GB of NVFP4 weights + scales at ~1.24 TB/s, i.e. it is far from its 190 us bandwidth floor and not bound by bytes.

## What changed, and where
- Worktree file: `.worktrees/muse_prefill_layer/experts-marlin-block-m-local-rows/muse_prefill_layer/submission.py`, `_experts`, the block_size_m selection loop (base lines 273-275).
- One expression changed: `num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9` becomes `num_tokens * TOP_K / N_EXPERTS / block_size_m < 0.9`. Candidate list (8, 16, 32, 48, 64) and the 0.9 rule unchanged. Added a two-line comment above the loop.
- The single `block_size_m` still feeds `moe_align_block_size(local_ids, block_size_m, N_EXPERTS)` and the `moe_block_size=block_size_m` in the shared `common` kwargs of both `moe_wna16_marlin_gemm` calls (w13 and w2); nothing else touched. The pick branches on `num_tokens` (a shape int) only.
- Picks, before (LOCAL_EXPERTS) -> after (N_EXPERTS), computed offline with the same loop:

  | T (num_tokens) | where | before | after |
  |---|---|---|---|
  | 753 | scored turn_1 (both shapes); test ctx 3000 q 753 | 64 | 32 |
  | 3012 | scored turn_4 (both shapes) | 64 | 64 |
  | 16384 | scored first_turn (all four shapes) | 64 | 64 |
  | 300 | test batch 1 q 300 | 64 | 16 |
  | 600 | test batch 3 q 200 (dist outliers) | 64 | 32 |
  | 1506 | test batch 2 q 753 | 64 | 64 |
  | 4096 | test q 4096 | 64 | 64 |

- Deviation from spec: none. Note the md's "T=300 -> 16, T=600 -> 32" matches; the "before" column is 64 everywhere under the parity rule (T*8/32/64 >= 0.9 for every T >= 231).
- Uncertainty: the T=600 test uses dist outliers, so hot experts there take several 32-row blocks; this is a tiling choice and does not affect correctness.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-marlin-block-m-local-rows diff` (1 file, +3/-1, one expression plus a comment).

- [x] CUDA-graph capturable: the diff adds no `.item()`/`.cpu()`/`synchronize()`/prints; the branch is on `num_tokens` (x.shape[0]), a shape, not on input values.
- [x] Nothing derived from `data` is cached across calls: the diff touches no `state` writes.
- [x] Weights: n/a: diff does not touch `prepare()` or weight storage; NVFP4 stays NVFP4.
- [x] Activations BF16 / KV FP8 / queries <= FP8: n/a: diff changes no dtype; activations stay as before.
- [x] Attention sees every key: n/a: diff is in the experts path only; attention untouched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds no imports or calls of any kind.
- [x] All work happens inside the timed call; no output caching or replay: the pick runs in `_experts` within `kernel_fn` every call.
- [x] First-line marker `# okc: problem=muse_prefill_layer` kept; no triton.autotune added.

Lint (`okc_lint.py ... --problem muse_prefill_layer`):

```
warn  host-sync    L261: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: submission.py
```

`python3 -I -m py_compile` OK; the `__pycache__` it left was deleted.

## Risks / what could fail
low
Routing is skewed (router rows aligned with the stream mean, router_bias std 0.055, reference.py _router), so hot local experts at T=753 can exceed 32 rows and need 2 blocks of 32 (2 weight passes at thread_m_blocks 2); if that offsets the per-block saving turn_1 comes in flat, and the sibling experts-marlin-block-m-48-turn1 (one block up to 48 rows) is the comparison. The extrapolation from the sister's two points (tmb 1->2 and 3->4 at tiny M) to tmb 4->2 at ~23 real rows is unmeasured. Tests T=300/600 also change block (16/32); correctness is unaffected (block size is a tiling choice).

## Result (after the run)
- job `8e9943ec1b004b0f` (check), status done, passed True, tests 5/5
- speedup 0.9836x, 95% CI [0.9688, 0.9983], score 393882.5052353208 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.result.json
- **outcome: regression** (regression)
- files: `muse_prefill_layer/hypotheses/done/regression/experts-marlin-block-m-local-rows.result.json`, `muse_prefill_layer/hypotheses/done/regression/experts-marlin-block-m-local-rows.patch`; branch `hyp/muse_prefill_layer/experts-marlin-block-m-local-rows`
