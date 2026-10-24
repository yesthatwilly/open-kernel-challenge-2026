---
problem: muse_prefill_layer
slug: experts-marlin-block-m-local-rows
created: 2026-10-24T075409Z
base: master
area: w13 marlin + w2 marlin (block_size_m pick)
change: >
  Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py). The parity
  loop walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and breaks on the first size where
  num_tokens * TOP_K / LOCAL_EXPERTS / size < 0.9. num_tokens * 8 / 32 is the routes per local
  expert only if every route were local; under EP8 only 32 of the 256 experts are local, so the
  expected local routes per local expert are num_tokens * TOP_K / N_EXPERTS (= T / 32), 8x fewer.
  At T=753 the parity rule computes 188 rows per expert, never breaks, and picks 64 although each
  local expert holds ~23.5 rows on average. Replace LOCAL_EXPERTS with N_EXPERTS in that one
  expression (keep the 0.9 rule and the candidate list). Picks become: T=753 -> 32 (23.5/32 = 0.73
  breaks), T=3012 -> 64 (94/64 = 1.47, unchanged), T=16384 -> 64 (unchanged); tests T=300 -> 16,
  T=600 -> 32, T=753 -> 32, T=1506 -> 64, T=4096 -> 64. The same block_size_m feeds
  moe_align_block_size and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls, as it
  must. Nothing else changes. Mechanism (empirical, from muse_moe_w4a16
  marlin-block-size-m-next-larger, job 2639e3ca7f5549de): Marlin's per-m-block cost grows with
  thread_m_blocks even with the m-block count unchanged (+28% for 1->2, +18% for 3->4 at about one
  block per expert); here 64 -> 32 is thread_m_blocks 4 -> 2 while most local experts (~23 rows)
  still need one block. In the parity profile (job 62025c368c3c4e8e) Marlin is 730/681 us of the
  turn_1 shapes (66%/45%) and streams ~0.9 GB of NVFP4 weights + scales at ~1.24 TB/s, i.e. it is
  far from its 190 us bandwidth floor and not bound by bytes.
expected:
  - shapes: "*load=turn_1*"
    direction: faster
    why: Marlin pair 730/681 us at block 64 (thread_m_blocks 4) on ~23.5 rows per local expert; block 32 halves thread_m_blocks with ~1 block per expert for most experts; sister data predicts ~20-30% off Marlin, i.e. ~10-15% per turn_1 shape
  - shapes: "*load=turn_4*"
    direction: flat
    why: T=3012 keeps 64 under both rules (control)
  - shapes: "*load=first_turn*"
    direction: flat
    why: T=16384 keeps 64 under both rules (control)
risk: low
depends_on: null
priority: 1
kind: primary
controls_for: null
review: proceed
status: done
review_at: 2026-10-24T083634Z
review_companions: null
stage: harvested
claimed_at: 2026-10-24T083634Z
base_sha: 67649c559d5a07d10b659c84bf80c1b1ec29c879
base_blob: e4931f050c7fb99835fe116d25bb454a5ea731e9
branch: hyp/muse_prefill_layer/experts-marlin-block-m-local-rows
worktree: .worktrees/muse_prefill_layer/experts-marlin-block-m-local-rows
mode: check
implemented_at: 2026-10-24T083755Z
impl_commit: ed2a571cf7bccd94cf9148aed7baa997bbe73c57
submission_sha256: 6d80a359d004ca1f07e7d4192af251108ed4a76502c1cb56ceb7b1df459c824e
norm_sha256: 14334b52ebd2b5045e65bf21d09f43476099101511d70f70b31dc07d8fa4cf5f
submission_blob: 3d09290ced5f8f540573ed73c5f64b3aaeb0cc99
lint_warnings: 1
attempt_file: 2026-10-24T083755Z-check-experts-marlin-block-m-local-rows.md
prepared_at: 2026-10-24T083755Z
job: 8e9943ec1b004b0f
dispatched_at: 2026-10-24T083825Z
outcome: regression
verdict_detail: regression
speedup: 0.9835519345256266
speedup_ci: [0.9687986555077422, 0.9983052135435109]
harvested_at: 2026-10-24T095814Z
best_at_harvest: 1.0038664293875537
promotion: null
---

# experts-marlin-block-m-local-rows

## Change

Change only the block_size_m pick in _experts (muse_prefill_layer/submission.py). The parity loop walks BLOCK_SIZE_M_CANDIDATES (8, 16, 32, 48, 64) and breaks on the first size where num_tokens * TOP_K / LOCAL_EXPERTS / size < 0.9. num_tokens * 8 / 32 is the routes per local expert only if every route were local; under EP8 only 32 of the 256 experts are local, so the expected local routes per local expert are num_tokens * TOP_K / N_EXPERTS (= T / 32), 8x fewer. At T=753 the parity rule computes 188 rows per expert, never breaks, and picks 64 although each local expert holds ~23.5 rows on average. Replace LOCAL_EXPERTS with N_EXPERTS in that one expression (keep the 0.9 rule and the candidate list). Picks become: T=753 -> 32 (23.5/32 = 0.73 breaks), T=3012 -> 64 (94/64 = 1.47, unchanged), T=16384 -> 64 (unchanged); tests T=300 -> 16, T=600 -> 32, T=753 -> 32, T=1506 -> 64, T=4096 -> 64. The same block_size_m feeds moe_align_block_size and the moe_block_size kwarg of both moe_wna16_marlin_gemm calls, as it must. Nothing else changes. Mechanism (empirical, from muse_moe_w4a16 marlin-block-size-m-next-larger, job 2639e3ca7f5549de): Marlin's per-m-block cost grows with thread_m_blocks even with the m-block count unchanged (+28% for 1->2, +18% for 3->4 at about one block per expert); here 64 -> 32 is thread_m_blocks 4 -> 2 while most local experts (~23 rows) still need one block. In the parity profile (job 62025c368c3c4e8e) Marlin is 730/681 us of the turn_1 shapes (66%/45%) and streams ~0.9 GB of NVFP4 weights + scales at ~1.24 TB/s, i.e. it is far from its 190 us bandwidth floor and not bound by bytes.

## Expected

- `*load=turn_1*` faster: Marlin pair 730/681 us at block 64 (thread_m_blocks 4) on ~23.5 rows per local expert; block 32 halves thread_m_blocks with ~1 block per expert for most experts; sister data predicts ~20-30% off Marlin, i.e. ~10-15% per turn_1 shape
- `*load=turn_4*` flat: T=3012 keeps 64 under both rules (control)
- `*load=first_turn*` flat: T=16384 keeps 64 under both rules (control)

## Risk

Routing is skewed (router rows aligned with the stream mean, router_bias std 0.055, reference.py _router), so hot local experts at T=753 can exceed 32 rows and need 2 blocks of 32 (2 weight passes at thread_m_blocks 2); if that offsets the per-block saving turn_1 comes in flat, and the sibling experts-marlin-block-m-48-turn1 (one block up to 48 rows) is the comparison. The extrapolation from the sister's two points (tmb 1->2 and 3->4 at tiny M) to tmb 4->2 at ~23 real rows is unmeasured. Tests T=300/600 also change block (16/32); correctness is unaffected (block size is a tiling choice).

## Adversarial review

Reviewed 2026-10-24T083634Z by an adversarial subagent (model opus).

**Verdict: proceed.** A one-expression, shape-only change that moves block_size_m 64->32 at T=753 only. It is well justified by the 8x local-row overcount (the same insight that won on decode, marlin-block-m-cap-16) and by Marlin dominating turn_1. The untouched turn_4 and first_turn shapes act as built-in controls, the sibling experts-marlin-block-m-48-turn1 is the natural A/B, and no extra companion is needed.

### Justification
Arithmetic checks out against submission.py lines 273-276. Under the parity rule, T*8/32/size never drops below 0.9 at T=753, so 64 is picked. With N_EXPERTS the per-size values are 23.5/8=2.9, /16=1.47 and /32=0.73, so the loop breaks at 32. T=1506 still breaks at 64 (0.735). T=3012 (1.47) and T=16384 fall through to 64. The scored set therefore changes only the two turn_1 shapes (T=753), and only Python-side selection on a shape int changes. Profile (job 62025c368c3c4e8e): the Marlin pair is 730.1/680.7 us of the turn_1 shapes (66%/45%), and the floors are 257/497 us. In turn_1-load terms, Marlin is about 33.6 ms + 10.9 ms of about 72.6 ms weighted, i.e. 61% of that load. Since the geomean is over three loads, a 20-30% Marlin cut is roughly 12-18% on the turn_1 load and roughly 4-6% on the headline. That is well above the 1.5-2.5% run-to-run drift, so one run can resolve it. Cross-shape scaling supports a padded-work-bound Marlin rather than a byte-bound one. Turn_1 (T=753) costs 730 us against 2248 us at T=3012: 4x the tokens for only about 3x the time. Shrinking the 64-row padding (about 2048 padded rows for about 750 real local routes, under 40% MMA utilisation) should therefore help, and decode's cap-16 result (+8-15% at T=128/256) shows the same mechanism. Weakest link: the cost model. Halving block size halves MMA rows per block, but dequant work per block is unchanged, and every expert pushed past 32 rows by skew (router bias std 0.055, 'soft' dist; Poisson alone gives about 4% of experts above 32 at mean 23.5, and skew makes it more) pays a second full weight pass with a second dequant. If Marlin at this M is dequant/ALU-bound rather than MMA-bound (as LESSONS found for the Triton W4A16 GEMV), the saving shrinks and the 2-block experts can cancel it. The 20-30% figure is also extrapolated from the sister's tiny-M points (tmb 1->2, 3->4) to tmb 4->2 at about 23 real rows, which is unmeasured. Expected directions match TASK.md: turn_4 and first_turn flat, turn_1 faster.

### Confounders on the current kernel
- sglang-pipeline-parity: it is the base that introduced the parity loop being edited. It does not confound, because baseline.py (lines 243-246) uses the identical rule, so the same-run SGLang time is an exact paired control for the old pick. Reverting it makes no sense, because it is the thing being modified.
- No other landed hypothesis touches the experts path (moe-align-side-stream is neutral and not on master), so a turn_1 Marlin delta attributes cleanly to block_size_m.

### Interference and A/B needs
No companion is warranted. Only one landed item exists and it is the base. The change is a single variable (block_size_m at T=753), and turn_4 and first_turn (picks unchanged) are in-run controls for drift. A gain shows that 32 beats 64 at T=753, and a loss shows that 2-block hot experts or per-block dequant dominate. Either outcome can be read from the Marlin kernel's own top_kernels time on the two turn_1 rows. The only real ambiguity, whether 48 would be better, is already covered by the queued sibling experts-marlin-block-m-48-turn1, which should run on the same base so the two are an A/B pair. An ab-tune at block 16 would only answer a further question, not disambiguate this result.

### Open questions
- What is the actual distribution of local rows per expert at T=753 under the 'soft' dist? Count it offline, outside the timed call, from reference.generate_input routing (or from n_padded at block 32 vs 64). It tells how many experts take 2 blocks at 32 and whether 48 is the better pick.
- The profile merges both Marlin calls (count 2). Is the saving split between w13 (size_n=2I) and w2 differently? The profile cannot show this, so read only the combined Marlin time.
- If experts-triton-w4a16-inkernel-fp4 (running) replaces Marlin on the turn shapes, this pick becomes moot for turn_1. Whichever lands second should be rebased onto the winner.
- Turn_4 (T=3012, 94 rows/expert) is left at 64 by this rule even though 48 fits 2 blocks of 96 almost exactly. That belongs to experts-marlin-block-m-48-turn4; do not fold it in here.


## Implementation notes

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

## Rules check

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

## Result

- job `8e9943ec1b004b0f` (check), status done, passed True, tests 5/5
- speedup 0.9836x, 95% CI [0.9688, 0.9983], score 393882.5052353208 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.0038664293875537, "ci": [0.9888084329467404, 1.018924425828367], "job": "62025c368c3c4e8e"}; base result: muse_prefill_layer/hypotheses/done/gain/sglang-pipeline-parity.result.json
- **outcome: regression** (regression)
- files: `muse_prefill_layer/hypotheses/done/regression/experts-marlin-block-m-local-rows.result.json`, `muse_prefill_layer/hypotheses/done/regression/experts-marlin-block-m-local-rows.patch`; branch `hyp/muse_prefill_layer/experts-marlin-block-m-local-rows`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 1078.66 | 1134.07 | 0.9475 | lost |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1434.85 | 1558.75 | 0.9491 | lost |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 3367.03 | 3283.49 | 0.9984 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 4949.82 | 4960.64 | 0.9971 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18279.1 | 17166.51 | 0.9957 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 16995.2 | 17319.99 | 0.9893 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 21535.66 | 21923.55 | 1.0059 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 25286.61 | 24648.64 | 1.0122 | flat |

lost on sliding turn_1; full turn_1. Expected hit: 2/3. Noise ±1.5%.

## Follow-ups

- record which shapes lost (Attribution) and do not retry this area in this form.

