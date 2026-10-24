# muse_bf16_proj · 2026-10-24T104355Z · check · router-splitk32-small-m-all-loads-in-flight-control-stages3

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T103959Z-router-splitk32-small-m-all-loads-in-flight-control-stages3.md` (branch `hyp/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight-control-stages3`).

- **Job:** `4685bac7a662454d` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `3a8ae33`

## Goal
From the parent's branch (router-splitk32-small-m-all-loads-in-flight), keep everything exactly as the parent: the new m<=16 bucket (BLOCK_M=16, BLOCK_N=32, SPLIT_K=32) and 17..32 bucket (BLOCK_M=32, BLOCK_N=32, SPLIT_K=32), grid (1, 8, 32) = 256 CTAs, K-slice 256, BLOCK_K=128, num_warps=4, the torch.zeros output with the relaxed fp32 atomic_add epilogue, the unchanged m>32 buckets, and _router_config returning (BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES). Change one thing: the two small-m buckets return NUM_STAGES=3 (master's ROUTER_NUM_STAGES) instead of 2, so every bucket launches with num_stages=3. ROUTER_WARM_MS is unchanged and still compiles both specializations in prepare().

## What changed, and where
Implemented inline by churnsubs on the parent's commit (60622cd): `ROUTER_SMALL_M_NUM_STAGES` 2 -> 3, so the m <= 32 buckets keep SPLIT_K=32 (grid 1x8x32) with master's num_stages=3. Nothing else differs from the parent. No deviation.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight-control-stages3 diff` (one constant).

- [x] CUDA-graph capturable: the edit is on config values only (a module constant); no `.item()`/`.cpu()`/`synchronize()` added; no control flow on input values.
- [x] Nothing derived from `data` is cached across calls: no state writes added.
- [x] Weights: n/a: untouched.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8 and queries FP8 E4M3: unchanged by this edit (both paths already exist on the parent and passed review).
- [x] Attention sees every key: n/a: no attention in this problem.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: none added.
- [x] All work happens inside the timed call; no output caching or replay: unchanged.
- [x] First line marker kept; no triton.autotune added.

okc_lint output:

```
CLEAN: .worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight-control-stages3/muse_bf16_proj/submission.py
```

## Risks / what could fail
low
Separates the SPLIT_K 16->32 grid-depth effect (control vs master) from the num_stages 3->2 effect (parent vs control). Without it, a loss could come from either the atomics or the stage change, and a gain cannot tell the semaphore-reduce follow-up which SPLIT_K to build on.

## Result (after the run)
- job `4685bac7a662454d` (check), status done, passed True, tests 9/9
- speedup 1.1518x, 95% CI [1.1345, 1.1691], score 4194.106092921194 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/neutral/router-splitk32-small-m-all-loads-in-flight.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-splitk32-small-m-all-loads-in-flight-control-stages3.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-splitk32-small-m-all-loads-in-flight-control-stages3.patch`; branch `hyp/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight-control-stages3`
