# muse_bf16_proj · 2026-10-24T104146Z · check · router-splitk32-small-m-all-loads-in-flight

Hypothesis: `muse_bf16_proj/hypotheses/running/2026-10-24T075025Z-router-splitk32-small-m-all-loads-in-flight.md` (branch `hyp/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight`).

- **Job:** `1e339074b1844a43` (`okc status <job>`)
- **File:** `muse_bf16_proj/submission.py` at git `60622cd`

## Goal
Built on master (dba4ffd). Change only the two small-m buckets of _router_config and the num_stages they launch with; the kernel body, the torch.zeros output + relaxed fp32 atomic_add epilogue, BLOCK_K=128, num_warps=4, and the m > 32 buckets (64,64,16) / (64,64,8) with num_stages=3 stay exactly as master. New small-m buckets: m <= 16: BLOCK_M=16, BLOCK_N=32, SPLIT_K=32; 17..32: BLOCK_M=32, BLOCK_N=32, SPLIT_K=32. Grid becomes (1, 8, 32) = 256 CTAs (was 128), K-slice 256 = 2 K-steps of BLOCK_K=128 (was 4), and these buckets launch with num_stages=2 so both K-steps' hi, lo and x tiles are issued before the first dot (per CTA ~40 KB smem, so ~2 CTAs per SM are co-resident and all 256 run in one wave). Implement by having _router_config return (BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES) and passing num_stages from it; _router_shape_ok's k % (128 * SPLIT_K) check still holds (8192 % 4096 == 0). Keep ROUTER_WARM_MS so both new specializations compile in prepare().

## What changed, and where
- Worktree file: `.worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight/muse_bf16_proj/submission.py` (base blob da55301). Diff: 1 file, 12 insertions, 11 deletions.
- `_router_config(m)` now returns `(BLOCK_M, BLOCK_N, SPLIT_K, NUM_STAGES)`:
  - m <= 16: `(16, 32, 32, 2)`; 17..32: `(32, 32, 32, 2)`. Both use the new constant `ROUTER_SMALL_M_NUM_STAGES = 2`. Grid is (1, 8, 32) = 256 CTAs, K_SLICE = 256, so 2 K-steps of BLOCK_K = 128.
  - m <= 128: `(64, 64, 16, 3)`; m > 128: `(64, 64, 8, 3)`. These are unchanged from master and use `ROUTER_NUM_STAGES = 3`.
  - Docstring updated to match.
- `_router_forward` unpacks `num_stages` and passes `num_stages=num_stages` to the launch, replacing the global `ROUTER_NUM_STAGES`.
- `_router_shape_ok` unpacks the 4-tuple and ignores the stages value. Its `k % (128 * SPLIT_K)` check holds: 8192 % 4096 == 0.
- Unchanged: the kernel body, the torch.zeros output with the relaxed fp32 atomic_add epilogue, BLOCK_K, num_warps=4, and the bf16 layers. `ROUTER_WARM_MS = (1, 8, 16, 32, 64, 128, 256)` is also unchanged, so prepare() compiles both new (K_SLICE=256, stages=2) specializations at m=1/8/16 and m=32.
- No deviation from the spec. The reviewer's open questions (whether Triton pipelines a 2-iteration loop at num_stages=2, occupancy and wave count, kernel-us vs memset) are about reading the results and cannot be checked on this node. Read them from the profile. The stages=3 control is queued separately as `router-splitk32-small-m-all-loads-in-flight-control-stages3`.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight diff`:

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints. The new branch picks num_stages from `m` (a shape) only. No new allocation; the zeroed output is the same as master.
- [x] Nothing derived from `data` is cached across calls. The diff adds no state writes.
- [x] Weights: unchanged. The router is still the BF16 hi + lo split built in prepare(), with FP32 accumulation. The bf16 weights are untouched.
- [x] Activations BF16 where SGLang keeps BF16. x is still bf16 into the kernel, and the router output is fp32 as before.
- [x] n/a: attention. This problem has no attention, KV cache or queries; it is dense projections only.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The diff touches only launch-config constants.
- [x] All work happens inside the timed call, with no output caching or replay. The memset and kernel run on every call, and prepare() only warms compiles on synthetic x.
- [x] First line is still `# okc: problem=muse_bf16_proj`. No triton.autotune.

okc_lint output:

```
CLEAN: /home/user/git/musespark/.worktrees/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight/muse_bf16_proj/submission.py
```

`python3 -I -m py_compile` passed. The `__pycache__` it left in the worktree was removed.

## Risks / what could fail
low
A pure one-axis retune (grid depth x pipeline depth) of the landed kernel on the four m <= 32 router rows; m=128/256 rows are in-run controls for noise. The idea is memory-level parallelism, not tile shape: master's 128 CTAs each hold ~3 x 16 KB of weight in flight for only 4 steps, so each CTA's life is ~2 DRAM round trips plus launch; doubling the CTA count and halving the steps puts the whole 8.4 MB in flight at once. Costs: atomic traffic doubles (m=32: 262K fp32 atomics, ~1 MB, into a 32 KB output that stays in L2), and x is re-read by 8 N-blocks (trivial at m <= 32). Per-shape router effect is ~10-15% on 4 shapes but router is ~11% of each decode load, so geomean ~+1%: read the per-shape rows and the kernel us, not the headline. If flat, the router's small-m floor is launch + memset, and router-splitk-semaphore-no-memset is the remaining lever. Stacking: independent of the memset hypothesis (different lines); if both gain, combine and re-run.

## Result (after the run)
- job `1e339074b1844a43` (check), status done, passed True, tests 9/9
- speedup 1.1427x, 95% CI [1.1256, 1.1599], score 4227.423420381475 µs vs SGLang 4830.8 µs
- best at harvest: {"speedup": 1.1432568936693237, "ci": [1.1261080402642838, 1.1604057470743634], "job": "240eb22ea0b6487e"}; base result: muse_bf16_proj/hypotheses/done/gain/router-triton-hilo-fused-control-splitk-all-m.result.json
- **outcome: neutral** (within-ci)
- files: `muse_bf16_proj/hypotheses/done/neutral/router-splitk32-small-m-all-loads-in-flight.result.json`, `muse_bf16_proj/hypotheses/done/neutral/router-splitk32-small-m-all-loads-in-flight.patch`; branch `hyp/muse_bf16_proj/router-splitk32-small-m-all-loads-in-flight`
