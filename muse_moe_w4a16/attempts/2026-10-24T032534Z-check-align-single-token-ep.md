# muse_moe_w4a16 · 2026-10-24T032534Z · check · align-single-token-ep

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-23T225502Z-align-single-token-ep.md` (branch `hyp/muse_moe_w4a16/align-single-token-ep`).

- **Job:** `a7bb0cecaf8d4de1` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `626e602`

## Goal
At num_tokens == 1 only, call SGLang's moe_align_single_token(local_ids, block_size_m) (from sglang.kernels.ops.moe.moe_align_single_token) with the expert_map-mapped ids (-1 for remote experts) instead of moe_align_block_size. baseline.py takes this path only when expert_map is None (the 'full' sharding), never under EP; this hypothesis tests whether the single-token kernel tolerates -1 ids and is cheaper than the general align (which at 256 or 32 experts runs a histogram + cumsum + scatter over a padded sorted_ids buffer) for the 8 ids of one token. Its outputs (sorted_ids, block_experts, n_padded) feed both Marlin GEMMs unchanged; all other shapes keep moe_align_block_size.

## What changed, and where
- Worktree `.worktrees/muse_moe_w4a16/align-single-token-ep`, file `muse_moe_w4a16/submission.py` only (diff stat: 1 file, +19 / -3).
- kernel_fn align branch: `if num_tokens == 1 and expert_map is None:` became `if num_tokens == 1 and moe_align_single_token is not None:`. The branch is shape-only (num_tokens) plus a module-level constant; it now fires under ep8. It is called exactly as baseline.py calls it, `moe_align_single_token(local_ids, block_size_m)`, on the expert_map-mapped ids (-1 for remote experts). sorted_ids / block_experts / n_padded go straight into both Marlin GEMMs unchanged; n_padded is never read on the host. All other shapes keep `moe_align_block_size(local_ids, block_size_m, bank["count"])`.
- Import hoisted from inside kernel_fn to module level, guarded with try/except ImportError (per review). On success it prints `[align-single-token-ep] bs=1 path: moe_align_single_token`; on failure it sets the name to None, prints `... unavailable (<err>); bs=1 path: moe_align_block_size (fallback)`, and bs=1 runs master's path. Check log_tail for which line appears: a fallback run is a master re-run on bs=1 and says nothing about the hypothesis.
- Docstring gained one paragraph describing the change.
- Deviation: none from the change paragraph. Not verified (no SGLang source or GPU on this node): whether the kernel tolerates -1 ids and whether it compacts or emits -1 blocks. The bs=1 pre-timing test covers correctness; the bs=1 per-kernel profile separates align time from Marlin m-block cost (see review).

## Rules check (each must be true before the run)
- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints in kernel_fn; the new branch is on `num_tokens` (shape) and a module constant; n_padded stays on device. The prints are at import time, strings only.
- [x] Nothing derived from `data` is cached across calls (no new state; module global is set at import from an import result only).
- [x] Weights: unchanged; prepare() untouched, NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged; only the int32 align outputs differ. KV cache / queries: n/a, no attention in this problem.
- [x] Attention sees every key: n/a, MoE problem with no attention.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the only addition is an import of an SGLang op baseline.py itself imports.
- [x] All work happens inside the timed call; no output caching or replay.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune added.

Lint (`okc_lint.py ... --problem muse_moe_w4a16`, exit 0):

```
warn  host-sync    L126: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_moe_w4a16/align-single-token-ep/muse_moe_w4a16/submission.py
```

The warning is the pre-existing synchronize() in prepare(), untouched by this diff. `py_compile` OK; __pycache__ removed.

## Risks / what could fail
medium
Risks: if moe_align_single_token does not skip -1 ids, the bs=1 pre-timing test fails outright (a cheap, informative failure: log it as a do-not-retry). Superseded on bs=1 if triton-w2-grouped-gemv-small-m lands, since that removes the align from the small-M path; still worth a run as a low-cost probe of SGLang's own fast path. No rule exposure.

## Result (after the run)
- job `a7bb0cecaf8d4de1` (check), status done, passed True, tests 5/5
- speedup 0.9951x, 95% CI [0.9802, 1.0101], score 13514.401146044253 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/align-single-token-ep.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/align-single-token-ep.patch`; branch `hyp/muse_moe_w4a16/align-single-token-ep`
