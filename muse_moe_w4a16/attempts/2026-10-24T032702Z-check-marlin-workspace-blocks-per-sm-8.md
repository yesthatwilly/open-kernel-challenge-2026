# muse_moe_w4a16 · 2026-10-24T032702Z · check · marlin-workspace-blocks-per-sm-8

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-23T225502Z-marlin-workspace-blocks-per-sm-8.md` (branch `hyp/muse_moe_w4a16/marlin-workspace-blocks-per-sm-8`).

- **Job:** `75805dc2f18a4d06` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `8274bfd`

## Goal
In prepare(), build a second lock workspace marlin_make_workspace(device, max_blocks_per_sm=8) and pass it to the w13 moe_wna16_marlin_gemm call only; the w2 call keeps the max_blocks_per_sm=4 workspace SGLang's MoE runner uses (its dense runner uses 1). The Marlin runner sizes the number of resident CTAs it launches from the workspace's lock count (sms * max_blocks_per_sm), capped by what the kernel config's shared memory allows, so a larger workspace lets more CTAs be resident and more k-slices run in parallel for the verify shapes where the w13 grid (2048 rows x 8192 cols) has work to spare. Nothing else changes.

## What changed, and where
Implemented 2026-10-24 in the worktree's `muse_moe_w4a16/submission.py` (diff: 1 file, +4/-1).

- `prepare()`: adds `state["workspace_w13"] = marlin_make_workspace(device, max_blocks_per_sm=8)` beside the existing `state["workspace"]` (`max_blocks_per_sm=4`, unchanged). Built once, untimed, depends on nothing in `data`.
- `kernel_fn()`: the w13 `moe_wna16_marlin_gemm` call passes `state["workspace_w13"]` as its workspace; the w2 call still passes `ws = state["workspace"]` (4 per SM), the built-in per-call control. Nothing else changes: same block_size_m choice, align, caches, flags, silu_and_mul and post-norm sum.
- No deviation from the spec.
- Expectation (per the adversarial review): the reviewer expects this to be **inert** on every shape. On their reading of the upstream Marlin MoE runner, the launch grid is sized by the kernel's exec config (`blocks = sms * blocks_per_sm`, with `blocks_per_sm = 1` when `thread_m_blocks > 1`, i.e. bs=128/256, and `min(allow_count, 4)` otherwise), and the workspace is only size-checked against that grid and used as the lock pool. A flat result should be read as "workspace size is not a lever", not as "shared memory caps residency"; do not spend the 2/16 ab-tune unless a verify-shape gain clears the CI.
- Note for the attempt file: the 620/467 us figures in the hypothesis are w13+w2 combined (the okc profile reports both Marlin launches together, calls=2).
- Not implemented (suggested follow-up, a separate idea): the reviewer's zero-cost prepare()-time probe, i.e. call `moe_wna16_marlin_gemm` once on dummy bs=4 and bs=128 inputs with a 1-per-SM workspace inside try/except and print the exception text, which would state the required workspace size (= blocks launched) per regime and settle the grid question without a timed run.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/marlin-workspace-blocks-per-sm-8 diff`.

- [x] CUDA-graph capturable: the diff only swaps which prebuilt state tensor the w13 call receives; no `.item()`/`.cpu()`/`synchronize()`/tensor prints, no allocation or control flow on input values added in kernel_fn. (The existing `torch.cuda.synchronize()` is in prepare(), unchanged.)
- [x] Nothing derived from `data` is cached across calls: the new workspace is built in prepare() from the device only.
- [x] Weights: unchanged; repacking still in prepare() via prepare_moe_nvfp4_layer_for_marlin, NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: unchanged (n/a for KV cache / queries: no attention in this problem).
- [x] n/a: attention-key rule does not apply (MoE problem, no attention); routing still computes every local route.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds none.
- [x] All work happens inside the timed call; no output caching or replay: the workspace is a zeroed lock pool reset by the kernel, not an output.
- [x] First line `# okc: problem=muse_moe_w4a16` kept; no triton.autotune added.

Lint (`okc_lint.py ... --problem muse_moe_w4a16`):

```
warn  host-sync    L110: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_moe_w4a16/marlin-workspace-blocks-per-sm-8/muse_moe_w4a16/submission.py
```

`python3 -I -m py_compile`: OK (the `__pycache__` it left was deleted).

## Risks / what could fail
low
Risks: the kernel may already cap residency by shared memory (Marlin stages use most of the 228 KB) so the knob is inert and the result flat; a larger lock array is harmless (zeroed once in prepare, reset by the kernel). If flat, record that occupancy is not Marlin's limiter at these shapes; if a gain, the ab-tune is 2 and 16, and the same workspace on w2.

## Result (after the run)
- job `75805dc2f18a4d06` (check), status done, passed True, tests 5/5
- speedup 0.9921x, 95% CI [0.9772, 1.0070], score 13556.004026602383 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/marlin-workspace-blocks-per-sm-8.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/marlin-workspace-blocks-per-sm-8.patch`; branch `hyp/muse_moe_w4a16/marlin-workspace-blocks-per-sm-8`
