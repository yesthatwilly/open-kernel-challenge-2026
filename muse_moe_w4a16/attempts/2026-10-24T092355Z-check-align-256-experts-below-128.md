# muse_moe_w4a16 · 2026-10-24T092355Z · check · align-256-experts-below-128

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-24T075232Z-align-256-experts-below-128.md` (branch `hyp/muse_moe_w4a16/align-256-experts-below-128`).

- **Job:** `e753d5b9df644ebb` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `245814a`

## Goal
In kernel_fn, for num_tokens < 128 only (bs=1, 8, 16, 32 and the bs=4/33 tests), call moe_align_block_size(local_ids, block_size_m, N_EXPERTS) exactly as baseline.py does (256 experts, the expert_map-mapped ids with -1 for remote experts unchanged), instead of master's moe_align_block_size(local_ids, block_size_m, bank['count']) (32). bs=128/256 keep master's 32-expert call as the in-run control. Nothing else changes: same block_size_m pick, same Marlin calls, silu_and_mul, post-norm. Why: master is 3.2 us slower than SGLang at bs=1 (80.2 vs 77.0 us, err 0.48 us, ~4%) and 1.5-2.6 us slower at bs=8/16/32, while it sits at parity at bs=128/256; the only GPU-side difference from baseline.py on master is the 32-expert align, so it is the prime suspect for the small-M deficit. Supporting datapoint: align-single-token-ep (neutral by geomean) replaced exactly this align at bs=1 and its decode_1 shape went to 0.999x (76.99 vs 76.91 us) against master's 0.960x, a per-shape move well outside the 0.5 us error that the geomean verdict hid. A plausible mechanism (unverified; no sgl_kernel source here) is that the align kernel's handling of -1 ids depends on num_experts (e.g. a signed index into a 32-entry shared counter array lands remote routes in a local bucket, adding a phantom m-block and one extra expert's weight pass to both Marlin GEMMs), so the attempt file should record, from the profile, the align kernel name and the Marlin time at bs=1 under each setting.

## What changed, and where
- File: `.worktrees/muse_moe_w4a16/align-256-experts-below-128/muse_moe_w4a16/submission.py`, `kernel_fn` only (plus one docstring sentence).
- Added an `elif num_tokens < 128:` branch between the existing `num_tokens == 1 and expert_map is None` single-token branch (unchanged, dead under ep8) and master's 32-expert `else`. It calls `moe_align_block_size(local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])`, which is baseline.py's call verbatim (under ep8 expert_map is always set, so this is N_EXPERTS=256 with the expert_map-mapped ids, -1 for remote, unchanged).
- bs>=128 keeps master's `moe_align_block_size(local_ids, block_size_m, bank["count"])` (drift control only, per the review: those shapes already use the regular kernel).
- Unchanged: block_size_m pick (still from bank["count"], as baseline.py), both Marlin calls and their args, silu_and_mul, the post-norm weighted sum. The branch is on `num_tokens` (a shape), not on input values.
- Deviation from spec: none. Per the adversarial review, the attempt file should record per shape the align-related kernels (name, calls, GPU time) instead of Marlin time, and note whether the 256-expert path at M<128 launches one kernel or two and whether it beats the small-batch kernel's 4.4-6.7 us.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/align-256-experts-below-128 diff` (1 file, +8/-1).

- [x] CUDA-graph capturable: no `.item()`/`.cpu()`/`synchronize()`/tensor prints added; the new branch is on `num_tokens` (shape), no allocation or control flow on input values. The only `synchronize()` is the pre-existing one in prepare().
- [x] Nothing derived from `data` is cached across calls: the diff writes nothing to `state`.
- [x] Weights: untouched by the diff; prepare() unchanged (NVFP4 stays NVFP4 via SGLang's lossless Marlin repack).
- [x] Activations BF16 where SGLang keeps BF16: no dtype changes in the diff. KV cache / queries: n/a: no attention in this problem.
- [x] Attention sees every key: n/a: MoE problem, no attention; no expert/route skipping introduced (the 256-expert align is exactly SGLang's).
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds only one moe_align_block_size call already imported.
- [x] All work happens inside the timed call; no output caching or replay: the align runs every call inside kernel_fn.
- [x] First-line marker `# okc: problem=muse_moe_w4a16` intact; no triton.autotune.
- [x] `python3 -I -m py_compile` passes; the __pycache__ it left in the worktree was deleted.

Lint (`okc_lint.py ... --problem muse_moe_w4a16`, exit 0):

```
warn  host-sync    L111: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_moe_w4a16/align-256-experts-below-128/muse_moe_w4a16/submission.py
```

## Risks / what could fail
low
This partially reverts the parity hypothesis's one structural change, on the shapes where it measured below SGLang; it can only bring those shapes to parity (identical to baseline.py there), so its value is a clean >=1.0 base to stack post-expert-norm-triton-small-m and block-size work on, plus the answer to why bs=1 lost 3.2 us. If decode_1 does not move, the 3.2 us is not the align and the next suspect is the Marlin call itself (block_experts length) or run drift. Low risk: the 256-expert call is what the judge's SGLang column runs every time on the same inputs.

## Result (after the run)
- job `e753d5b9df644ebb` (check), status done, passed True, tests 5/5
- speedup 0.9998x, 95% CI [0.9848, 1.0147], score 13451.839641703164 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_moe_w4a16/hypotheses/done/neutral/align-256-experts-below-128.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/align-256-experts-below-128.patch`; branch `hyp/muse_moe_w4a16/align-256-experts-below-128`
