# muse_moe_w4a16 · 2026-10-24T040655Z · check · w13-marlin-bf16-reduce-verify

Hypothesis: `muse_moe_w4a16/hypotheses/running/2026-10-23T225502Z-w13-marlin-bf16-reduce-verify.md` (branch `hyp/muse_moe_w4a16/w13-marlin-bf16-reduce-verify`).

- **Job:** `8d389f3d3e2d473f` (`okc status <job>`)
- **File:** `muse_moe_w4a16/submission.py` at git `429b078`

## Goal
For num_tokens >= 128 (verify_16 and verify_32), pass use_fp32_reduce=False to the w13 moe_wna16_marlin_gemm call only (use_atomic_add stays False; w2 and all decode shapes keep use_fp32_reduce=True). With fp32 reduce, Marlin writes each k-slice's partial C tile to an fp32 scratch and the last slice reads them back: at bs=256 the w13 output is M*8 x 2I = 2048 x 8192, so each slice pass moves 67 MB of fp32 scratch (~14 us at HBM speed per pass) on a 620 us shape; without it partials are combined in bf16 in the output tile. Nothing else changes.

## What changed, and where
- Worktree file: `.worktrees/muse_moe_w4a16/w13-marlin-bf16-reduce-verify/muse_moe_w4a16/submission.py` (diff: 1 file, +6/-2).
- `kernel_fn`: the w13 `moe_wna16_marlin_gemm` call now takes `**w13_args`, where
  `w13_args = dict(common, use_fp32_reduce=num_tokens < 128)`. For num_tokens >= 128 (verify 16, bs=128; verify 32, bs=256 — the only scored shapes at or above 128 per TASK.md) the w13 call runs with `use_fp32_reduce=False`; every other shape keeps `True`. `use_atomic_add` stays False for all calls (per review: atomic add into a bf16 output would need a zeroed cache1). The w2 call still uses `**common` unchanged. The branch is on `num_tokens` (a shape) only, so it is CUDA-graph safe and nothing is derived from input values.
- Module docstring: the one-line w13 description updated to say BF16 reduce at >= 128 tokens. Nothing else touched.
- Precision note (per review): with fp32 reduce off, intermediate k-slice partials are rounded to bf16 at each hand-off between CTAs that share a split column (Marlin's bf16 global-reduce path, which still round-trips the partial tile through C in global memory). This is a kernel-internal accumulation choice: within a CTA the MMA accumulation is still fp32, and the output cache1 is bf16 either way, so no activation is stored below the BF16 SGLang keeps. Only tiles whose column straddles a CTA boundary (slice_count > 1) are affected.
- Error margin: the harness does not report per-row error margins (it reports pass/fail against the relative-L2 bound), so no per-row margin is claimed; the Risk section's "attempt file must show the per-row margin" cannot be satisfied from the harness output and is replaced by the pass/fail result.
- Open question carried forward: if sgl_kernel's binding rejects use_fp32_reduce=False with use_atomic_add=False (no source on this node to check), record the run as rejected, not failed. Retirement text if flat: "knob below CI on master", not "Marlin not traffic-bound".
- Not run (no GPU/torch on this node); not committed or staged.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_moe_w4a16/w13-marlin-bf16-reduce-verify diff`:

- [x] CUDA-graph capturable: the diff adds only a Python-level dict built from `num_tokens` (a shape) and a comment; no `.item()`/`.cpu()`/`synchronize()`/prints, no allocation or control flow on input values.
- [x] Nothing derived from `data` is cached across calls: `w13_args` is a local in `kernel_fn`; `state` untouched.
- [x] Weights: n/a to this diff — `prepare()` and the repack are unchanged; NVFP4 stays NVFP4.
- [x] Activations BF16 where SGLang keeps BF16: cache1 is bf16 in both paths; only the kernel-internal combination of split-k partials is bf16 instead of fp32 (see Implementation notes). KV cache / queries: n/a (MoE problem, no attention).
- [x] Attention sees every key: n/a — no attention in this problem; no skipping of experts, tokens or weights introduced.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: diff adds none.
- [x] All work happens inside the timed call; no output caching or replay: both GEMMs still run every call.
- [x] First-line marker `# okc: problem=muse_moe_w4a16` intact; no triton.autotune added.

Lint (`okc_lint.py ... --problem muse_moe_w4a16`):

```
warn  host-sync    L110: torch.cuda.synchronize()  (outside kernel_fn's call graph; fine in prepare())
OK with warnings: /home/user/git/musespark/.worktrees/muse_moe_w4a16/w13-marlin-bf16-reduce-verify/muse_moe_w4a16/submission.py
```

`python3 -I -m py_compile`: OK (worktree __pycache__ removed). The one warning is pre-existing (in `prepare()`, unchanged by this diff).

## Risks / what could fail
medium
Risks: bf16 combination of k-slice partials adds ~2^-8 relative error per slice (2-4 slices): well inside the 1.5% rel-L2 bound, but it is a precision knob, so the attempt file must show the per-row margin from the run; it does not lower activation precision below bf16, which SGLang keeps. The expected gain is small; priority 5. If flat, retire: Marlin is then not traffic-bound at bs=256 and the verify gap (467/620 us vs 101/109 us floors) needs a different kernel, not a knob.

## Result (after the run)
- job `8d389f3d3e2d473f` (check), status done, passed True, tests 5/5
- speedup 0.9840x, 95% CI [0.9693, 0.9988], score 13666.550771281678 µs vs SGLang 13448.5 µs
- best at harvest: {"speedup": 0.9887547325362586, "ci": [0.9739234115482147, 1.0035860535243024], "job": "fb2cc736613747df"}; base result: muse_moe_w4a16/hypotheses/done/gain/sglang-marlin-path-parity.result.json
- **outcome: neutral** (within-ci)
- files: `muse_moe_w4a16/hypotheses/done/neutral/w13-marlin-bf16-reduce-verify.result.json`, `muse_moe_w4a16/hypotheses/done/neutral/w13-marlin-bf16-reduce-verify.patch`; branch `hyp/muse_moe_w4a16/w13-marlin-bf16-reduce-verify`
