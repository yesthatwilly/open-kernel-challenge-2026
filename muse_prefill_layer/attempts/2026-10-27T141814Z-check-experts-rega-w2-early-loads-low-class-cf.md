# muse_prefill_layer · 2026-10-27T141814Z · check · experts-rega-w2-early-loads-low-class-cf

Hypothesis: `muse_prefill_layer/hypotheses/running/2026-10-27T141744Z-experts-rega-w2-early-loads-low-class-cf.md` (branch `hyp/muse_prefill_layer/experts-rega-w2-early-loads-low-class-cf`).

- **Job:** `8716d98c405f417d` (`okc status <job>`)
- **File:** `muse_prefill_layer/submission.py` at git `3905a2f8`

## Goal
Confirmation rerun of experts-rega-w2-early-loads-low-class: its implementation commit 52169f1783ec72c5ee8b2e97b0648a0e91426c87 unchanged except one trailing comment, run as a check and pooled with job 59eb665f162a4ac9 at harvest.

## What changed, and where
No code change: `experts-rega-w2-early-loads-low-class` implementation commit `52169f1783ec72c5ee8b2e97b0648a0e91426c87` plus one trailing comment, so okc times it again instead of returning the earlier job. Written by churn.py claim.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_prefill_layer/experts-rega-w2-early-loads-low-class diff HEAD`:

- [x] CUDA-graph capturable: the only kernel_fn-path change (`_experts_rega`) is a dict lookup on prepare-time
      state and the class from `num_tokens` (shape), then one launch with the same arguments as master's; every
      `.cpu()` / `synchronize()` / `torch.equal` added is in prepare()-only w2st / fsilu helpers.
- [x] Nothing derived from `data` is cached: state gains only `rega_w2st_ok` (plain bools per class) and the module.
- [x] Weights: untouched (same `bank["q2"]` / `gs2` NVFP4 bank as master's w2).
- [x] Activations: bf16 y with master's values, addresses and rounding (bit-exactness asserted in the gate); KV cache
      / query precision untouched.
- [x] Attention: n/a: no attention code touched.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network: the
      nvcc build is the existing fsilu `load_inline` path; no new imports.
- [x] All work inside the timed call; no output caching or replay: arm a recomputes y every call; the prepare-time
      gate only picks which module's w2 launches.
- [x] First line `# okc: problem=muse_prefill_layer` kept; no triton.autotune added.

okc_lint (`python3 -I .claude/skills/churnsubs/bin/okc_lint.py <worktree>/muse_prefill_layer/submission.py
--problem muse_prefill_layer`), exit 0:

```
107 x warn host-sync   (all "outside kernel_fn's call graph; fine in prepare()")
  5 x warn dtype       L3470, L3471, L3761, L4308, L6794 torch.uint8 buffers (pre-existing master lines)
OK with warnings: muse_prefill_layer/submission.py
```

`python3 -I -m py_compile` OK; `__pycache__` removed.

Copied from `experts-rega-w2-early-loads-low-class`, whose code this is; the only change is a comment.

## Risks / what could fail
low

## Result (after the run)
- job `8716d98c405f417d` (check), status done, passed True, tests 5/5
- speedup 1.4994x, 95% CI [1.4769, 1.5219], score 258367.82977029568 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.488912488847924, "ci": [1.466578801515205, 1.5112461761806426], "job": "ec3bd01859574bf1"}; base result: muse_prefill_layer/hypotheses/done/neutral/experts-rega-w2-early-loads-low-class.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-w2-early-loads-low-class-cf.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-w2-early-loads-low-class-cf.patch`; branch `hyp/muse_prefill_layer/experts-rega-w2-early-loads-low-class-cf`
- confirm of `experts-rega-w2-early-loads-low-class`: pooled 1.4986x (job 59eb665f162a4ac9) and 1.4994x (job 8716d98c405f417d) → 1.4990x; best 1.4889x, two-run margin 1.07% → neutral (unproven-vs-best)
