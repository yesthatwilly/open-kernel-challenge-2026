---
problem: muse_prefill_layer
slug: experts-rega-w2-early-loads-low-class-cf
created: 2026-10-27T141744Z
base: master
area: "rega w2 epilogue (early gs/sid loads, arm a) routed at the LOW class"
change: "Confirmation rerun of experts-rega-w2-early-loads-low-class: its implementation commit 52169f1783ec72c5ee8b2e97b0648a0e91426c87 unchanged except one trailing comment, run as a check and pooled with job 59eb665f162a4ac9 at harvest."
expected: []
risk: low
depends_on: experts-rega-w2-early-loads-low-class
priority: 1
kind: confirm
controls_for: experts-rega-w2-early-loads-low-class
confirm_of: experts-rega-w2-early-loads-low-class
confirm_of_job: 59eb665f162a4ac9
review: proceed
status: done
stage: harvested
claimed_at: 2026-10-27T141759Z
base_sha: 52169f1783ec72c5ee8b2e97b0648a0e91426c87
base_blob: 11a08b3ff3cc8cb1ac2d0a96e19c7e94b92ab27d
branch: hyp/muse_prefill_layer/experts-rega-w2-early-loads-low-class-cf
worktree: .worktrees/muse_prefill_layer/experts-rega-w2-early-loads-low-class-cf
mode: check
restack_of: null
stale_base: null
implemented_at: 2026-10-27T141814Z
impl_commit: 3905a2f8da54c88e19af1e5e76857428ff1ef83b
submission_sha256: 703bd19c2ff0f8eea3d9dbc9e4e6f4df5df824620a8a23963ce73d7790f68031
norm_sha256: 43002eb32fb0b8a658dfbb49423583e797b09ba45f2c3b0b918629233dc73da4
submission_blob: 50b7846d13be612454d82662a08fd3c6842b48af
lint_warnings: 112
attempt_file: 2026-10-27T141814Z-check-experts-rega-w2-early-loads-low-class-cf.md
prepared_at: 2026-10-27T141815Z
job: 8716d98c405f417d
dispatched_at: 2026-10-27T141829Z
outcome: neutral
verdict_detail: unproven-vs-best
speedup: 1.4994277745198583
speedup_ci: [1.4769363579020605, 1.521919191137656]
harvested_at: 2026-10-27T142520Z
best_at_harvest: 1.488912488847924
promotion: null
---

# experts-rega-w2-early-loads-low-class-cf

## Change

Confirmation rerun of experts-rega-w2-early-loads-low-class: its implementation commit 52169f1783ec72c5ee8b2e97b0648a0e91426c87 unchanged except one trailing comment, run as a check and pooled with job 59eb665f162a4ac9 at harvest.

## Why

Queued by churn.py (harvest): `experts-rega-w2-early-loads-low-class` ran 1.4986x against best 1.4889x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.

## Implementation notes

No code change: `experts-rega-w2-early-loads-low-class` implementation commit `52169f1783ec72c5ee8b2e97b0648a0e91426c87` plus one trailing comment, so okc times it again instead of returning the earlier job. Written by churn.py claim.

## Rules check

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

## Result

- job `8716d98c405f417d` (check), status done, passed True, tests 5/5
- speedup 1.4994x, 95% CI [1.4769, 1.5219], score 258367.82977029568 µs vs SGLang 387403.9 µs
- best at harvest: {"speedup": 1.488912488847924, "ci": [1.466578801515205, 1.5112461761806426], "job": "ec3bd01859574bf1"}; base result: muse_prefill_layer/hypotheses/done/neutral/experts-rega-w2-early-loads-low-class.result.json
- **outcome: neutral** (unproven-vs-best)
- files: `muse_prefill_layer/hypotheses/done/neutral/experts-rega-w2-early-loads-low-class-cf.result.json`, `muse_prefill_layer/hypotheses/done/neutral/experts-rega-w2-early-loads-low-class-cf.patch`; branch `hyp/muse_prefill_layer/experts-rega-w2-early-loads-low-class-cf`
- confirm of `experts-rega-w2-early-loads-low-class`: pooled 1.4986x (job 59eb665f162a4ac9) and 1.4994x (job 8716d98c405f417d) → 1.4990x; best 1.4889x, two-run margin 1.07% → neutral (unproven-vs-best)

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft layer=sliding load=turn_1 page=1 q_len=753 repeats=6 | 46 | 730.62 | 727.27 | 1.0022 | flat |
| batch=1 ctx=80000 dist=soft layer=full load=turn_1 page=1 q_len=753 repeats=6 | 16 | 1114.09 | 1134.44 | 0.9962 | flat |
| batch=4 ctx=80000 dist=soft layer=sliding load=turn_4 page=1 q_len=753 repeats=6 | 46 | 2088.47 | 2099.71 | 1.0188 | flat |
| batch=4 ctx=80000 dist=soft layer=full load=turn_4 page=1 q_len=753 repeats=6 | 16 | 3626.99 | 3631.09 | 0.9994 | flat |
| batch=1 ctx=0 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 11506.74 | 11076.19 | 0.9877 | flat |
| batch=1 ctx=32768 dist=soft layer=sliding load=first_turn page=1 q_len=16384 repeats=6 | 138 | 10748.23 | 10875.2 | 0.9897 | flat |
| batch=1 ctx=32768 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 14529.77 | 14256.76 | 0.9902 | flat |
| batch=1 ctx=65536 dist=soft layer=full load=first_turn page=1 q_len=16384 repeats=6 | 16 | 18001.25 | 18014.49 | 0.988 | flat |

no shape moved outside noise. Noise ±1.5%.

## Follow-ups

- pooled verdict for `experts-rega-w2-early-loads-low-class`: neutral (unproven-vs-best); this file is the record of the rerun only.
- unproven-vs-best: faster than the best but inside the drift margin; churn queues one confirm rerun when the margin allows (CONFIRM_MIN), and pools the two runs.

