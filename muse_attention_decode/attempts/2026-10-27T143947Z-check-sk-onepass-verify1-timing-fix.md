# muse_attention_decode · 2026-10-27T143947Z · check · sk-onepass-verify1-timing-fix

Hypothesis: `muse_attention_decode/hypotheses/running/2026-10-27T143409Z-sk-onepass-verify1-timing-fix.md` (branch `hyp/muse_attention_decode/sk-onepass-verify1-timing-fix`).

- **Job:** `bfb0457c9ae84322` (`okc status <job>`)
- **File:** `muse_attention_decode/submission.py` at git `58f460d3`

## Goal
Restack of sk-ticket-merge-onepass (neutral 1.6177 in check ba8c8a61; twins compiled, merge canary exact, full decode measured only x1.002-1.005) with ONE fix: its two target cells (sliding verify B=1 twin 9, sliding decode B=17..32 twin 10) were never timed because the INCUMBENT failed the [op] phase's own timing-data check (rel L2 9.64e-02 / 3.65e-02): the timing inputs the phase built do not match the reference the check compares against. Fix the [op] timing inputs: reuse the exact inputs (and reference) that each cell's own correctness cases pass on, or build timing data with the same builder the incumbent's routing gate (_ps_setup / _sd_setup) uses; print the incumbent's own timing-data rel L2 so the fix is visible. Drop the full-decode cells' twin 8 timing (x1.002-1.005 measured, not routable) to save prepare time, keeping twin 8 compiled only if removing it would change the build. Everything else as the parent: 1.01x in both rounds, IQR guard, MB-4 readout at verify B=1, one grid step up print-only.

## What changed, and where
Started from the parent sk-ticket-merge-onepass patch, already applied and staged by the restack. This hypothesis's
own edits are unstaged on top of it (`git -C <worktree> diff` shows only them; `git diff HEAD` shows parent + fix:
1 file, +599/-17 vs master 6c6219d2). Every edit is in the import-time [op] phase (`_op_*`) of
`muse_attention_decode/submission.py` plus two comments. Nothing in kernel_fn or the CUDA/Triton sources changed.

1. **The fix (review item 1).** In both `_op_cell_ps` and `_op_cell_sd`, after `data = _ps_synthetic(...)` and before
   `ref = _ps_reference(...)`, the code now runs `_ps_prepare_q(data); torch.cuda.synchronize()`. The reference then
   reads the new-token K/V rows that the pre-attention kernel has normed, RoPE'd and stored, not the random pre-generated
   rows. This matches `_op_run_cases` and `_ps_case`. The timed lambdas, `_ps_launch(data, *_ps_prepare_q(data), ...)` and
   `_sd_run`, re-run the same idempotent pre-attention writes. The step-up readouts reuse the same `ref`. I applied the
   ordering fix directly and did not swap builders.
2. **Visibility (review item 2).** `_op_time_cell` now prints `[op] <cell>: incumbent timing-data rel L2 X (ok|FAIL)`
   and `... onepass twin timing-data rel L2 X`. The returned detail, which ends up in the PATH CHOSEN line, now starts
   with `timing-data rel L2 incumbent X / onepass Y; `. That keeps it ahead of the 398-char truncation. If the
   incumbent fails, an explicit line `[op] <cell>: INCUMBENT FAILED the timing-data check (rel L2 X): cell not timed,
   incumbent kept` prints and the detail says `incumbent failed the timing-data check (rel L2 X): not timed`. A twin
   failure gets its own `ONEPASS TWIN FAILED ...` line. The step-up `_op_readout` lines keep their own
   `readout <label>: failed ...` wording, so a step-up failure can't be mistaken for the decision cell's failure.
3. **Full-decode timing dropped (review item 3).** I removed `_op_cell_full` and its `_OP_FULL_CHECKED` cache. The
   B=2..16 and B=17..64 jobs in `_op_setup` now return `(False, None, _OP_FULL_NOT_TIMED)` without doing any work, so the
   final loop prints `PATH CHOSEN: sk ticket merge [full decode B=2..16]: incumbent (not timed in this restack)` and the
   same line for `[full decode B=17..64]`. Twin 8 is still compiled. The CUDA source, `_SK_LABELS`, the variant table,
   the attrs line for v8 and the merge canary (MB 8 / MB 4) are all unchanged. The band-swap code in `_op_setup` is
   still there but can't fire for those bands.
4. **Unchanged from the parent:** the 1.01x gate in both rounds plus the IQR guard (1.5x, 0.5 us floor), the MB-4 twin
   11 readout at verify B=1, the single grid-step-up readouts (print only; next PSEUDO spec after the routed one for
   verify, next `_SD_SPECS` spec for sd), the import-clock guards at 430/460/480 s, `_op_setup` running last, and the
   PATH CHOSEN lines being re-printed at the end of import.

Deviations: none from the review's binding items. I did delete `_op_cell_full` instead of leaving it as dead code;
it remains on the parent's branch if it's needed again.
Uncertainty: I couldn't run anything on this node (no GPU). The expected incumbent timing-data rel L2 of ~1.7e-03
comes from the review and was not measured here.

## Rules check (each must be true before the run)
Checked against `git -C .worktrees/muse_attention_decode/sk-onepass-verify1-timing-fix diff` (this fix) and
`diff HEAD` (parent + fix):
- [x] CUDA-graph capturable: the new `torch.cuda.synchronize()` calls (two of them) and the rel L2 prints sit in
      `_op_cell_ps` / `_op_cell_sd` / `_op_time_cell`, which run at import time (lint: "outside kernel_fn's call
      graph"). kernel_fn and the routed launch paths are untouched. Nothing allocates or branches on input values.
- [x] Nothing derived from `data` is cached across calls: the [op] timing data and ref are local and deleted after
      each cell. I removed `_OP_FULL_CHECKED`, which only ever held a prepare-time status string anyway.
- [x] n/a: weights. This problem has no weights, and no repacking changed.
- [x] Activations BF16 where SGLang keeps BF16; KV cache FP8; queries e4m3. No dtype path changed (the twins are the
      parent's, already canaried).
- [x] Every key: the window and context coverage are unchanged. The fix only changes which cache rows the [op]
      phase's own reference reads.
- [x] No timing patches, harness frames/I/O/secrets, module edits, reference import, subprocesses or network. The
      [op] timing is our own import-time A/B on our own synthetic data.
- [x] All work happens inside the timed call; no output caching or replay. Routing is frozen at import from our own
      synthetic A/B, and nothing is cached from the judge's inputs.
- [x] First line is still `# okc: problem=muse_attention_decode`; no `triton.autotune` was added.

Lint (`okc_lint.py <worktree>/muse_attention_decode/submission.py --problem muse_attention_decode`):

```
OK with warnings: muse_attention_decode/submission.py
  102 x warn host-sync  (all "outside kernel_fn's call graph; fine in prepare()"; includes the 2 new
                         torch.cuda.synchronize() at the [op] timing-data prepare, L9814 and L9889)
    1 x warn dtype      L7786 q = torch.full(..., 0x7F, dtype=torch.uint8).view(FP8)  (pre-existing, unchanged)
lint rc=0, 0 errors
```

`python3 -I -m py_compile`: ok, and I removed the `__pycache__` it created in the worktree.

## Risks / what could fail
low
The parent answered half its question (the merge does not bind full decode); this run answers the other half. Low risk: the twins already compiled and passed the merge canary on the judge.

## Result (after the run)
- job `bfb0457c9ae84322` (check), status done, passed True, tests 16/16
- speedup 1.6219x, 95% CI [1.5976, 1.6516], score 2322.096208253241 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.6339575246416982, "ci": [1.6094481617720726, 1.6584668875113235], "job": "4e386647edd845f7"}; base result: SGLang (no base result)
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/sk-onepass-verify1-timing-fix.result.json`, `muse_attention_decode/hypotheses/done/neutral/sk-onepass-verify1-timing-fix.patch`; branch `hyp/muse_attention_decode/sk-onepass-verify1-timing-fix`
