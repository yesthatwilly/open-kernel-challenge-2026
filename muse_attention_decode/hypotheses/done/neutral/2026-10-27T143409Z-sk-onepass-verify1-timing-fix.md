---
problem: muse_attention_decode
slug: sk-onepass-verify1-timing-fix
created: 2026-10-27T143409Z
base: master
area: "stream-K ticket merge onepass twin at sliding verify B=1 (the gate's timing inputs)"
change: >
  Restack of sk-ticket-merge-onepass (neutral 1.6177 in check ba8c8a61; twins compiled, merge
  canary exact, full decode measured only x1.002-1.005) with ONE fix: its two target cells
  (sliding verify B=1 twin 9, sliding decode B=17..32 twin 10) were never timed because the
  INCUMBENT failed the [op] phase's own timing-data check (rel L2 9.64e-02 / 3.65e-02): the timing
  inputs the phase built do not match the reference the check compares against. Fix the [op]
  timing inputs: reuse the exact inputs (and reference) that each cell's own correctness cases
  pass on, or build timing data with the same builder the incumbent's routing gate (_ps_setup /
  _sd_setup) uses; print the incumbent's own timing-data rel L2 so the fix is visible. Drop the
  full-decode cells' twin 8 timing (x1.002-1.005 measured, not routable) to save prepare time,
  keeping twin 8 compiled only if removing it would change the build. Everything else as the
  parent: 1.01x in both rounds, IQR guard, MB-4 readout at verify B=1, one grid step up
  print-only.
expected:
  - shapes: layer=sliding load=verify_1 *
    direction: faster
    why: "the ticket merge is on the latency-bound tail at p8 (~4 tiles per CTA): the parent review estimated -0.5..-1.0 us on a ~13 us row (0.68%/us), up to ~0.4-0.7% headline if it routes"
  - shapes: layer=sliding load=decode_32 *
    direction: faster
    why: twin 10 at p4, smaller (~0.2-0.4 us)
  - shapes: "*"
    direction: flat
    why: full decode keeps the incumbent
risk: low
depends_on: sk-ticket-merge-onepass
priority: 2
kind: primary
controls_for: null
prior: "sk-ticket-merge-onepass: the same twins; its verify B=1 and sliding decode cells were never timed because the incumbent failed the timing-data check (a harness bug in the [op] phase's timing inputs), so the mechanism there is untested"
review: proceed
status: done
review_at: 2026-10-27T143715Z
review_companions: null
stage: harvested
claimed_at: 2026-10-27T143715Z
base_sha: 6c6219d272efce7c7cfb16581e6a5b87ab96f188
base_blob: 2eb03f76ceefa9bf81e28616ced3b33be3a70443
branch: hyp/muse_attention_decode/sk-onepass-verify1-timing-fix
worktree: .worktrees/muse_attention_decode/sk-onepass-verify1-timing-fix
mode: check
restack_of: sk-ticket-merge-onepass
stale_base: null
implemented_at: 2026-10-27T143946Z
impl_commit: 58f460d3974b8b4cbbfa3d1a720b56fe52497dd0
submission_sha256: 3b29647036aa1a23f49c3f95549f1d3c219497fa1b3a2c0ca5442e32b1ea5a29
norm_sha256: 8441e7f05b907f63dc88657fd55557af77ed067b4d7af91390e85f4f5c2ddc9b
submission_blob: 21166a140086e6d4431a5dd0a219311831020632
lint_warnings: 103
attempt_file: 2026-10-27T143947Z-check-sk-onepass-verify1-timing-fix.md
prepared_at: 2026-10-27T143947Z
job: bfb0457c9ae84322
dispatched_at: 2026-10-27T144003Z
outcome: neutral
verdict_detail: within-ci
speedup: 1.6219396882066044
speedup_ci: [1.5976105928835054, 1.6516121269561428]
harvested_at: 2026-10-27T145256Z
best_at_harvest: 1.6339575246416982
promotion: null
---

# sk-onepass-verify1-timing-fix

## Change

Restack of sk-ticket-merge-onepass (neutral 1.6177 in check ba8c8a61; twins compiled, merge canary exact, full decode measured only x1.002-1.005) with ONE fix: its two target cells (sliding verify B=1 twin 9, sliding decode B=17..32 twin 10) were never timed because the INCUMBENT failed the [op] phase's own timing-data check (rel L2 9.64e-02 / 3.65e-02): the timing inputs the phase built do not match the reference the check compares against. Fix the [op] timing inputs: reuse the exact inputs (and reference) that each cell's own correctness cases pass on, or build timing data with the same builder the incumbent's routing gate (_ps_setup / _sd_setup) uses; print the incumbent's own timing-data rel L2 so the fix is visible. Drop the full-decode cells' twin 8 timing (x1.002-1.005 measured, not routable) to save prepare time, keeping twin 8 compiled only if removing it would change the build. Everything else as the parent: 1.01x in both rounds, IQR guard, MB-4 readout at verify B=1, one grid step up print-only.

## Expected

- `layer=sliding load=verify_1 *` faster: the ticket merge is on the latency-bound tail at p8 (~4 tiles per CTA): the parent review estimated -0.5..-1.0 us on a ~13 us row (0.68%/us), up to ~0.4-0.7% headline if it routes
- `layer=sliding load=decode_32 *` faster: twin 10 at p4, smaller (~0.2-0.4 us)
- `*` flat: full decode keeps the incumbent

## Risk

The parent answered half its question (the merge does not bind full decode); this run answers the other half. Low risk: the twins already compiled and passed the merge canary on the judge.

## Adversarial review

Reviewed 2026-10-27T143715Z by an adversarial subagent (model opus).

**Verdict: proceed.** The parent's two sliding cells were never timed because of a real ordering bug in its [op] timing data. The fix is small and the question was never answered: does the one-pass ticket merge pay at PSEUDO verify B=1 and at sd decode B=32? The in-prepare interleaved A/B is enough; no companion. Judge on the PATH CHOSEN ratios and the verify_1 splitkv_attn_sk_kernel row, not the headline.

### Justification
Root cause (parent patch, _op_cell_ps and _op_cell_sd): both build data = _ps_synthetic(...) then ref = _ps_reference(data, ...) BEFORE any _ps_prepare_q(data); _ps_reference attends over the cache as the pre-attention kernel left it, so it sees random pre-generated K/V at the new-token rows instead of the RMS-normed, RoPE'd rows. The error tracks the number of new rows (verify L=8 9.64e-02, decode L=1 3.65e-02), identical for incumbent and twin and at the step-up grid: a harness mismatch. The real per-case runner calls prepare first and passes at 1.7e-03. Fix: _ps_prepare_q(data) once (sync) before _ps_reference in both cells (pre-attention writes are idempotent). Mechanism: verify_1 sliding 13.50 us (sk kernel 9.8 + pre-attn 1.8), floor 0.12 us, latency-bound; at p8 (~8 slots) the winner's merge is on the tail: ~4 dependent round trips -> ~1, saving ~0.4-0.75 us; the 1.01x gate needs ~0.12 us. Headline +0.3..0.5%. sd decode_32 at p4 saves ~0.2 us, likely at or under the bar. Weakest link: full decode measured the merge's price as tiny (x1.002-1.005); twins 9/11 carry 127/117 regs vs 121 (no spill).

### Confounders on the current kernel
- restack-sk-pseudo-cluster-plus-v3-fused-preattn: installed PSEUDO verify B=1 at p8 under the two-pass merge; read the print-only step-up (sm) readout.
- restack-sliding-decode-sk-window-on-cluster-master: sd route variant 6 at p4; a flat result there does not refute the mechanism; read the p8 step-up readout.
- full-decode1-streamk-s3-two-level-ticket: the host kernel; twin differs only in the merge template flag (canary exact).
- sk-full-line-gather: check that the gather PATH CHOSEN line reads full-line.
- sliding-verify-s2b-b17-32-bar103: the sw gate takes the import clock to ~295 s; [op] must stay inside its 430/460/480 s guards.

### Interference and A/B needs
No companion: in-prepare interleaved A/B (variant 4 vs 9, 6 vs 10), same module, cold-L2, empty graph subtracted, 2 x 21 reps, 1.01x both rounds + IQR guard; the MB-4 twin 11 readout separates pass fusion from batch depth. Read the PATH CHOSEN ratios, the incumbent's newly printed timing-data rel L2 (should be ~1.7e-03) and the scored verify_1 sk kernel row (9.8 us in ba8c8a61).

### Open questions
- Implement as _ps_prepare_q(data); torch.cuda.synchronize(); ref = _ps_reference(...) in both _op_cell_ps and _op_cell_sd (the step-up readouts reuse ref); print the incumbent's timing-data rel L2.
- Print a failure line when _op_time_cell's incumbent fails, distinct from the step-up readouts.
- combo-tail-onepass-push-fastepi inherits the same broken timing data: should it take this fix?
- If twin 9 routes and the sm step-up beats p8, re-gate the PSEUDO grid with the one-pass merge in a follow-up.
- Both full-decode bands must still print an explicit 'incumbent (not timed in this restack)' PATH CHOSEN line.


## Implementation notes

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

## Rules check

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

## Result

- job `bfb0457c9ae84322` (check), status done, passed True, tests 16/16
- speedup 1.6219x, 95% CI [1.5976, 1.6516], score 2322.096208253241 µs vs SGLang 3766.3 µs
- best at harvest: {"speedup": 1.6339575246416982, "ci": [1.6094481617720726, 1.6584668875113235], "job": "4e386647edd845f7"}; base result: SGLang (no base result)
- **outcome: neutral** (within-ci)
- files: `muse_attention_decode/hypotheses/done/neutral/sk-onepass-verify1-timing-fix.result.json`, `muse_attention_decode/hypotheses/done/neutral/sk-onepass-verify1-timing-fix.patch`; branch `hyp/muse_attention_decode/sk-onepass-verify1-timing-fix`

## Attribution

| shape | w | base µs | hyp µs | delta | verdict |
|---|---|---|---|---|---|
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_1 page=1 | 46 | 20.22 | 10.88 | 1.8591 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_1 page=1 | 16 | 31.66 | 15.55 | 2.0369 | gained |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_16 page=1 spread=agentic | 46 | 26.85 | 15.18 | 1.7684 | gained |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_16 page=1 spread=agentic | 16 | 200.85 | 104.34 | 1.925 | gained |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=decode_32 page=1 spread=agentic | 46 | 31.5 | 17.25 | 1.8261 | gained |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=decode_32 page=1 spread=agentic | 16 | 385.17 | 189.52 | 2.0323 | gained |
| batch=1 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_1 page=1 q_len=8 | 46 | 22.77 | 12.79 | 1.7805 | gained |
| batch=1 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_1 page=1 q_len=8 | 16 | 35.76 | 29.75 | 1.2022 | gained |
| batch=16 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_16 page=1 q_len=8 spread=agentic | 46 | 29.28 | 20.1 | 1.4566 | gained |
| batch=16 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_16 page=1 q_len=8 spread=agentic | 16 | 205.54 | 164.02 | 1.2531 | gained |
| batch=32 ctx=80000 dist=soft heads=tp8 layer=sliding load=verify_32 page=1 q_len=8 spread=agentic | 46 | 34.43 | 27.93 | 1.233 | gained |
| batch=32 ctx=80000 dist=coherent heads=tp8 layer=full load=verify_32 page=1 q_len=8 spread=agentic | 16 | 391.3 | 303.51 | 1.2892 | gained |

gained on sliding decode_1, decode_16, decode_32, verify_1, verify_16, verify_32; full decode_1, decode_16, decode_32, verify_1, verify_16, verify_32. Expected hit: 0/3. Noise ±1.7%.

## Follow-ups

- within-ci: the change is not proven either way; a stacked hypothesis (`depends_on: sk-onepass-verify1-timing-fix`) that adds a further change in the same area is the only way to learn more.

