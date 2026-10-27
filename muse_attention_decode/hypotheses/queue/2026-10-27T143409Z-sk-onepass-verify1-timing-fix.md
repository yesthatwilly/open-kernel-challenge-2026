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
status: queued
review_at: 2026-10-27T143715Z
review_companions: null
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

