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
review: required
status: queued
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
