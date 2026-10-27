---
problem: muse_bf16_proj
slug: router-pdl-plus-khalf-pooled-iqr-gate
created: 2026-10-27T143409Z
base: master
area: router m=9..256 PDL (restack) + pooler m=128 khalf gate rule
change: >
  Restack of router-zero-pdl-overlap-m9-m256 (neutral 1.2715 in check a20f8048, but ROUTED in all
  4 classes: m=16 1.040/1.091, m=32 1.086/1.101, m=128 1.031/1.059, m=256 1.057/1.041 on the
  read-flush gate; the nopdl arm equals master, so the gain is the PDL overlap; race checks clean)
  on current master, unchanged, PLUS one gate-rule change for the khalf (b2) pooler cell at m=128,
  whose miss pulled that run's headline down (pooler_khalf128_ok=False; it also missed in c65ffcb6
  and in the board run bc147c5b). The khalf adversary (khalf-gate-three-rounds-pooled, held)
  showed both observed misses failed the per-round 41-rep IQR rule (cand IQR <= 1.5x max(master
  IQR, 0.5 us)), not the ratio rule, because a 41-sample IQR has ~25-30% relative error. CHANGE to
  _khalf_prepare's gate only: run 3 read-flush rounds; route when every round's master/cand >=
  1.01, the MEDIAN ratio >= 1.03, and the IQR test passes on the POOLED raw per-rep times across
  the 3 rounds (123 reps each side; cand IQR <= 1.5x max(pooled master IQR, KHALF_IQR_FLOOR_US)).
  To get the raw times, _cell_gate stores each side's per-rep list in its optional stats dict
  (return value and other callers unchanged). Print all three rounds' ratios and per-round IQRs,
  the pooled IQRs, the median, and which rule decided, in the PATH CHOSEN K line (kept in the
  DECISIONS re-print). Candidate, canary, A/B copy check and deadlines unchanged.
expected:
  - shapes: layer=router *
    direction: faster
    why: "router PDL routed at m=16/32/128/256 in a20f8048: -0.2 to -0.45 us per router row (x1.03-1.10 gate)"
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: "khalf routes on most runs under the pooled rule (b2 won every one of 8 recorded rounds at x1.025-1.054): keeps the ~-1.2 us judged pooler m=128 row"
  - shapes: "*"
    direction: flat
    why: every other cell unchanged
risk: low
depends_on: router-zero-pdl-overlap-m9-m256
priority: 1
kind: primary
controls_for: null
prior: "router-zero-pdl-overlap-m9-m256: same router code, which routed everywhere but lost the headline to a khalf gate miss in the same run; the khalf part is the pooled-IQR rule the khalf-gate-three-rounds-pooled adversary prescribed (that parent was held because its rule kept the per-round IQR test)"
review: proceed
status: queued
review_at: 2026-10-27T143812Z
review_companions: null
---

# router-pdl-plus-khalf-pooled-iqr-gate

## Change

Restack of router-zero-pdl-overlap-m9-m256 (neutral 1.2715 in check a20f8048, but ROUTED in all 4 classes: m=16 1.040/1.091, m=32 1.086/1.101, m=128 1.031/1.059, m=256 1.057/1.041 on the read-flush gate; the nopdl arm equals master, so the gain is the PDL overlap; race checks clean) on current master, unchanged, PLUS one gate-rule change for the khalf (b2) pooler cell at m=128, whose miss pulled that run's headline down (pooler_khalf128_ok=False; it also missed in c65ffcb6 and in the board run bc147c5b). The khalf adversary (khalf-gate-three-rounds-pooled, held) showed both observed misses failed the per-round 41-rep IQR rule (cand IQR <= 1.5x max(master IQR, 0.5 us)), not the ratio rule, because a 41-sample IQR has ~25-30% relative error. CHANGE to _khalf_prepare's gate only: run 3 read-flush rounds; route when every round's master/cand >= 1.01, the MEDIAN ratio >= 1.03, and the IQR test passes on the POOLED raw per-rep times across the 3 rounds (123 reps each side; cand IQR <= 1.5x max(pooled master IQR, KHALF_IQR_FLOOR_US)). To get the raw times, _cell_gate stores each side's per-rep list in its optional stats dict (return value and other callers unchanged). Print all three rounds' ratios and per-round IQRs, the pooled IQRs, the median, and which rule decided, in the PATH CHOSEN K line (kept in the DECISIONS re-print). Candidate, canary, A/B copy check and deadlines unchanged.

## Expected

- `layer=router *` faster: router PDL routed at m=16/32/128/256 in a20f8048: -0.2 to -0.45 us per router row (x1.03-1.10 gate)
- `layer=pooler load=verify_16 m=128` faster: khalf routes on most runs under the pooled rule (b2 won every one of 8 recorded rounds at x1.025-1.054): keeps the ~-1.2 us judged pooler m=128 row
- `*` flat: every other cell unchanged

## Risk

Headline effect ~+0.3-0.6% combined (router ~0.3% + khalf ~0.27% when it would otherwise miss): near the single-run readability line, so read the R-PDL and K PATH CHOSEN lines. The 1.2735 board run had khalf OFF, so a run with both ON can plausibly beat it; a check that beats the board gets a scored run.

## Adversarial review

Reviewed 2026-10-27T143812Z by an adversarial subagent (model opus).

**Verdict: proceed.** A reasonable bid to beat the board: it restacks the router PDL change (routed in all four classes) and adds the pooled-IQR khalf rule; the parent patch applies to the same base blob and breaks no rule. One run cannot read the headline (~+0.5% vs CI ~1.5%): judge on the R-PDL and K PATH CHOSEN lines and the judged rows. The khalf half is the same routing rule the queued control khalf-gate-three-rounds-pooled-control-gate-stats-pooled-iqr implements: fold that control's diagnostic in here or run them in sequence.

### Justification
Router PDL is real: gates passed in all four classes, the nopdl arm equals master, the race checks are clean and the no-wait control was wrong 32/32 times. Judged rows only partly agree vs three master-lineage runs: m=16 5.27 vs 5.48-5.73 (~-0.36 us, clear); m=128 7.06 vs 7.39-7.95 (likely); m=32 5.96 vs 5.94-6.10 (flat despite ~0.4 us at the gate); m=256 9.06 inside master's spread: ~0.6-1.0 us summed, ~0.2-0.3% headline. Khalf: the IQR clause binds (>= 5 of 10 logged rounds failed it), but a20f8048 r1 was 1.027 (< 1.03), so that miss was ratio and IQR. Weakest link: the candidate is genuinely wider (summed IQRs cand/master ~1.42 vs a 1.5 bar), so a 123-rep pooled test at 1.5x is near a coin flip (~55-70%). Median-of-3 >= 1.03 passes ~90%. 'layer=router *' is too broad: m=1/8 take the untouched cluster path.

### Confounders on the current kernel
- 8f233e88 (read-flush gates + khalf b2): R-PDL classes are themselves marginal (m=128 r1 1.031): read FINAL FLAGS before any row.
- pooler-nn-layout-widen-m256 / restack-three-cell-wins: the khalf master side's IQR (0.45-0.90 us) and the 0.5 us floor set the bar.
- router hilo / swapab designs are preconditions; the flat judged m=32 row suggests less of the fill is hidden there.
- router_kernel<8,8> at m<=8: m=1/8 rows cannot move.
- bulk12 m=1: whole-run drift 1.7-3.6% exceeds this effect; headline comparisons with bc147c5b mean nothing.

### Interference and A/B needs
No new companion. Fold the queued control's diagnostic into this build: KHALF_ROUNDS=10, route on POOLED over rounds 1-3, print the per-round IQR-fail and ratio<1.03 fractions plus OLD/PARENT/POOLED over triples 1-3, 4-6, 7-9 (~0.7 s); then retire the control.

### Open questions
- Re-base the pooled KHALF_MAX_IQR: the candidate's true IQR is ~1.42x master's vs a 1.5x bar; a pooled 2.0x bar or p90(cand) < median(master) guards bimodality without failing ~40% of real wins.
- Why did the judged m=32 router row not move? If repeated, decide whether router_pdl_ok_32 should route.
- Give R-PDL the same median-of-3 rule so its classes do not flicker.
- Store raw times in the parent's modified _cell_gate (extra-arm version) for cand and master only; keep R-PDL callers' prints and stats keys.
- A scored run's FINAL FLAGS must show pooler_khalf128_ok and all four router_pdl_ok flags.

