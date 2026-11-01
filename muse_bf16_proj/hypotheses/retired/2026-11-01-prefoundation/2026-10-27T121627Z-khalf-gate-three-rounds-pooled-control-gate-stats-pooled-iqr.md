---
problem: muse_bf16_proj
slug: khalf-gate-three-rounds-pooled-control-gate-stats-pooled-iqr
created: 2026-10-27T121627Z
base: master
area: pooler m=128 khalf (b2) cell gate (_khalf_prepare gate only)
change: >
  From the parent's branch (master 8f233e88 lineage), change only _khalf_prepare's gate block and
  _cell_gate's optional stats output; candidate, canary, _cell_check, kernel-name assertion, A/B
  copy check, deadlines and every other cell stay byte-identical. (1) In _cell_gate, when stats is
  not None, also store each side's raw per-rep times: stats[name]['times'] = list(v); the return
  value and the print for other callers do not change. (2) Set KHALF_ROUNDS = 10 (about 1 s of
  prepare); for each round print its ratio (master/cand net), cand IQR, master IQR and the IQR
  bar. (3) Evaluate three rules on the same rounds and print one line each with PASS/FAIL and the
  reason: OLD: rounds 1-2 only, both ratios >= 1.03 and both IQR tests pass. PARENT: rounds 1-3
  only, every ratio >= 1.01, median ratio >= 1.03, IQR per round in all 3. POOLED: rounds 1-3
  only, every ratio >= 1.01 and the median of the three ratios >= 1.03; pool each side's raw times
  across the 3 rounds (123 reps) and require pooled cand IQR <= 1.5 x max(pooled master IQR,
  KHALF_IQR_FLOOR_US). (4) Over rounds 1-10 also print the empirical per-round IQR-fail fraction
  and ratio < 1.03 fraction, and the OLD / PARENT / POOLED decisions over the 3 disjoint triples
  1-3, 4-6, 7-9. (5) Route the flag with POOLED. Put all of this in the PATH CHOSEN K line and
  keep it in the end-of-prepare re-print (16,000-character log tail).
expected:
  - shapes: layer=pooler load=verify_16 m=128
    direction: faster
    why: POOLED should route on most runs because the candidate wins about 4% in every round seen (8/8 >= 1.025); when routed, the judged row should match a6424247/670ad7c6 (about 21.9 us vs about 23.1 us). The evidence is the gate lines, not the headline.
  - shapes: "*"
    direction: flat
    why: gate-only change; the other cells are identical, and the about 0.27% headline effect is inside the CI
risk: low
depends_on: khalf-gate-three-rounds-pooled
priority: 1
kind: control
controls_for: khalf-gate-three-rounds-pooled
prior: null
review: required
status: hold
hold_reason: folded into router-pdl-plus-khalf-pooled-iqr-gate (its 10-round diagnostic and OLD/PARENT/POOLED readout run there), per that hypothesis's adversarial review 2026-10-27
---

# khalf-gate-three-rounds-pooled-control-gate-stats-pooled-iqr

## Change

From the parent's branch (master 8f233e88 lineage), change only _khalf_prepare's gate block and _cell_gate's optional stats output; candidate, canary, _cell_check, kernel-name assertion, A/B copy check, deadlines and every other cell stay byte-identical. (1) In _cell_gate, when stats is not None, also store each side's raw per-rep times: stats[name]['times'] = list(v); the return value and the print for other callers do not change. (2) Set KHALF_ROUNDS = 10 (about 1 s of prepare); for each round print its ratio (master/cand net), cand IQR, master IQR and the IQR bar. (3) Evaluate three rules on the same rounds and print one line each with PASS/FAIL and the reason: OLD: rounds 1-2 only, both ratios >= 1.03 and both IQR tests pass. PARENT: rounds 1-3 only, every ratio >= 1.01, median ratio >= 1.03, IQR per round in all 3. POOLED: rounds 1-3 only, every ratio >= 1.01 and the median of the three ratios >= 1.03; pool each side's raw times across the 3 rounds (123 reps) and require pooled cand IQR <= 1.5 x max(pooled master IQR, KHALF_IQR_FLOOR_US). (4) Over rounds 1-10 also print the empirical per-round IQR-fail fraction and ratio < 1.03 fraction, and the OLD / PARENT / POOLED decisions over the 3 disjoint triples 1-3, 4-6, 7-9. (5) Route the flag with POOLED. Put all of this in the PATH CHOSEN K line and keep it in the end-of-prepare re-print (16,000-character log tail).

## Expected

- `layer=pooler load=verify_16 m=128` faster: POOLED should route on most runs because the candidate wins about 4% in every round seen (8/8 >= 1.025); when routed, the judged row should match a6424247/670ad7c6 (about 21.9 us vs about 23.1 us). The evidence is the gate lines, not the headline.
- `*` flat: gate-only change; the other cells are identical, and the about 0.27% headline effect is inside the CI

## Risk

Shows from one check which clause makes the khalf gate flaky (per-round IQR fail fraction vs ratio < 1.03 fraction) and gives each candidate rule's miss rate on identical data (OLD vs PARENT vs POOLED, 3 triples each). If the IQR fail fraction is ~0.25 per round as the 8 logged rounds suggest, PARENT is refuted and POOLED is the rule to land.
