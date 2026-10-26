---
problem: muse_bf16_proj
slug: combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16
created: 2026-10-25T211857Z
base: master
area: combination (pooler K-halves m=128/256 + qkv row-owner m=8/16 + per-cell Lt plans m=128/256)
change: >
  Stack for the board. Build on the branch of pooler-k-halves-fp32-partials-restack-m128-m256
  (depends_on). Port, by 3-way git merge-file of their patches, the cells of
  qkv-rowowner-a2-widen-m8-m16 and bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpooler,
  but only the cells whose own run printed PATH CHOSEN candidate AND whose judged row moved the
  right way by more than the 1.5% per-row noise (read their Attribution tables); a cell whose own
  run kept master or lost its row is omitted, and the change says which. Every ported cell keeps
  its own flag, canary, correctness check and hardened gate (two rounds, >= 1.03, IQR <= 1.5x
  master), so the combo can only route what re-wins in this run's prepare. Do not touch m=1 cells,
  m<=32 streams or the router. Print one FINAL FLAGS line listing every cell's route. If neither
  sibling has harvested when this is claimed, wait (do not run it as a pooler-only rerun).
expected:
  - shapes: layer=pooler load=verify_* m=*
    direction: faster
    why: K-halves fp32 partials (from depends_on)
  - shapes: layer=qkv_gate load=verify_1 m=8
    direction: faster
    why: A2 row-owner at m=8 if its run routed
  - shapes: layer=qkv_gate load=decode_16 m=16
    direction: faster
    why: NT=16 row-owner if its run routed
  - shapes: layer=o load=verify_* m=*
    direction: faster
    why: per-cell Lt plan if its run routed
risk: medium
depends_on: pooler-k-halves-fp32-partials-restack-m128-m256
priority: 2
kind: primary
controls_for: null
prior: "combo-rowowner-reduce-free-m1-qkv-o-plus-naive: that stack combined m=1 row-owner wins with cell B and set the current best 1.2566; this stacks the next three per-cell changes, each individually below the 1.24% margin, so together they can clear it"
review: proceed
status: hold
review_at: 2026-10-26T215633Z
review_companions: null
hold_reason: "sprint 2026-10-26 21:58Z: after the adversary's narrowing only pooler K-halves kind (a) can route, and kind (a) rounds each K-half's fp32 partial to bf16 before the final add (lower precision than SGLang's fp32 split-K; fails our fp32-accumulation rules gate); kind (b) fp32 partials never passed a gate. Not run."
---

# combo-verify-cells-pooler-khalf-plus-qkv-rowowner-m8-m16

## Change

Stack for the board. Build on the branch of pooler-k-halves-fp32-partials-restack-m128-m256 (depends_on). Port, by 3-way git merge-file of their patches, the cells of qkv-rowowner-a2-widen-m8-m16 and bf16-verify-per-cell-cublaslt-plan-selector-o-qkv256-unpooler, but only the cells whose own run printed PATH CHOSEN candidate AND whose judged row moved the right way by more than the 1.5% per-row noise (read their Attribution tables); a cell whose own run kept master or lost its row is omitted, and the change says which. Every ported cell keeps its own flag, canary, correctness check and hardened gate (two rounds, >= 1.03, IQR <= 1.5x master), so the combo can only route what re-wins in this run's prepare. Do not touch m=1 cells, m<=32 streams or the router. Print one FINAL FLAGS line listing every cell's route. If neither sibling has harvested when this is claimed, wait (do not run it as a pooler-only rerun).

## Expected

- `layer=pooler load=verify_* m=*` faster: K-halves fp32 partials (from depends_on)
- `layer=qkv_gate load=verify_1 m=8` faster: A2 row-owner at m=8 if its run routed
- `layer=qkv_gate load=decode_16 m=16` faster: NT=16 row-owner if its run routed
- `layer=o load=verify_* m=*` faster: per-cell Lt plan if its run routed

## Risk

Sizing: pooler +0.5-0.9%, qkv m=8/16 +0.45-0.7%, Lt cells +0.4-1.0%: about +1.3-2.5% combined, enough to clear the ~1.24% margin and the cause of this being the submit candidate. Inherits the fault risk of its parts; if the qkv m=16 instance or the Lt workspace path has never run in a judge, the first measurement is an okc check. Risk of dilution (LESSONS: adding the M128 Lt cell diluted the M=1 wins): each cell is gated in-run.

## Adversarial review

Reviewed 2026-10-26T215633Z by an adversarial subagent (model opus).

**Verdict: proceed.** Proceed only as a narrowed restack onto master 2bcfe7d4 (bulk12, 1.2664): of the three cells only pooler K-halves survives (m=128 a robust row win, m=256 a gate coin flip), plus at most the noise-level o m=128 Lt plan; the claimed +1.3-2.5% is really about +0.3-1.0%, below the 1.24% one-run margin.

### Justification
Pooler K-halves kind (a) (bf16 partials) routed m=128 in both runs (21.36 / 21.81 us vs master 22.86-23.10); kind (b) (fp32 partials) never passed a gate (0.934-0.991). qkv row-owner m8/m16 lost its own gates (A2w8 0.949/0.959, A2w16 0.773/0.779) and bulk12's b12x qkv m=8 also lost; the per-cell Lt qkv256 lost on the judge for the third time; o128 is noise-level (7.68 -> 7.24 inside master's band). The khalf patch needs 2 keep-both hunks on bulk12.

### Confounders on the current kernel
- bulk12 section (10) landed after this hypothesis; build on master 2bcfe7d4, not the pre-bulk12 depends_on branch
- BEST.json still records 741072dc / 1.2566: register a base result for blob 2bcfe7d4 (scored 3834ceec, 1.2664) before harvest
- pooler-nn-layout-widen-m256 owns the m=256 incumbent
- restack-three-cell-wins owns nn_ok shared at m=128

### Interference and A/B needs
No companion: the touched cells share no (layer, m) with landed cells; PATH CHOSEN and FINAL FLAGS state what routed.

### Open questions
- Port o128 or not (+0.07% headline)?
- Register a base result for blob 2bcfe7d4 before harvest
- Decide m=256 deterministically?
- Is the user comfortable putting bf16 K-half partials (kind a) on the board?

