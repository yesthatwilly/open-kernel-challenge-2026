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
review: required
status: queued
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
