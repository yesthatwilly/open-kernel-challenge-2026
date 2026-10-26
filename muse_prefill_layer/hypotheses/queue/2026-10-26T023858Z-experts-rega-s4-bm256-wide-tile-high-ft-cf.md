---
problem: muse_prefill_layer
slug: experts-rega-s4-bm256-wide-tile-high-ft-cf
created: 2026-10-26T023858Z
base: master
area: "w13 + w2 expert GEMMs at 1024 < T <= 16384 (rega high + ft classes): 256-weight-row CTA tile with in-warpgroup dequant/MMA overlap"
change: "Confirmation rerun of experts-rega-s4-bm256-wide-tile-high-ft: its implementation commit 97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2 unchanged except one trailing comment, run as a check and pooled with job a6f6063051f14750 at harvest."
expected: []
risk: low
depends_on: experts-rega-s4-bm256-wide-tile-high-ft
priority: 1
kind: confirm
controls_for: experts-rega-s4-bm256-wide-tile-high-ft
confirm_of: experts-rega-s4-bm256-wide-tile-high-ft
confirm_of_job: a6f6063051f14750
review: proceed
status: queued
---

# experts-rega-s4-bm256-wide-tile-high-ft-cf

## Change

Confirmation rerun of experts-rega-s4-bm256-wide-tile-high-ft: its implementation commit 97bc07f18b4cbc9aa9b228fdea23c980cefe6fc2 unchanged except one trailing comment, run as a check and pooled with job a6f6063051f14750 at harvest.

## Why

Queued by churn.py (harvest): `experts-rega-s4-bm256-wide-tile-high-ft` ran 1.4758x against best 1.4677x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.
