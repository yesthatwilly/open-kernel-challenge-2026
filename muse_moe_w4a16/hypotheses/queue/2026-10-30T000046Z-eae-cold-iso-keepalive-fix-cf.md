---
problem: muse_moe_w4a16
slug: eae-cold-iso-keepalive-fix-cf
created: 2026-10-30T000046Z
base: master
area: "prepare-only correctness: [ea-enc] T=256 cold-half iso timing graphs"
change: "Confirmation rerun of eae-cold-iso-keepalive-fix: its implementation commit 92e1dcd354eded46c17b1dbeba6fa86cbec59b46 unchanged except one trailing comment, run as a check and pooled with job 582bb33467dc4663 at harvest."
expected: []
risk: low
depends_on: eae-cold-iso-keepalive-fix
priority: 1
kind: confirm
controls_for: eae-cold-iso-keepalive-fix
confirm_of: eae-cold-iso-keepalive-fix
confirm_of_job: 582bb33467dc4663
review: proceed
status: queued
---

# eae-cold-iso-keepalive-fix-cf

## Change

Confirmation rerun of eae-cold-iso-keepalive-fix: its implementation commit 92e1dcd354eded46c17b1dbeba6fa86cbec59b46 unchanged except one trailing comment, run as a check and pooled with job 582bb33467dc4663 at harvest.

## Why

Queued by churn.py (backlog): `eae-cold-iso-keepalive-fix` ran 2.0469x against best 2.0201x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.
