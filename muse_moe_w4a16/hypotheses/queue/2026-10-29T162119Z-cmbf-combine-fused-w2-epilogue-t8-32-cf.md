---
problem: muse_moe_w4a16
slug: cmbf-combine-fused-w2-epilogue-t8-32-cf
created: 2026-10-29T162119Z
base: master
area: "post-expert norm+combine"
change: "Confirmation rerun of cmbf-combine-fused-w2-epilogue-t8-32: its implementation commit 42ee20ddf03e82dd4a76c80a7d6185687e5a16ea unchanged except one trailing comment, run as a check and pooled with job a5aee75176174ce4 at harvest."
expected: []
risk: low
depends_on: cmbf-combine-fused-w2-epilogue-t8-32
priority: 1
kind: confirm
controls_for: cmbf-combine-fused-w2-epilogue-t8-32
confirm_of: cmbf-combine-fused-w2-epilogue-t8-32
confirm_of_job: a5aee75176174ce4
review: proceed
status: queued
---

# cmbf-combine-fused-w2-epilogue-t8-32-cf

## Change

Confirmation rerun of cmbf-combine-fused-w2-epilogue-t8-32: its implementation commit 42ee20ddf03e82dd4a76c80a7d6185687e5a16ea unchanged except one trailing comment, run as a check and pooled with job a5aee75176174ce4 at harvest.

## Why

Queued by churn.py (harvest): `cmbf-combine-fused-w2-epilogue-t8-32` ran 1.9675x against best 1.9521x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.
