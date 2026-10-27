---
problem: muse_prefill_layer
slug: experts-rega-w2-epilogue-prefetch-bulk-store-cf
created: 2026-10-27T140020Z
base: master
area: "w13 + w2 expert GEMMs (rega w2 consumer epilogue: loads and stores)"
change: "Confirmation rerun of experts-rega-w2-epilogue-prefetch-bulk-store: its implementation commit 1f1f74edd899b3edf61b517904e57d665ccfb412 unchanged except one trailing comment, run as a check and pooled with job 665c32dd8c794c0b at harvest."
expected: []
risk: low
depends_on: experts-rega-w2-epilogue-prefetch-bulk-store
priority: 1
kind: confirm
controls_for: experts-rega-w2-epilogue-prefetch-bulk-store
confirm_of: experts-rega-w2-epilogue-prefetch-bulk-store
confirm_of_job: 665c32dd8c794c0b
review: proceed
status: queued
---

# experts-rega-w2-epilogue-prefetch-bulk-store-cf

## Change

Confirmation rerun of experts-rega-w2-epilogue-prefetch-bulk-store: its implementation commit 1f1f74edd899b3edf61b517904e57d665ccfb412 unchanged except one trailing comment, run as a check and pooled with job 665c32dd8c794c0b at harvest.

## Why

Queued by churn.py (harvest): `experts-rega-w2-epilogue-prefetch-bulk-store` ran 1.5010x against best 1.4889x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.
