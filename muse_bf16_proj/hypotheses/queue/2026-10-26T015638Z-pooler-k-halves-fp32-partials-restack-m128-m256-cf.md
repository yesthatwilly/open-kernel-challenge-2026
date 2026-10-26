---
problem: muse_bf16_proj
slug: pooler-k-halves-fp32-partials-restack-m128-m256-cf
created: 2026-10-26T015638Z
base: master
area: "pooler GEMM (m=128/256, K-split as a 2-batch cuBLAS GEMM)"
change: "Confirmation rerun of pooler-k-halves-fp32-partials-restack-m128-m256: its implementation commit 659ff4783582d8103e09cf89dd5fae13e57f96cf unchanged except one trailing comment, run as a check and pooled with job de95f81782374124 at harvest."
expected: []
risk: low
depends_on: pooler-k-halves-fp32-partials-restack-m128-m256
priority: 1
kind: confirm
controls_for: pooler-k-halves-fp32-partials-restack-m128-m256
confirm_of: pooler-k-halves-fp32-partials-restack-m128-m256
confirm_of_job: de95f81782374124
review: proceed
status: queued
---

# pooler-k-halves-fp32-partials-restack-m128-m256-cf

## Change

Confirmation rerun of pooler-k-halves-fp32-partials-restack-m128-m256: its implementation commit 659ff4783582d8103e09cf89dd5fae13e57f96cf unchanged except one trailing comment, run as a check and pooled with job de95f81782374124 at harvest.

## Why

Queued by churn.py (harvest): `pooler-k-halves-fp32-partials-restack-m128-m256` ran 1.2683x against best 1.2566x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.
