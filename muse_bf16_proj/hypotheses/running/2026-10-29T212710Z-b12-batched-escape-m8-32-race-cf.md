---
problem: muse_bf16_proj
slug: b12-batched-escape-m8-32-race-cf
created: 2026-10-29T212710Z
base: master
area: "bf16 dense GEMM cells at m=8..32 (pooler, qkv_gate, o): bulk12 kernels"
change: "Confirmation rerun of b12-batched-escape-m8-32-race: its implementation commit 84995d9e32799d8c940b71c9d279bedaad12e6d1 unchanged except one trailing comment, run as a check and pooled with job bced57b42b87444d at harvest."
expected: []
risk: low
depends_on: b12-batched-escape-m8-32-race
priority: 1
kind: confirm
controls_for: b12-batched-escape-m8-32-race
confirm_of: b12-batched-escape-m8-32-race
confirm_of_job: bced57b42b87444d
review: proceed
status: running
stage: claimed
claimed_at: 2026-10-29T212727Z
base_sha: 84995d9e32799d8c940b71c9d279bedaad12e6d1
base_blob: 5ef62a2078f6922e02a5c5c4e9aa9c7a1b3e31ab
branch: hyp/muse_bf16_proj/b12-batched-escape-m8-32-race-cf
worktree: .worktrees/muse_bf16_proj/b12-batched-escape-m8-32-race-cf
mode: null
restack_of: null
stale_base: null
---

# b12-batched-escape-m8-32-race-cf

## Change

Confirmation rerun of b12-batched-escape-m8-32-race: its implementation commit 84995d9e32799d8c940b71c9d279bedaad12e6d1 unchanged except one trailing comment, run as a check and pooled with job bced57b42b87444d at harvest.

## Why

Queued by churn.py (harvest): `b12-batched-escape-m8-32-race` ran 1.2871x against best 1.2794x, above the best but inside the one-run margin (1.24%). Two pooled runs need 1.07%. No new idea is tested, so it skips the adversarial review; the user sanctioned confirm reruns on 2026-10-25.
