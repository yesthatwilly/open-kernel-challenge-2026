## Now

- **Quota policy (the user, 2026-10-26 ~19:05Z, PLAN §6 top and §11; keep this bullet first):** never idle until a
  reset, and never pre-build hours ahead of one. When a problem's checks run out, write its reset plan, then work on the
  highest-gain problem that still has checks today, with immediate check feedback. Order: decode > moe > bf16_proj >
  attention_decode > prefill (last). On bf16 / attention, favour kernels that port into decode, and submit when a check
  beats that board. Every check must answer a new question; using every check is not required; no identical bytes
  re-rolled.
- updated: 2026-10-26T20:57Z
- step: **bf16_proj**.
  - Done: bulk12 m=1 scored at 1.2664 (board 1.2566 -> 1.2664, rank 381 -> 355) and landed on master.
  - Building: bulk12 for m = 8 / 16 / 32 (branch campaign/expert-bandwidth after 3acb748e, uncommitted). b12x_kernel
    streams the NT token rows of x with each chunk.
    - Cells: pooler / qkv / o at m=8, then pooler / o / full unpooler at m=16 / 32.
    - Each layer's codec stats, instance gates and PATH CHOSEN lines are re-printed at the end of prepare.
  - Next: audit, then one check. Submit if it beats 1.2664.
- in flight: nothing
- at 00:00Z (10-27), decode / moe, from the drafts in the session scratchpad (submit_attempt.md, dqab_attempt.md,
  anat_attempt.md):
  1. `okc submit` of `submission.edoff-27ef83df.py` (check f59cb301, 1.5347). Verify that the content hash is
     613b6c2d..., then land it on master.
  2. moe check at eacf0722:
     - the ladder dq A/B (dq2 / dq3 / dq4);
     - H=-8;
     - v4c.
  3. decode check at eacf0722: H=-8, v4c, and the anatomies.
- quota left 10-26: decode 0 checks / 0 scored, moe 0 / 4, bf16 11 / 9, attention 14 / 10, prefill 16 / 10
- boards: decode **1.4956** (rank 347 of 740; 1.5347 would be about 318), moe **1.6851** (rank 370 of 844),
  bf16 **1.2664** (rank 355 of 1090; rank 300 = 1.2865, 200 = 1.3503), overall 347 of 1228
