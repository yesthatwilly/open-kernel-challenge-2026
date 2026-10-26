# Campaign log: 2026-10-26-decode-first-rewrite

Append-only. Newest entries at the bottom. Every check and scored run gets an entry: question asked, job,
rows that moved, decision. A new session resumes from the last entry.

## 2026-10-26T04:38Z: campaign approved, execution started

- The user approved PLAN.md ("execute the plan").
- Baseline: decode board 1.3164 (eaa8cc0f), master 1.3142 (a903b9d0). moe 1.4021, attention 1.5347, bf16 1.2566, prefill 1.468. Overall rank 480 of 1163.
- State at start: master d89b52b1 clean. One in-flight churn run: prefill `attn-full-prefill-wgmma-s2-warpspec-pingpong-expoffload`.
- Quota day 10-26 began at 00:00Z. Plan for today: 2 decode checks, 1 moe check, nothing scored.

## 2026-10-26T05:10Z: Phase 0 done; W0 infra built; first decode check dispatched

- Campaign docs committed on master (cd7fe010). Churn paused (11a14fc8): the harvest-only tick found nothing in flight.
- Branch `campaign/decode-first-rewrite`, worktree `.worktrees/decode-first-rewrite`, commit a1abc1d1:
  - `rewrite/build.py` generates each `<p>/submission.py` from `rewrite/src/<p>.py` by expanding `#@include`, then lints.
  - `rewrite/lib/rt.py` (W0): read flush, graph timing with an empty-graph baseline, selection rule, judge-form checks.
  - `rewrite/bin/report.py`: per-shape and per-load comparison against a base job plus the PATH CHOSEN lines.
- No nvcc or torch on this node; CUDA is validated on the judge, behind an Opus audit for each new module.
- **decode W1 check, job 9dacd0f59a174eaf.** Base = SKW branch a48a10b3 (master + SKW). Attention gate now bounds new o
  by 1.75x the fp64 FP8-query reference error (judge allows 2.5x) with the judge row form; S3 front code-gap gate off;
  routing-aware layer checks (SKW, and T=8 when an S3 T=8 cell is live); read-flush timers; SKW start 200 s.
  Question: which stranded cells (S1 A/B, S3 x7, SKW b32/v1) pass and win, and the headline effect.
- **moe ladder (W2 step 1)**: built (ea_common.cuh, ea_ladder.cuh, ea_ladder.py), audit running before its check.
