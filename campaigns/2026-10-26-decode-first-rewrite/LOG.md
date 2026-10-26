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

## 2026-10-26T05:15Z: decode W1 check 9dacd0f59a174eaf: 1.3371 [1.3170, 1.3571], 13/13. M0 reached; scored submit sent

- vs board 1.3164: +1.57% (margin 1.24%). vs master a903b9d0 1.3142: +1.74%.
- Every stranded cell passed the judge-faithful gate and won its selector: S1 A 28.6 vs 39.5 us, S1 B 38.2 vs 45.6,
  S3 c1-b16/c1-b32/c2-t8/c2-t128/c2-t256/c3-t128/c3-t256/c4-t8 all NEW (3-18 us each), SKW b32 38.4 vs 45.7, SKW v1
  33.5 vs 38.1. S2a/S2b unchanged (NEW).
- Rows (base a903b9d0 -> run): v16 full 593.4 -> 547.0, v1 sliding 181.1 -> 177.0, d32 sliding 239.5 -> 236.2,
  v32 full 866.6 -> 851.4; d16 full 290.5 -> 302.3 (drift: same cells as master there).
- Loads: d1 1.441 (1.449), d16 1.359 (1.325), d32 1.250 (1.206), v1 1.502 (1.447), v16 1.282 (1.266), v32 1.213 (1.214).
- Decision: scored submit of the same bytes, job 69db8aeb251944c5 (PLAN §6, M0).
