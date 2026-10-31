# Public-plus: LOG

## Now (2026-10-31T00:29Z)

- **ACTIVE.** The user approved the plan at ~00:29Z. Before that the user asked for "a durable MD plan that can be run
  cold" (the reply when ExitPlanMode was rejected at ~00:20Z).
- **Step 0 is in progress.** On master: PLAN, FINDINGS, the inventory export and this LOG. Next:
  - create the branch and worktree;
  - run the B0/M0 raw controls (bf16, moe): 4 checks;
  - launch the build workflow `public-plus-build`.
- **Quota 10-31 (read 00:24Z):** 20/20 checks on every problem; scored runs 10/10 (attention 9/10). This track never
  submits.
- **union-confirm:** IDLE at master d6a0d421 (attention landed 2.005). Its next item is the decode swap-AB root cause
  on the H100, on the user's go. The default reserve for its checks: 6 each on attention and decode, 2 on the others.
- **Agents:** up to 6 (user, 10-30 ~23:45Z).

## History

- 2026-10-30 23:28Z - 10-31 00:14Z: read-only inventory workflow wf_1949358b-dd0 (11 agents, ~3.8M subagent tokens,
  45 min). The result is in FINDINGS.md and PLAN §3-4.
  - Ours is faster on 0 of 68 rows; the public file is faster on every one.
  - Of 48 graft candidates, 1 speed graft survived (attention W2).
  - The public attention, decode and prefill files fail our FA3-precision ruling; moe and bf16 are clean.
- 2026-10-30 ~23:45Z: user decisions (AskUserQuestion):
  - public base plus grafts;
  - MIT routes kept;
  - the main line gets checks first;
  - up to 6 agents.
