# Public-plus: LOG

## Now (2026-10-31T00:34Z)

- **USER STOP, ~00:32Z (verbatim):** "do not launch any new work. get the four control runs back, harvest, process,
  plan out. but do not launch more agents or checks submissions or h100 work". Holds until the user lifts it.
  - No new agents, checks or H100 work. This track never submits anyway.
  - Allowed: harvest the 4 controls, process them (RESULTS.tsv, runs/, attempt Results), and write the plan for what
    comes next.
- **In flight:** 4 raw control checks, all queued for a GPU at 00:33Z:
  - bf16 B0-aa1 `a95479284a204d87`, B0-aa2 `280a3622e2384023`;
  - moe M0-aa1 `b3510dcc6553442b`, M0-aa2 `99b14fedc99142e9`.
  - Attempt files: branch campaign/public-plus e18ff591.
- **Build workflow `public-plus-build` (wf_b31b245a-546):**
  - Launched at 00:31Z and **stopped at 00:32Z** on the user's message.
  - Its 6 builders had only read PLAN/FINDINGS and created empty worktrees. No edits, no commits, no H100, no okc.
  - Worktrees `pp-*` removed and branches `pp/*` deleted (all were at e18ff591).
  - Script kept: `~/.claude/projects/-home-user-git-musespark/e0c4de02-bc78-4108-8476-c8340385be4b/workflows/scripts/public-plus-build-wf_b31b245a-546.js`.
    To restart it on the user's go: `Workflow({scriptPath})`, fresh, not resume.
- **Quota 10-31 after these 4:** bf16 18/20 checks, moe 18/20, the others 20/20. Scored runs unchanged.
- union-confirm: IDLE at master d6a0d421 (attention 2.005).

## History
- 2026-10-31 00:28-00:31Z: step 0 committed (master e241fdfd); branch campaign/public-plus + worktree; B0/M0 controls dispatched 00:30Z (e18ff591).

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
