---
name: 3to1
description: The 3:1 daily rhythm for the okc kernel challenge, run autonomously. 00:00Z onward builds and checks the biggest decode rewrites on Opus 5.5 ultracode workflows; once decode's reasonable checks are spent it turns to the other problems' minor gains; 18:00Z-24:00Z spends every remaining check and runs /hypothesize. Scored submits whenever a check beats a board. Use for "/3to1", "3:1 time plan", "get back on schedule", "back on the daily rhythm", or under "/loop /3to1".
user-invocable: true
---

# /3to1: one tick of the 3:1 daily rhythm

The policy is `campaigns/DAILY-RHYTHM.md` (the user's standing steer, 2026-10-27). Read it at the first tick of a
session. Invoking this skill is the user's authorization to run that rhythm autonomously, ultracode workflows in
phase A included. Never ask the user a question during a tick: record open decisions in the active campaign's LOG
and pick the default.

## Each tick, in order

1. **Orient (cheap).**
   - `date -u` (never estimate a stamp). Phase: before 18:00Z is A or B; 18:00Z onward is C.
   - The active campaign is the `status: active` row of `campaigns/INDEX.md`.
   - Check the steer banners: `git status` / `git diff` on `campaigns/INDEX.md`, the active `PLAN.md` and the active
     `CLAUDE.md`. Act on a new banner this tick and acknowledge it with a dated LOG entry.
   - Read the LOG `## Now` block before rewriting it, and carry forward every user bullet.
   - Run `okc problems` for the quota. Run `okc leaderboard <p> --json` only at the day's first tick and after a
     scored run.
2. **Harvest** every finished job, using a subagent when the record is large. Archive the run (`rewrite/bin/archive.py`),
   compare it with `report.py` and the campaign scratchpad's `rowcmp2.py`, and fill the attempt file's result. Add at
   most 10 lines to the LOG per run.
   - **A check beats its board:** send a scored run of the same bytes (a new attempt file plus the rules checklist).
   - **A scored run beats the board:** land the file on master as `<problem>/submission.py` and log it.
   - Register campaign scored runs in churn's BEST.json before any churn run on that problem.
3. **Act for the phase.**
   - **Day start** (the first tick at or after 00:00Z with no phase-A workflow running today):
     - Record the boards in the LOG.
     - Take the phase-A plan written the night before. If there is none, re-rank the levers in the active campaign's
       DISTILL/PLAN and FINDINGS.
     - Choose the 1-3 biggest, most promising decode rewrites (moe counts as decode's expert-block testbed).
     - Launch an **ultracode Workflow** for them before anything else. Load `workflow-authoring` first. Shape it as a
       design panel → a build in worktrees (diagnostic rung first when a new engine depends on an unmeasured rate) →
       an adversarial audit of every new CUDA module → `cuda_syntax.py` / lint → the attempt file and rules checklist.
       Each build returns a check-ready commit, and the main thread dispatches the checks.
     - **Every phase-A workflow agent runs at max effort:** pass `effort: 'max'` on every `agent()` call (design,
       judge, build, audit, fix). The user asked for this on 2026-10-27, and it applies regardless of the session's
       /effort.
   - **Phase A** (decode's reasonable checks not yet spent):
     - Dispatch the decode/moe checks that phase-A builds make ready, one new routable cell per GEMM per check.
     - Start the next ultracode phase (build or fix) from the harvests.
     - Decode's reasonable checks are spent when decode has no checks left, or when no build is ready and none will
       be within ~2 h. Then go to phase B; builds still in flight keep running.
   - **Phase B** (until 18:00Z): the other problems' minor gains, in the order moe > bf16_proj > attention_decode >
     prefill.
     - Favour decode-portable work.
     - Sources: the "run first" verdicts in the campaign's DISTILL §4.4, the churn queues (`/churnsubs`) and the
       campaign items.
     - Use at most 3 agents.
   - **Phase C** (18:00Z-24:00Z): **spend every remaining check on every problem.**
     - Count each problem's checks left and fill them: churn-queued hypotheses, cheap diagnostic questions, rungs.
     - When the queues are shallower than the checks left, run `/hypothesize <p>`.
     - A cheap question is fine; re-rolling identical bytes is not.
     - By ~23:00Z, write tomorrow's phase-A plan in the LOG: the levers, the workflow scope, and what tonight's checks
       answered.
4. **Close the tick.**
   - Commit on the campaign branch at every working step, and land scored files on master immediately.
   - Rewrite the LOG Now block with the phase, what is in flight and the next action.
   - Under `/loop`, end with `ScheduleWakeup`:
     - while checks are in flight, run `okc status <job> --wait` in the background and set a 1200-1800 s fallback;
     - when idle in phase A or B, wake in 1800-3000 s;
     - in phase C, set the wake that keeps checks flowing, no longer than the next check's expected finish;
     - at the day boundary, wake at about 00:00Z.

## Standing rules (unchanged by this rhythm)

- Write an attempt file in `<problem>/attempts/` before every check or submit. It needs the `**Job:** \`<job id>\``
  placeholder (the hook enforces it) and the rules checklist ticked against the actual diff. If any item is false or
  unsure, do not run.
- Re-fetch the rules version at the day start (2026-10-22.1). Withdraw a rule-breaking scored run at once.
- Use at most 3 concurrent agents outside the phase-A ultracode workflow. Delegate builds, greps over generated
  files, audits and harvests, and keep the main context to decisions and the LOG.
- The working deadline is 11-05 06:59Z, with the last dispatch at ~06:30Z. DISTILL §4.5's end game applies on 11-04.
