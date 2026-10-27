# The 3:1 daily rhythm (the user, 2026-10-27 ~00:07Z; standing until the challenge closes)

Invoke with `/3to1` (one tick) or `/loop /3to1` (self-paced, autonomous). It supersedes every earlier quota policy
(the 19:05Z "using every check is not required" rule and the ~23:00Z STOP banner are retired). The challenge rules,
the attempt-file gate, the pre-submission rules checklist and "never re-roll identical bytes" are unchanged.

The user's words: "at the start of the day, i want big rewrites and overhauls to be planned, built, and scheduled.
once decode problem has exhausted its reasonably-dispatched checks/submits, i want attention redirected to the other
problems for their minor gains. [...] the last 6 or so hours of every zulu day can therefore be used for either the
normal /hypothesize or for just other work advancing these solutions. but at the start of a zulu day, i want the
biggest and most promising approaches developed and deployed and checked; i want it to be on opus5.5 ultracode.
after 18 hours or so of the day, with six remaining, heartbeats should be focusing on using up all CHECKS. not using
all scoring submissions is OK, but use a scoring submission any time there is a gain to be secured."

## Phases (by `date -u` hour; the quota resets at 00:00Z)

| phase | UTC | focus | how |
|---|---|---|---|
| **A: overhaul** | 00:00Z until decode's reasonable checks are spent (at latest 18:00Z) | the biggest, most promising decode rewrites (moe is decode's expert-block testbed and counts here) | Opus 5.5 **ultracode**: a Workflow per phase (design panel → build in worktrees → adversarial audit → check). Plan and launch at the start of the day, not later. |
| **B: minor gains** | after phase A, until 18:00Z | the other problems' smaller gains, in order moe > bf16_proj > attention_decode > prefill; favour decode-portable work | queued hypotheses (`/churnsubs`) and the campaign's per-problem items; ≤ 3 agents |
| **C: spend checks** | 18:00Z-24:00Z | **use up every remaining CHECK on every problem**; `/hypothesize` to keep queues deep enough; other work that advances the solutions (and the next day's phase-A plan) | heartbeats dispatch checks; ≤ 3 agents |

**"Decode's reasonable checks are spent"** means one of: decode has no checks left; or no decode build is ready that
answers a new question and none will be ready within ~2 h. Moe (testbed) checks follow the same test. Phase A builds
still in flight continue in the background while phase B runs.

## Rules for every phase

- **Scored submits:** send one whenever a check beats that problem's board (the same bytes, attempt file first).
  Leaving scored runs unused is fine. A check under ~1.2% above the board is borderline (scored spread); still
  submit if no better candidate is due that day.
- **Checks:** every check answers a question (phase C included); never re-roll identical bytes. In phase C, a
  cheap question is acceptable: the goal is to spend all checks.
- **Agents:** ultracode workflows in phase A (the session's workflow size guideline applies); otherwise at most 3
  concurrent agents. Delegate execution; keep the main context lean.
- **Effort (the user, 2026-10-27 ~03:30Z: "use max effort for phase A builds, bake it in"):** every agent in a
  phase-A workflow (design panel, judge, builder, auditor, fixer) runs at `effort: 'max'`: pass it on each `agent()`
  call, whatever the session's /effort is. Phase B/C agents, harvests and routine dispatches keep the session effort.
- **Heartbeats:** wake on job completion while checks are in flight (fallback 1200-1800 s); idle 1800-3000 s in phase
  A/B; in phase C, no longer than needed to keep checks flowing (checks take ~10-20 min).
- **Every tick:** `date -u`; check the steer banners (INDEX.md top, active PLAN.md, active campaign CLAUDE.md);
  read LOG Now before rewriting it; harvest finished jobs; then act for the current phase.
- **Day boundary (first tick at/after 00:00Z):** record the boards, then start phase A: re-rank the levers (the active
  campaign's DISTILL/PLAN), choose the 1-3 biggest rewrites, and launch the ultracode workflow before anything else.
- **Phase C close (~23:00Z):** write the next day's phase-A plan into the active campaign LOG (the levers, the
  workflow's scope), so 00:00Z starts building immediately.
- **Close of the challenge:** the working deadline is 11-05 06:59Z (last dispatch ~06:30Z); DISTILL §4.5's end game
  applies on 11-04.
