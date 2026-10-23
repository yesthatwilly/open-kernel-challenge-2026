---
name: churnsubs
description: Execute queued kernel hypotheses autonomously and spend the day's okc quota (checks and scored submits) to 100% when prudent. Each tick harvests finished runs into <problem>/hypotheses/done/, promotes proven gains to master and the board, decides how many new runs are prudent, and dispatches hypotheses (worktree → code change → lint → attempt file → okc). Never writes hypotheses (that is /hypothesize). Use for "/churnsubs", "churn the queue", "spend today's runs", "run the hypotheses", or under "/loop churnsubs".
user-invocable: true
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash(python3 -I */.claude/skills/churnsubs/bin/churn.py *)
  - Bash(python3 -I */.claude/skills/churnsubs/bin/okc_lint.py *)
  - Bash(okc check *)
  - Bash(okc submit *)
  - Bash(okc status *)
  - Bash(okc problems *)
  - Bash(okc jobs *)
  - Bash(okc leaderboard *)
  - Bash(git -C */.worktrees/* *)
  - Bash(git log *)
  - Bash(git diff *)
  - Bash(git status *)
  - Bash(cat *)
  - Bash(ls *)
---

# churnsubs — execute the hypothesis queue, spend the quota prudently

One invocation = one **tick**. Under `/loop churnsubs` the tick ends by scheduling the next wake
with the `wake_s` the budget returns. All state is in files, git and okc; a tick that dies
anywhere is repaired by the next one's `recover`. `churn.py` does every mechanical step and
**never runs okc itself**: it prints the exact `okc` command and you run it as a bare Bash call so
the repo's attempt gate fires.

```
C="python3 -I $CLAUDE_PROJECT_DIR/.claude/skills/churnsubs/bin/churn.py"
```

## Fixed rules

- **Never create a hypothesis file yourself.** The only hypotheses churnsubs causes to exist are
  the A/B companions the adversarial reviewer specifies (step 4). If the queue is empty, report
  the shortfall and suggest `/hypothesize`. Never "fix" a failed hypothesis by re-running it: identical code is never re-run,
  and a fix is a new hypothesis for /hypothesize to mint.
- **One run per hypothesis.** `check` while the problem has checks left, else a scored `submit`
  (never a scored run for `risk: high`). The board shows only the best run, so a slow scored run
  costs nothing in rank.
- **Spend to 100% when prudent.** Prudent means: lint-clean, rules reviewed, not superseded by a
  landed finding, not a duplicate. It never means spending a run on a hypothesis you judge weak:
  `hold` it with a reason and let the shortfall show in the index.
- **Rules compliance is a gate, not paperwork.** Before `commit-impl` you must have ticked every
  box of the `## Rules check` section in the hypothesis file against the actual diff (see
  `ATTEMPT_TEMPLATE.md` for the list and `rules/technical-rules.llms.txt` for the source). If any
  box cannot be ticked, `hold` the hypothesis instead.
- **Only churnsubs commits on master.** Work on `submission.py` happens in the hypothesis's
  worktree, never in the main tree. A dirty `<p>/submission.py` on master only defers promotion.

## The tick

1. **Plan.** `$C tick` → JSON with `promote` (scored submits to run), `dispatch` (per-problem
   picks with mode), `wake_s`, `needs_human`, and the sub-reports. It already ran preflight (lock,
   master, okc signed in), intake (commits new queue files that pass lint), recover, harvest,
   harvest-promotions, promote (merged gains + wrote their submit attempt files), budget, index.
   Exit 2 = refused (another tick holds the lock, or the main tree is not on master): stop and say
   why. Anything in `needs_human`: report it and still continue with the rest.
2. **Report the harvest** in one or two lines per result: slug → outcome, speedup and CI, the
   attribution summary from its `.result.json`. Mention gains that were merged or are `unmerged`.
3. **Promotions.** For each entry in `promote`, run its `run` command verbatim, then
   `$C record-submit <p> <slug> '<okc output>'`.
4. **Adversarial review (Fable), one subagent per pick, in parallel.** Every hypothesis is
   attacked by a fresh Fable context before it is claimed; `claim` refuses anything whose
   frontmatter is not `review: proceed`. For each pick whose `review` is not `proceed`:
   - Gather its inputs: `$C landed <p>` (hypotheses already on master's kernel, with patch
     paths), the hypothesis path, `<p>/hypotheses/FINDINGS.md`, `BEST.json`, the base
     `submission.py` (master, or the parent's worktree for an A/B companion), and the other
     queued/running slugs.
   - Spawn `Agent` with `subagent_type: general-purpose`, **`model: fable`**, and the prompt in
     `.claude/skills/churnsubs/ADVERSARY_PROMPT.md` with its `{PLACEHOLDERS}` filled in. The
     adversary's specific job: judge the justification, name what already-landed hypotheses
     confound it, and decide whether *this* hypothesis needs an A/B companion (the change with a
     landed hypothesis reverted or tuned differently, or a control isolating a sub-change) for
     its one result to mean anything. It returns JSON only.
   - Save the JSON to the scratchpad and run `$C review-record <p> <slug> <file>`. That writes
     `## Adversarial review` into the hypothesis, sets `review: proceed|hold` (hold → status
     hold with the reason), and **queues each companion** as a real hypothesis file via
     `hyp.py new --kind ab-revert|ab-tune|control --controls-for <slug> --depends-on <slug>`.
     Companions are the one case where churnsubs causes hypothesis files to exist; they are
     always the adversary's, never yours.
   - Companions base on the parent's implementation commit and become eligible as soon as the
     parent is implemented, so the A/B pair runs the same day. If the adversary held the pick,
     re-run `$C budget` to refill it. Do not argue with a hold in the same tick; the reason is
     in the file for the human.
   - `$C claim --skip-review` exists for a human who has read the hypothesis and says so; never
     use it on your own.
5. **Dispatch each pick** (parallel is fine: run the implementation step for several picks as
   parallel subagents, one per worktree; keep steps 5a and 5c–5e in this thread, in order):
   - 5a `$C claim <p> <slug>` → `worktree`, `branch`, `submission`, `spec`, `base_sha`.
   - 5b **Implement** the `change` in `<worktree>/<p>/submission.py` exactly as specified,
     modestly, touching only the named area; keep the first-line marker. Then edit the hypothesis
     file (its `md` path, on master) and write `## Implementation notes` (what you changed, where,
     any deviation from the spec) and `## Rules check` with every box ticked only after checking
     the diff. Run `python3 -I .../okc_lint.py <submission> --problem <p>` and paste its findings
     under Rules check. Read AGENTS.md for the problem if you have not this session.
   - 5c `$C commit-impl <p> <slug>` — lints, rejects duplicates and no-change diffs, commits on
     the branch. Exit 1 = rejected without a run (it already filed the hypothesis under
     `done/rejected/`); move on.
   - 5d `$C prepare-run <p> <slug>` → writes the attempt file on the branch, prints `run`.
   - 5e Run `run` verbatim, then `$C record <p> <slug> '<okc output>'`. If okc refused (quota,
     rate limit, the gate), do not retry blindly: say what it said; `recover` on the next tick
     resumes from `implemented`.
6. **Close.** `$C index`, then `$C unlock`. Under `/loop`, call `ScheduleWakeup` with `wake_s`
   and a reason naming in-flight jobs and the next planned dispatch. Outside `/loop`, say when
   the next tick should run.

## What to tell the user each tick

Three lines unless something needs them: what was harvested (outcome per slug), what the
adversary said (verdict per slug, companions queued), what was dispatched (slug, mode, job id),
and the budget line per problem (checks left / scored left /
in flight / shortfall + reason). `needs_human` items and `unmerged` gains always get their own
line with the file path.

## Reviewing one hypothesis without dispatching it

`/churnsubs review <problem> <slug>` runs step 4 alone on a queued hypothesis (no lock, no
claim), so a batch from /hypothesize can be reviewed and its companions queued before the loop
spends anything.

## Reading results by hand

- `<p>/hypotheses/INDEX.md` — every hypothesis, sortable columns; `hypotheses/INDEX.md` — roll-up
  and today's budget. `<p>/hypotheses/FINDINGS.md` — learnings by area, open follow-ups,
  do-not-retry list, where the current best spends its time.
- `<p>/hypotheses/done/<outcome>/<slug>.{md,patch,result.json}` and the branch
  `hyp/<p>/<slug>` (kept forever; worktree removed after harvest, kept for `unmerged` gains).
- `okc status <job>` and `profiles/submission.<job>.txt` in the worktree while it exists.
