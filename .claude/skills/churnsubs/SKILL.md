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
  the A/B companions the adversarial reviewer specifies (step 4) and the `kind: confirm` reruns
  churn.py queues itself (see Judging). If the queue is empty, report the shortfall and suggest
  `/hypothesize`. Never "fix" a failed hypothesis by re-running it: a fix is a new hypothesis for
  /hypothesize to mint.
- **One run per hypothesis**, plus at most one confirm rerun of an unproven result (the user,
  2026-10-25: "yes, auto-confirm"). `check` while the problem has checks left, else a scored
  `submit` (never a scored run for `risk: high`, never for a confirm). The board shows only the
  best run, so a slow scored run costs nothing in rank.
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
   picks with mode), `diagnose` (problems due a diagnosis), `wake_s`, `needs_human`, and the
   sub-reports. It already ran preflight (lock, master, okc signed in), intake (commits new queue
   files that pass lint, and any new DIAGNOSIS.md), recover, harvest (pools confirms, queues new
   ones), harvest-promotions, promote (merged gains + wrote their submit attempt files),
   reclassify --queue-confirms (backlog results on today's master, 3 per problem per day),
   budget, index; `release-stale` runs after recover and moves zombie claims (claimed or implemented, no job
   for 24 h) back to the queue as holds. The **freshness gate** runs in budget and in `claim`: a queued
   primary is held with a typed `hold_kind` (`stale-board` when BEST moved more than `margin()` since its
   unhold / review / mint, `stale-blob` when master's submission.py is no longer the blob it was judged on,
   even if BEST barely moved, `no-value-case` when minted after the value bar without one). Companions,
   confirms and rewrite stages on a rewrite parent are exempt; a file with an explicit non-master `base`
   skips the blob check. A fresh adversarial review (`review-record`, proceed) re-queues a stale hold, and so
   does `unhold` (it re-bases the gate on today's board); `unhold` refuses a no-value-case file unless
   `--force`. Budget lists up to 2 stale holds per problem as `rereview` when checks would otherwise go
   unspent: send those to the adversary in step 4. Typed holds not re-reviewed in 72 h expire (no claim:
   they never ran). `budget --no-hold` reports `gated` without holding (index uses it);
   `claim --skip-freshness` bypasses the gate deliberately. `expire-holds` is not part of the tick: run `$C expire-holds --dry-run`, read
   the list, and only then run it without `--dry-run` (it never expires holds that wait on the
   user, a standing answer, another run or a condition, `HOLD_PROTECT`).
   Exit 2 = refused (another tick holds the lock, or the main tree is not on master): stop and say
   why. Anything in `needs_human`: report it and still continue with the rest.
1b. **Diagnosis (Opus), once per problem per UTC day.** For each problem in `diagnose`, spawn
   `Agent` (`subagent_type: general-purpose`, **`model: opus`**) with
   `.claude/skills/hypothesize/RESEARCH_PROMPT.md`, `{PROBLEM}` and `{REPO}` filled in, in
   parallel with step 4. Quote the predicted cost from COSTS.md first, and run
   `$C diagnose-start <p>` before spawning, so later ticks do not spawn it again for 3 h while it
   runs. It writes
   `<p>/hypotheses/DIAGNOSIS.md`; the next tick's intake commits it. Apply its "LESSONS edits"
   to `hypotheses/LESSONS.md` yourself, one `churn.py note <p> diagnosis "<edit and why>"` per
   edit, and append each line of its "Claims (proposed)" with `$C claim-add` (on master, main checkout). Its
   Actions are what `/hypothesize <p>` mints first when you refill the queue.
2. **Report the harvest** in one or two lines per result: slug → outcome, speedup and CI, the
   **mechanism verdict** (`mechanism:` in its frontmatter: `hit`, `miss`, `flat`, `no-base`) and the
   attribution summary from its `.result.json`. Mention gains that were merged or are `unmerged`. Then
   **record one CLAIMS line per result** (see "Claims and the shelf"); failed and rejected runs need none
   unless they settled a mechanism.
3. **Promotions.** For each entry in `promote`, run its `run` command verbatim, then
   `$C record-submit <p> <slug> '<okc output>'`. A landing made outside churn (a campaign, by hand) is
   registered with `$C register-landing <p> <job> --mechanism "<what landed> (<slug>)" --rows "<rows it owns>"`
   on master: BEST.json with `result` (mode from the run record), BEST.result.json, a `[landed]` CLAIMS line
   and a DECISIONS line, so `landed` and the adversary see it. It skips the claim when a `[landed]` line
   already cites the job; `--no-claim` backfills BEST.json `result` only.
4. **Adversarial review (Opus), one subagent per pick, in parallel.** Every hypothesis is
   attacked by a fresh Opus context before it is claimed; `claim` refuses anything whose
   frontmatter is not `review: proceed`. Confirms arrive as `review: proceed` (they test no new
   idea). For each pick whose `review` is not `proceed`, and each budget `rereview` item (a stale
   hold: the adversary re-derives its value case on today's board; proceed re-queues it):
   - Gather its inputs: `$C landed <p>` (hypotheses already on master's kernel, with patch
     paths, `[landed]` claims included), the hypothesis path, `<p>/hypotheses/CLAIMS.md`,
     `<p>/hypotheses/FINDINGS.md`, the active campaign's FINDINGS/DISTILL, `BEST.json` with the
     file's freshness, the base `submission.py` (master, or the parent's worktree for an A/B
     companion), and the other queued/running slugs. `python3 -I .claude/skills/churnsubs/bin/fill_prompt.py
     adversary <p> <slug>` fills all of them and prints the prompt's path (under the git-ignored
     `hypotheses/prompts/filled/`, never /tmp).
   - Spawn `Agent` with `subagent_type: general-purpose`, **`model: opus`**, and the prompt in
     `.claude/skills/churnsubs/ADVERSARY_PROMPT.md` with its `{PLACEHOLDERS}` filled in. The
     adversary's specific job: judge the justification, name what already-landed hypotheses
     confound it, and decide whether *this* hypothesis needs an A/B companion (the change with a
     landed hypothesis reverted or tuned differently, or a control isolating a sub-change) for
     its one result to mean anything. It holds a pick whose mechanism has a newer `[refuted]` /
     `[superseded]` claim, whose rows a `[landed]` claim owns, or whose board moved past the margin
     since `best_at_mint` / its review without a re-derived value case. It returns JSON only.
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
     A hypothesis whose `depends_on` parent finished without landing (neutral, regression, unmerged) is a
     restack: claim bases it on current master and 3-way applies the parent's patch (`restack: applied`,
     change already staged). On `restack: conflicts` the worktree is clean master and the implementer
     must hand-merge the parent's patch (`restack_patch`) before the hypothesis's own change. Companions
     and rewrite stages still build on the parent's commit; `stale_base: true` flags one whose parent
     predates master's current submission.py.
   - A `kind: confirm` pick skips 5b: `claim` already based it on the parent's `impl_commit`,
     appended the marker comment and wrote its notes and Rules check. Go straight to 5c.
   - 5b **Implement** (a subagent per pick, briefed with `IMPLEMENT_PROMPT.md`) the `change` in `<worktree>/<p>/submission.py` exactly as specified,
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

## Rewrites

`kind: rewrite` hypotheses replace a whole subsystem over staged runs. They are dispatched ahead of
primaries, harvested like any run, and read on the replaced kernels' own time per shape against the
incumbent and the floor (their `## Milestones`). A rewrite's next stage (`depends_on` a rewrite)
is eligible once the previous stage has run, whatever its outcome, because it builds on that code; a
regression is not a do-not-retry for a rewrite. Promotion is unchanged: only a stage whose headline
CI clears the floor lands on master.

**The 40% rule** (the user, 2026-10-24): `REWRITE_SHARE` (40%) of each problem's daily checks is
kept for rewrites and the rest for everything else; `budget` enforces it from the day's dispatched
checks. A side with nothing eligible releases its share to the other, and both shares open in the
last 2 hours, so the split never leaves a check unspent. When a problem has no eligible rewrite,
that is a shortfall to fix by minting one (`/hypothesize <p> --kind rewrite`), not by spending the
share on primaries all day.

## Claims and the shelf

`<p>/hypotheses/CLAIMS.md` is the curated findings file: append-only, one line per claim, never
regenerated (FINDINGS.md is, and prints the newest claim per mechanism at its top). Write claims only on
master in the main checkout (`claim-add` and `register-landing` write there even from a worktree); never edit
a line, append a newer one about the same mechanism instead. Each harvest records one line per result with
`$C claim-add <p> --tag T --mechanism "<area>: <what changed> (<slug>)" --rows "<expected shapes>" --evidence
"<job> (<p>/hypotheses/done/<outcome>/<file>.md)" [--reopen "<condition>"]`; the tag comes from the outcome
and the mechanism verdict:

| outcome | mechanism | tag |
|---|---|---|
| gain | hit | `landed` once promoted; `lead` when a confirm pooled it (2+ runs) and it is not on master; one run only: `shelf` |
| gain | flat | `unresolved` (a draw: the predicted rows did not move) |
| neutral | hit | `shelf` (stack only to >= 2.5% headline; FINDINGS lists the Shelf) |
| neutral | flat | `unresolved` after one run (inside row noise); `refuted` once 2+ runs (a confirm) agree |
| neutral | miss | `refuted`, with a `--reopen` rule |
| regression | flat or miss | `refuted`, with a `--reopen` rule |
| regression | hit | `unresolved` (its rows moved; the loss is elsewhere: name those rows in `--reopen`) |
| any | no-base | `unresolved` (rows were not compared with a base) |

A row hit from one run earns `shelf`, never `lead` (rows drift 1.5-2.5% per shape between runs). A rewrite
stage is not refuted by one slow run: tag it `unresolved` with the next stage as its reopen rule. `within-ci`
alone is no longer a Do-not-try: a neutral whose rows hit is shelf. Readers take the newest line per
mechanism, and a line is about the same mechanism when it **starts with the same hypothesis slug or names it
first in parentheses** (else only identical text supersedes), so always put the slug there.
`hyp.py claims <p>` prints the newest line per mechanism; `hyp.py new` refuses to mint a
`[refuted]`/`[superseded]` mechanism (it names the slug, or shares 2+ distinctive words with it) unless
`--prior` says what differs.

## Judging and confirm reruns

`classify` judges a run against the best on measured drift, not on overlap of okc's CIs (okc's
95% CI is a near-constant ±1.5%, 3-4x wider than real drift). One run's headline has sd
`SIGMA_RUN` ≈ 0.35% (22 identical-code pairs, check vs scored submit, 2026-10-25); a **gain** must
beat the best by `margin()` = 2.5 sd of the difference: 1.24% for one run, 1.07% for two pooled.
A result above the best but inside that margin is `unproven-vs-best`; if it clears the best by
`CONFIRM_MIN` (0.5%), harvest queues one `<slug>-cf` confirm: the parent's code plus a trailing
comment (okc dedups identical bytes within a mode, check after check, so without it the judge
returns the old job). Confirms run first, as checks only (`prepare-run` refuses a scored run for
one), within `CONFIRM_SHARE` (25%) of the day's checks, carved out of the non-rewrite side. Their
harvest pools the two runs; a pooled gain whose confirm run alone also beats the best moves the
parent to `done/gain` (`gain-pooled`) for the normal promote path, which lands it with the pooled
number; logged in DECISIONS.md. A parent landed by hand meanwhile is left where it is. A
deduplicated rerun pools nothing. `churn.py reclassify` shows what the rule would change on past
results without moving them.

## What to tell the user each tick

Three lines unless something needs them: what was harvested (outcome and mechanism verdict per
slug, the CLAIMS tag recorded for it, and for a confirm the pooled verdict of its parent), what the
adversary said (verdict per slug, companions queued, holds on claims or freshness), the freshness
gate's holds (slug and `hold_kind`), diagnoses written (strategy and top action), what was dispatched (slug, mode, job id),
and the budget line per problem (checks left / scored left /
in flight / shortfall + reason). `needs_human` items and `unmerged` gains always get their own
line with the file path.

## Reviewing one hypothesis without dispatching it

`/churnsubs review <problem> <slug>` runs step 4 alone on a queued hypothesis (no lock, no
claim), so a batch from /hypothesize can be reviewed and its companions queued before the loop
spends anything.

## Records kept

Every harvested job's raw okc record and profile files are archived to
`<p>/hypotheses/archive/<slug>/` (`<job>.okc.json.gz`, `.profile.txt`, `.profile.json.gz`), because the
profiles otherwise vanish with the worktree. Keep everything with reproducibility value, not full
transcripts:
- the filled adversary prompt: `review-record ... --prompt <file>` archives it beside the raw review JSON;
- the per-pick implementer brief: write it to a file, then `churn.py archive-file <p> <slug> <file> --as implement-brief.md`;
- every judgement call (hold, deferral, merge by hand, overriding a reviewer, a deviation you
  approved): `churn.py note <p> <slug> "<what and why>"` appends to `hypotheses/DECISIONS.md` (hold
  writes there itself);
- each tick's full plan and report is saved to `hypotheses/ticks/<ts>.json` by `tick`.
Cross-problem lessons go in `hypotheses/LESSONS.md`:
after a harvest that teaches something general (a measured hardware fact, a knob that is dead), add
one line there with the slug that showed it.

## Reading results by hand

- `<p>/hypotheses/INDEX.md` — every hypothesis, sortable columns; `hypotheses/INDEX.md` — roll-up
  and today's budget. `<p>/hypotheses/FINDINGS.md` — learnings by area, open follow-ups,
  do-not-retry list, where the current best spends its time.
- `<p>/hypotheses/done/<outcome>/<slug>.{md,patch,result.json}` and the branch
  `hyp/<p>/<slug>` (kept forever; worktree removed after harvest, kept for `unmerged` gains).
- `okc status <job>` and `profiles/submission.<job>.txt` in the worktree while it exists.
