# Campaigns

A **campaign** is a multi-day, multi-problem program of work that is larger than one hypothesis.
It has its own plan, findings, running log, code branch and run archive.

A **hypothesis** (`<problem>/hypotheses/`) is one attributable change, minted by `/hypothesize` and run by
`/churnsubs`. A campaign may pause churn while it runs, and it may later hand pieces back to churn as
hypotheses. Campaign-level plans live here, not in `hypotheses/`.

## Layout

One directory per campaign, named `campaigns/<YYYY-MM-DD>-<slug>/` (UTC start date):

| path | contents |
|---|---|
| `PLAN.md` | the approved plan. Frontmatter: `id`, `status`, `objective`, `kpi`, `branch`, `worktree`, `start`, `deadline` |
| `FINDINGS.md` | the evidence base the plan rests on, kept current as checks land |
| `research/` | source reports (verbatim), analysis scripts with their printed outputs, data snapshots |
| `LOG.md` | append-only running state: every check and scored run, its job, the rows that moved, the decision. A new session resumes from here |
| `runs/<job>.*` | okc json, profile and log for the campaign's runs (like churn's `archive/`) |
| `RETRO.md` | written when the campaign closes |

## Conventions

- Status is one of `planned`, `active`, `paused`, `closed`. Keep it current in the PLAN.md frontmatter and in `INDEX.md`.
- Code lives on a `campaign/<slug>` branch in `.worktrees/<slug>`. Campaign docs and run archives are committed on master from the main tree.
- Every okc run still needs its attempt file (`<problem>/attempts/`) and the rules checklist, as for churn.
- When a milestone lands on master, record it in `hypotheses/DECISIONS.md` and in the campaign `LOG.md`.
- `INDEX.md` lists every campaign with its status and best results.
