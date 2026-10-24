# churnsubs / hypothesize — notes

## Provenance
Built 2026-10-23 from the plan at ~/.claude/plans/i-want-to-attempt-fizzy-locket.md, after the
attempts/ convention and the okc attempt gate were in place. The gate was fixed the same day to
resolve the checkout from the submission file's path (worktrees have a `.git` file).

## Parameters (churn.py constants)
- `MAX_IN_FLIGHT = 8` per problem, `PER_TICK_CAP = 4` per problem, `GLOBAL_TICK_CAP = 6` per tick
  (API limit is 6 submissions/min/token; dispatches are sequential anyway).
- Target = 100% of both pools. Promotion hold `Ps = max(gains pending, 1 if a check is in flight)`
  scored runs, 0 in the last 2 h before the 00:00 UTC reset.
- `SUBMIT_FLOOR = 1.0`: a check-proven gain is merged into master regardless, but a scored run is
  spent on it only if its CI low end is ≥ 1.0x.
- `STALE_H = 6` (queued run stops counting as in flight), `LOST_H = 24` (judge-lost → failed),
  `LOCK_STALE_H = 2`, attempt-file freshness 2 h (the gate's window).
- Classification noise: `max(1.5%, half the CI width relative to the speedup)`; a shape moved if
  its delta vs base is outside ±2·noise.

## Validation (2026-10-23)
- okc_lint: clean on all five starters; rejects `.item()`, `subprocess`, `from reference`; warns on
  fp16 casts and on `synchronize()` outside kernel_fn's call graph (prepare is untimed).
- best-init seeded four problems from today's finished starter checks; prefill still queued.
- budget at 02:00/14:00/22:00 UTC with the live quotas gives pace 2/4/28 and N bounded by queue.
- classify: SGLang's own baseline run → gain over the 0.902x starter best; starter vs that
  baseline as base → regression, "lost on router verify_*, o verify_*, qkv_gate verify_*".
- End-to-end on one real check (2026-10-23T2220Z): `sglang-gemm-dispatch-parity` claimed into
  `.worktrees/muse_bf16_proj/...` on `hyp/muse_bf16_proj/sglang-gemm-dispatch-parity`, implemented,
  lint clean, commit-impl + prepare-run, the gate allowed the printed `okc check`, job
  `017804b34fed40b3` recorded on branch and master. Recovery drill: job id erased from the running
  md → `recover` re-adopted it from ~/.config/okc/jobs.json. Second tick: pending, N=0, shortfall
  reason names /hypothesize. Lock: a second preflight during a tick exits 2.
- Found and fixed during the run: the gate's filename regex only took HHMM; now HHMM[SS].

## Adversarial review (added 2026-10-23)
Every hypothesis is reviewed by a fresh `model: opus` subagent before `claim` (which refuses
`review != proceed`). Prompt: `ADVERSARY_PROMPT.md`. The reviewer's defined job: justification,
confounding by landed hypotheses (`churn.py landed`), and whether an A/B companion (ab-revert /
ab-tune / control, `controls_for`, `depends_on` the parent, based on the parent's `impl_commit`)
is needed for the single result to be interpretable. `review-record` writes the verdict section,
sets `review`/`hold`, and queues companions through `hyp.py new`. Companions are eligible as
soon as the parent is implemented (even if the parent later regresses: that is what they
explain), but not if the parent failed or was rejected.

## Normalized duplicate check (added 2026-10-23)
`commit-impl` rejects, without spending a run, an implementation whose docstring-, annotation-
and TypedDict-stripped AST (common local names collapsed) equals SGLang's `baseline.py` or an
already-run hypothesis. Motivated by the first adversarial review: the parity hypothesis was
baseline.py with different prose, and the judge's byte-level dedup did not catch it. A `control`
companion is exempt. Renames beyond the collapsed set still defeat it; the adversary is the
backstop.

## Open questions
- `okc status` has no `problem` field; the hypothesis file is the source of truth for it.
- `content_hash` is not sha256 of the file; local duplicate detection uses sha256 of the file in
  the worktree, judge duplicates are reported by okc (`duplicate_of`).
- Whether a scored submit of code already judged by a check is deduplicated by the judge is
  unknown; harvest-promotions treats `duplicate` as passed and snapshots the leaderboard.
- best-init attributes a seeded run to the newest master commit at or before its submit time;
  that is right for the starter runs and approximate in general.

## Delegation
Schedulable: `churn.py tick` is read-mostly and idempotent; the dispatch steps need the LLM and
must run inside a Claude session (`/loop churnsubs`). Never schedule anything that spends a run
outside the skill.

## Frontmatter flow lists (fixed 2026-10-23)
`_scalar` now parses `[a, b]` flow lists, so `speedup_ci: [lo, hi]` reads back as floats.
Before the fix `promote` crashed on the first real gain ("'<=' not supported between
instances of 'str' and 'float'") and the gain sat in `done/gain` unmerged for a tick.
