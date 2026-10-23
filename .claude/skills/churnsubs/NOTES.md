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
- End-to-end on one real check: see the first hypothesis in muse_bf16_proj/hypotheses/.

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
