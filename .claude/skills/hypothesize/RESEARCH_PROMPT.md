# Daily clean-context diagnosis for {PROBLEM}

You have **no** knowledge of today's attempts beyond what is on disk, and that is the point: the
loop that minted and ran them is biased toward its own recent ideas. Form every conclusion from the
files. Do not write kernel code, do not edit any file except the one output below, and do not run
okc, git commits or churn.py subcommands that change state. Read-only `churn.py`/`hyp.py` calls
(`hyp.py list`, `hyp.py context`, `churn.py reclassify`) are fine.

Repo: {REPO}. Adapted from the research agent of github.com/Dogacel/auto-gpu-kernel, whose loop
found that a fresh context reading only the artifacts is better at proposing a pivot than the
optimizer is at proposing one to itself.

## Read

- `{PROBLEM}/hypotheses/FINDINGS.md` (current best, per-shape profile vs SGLang and SoL, by area,
  open follow-ups, Do not try), `{PROBLEM}/hypotheses/INDEX.md`, `{PROBLEM}/TASK.md`.
- `hypotheses/LESSONS.md` (cross-problem) and this problem's lines in `hypotheses/DECISIONS.md`.
- Every `{PROBLEM}/hypotheses/REWRITE_READ_*.md`, and the previous `DIAGNOSIS.md` if present.
- For the claims you make, the hypothesis files under `{PROBLEM}/hypotheses/done/*/` (including
  their `## Adversarial review` and `## Attribution` sections) and their `.result.json`.
- `rules/technical-rules.llms.txt` before recommending anything near a rule.

## Diagnose: go through each item, check the ones that apply, cite slugs

1. **Repetition loop**: variants of one idea run again and again (cite the slugs).
2. **Local minimum**: many results in one area within the drift margin (churn.py
   `margin()`, ~1.2% for one run) on the same design.
3. **Build-churn wall**: failed/rejected counts by language or toolchain (CUDA load_inline vs
   Triton). Repeated build-fix rounds cause plan amnesia; say whether a lighter abstraction would
   reach the same mechanism.
4. **Wrong bottleneck**: per shape, is the incumbent launch-bound, memory-bound or compute-bound
   against its SoL floor? Are recent hypotheses aimed at the binding one, and at the shapes that
   carry the weight?
5. **Missing fundamental**: a standard technique for this operator that no hypothesis tried.
6. **Ignored recommendations**: open follow-ups, reviewer suggestions or earlier DIAGNOSIS actions
   that were never minted.
7. **Unstacked wins**: per-shape wins on disjoint shapes that were each neutral on the headline
   and were never combined.

Judge each direction by its **ceiling, not its first run**: a rewrite at stage 1 is slower than a
tuned incumbent at stage 20. Recommend a pivot when the incumbent's ceiling (its distance to the
floor that tuning can still close) is lower than an alternative's.

Hard limits on anything you recommend: no top-k or sparse attention, no precision below the rules,
no caching of anything derived from inputs, no host syncs, no work outside the timed call, no
control flow on input values (shapes and `config` fields are fine). Held-out data is rerun.

## Write `{PROBLEM}/hypotheses/DIAGNOSIS.md` (overwrite)

```markdown
---
diagnosed_at: <YYYY-MM-DDTHHMMSSZ, now, UTC>
problem: {PROBLEM}
best: <current best speedup and job>
---

# {PROBLEM} diagnosis

## Diagnosis
2-4 sentences naming the checklist items that apply, with slugs.

## Strategy
One of **pivot** | **refactor** | **targeted**, and why in one sentence.

## Actions (priority ordered; /hypothesize mints these first)
1. **What:** the change, named precisely (kernel, function, tile, shapes). **Why:** the evidence
   (slugs, profile µs, floor). **Impact:** µs on which shapes, weighted. **Prior:** the closest
   earlier slug and what differs.
2. ...

## Do not try
- `<slug>`(s): the mechanism and the measured reason it is dead. Rewrite stages are not dead ends
  because a stage was slow.

## LESSONS edits (proposed; churnsubs applies them)
- retire | merge | correct: `<the line, quoted>` → why (slug or file evidence).
```

## Return

Under 120 words: the path, the strategy, the top action, and the most important Do-not-try entry.
