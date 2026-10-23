# muse_attention_decode · 2026-10-23T2105Z · check · starter-calibration

*Backfilled 2026-10-23T2135Z from `~/.config/okc/jobs.json` and git history; no explanation was
written before this run.*

- **Job:** `a8d647772fa945e5`
- **File:** `muse_attention_decode/submission.py` at git `ab21d70` (the committed starter kit)

## Goal
Calibration: establish that the untouched starter passes the 16 tests and record its time against SGLang, as the baseline to measure every later attempt against.

## What changed, and where
Nothing: the file as `okc init` wrote it (plus the commit's own edits, if any).

## Rules check
Not performed before the run. The starter is a plain PyTorch reference and uses no caching,
no reduced precision and no harness access, so no rule was at risk.

## Result (after the run)
See `okc status a8d647772fa945e5`; the `profiles/` directory beside submission.py holds the profile once
the run is done.
