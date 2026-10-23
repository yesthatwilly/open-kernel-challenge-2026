# muse_bf16_proj · 2026-10-23T2105Z · check · starter-calibration

*Backfilled 2026-10-23T2135Z from `~/.config/okc/jobs.json` and git history; no explanation was
written before this run.*

- **Job:** `366189f0f1114da1`
- **File:** `muse_bf16_proj/submission.py` at git `ab21d70` (the committed starter kit)

## Goal
Calibration: establish that the untouched starter passes the 16 tests and record its time against SGLang, as the baseline to measure every later attempt against.

## What changed, and where
Nothing: the file as `okc init` wrote it (plus the commit's own edits, if any).

## Rules check
Not performed before the run. The starter is a plain PyTorch reference and uses no caching,
no reduced precision and no harness access, so no rule was at risk.

## Result (after the run)
Passed all 9 tests, no flags. Speedup 0.902x, CI [0.889, 0.916]: the plain-PyTorch starter is about 10% slower than SGLang's kernel (scored 5355.2 µs vs 4830.8 µs). That is the floor every later bf16_proj attempt must beat.
